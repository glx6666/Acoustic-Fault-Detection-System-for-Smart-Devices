# agent_server.py
import os
import json
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from anomaly_detector import ResNetEmbedding, load_image_tensor, KNNDetector
from retriever import FaissRetriever
from memory import RedisMemory

# === Config ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    print("Warning: OPENAI_API_KEY not set. LLM calls will fail until set.")
openai.api_key = OPENAI_API_KEY

RESNET_CKPT = "models/resnet_model.pth"
KNN_PKL = "models/knn_detector.pkl"
THRESHOLD_PATH = "models/threshold.txt"
FAISS_INDEX = "faiss_index.index"
FAISS_META = "faiss_index_meta.json"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load components (lazy load with try/except to avoid crash if missing)
resnet = None
knn = None
threshold = None
retriever = None

def load_components(device="cpu"):
    global resnet, knn, threshold, retriever
    if resnet is None:
        resnet = ResNetEmbedding(pretrained=False)
        if os.path.exists(RESNET_CKPT):
            ckpt = torch.load(RESNET_CKPT, map_location=device)
            try:
                resnet.load_state_dict(ckpt)
            except Exception:
                # maybe ckpt saved as dict with state_dict
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    resnet.load_state_dict(ckpt["state_dict"])
                else:
                    raise
        resnet.eval()

    if knn is None and os.path.exists(KNN_PKL):
        import pickle
        knn = pickle.load(open(KNN_PKL, "rb"))

    if threshold is None and os.path.exists(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH, "r") as f:
                threshold = float(f.read().strip())
        except Exception:
            threshold = None

    if retriever is None and os.path.exists(FAISS_INDEX) and os.path.exists(FAISS_META):
        retriever = FaissRetriever(index_path=FAISS_INDEX, meta_path=FAISS_META)

# Pydantic model for JSON calls
class Query(BaseModel):
    question: str
    # optionally you can pass an image as base64 or send spec via file upload. For simplicity this accepts None
    # if you want to pass inline spec (2D array), use 'spec' as nested list
    spec: list = None   # optional 2d list representing image pixels / spectrogram

@app.post("/ask")
def ask(req: Query, session_id: str = Form("default")):
    load_components()
    memory = RedisMemory(session_id=session_id)

    # 1) retrieve docs
    docs_text = ""
    if retriever:
        docs = retriever.retrieve(req.question, top_k=3)
        docs_text = "\n---\n".join(docs)
    else:
        docs_text = "(no retriever available)"

    # 2) audio/image analysis (if spec provided)
    anomaly_score = None
    emb = None
    if req.spec:
        # assume req.spec is a nested list representing RGB image pixels or grayscale
        arr = np.array(req.spec, dtype=np.float32)
        # if shape is (H, W, 3) convert to PIL-like tensor; if (H,W) convert to single-channel -> replicate 3
        if arr.ndim == 3 and arr.shape[2] == 3:
            # convert to 0-1 range if given 0-255
            if arr.max() > 2.0:
                arr = arr / 255.0
            # convert to tensor format expected by transform pipeline by saving to temp PIL is simpler
            from PIL import Image
            im = Image.fromarray((arr * 255).astype("uint8"))
            x = load_image_tensor(im) if isinstance(im, str) else None
            # BUT load_image_tensor expects path; so instead transform manually:
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
            x = preprocess(im).unsqueeze(0)
        else:
            # fallback: treat as grayscale image array (H,W)
            from PIL import Image
            arr2 = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            im = Image.fromarray((arr2 * 255).astype("uint8")).convert("RGB")
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
            x = preprocess(im).unsqueeze(0)

        with torch.no_grad():
            emb = resnet(x).numpy()[0]  # (D,)
        if knn:
            anomaly_score = knn.score(emb)
        else:
            anomaly_score = None

    # 3) assemble prompt (include memory)
    history = memory.get_history(limit=10)
    convo = "\n".join([f"{m['role']}: {m['content']}" for m in history])

    prompt = f"""你是一个工业异常检测助手。
历史对话：
{convo}

检索到的知识（Top-3）：
{docs_text}

音频/图像分析结果：
- 异常分数: {anomaly_score}
- 阈值: {threshold}

用户问题：{req.question}

请结合以上信息，给出：
1) 是否异常及原因推断（若无法确定，请回答不知道）
2) 推荐的下一步检查或修复动作（短期/长期）
3) 置信度评估并说明理由
"""

    # 4) call LLM
    if openai.api_key:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是工业异常检测专家助手，回答要简洁、按步骤、给出置信度与理由。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=600,
        )
        answer = resp["choices"][0]["message"]["content"]
    else:
        answer = "OpenAI API key not set on server; cannot call LLM. Provide OPENAI_API_KEY env var."

    # store memory
    memory.append("user", req.question)
    memory.append("assistant", answer)

    return {"answer": answer, "anomaly_score": anomaly_score, "threshold": threshold}

