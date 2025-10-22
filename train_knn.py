# backend/train_knn.py
"""
从 data/normal 文件夹读取 RGB 时频图（png/jpg），使用 resnet_model.pth 提取 embedding，
训练 KNNDetector 并保存到 models/knn_detector.pkl
"""
import os
import pickle
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from anomaly_detector import ResNetEmbedding, IMG_TRANSFORM, extract_embedding_from_tensor, KNNDetector

DATA_DIR = "backend/data/normal"
MODEL_PATH = "backend/models/resnet_model.pth"
OUT_PATH = "backend/models/knn_detector.pkl"

def list_image_files(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    x = IMG_TRANSFORM(img).unsqueeze(0)
    return x

def main(data_dir=DATA_DIR, model_path=MODEL_PATH, out_path=OUT_PATH, device=None, k=5):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load resnet
    model = ResNetEmbedding()
    state = torch.load(model_path, map_location=device)
    # Try to adapt if saved state dict has prefixes
    try:
        model.load_state_dict(state)
    except Exception:
        # assume it was a dict with 'model_state_dict'
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    model.to(device)
    model.eval()

    files = list_image_files(data_dir)
    if len(files) == 0:
        raise RuntimeError(f"No image files found in {data_dir}")

    embeddings = []
    for p in files:
        x = load_image_tensor(p).to(device)
        with torch.no_grad():
            emb = model(x).cpu().numpy().reshape(-1)
        embeddings.append(emb)
    embeddings = np.stack(embeddings, axis=0)
    print("Extracted embeddings:", embeddings.shape)

    knn = KNNDetector(ref_embeddings=embeddings, k=k)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({'knn': knn, 'ref_embeddings': embeddings}, f)

    print("Saved KNN detector to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--out_path", default=OUT_PATH)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(data_dir=args.data_dir, model_path=args.model_path, out_path=args.out_path, k=args.k)
