# backend/demo_infer_rgb.py
"""
Standalone script to run inference on a single RGB spectrogram image file
and print anomaly score and a simple message.
"""
import sys
import numpy as np
from PIL import Image
import torch
import pickle
from anomaly_detector import ResNetEmbedding, extract_embedding_from_tensor, IMG_TRANSFORM
import os

RESNET_PATH = "backend/models/resnet_model.pth"
KNN_PATH = "backend/models/knn_detector.pkl"

def pil_to_tensor(img_pil):
    return IMG_TRANSFORM(img_pil).unsqueeze(0)

def load_models(device="cpu"):
    resnet = ResNetEmbedding()
    state = torch.load(RESNET_PATH, map_location=device)
    try:
        resnet.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'model_state_dict' in state:
            resnet.load_state_dict(state['model_state_dict'])
        else:
            resnet.load_state_dict(state)
    resnet.to(device).eval()
    with open(KNN_PATH, "rb") as f:
        knn_bundle = pickle.load(f)
    knn = knn_bundle.get('knn') if isinstance(knn_bundle, dict) else knn_bundle
    return resnet, knn

def infer(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet, knn = load_models(device=device)
    img = Image.open(image_path).convert("RGB")
    x = pil_to_tensor(img).to(device)
    with torch.no_grad():
        emb = resnet(x).cpu().numpy().reshape(-1)
    score = knn.score(emb)
    print("Anomaly score:", score)
    return score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage")
        sys.exit(1)
    infer(sys.argv[1])

