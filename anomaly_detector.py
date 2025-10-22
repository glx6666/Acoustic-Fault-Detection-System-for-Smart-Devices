
"""
ResNetEmbedding: 用于从 RGB 时频图提取 embedding
KNNDetector: 近邻异常检测器（封装 sklearn NearestNeighbors）
此模块同时包含命令行接口，用于：
 - 从 data/normal/*.png 提取 embedding 并训练 KNN
 - 保存 knn_detector.pkl 和 threshold.txt
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import pickle

# === 配置 ===
IMAGE_SIZE = (224, 224)   # ResNet 默认输入
EMBED_DIM = 128           # projection dim
K = 5                     # KNN中的k
THRESHOLD_PERCENTILE = 99  # 用 normal scores 的 99th percentile 做阈值（可调整）

# === 图像预处理 ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    t = transform(img)  # (3,H,W)
    return t.unsqueeze(0)  # (1,3,H,W)

# === ResNetEmbedding ===
class ResNetEmbedding(nn.Module):
    def __init__(self, backbone="resnet18", embedding_dim=EMBED_DIM, pretrained=False):
        super().__init__()
        if backbone == "resnet18":
            net = models.resnet18(pretrained=pretrained)
        else:
            raise NotImplementedError("only resnet18 supported")
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        # projection head
        self.proj = nn.Linear(feat_dim, embedding_dim)

    def forward(self, x):
        feats = self.backbone(x)           # (B, feat_dim)
        emb = self.proj(feats)            # (B, embedding_dim)
        emb = nn.functional.normalize(emb, dim=-1)
        return emb

# === KNNDetector ===
class KNNDetector:
    def __init__(self, ref_embeddings: np.ndarray = None, k: int = K):
        """
        ref_embeddings: (N, D)
        """
        self.k = k
        self.ref_embeddings = None
        self.nn = None
        if ref_embeddings is not None:
            self.fit(ref_embeddings)

    def fit(self, ref_embeddings: np.ndarray):
        self.ref_embeddings = np.array(ref_embeddings)
        self.nn = NearestNeighbors(n_neighbors=self.k)
        self.nn.fit(self.ref_embeddings)

    def score(self, x: np.ndarray):
        """
        x: (D,) or (1, D)
        return: average distance to k neighbors
        """
        v = x.reshape(1, -1)
        dist, _ = self.nn.kneighbors(v)
        return float(dist.mean())

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

# === 辅助：从文件夹提取 embeddings ===
def extract_embeddings_from_folder(model: ResNetEmbedding, folder: str, device="cpu"):
    files = [os.path.join(folder, p) for p in sorted(os.listdir(folder))]
    embeddings = []
    paths = []
    model.to(device).eval()
    with torch.no_grad():
        for p in files:
            if not p.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            x = load_image_tensor(p).to(device)
            emb = model(x)  # (1, D)
            embeddings.append(emb.cpu().numpy()[0])
            paths.append(p)
    return np.array(embeddings), paths

# === 命令行流程: 训练 KNN、保存 knn_detector.pkl 和 threshold.txt ===
def train_knn_from_normal_images(resnet_ckpt: str = "models/resnet_model.pth",
                                 normal_dir: str = "data/normal",
                                 out_knn_path: str = "models/knn_detector.pkl",
                                 threshold_path: str = "models/threshold.txt",
                                 device: str = None,
                                 k: int = K,
                                 percentile: int = THRESHOLD_PERCENTILE):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # load resnet embedding model
    model = ResNetEmbedding(pretrained=False)
    print("Loading resnet checkpoint:", resnet_ckpt)
    ckpt = torch.load(resnet_ckpt, map_location=device)
    # allow state_dict vs whole model
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        try:
            model.load_state_dict(ckpt)
        except Exception:
            # try compatibility: ckpt may be whole model
            model = ckpt
    model.to(device).eval()

    # extract embeddings
    print("Extracting embeddings from", normal_dir)
    embeddings, paths = extract_embeddings_from_folder(model, normal_dir, device=device)
    if embeddings.shape[0] == 0:
        raise RuntimeError("No images found in normal_dir")

    print(f"Extracted {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

    # build and save knn
    knn = KNNDetector(ref_embeddings=embeddings, k=k)
    print("Saving KNN detector to", out_knn_path)
    knn.save(out_knn_path)

    # compute normal sample self-scores (for threshold)
    scores = []
    for e in embeddings:
        scores.append(knn.score(e))
    scores = np.array(scores)
    thr = float(np.percentile(scores, percentile))
    with open(threshold_path, "w") as f:
        f.write(str(thr))
    print(f"Saved threshold (percentile {percentile}) = {thr} to {threshold_path}")

    # also save embeddings for inspection
    np.savez("models/normal_embeddings.npz", embeddings=embeddings, paths=paths)
    print("Saved normal embeddings to models/normal_embeddings.npz")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet_ckpt", default="models/resnet_model.pth")
    parser.add_argument("--normal_dir", default="data/normal")
    parser.add_argument("--out_knn", default="models/knn_detector.pkl")
    parser.add_argument("--threshold", default="models/threshold.txt")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--percentile", type=int, default=THRESHOLD_PERCENTILE)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_knn_from_normal_images(
        resnet_ckpt=args.resnet_ckpt,
        normal_dir=args.normal_dir,
        out_knn_path=args.out_knn,
        threshold_path=args.threshold,
        device=args.device,
        k=args.k,
        percentile=args.percentile
    )
