# retriever.py
import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np

EMB_MODEL = "all-MiniLM-L6-v2"

class FaissRetriever:
    def __init__(self, index_path="faiss_index.index", meta_path="faiss_index_meta.json", emb_model=EMB_MODEL):
        self.model = SentenceTransformer(emb_model)
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def retrieve(self, query, top_k=3):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.meta):
                results.append(self.meta[idx]["text"])
        return results
