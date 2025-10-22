# ingest_docs.py
import os
import glob
import json
from sentence_transformers import SentenceTransformer
import faiss

EMB_MODEL = "all-MiniLM-L6-v2"
OUT_PREFIX = "faiss_index"

def load_texts(folder="docs"):
    docs = []
    for path in glob.glob(os.path.join(folder, "**", "*.md"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        docs.append({"path": path, "text": txt})
    return docs

def build_index(docs, out_prefix=OUT_PREFIX, emb_model=EMB_MODEL):
    if len(docs) == 0:
        raise ValueError("no docs found")
    model = SentenceTransformer(emb_model)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine
    index.add(embeddings)

    faiss.write_index(index, out_prefix + ".index")
    with open(out_prefix + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"saved faiss index: {out_prefix}.index and meta: {out_prefix}_meta.json")

if __name__ == "__main__":
    docs = load_texts("docs")
    build_index(docs)
