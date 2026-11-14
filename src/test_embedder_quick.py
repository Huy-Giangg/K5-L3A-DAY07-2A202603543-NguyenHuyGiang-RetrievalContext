import sys
from pathlib import Path

# Add project root to sys.path so running directly or via VS Code Run button works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
from src.embeddings import get_embedder

def main():
    print("Loading embedder...")
    embedder = get_embedder()
    
    query = "What was the debt-to-income ratio of IDA-eligible countries in 2025?"
    print(f"Embedding query: '{query}'")
    q_vec = np.array(embedder.embed_query(query), dtype=np.float32)
    print(f"Query vector generated! Dimension: {len(q_vec)}")
    
    chunks_path = PROJECT_ROOT / "data" / "chunks.json"
    vectors_path = PROJECT_ROOT / "data" / "vectors.npy"
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    doc_vecs = np.load(vectors_path)
    print(f"Loaded {len(chunks)} chunks and {doc_vecs.shape} document vectors.")
    
    # Cosine similarity via dot product (both query and doc vectors are normalized)
    scores = np.dot(doc_vecs, q_vec)
    top_indices = np.argsort(scores)[::-1][:3]
    
    print("\n--- TOP 3 CHUNKS RETRIEVED (DENSE RETRIEVAL) ---")
    for rank, idx in enumerate(top_indices, start=1):
        c = chunks[idx]
        score = scores[idx]
        print(f"#{rank} [Score: {score:.4f}] Chunk ID: {c['id']} | Page: {c['page']}")
        preview = c['text'].replace('\n', ' ')[:120]
        print(f"    Snippet: {preview}...")

if __name__ == "__main__":
    main()
