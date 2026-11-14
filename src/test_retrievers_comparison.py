import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
from src.embeddings import get_embedder
from src.retrievers import DenseRetriever, LexicalRetriever, HybridRetriever

def main():
    print("=== TESTING & COMPARING 3 RETRIEVAL METHODS ===")
    
    # 1. Load data
    with open(PROJECT_ROOT / "data" / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    doc_vecs = np.load(PROJECT_ROOT / "data" / "vectors.npy")
    embedder = get_embedder()
    
    # 2. Initialize the 3 retrievers
    dense = DenseRetriever(chunks=chunks, doc_vectors=doc_vecs, embedder=embedder)
    lexical = LexicalRetriever(chunks=chunks)
    hybrid = HybridRetriever(dense_retriever=dense, lexical_retriever=lexical)
    
    # Test Query 1: Conceptual / Semantic
    q1 = "What was the debt-to-income ratio of IDA-eligible countries in 2025?"
    print(f"\n[Query 1]: {q1}")
    print("\n--- Dense Retrieval (Top 3) ---")
    for r in dense.search(q1, top_k=3):
        print(f"  [{r['score']:.4f}] {r['id']} (Page {r['page']}) -> {r['text'][:90]}...")
        
    print("\n--- Lexical BM25 (Top 3) ---")
    for r in lexical.search(q1, top_k=3):
        print(f"  [{r['score']:.4f}] {r['id']} (Page {r['page']}) -> {r['text'][:90]}...")
        
    print("\n--- Hybrid RRF (Top 3) ---")
    for r in hybrid.search(q1, top_k=3):
        print(f"  [{r['score']:.4f}] {r['id']} (Page {r['page']}) -> {r['text'][:90]}...")

    # Test Query 2: Exact Keyword / Specific Name
    q2 = "Which borrower led the increase in borrowing levels in 2025 among MICs other than China?"
    print(f"\n[Query 2]: {q2}")
    print("\n--- Hybrid RRF (Top 3) ---")
    for r in hybrid.search(q2, top_k=3):
        print(f"  [{r['score']:.4f}] {r['id']} (Page {r['page']}) -> {r['text'][:90]}...")

if __name__ == "__main__":
    main()
