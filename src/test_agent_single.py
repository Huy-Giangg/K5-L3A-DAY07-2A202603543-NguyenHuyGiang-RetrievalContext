import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
from starter.agentic import run_agent
from src.embeddings import get_embedder
from src.retrievers import DenseRetriever, LexicalRetriever, HybridRetriever
from src.agent import get_default_controller


def main():
    print("=== TESTING AGENTIC RAG WITH GEMINI ===")
    
    # 1. Load components
    with open(PROJECT_ROOT / "data" / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    doc_vecs = np.load(PROJECT_ROOT / "data" / "vectors.npy")
    embedder = get_embedder()
    
    dense = DenseRetriever(chunks=chunks, doc_vectors=doc_vecs, embedder=embedder)
    lexical = LexicalRetriever(chunks=chunks)
    hybrid = HybridRetriever(dense_retriever=dense, lexical_retriever=lexical)
    
    controller = get_default_controller()
    
    # Test 1: Answerable Question
    q1 = "What was the debt-to-income ratio of IDA-eligible countries in 2025?"
    print(f"\n[TEST 1 - Answerable]: {q1}")
    result1 = run_agent(q1, controller, hybrid, max_searches=3, top_k=3, budget=1200)
    print(f"Status:    {result1['status']}")
    print(f"Answer:    {result1['answer']}")
    print(f"Citations: {result1['citations']}")
    print(f"Events:    {len(result1['events'])} steps executed")

    # Test 2: Unanswerable Question (Lack of evidence)
    q2 = "What was the exact external debt stock of Vietnam at the end of 2025 according to the report?"
    print(f"\n[TEST 2 - Unanswerable]: {q2}")
    result2 = run_agent(q2, controller, hybrid, max_searches=3, top_k=3, budget=1200)
    print(f"Status:    {result2['status']}")
    print(f"Answer:    {result2['answer']}")
    print(f"Citations: {result2['citations']}")
    print(f"Events:    {len(result2['events'])} steps executed")


if __name__ == "__main__":
    main()
