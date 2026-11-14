"""Benchmark script comparing 3 retrieval methods on the Dev benchmark.

Evaluates:
- Hit@1: Gold evidence in rank 1
- Hit@3: Gold evidence in top 3
- Context Hit: Gold evidence included within context budget (800 chars)
- Latency: median time per query (ms)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import statistics
import time
import numpy as np
from starter.baseline import hit, pack_context
from src.embeddings import get_embedder
from src.retrievers import DenseRetriever, LexicalRetriever, HybridRetriever


def evaluate_retriever(retriever, questions, budget=800, top_k=3):
    traces = []
    latencies = []
    
    for q in questions:
        filters = q.get("filter")
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        
        context, ids = pack_context(results, budget=budget)
        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]
        
        hit1 = bool(marked and marked[0]["hit"])
        hit3 = any(c["hit"] for c in marked)
        context_hit = any(c["hit"] and c["id"] in ids for c in marked)
        
        traces.append({
            "query_id": q["id"],
            "question": q["question"],
            "filter": filters,
            "results": marked,
            "context": context,
            "context_ids": ids,
            "elapsed_ms": elapsed_ms,
            "hit1": hit1,
            "hit3": hit3,
            "context_hit": context_hit,
        })
        
    n = len(traces)
    summary = {
        "n": n,
        "hit1": sum(t["hit1"] for t in traces),
        "hit1_rate": sum(t["hit1"] for t in traces) / n if n else 0,
        "hit3": sum(t["hit3"] for t in traces),
        "hit3_rate": sum(t["hit3"] for t in traces) / n if n else 0,
        "context_hit": sum(t["context_hit"] for t in traces),
        "context_hit_rate": sum(t["context_hit"] for t in traces) / n if n else 0,
        "median_ms": statistics.median(latencies) if latencies else 0,
        "mean_ms": statistics.mean(latencies) if latencies else 0,
    }
    return summary, traces


def main():
    with open(PROJECT_ROOT / "data" / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(PROJECT_ROOT / "data" / "questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    dev_questions = [q for q in questions if q["split"] == "dev"]
    doc_vecs = np.load(PROJECT_ROOT / "data" / "vectors.npy")
    embedder = get_embedder()
    
    dense = DenseRetriever(chunks=chunks, doc_vectors=doc_vecs, embedder=embedder)
    lexical = LexicalRetriever(chunks=chunks)
    hybrid = HybridRetriever(dense_retriever=dense, lexical_retriever=lexical)
    
    retrievers = {
        "Dense Retrieval": dense,
        "BM25 Lexical": lexical,
        "Hybrid (RRF)": hybrid,
    }
    
    results = {}
    print("\n" + "=" * 70)
    print(f"BENCHMARK RETRIEVAL ON DEV SET ({len(dev_questions)} questions)")
    print("=" * 70)
    
    for name, r in retrievers.items():
        summary, traces = evaluate_retriever(r, dev_questions)
        results[name] = summary
        print(f"\n--- {name} ---")
        print(f"  Hit@1:       {summary['hit1']}/{summary['n']} ({summary['hit1_rate']*100:.1f}%)")
        print(f"  Hit@3:       {summary['hit3']}/{summary['n']} ({summary['hit3_rate']*100:.1f}%)")
        print(f"  Context Hit: {summary['context_hit']}/{summary['n']} ({summary['context_hit_rate']*100:.1f}%)")
        print(f"  Latency:     {summary['median_ms']:.2f} ms (median)")

    # Save results to results/retrieval_comparison.json
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "retrieval_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 70)
    print(f"Saved benchmark results to {out_dir / 'retrieval_comparison.json'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
