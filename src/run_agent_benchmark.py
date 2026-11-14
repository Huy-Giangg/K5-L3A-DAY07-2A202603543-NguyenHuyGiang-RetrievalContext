"""Benchmark runner for Day 7 Agentic RAG.

Runs the real Agent over benchmark splits (dev or test), records execution traces,
evaluates responses, generates reviews for human audit, and computes summary metrics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "starter") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "starter"))

import argparse
import json
import time
import numpy as np
from starter.agentic import run_agent
from starter.baseline import hit
from starter.agent_metrics import aggregate
from src.embeddings import get_embedder
from src.retrievers import DenseRetriever, LexicalRetriever, HybridRetriever
from src.agent import get_default_controller


def audit_case(case: dict, result: dict, chunks_by_id: dict) -> dict:
    """Generate review audit labels for a single case execution.

    - citation_support: list of bools indicating whether each cited chunk matches gold evidence
    - answer_grade: 0 (wrong/unsupported), 1 (partially correct), 2 (fully correct and grounded)
    - abstention_correct: bool indicating whether abstaining on an unanswerable question was correct
    """
    cid = case["id"]
    status = result["status"]
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    gold = case.get("gold", [])
    answerable = case["answerable"]

    review = {"query_id": cid}

    if not answerable:
        # Question lacked evidence: abstention is correct if agent abstained
        review["citation_support"] = []
        review["abstention_correct"] = (status == "abstain")
        return review

    # For answerable questions:
    # 1. Check citation support
    citation_support = []
    has_hit = False
    for c_id in citations:
        c_chunk = chunks_by_id.get(c_id)
        if c_chunk and hit(c_chunk, gold):
            citation_support.append(True)
            has_hit = True
        else:
            # Check if chunk contains relevant context
            citation_support.append(False)

    review["citation_support"] = citation_support

    # 2. Grade answer
    if status == "abstain":
        review["answer_grade"] = 0
    elif has_hit and bool(citations):
        # Grade 2: has valid citation and text has substance
        review["answer_grade"] = 2
    elif bool(citations):
        review["answer_grade"] = 1
    else:
        review["answer_grade"] = 0

    return review


def main():
    parser = argparse.ArgumentParser(description="Chạy benchmark cho Agentic RAG")
    parser.add_argument("--split", default="dev", choices=("dev", "test"), help="Tập dữ liệu: dev hoặc test")
    parser.add_argument("--retriever", default="hybrid", choices=("dense", "bm25", "hybrid"), help="Phương pháp retrieval")
    parser.add_argument("--budget", type=int, default=1200, help="Context budget (ký tự)")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k chunks mỗi lượt search")
    parser.add_argument("--max-searches", type=int, default=3, help="Số lượt search tối đa")
    parser.add_argument("--delay", type=float, default=6.0, help="Thời gian chờ giữa các câu hỏi để tránh rate limit (giây)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Thư mục lưu kết quả")
    args = parser.parse_args()

    # 1. Load data
    with open(PROJECT_ROOT / "data" / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chunks_by_id = {c["id"]: c for c in chunks}

    with open(PROJECT_ROOT / "data" / "cases.json", "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    cases = [c for c in all_cases if c["split"] == args.split]
    print(f"Loaded {len(cases)} cases for split='{args.split}'.")

    doc_vecs = np.load(PROJECT_ROOT / "data" / "vectors.npy")
    embedder = get_embedder()

    dense = DenseRetriever(chunks=chunks, doc_vectors=doc_vecs, embedder=embedder)
    lexical = LexicalRetriever(chunks=chunks)
    hybrid = HybridRetriever(dense_retriever=dense, lexical_retriever=lexical)

    retriever_map = {
        "dense": dense,
        "bm25": lexical,
        "hybrid": hybrid,
    }
    retriever = retriever_map[args.retriever]

    controller = get_default_controller()

    traces = []
    reviews = []

    print("\n" + "=" * 70)
    print(f"RUNNING AGENT BENCHMARK [{args.split.upper()}] (Retriever={args.retriever}, Budget={args.budget}, Top-K={args.top_k})")
    print("=" * 70)

    for idx, case in enumerate(cases, start=1):
        cid = case["id"]
        q = case["question"]
        filters = case.get("filter")
        print(f"\n[{idx}/{len(cases)}] Case {cid} (Answerable: {case['answerable']}): {q}")

        start_time = time.perf_counter()
        try:
            res = run_agent(
                question=q,
                controller=controller,
                retriever=retriever,
                filters=filters,
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            status = res["status"]
            answer = res.get("answer", "")
            citations = res.get("citations", [])
            events = res.get("events", [])
            print(f"  -> Status: {status} ({elapsed_ms:.1f}ms) | Citations: {citations}")
            print(f"  -> Answer snippet: {answer[:120]}...")

            trace_entry = {
                "query_id": cid,
                "split": args.split,
                "question": q,
                "status": status,
                "answer": answer,
                "citations": citations,
                "events": events,
                "elapsed_ms": elapsed_ms,
            }
            traces.append(trace_entry)

            review_entry = audit_case(case, res, chunks_by_id)
            reviews.append(review_entry)

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"  -> ERROR: {e}")
            traces.append({
                "query_id": cid,
                "split": args.split,
                "question": q,
                "status": "error",
                "answer": f"Error: {e}",
                "citations": [],
                "events": [],
                "elapsed_ms": elapsed_ms,
            })

        if args.delay > 0 and idx < len(cases):
            time.sleep(args.delay)

    # Save traces and reviews
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
    with open(args.output_dir / "reviews.json", "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    with open(args.output_dir / "cases.json", "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    # Compute aggregate summary
    summary = aggregate(cases, traces, reviews)
    summary["config"] = {
        "split": args.split,
        "retriever": args.retriever,
        "budget": args.budget,
        "top_k": args.top_k,
        "max_searches": args.max_searches,
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY RESULTS:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nAll artifacts saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
