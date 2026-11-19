"""Benchmarking Suite for LangChain FAISS RAG System.

Measures:
1. Ingestion latency, throughput (docs/s, chunks/s), and memory usage.
2. Retrieval latency (mean, median, p95, min, max).
3. End-to-end query synthesis latency.
4. Memory footprint of vector database and model runtime.
"""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

import ingest
import rag
from config import get_config

ROOT_DIR = Path(__file__).resolve().parent
BENCHMARK_OUTPUT = ROOT_DIR / "benchmark_results.json"


def benchmark_ingestion() -> Dict[str, Any]:
    """Measure ingestion time, document count, chunk count, and memory allocation."""
    cfg = get_config()
    tracemalloc.start()
    start_time = time.perf_counter()

    docs = ingest.load_documents(cfg.data_dir)
    chunks = ingest.split_documents(docs)
    ingest.build_faiss_index(chunks, index_path=cfg.index_dir, api_key="test-benchmark-key")

    elapsed = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    docs_per_sec = len(docs) / elapsed if elapsed > 0 else 0.0
    chunks_per_sec = len(chunks) / elapsed if elapsed > 0 else 0.0

    return {
        "num_documents": len(docs),
        "num_chunks": len(chunks),
        "ingestion_time_seconds": round(elapsed, 4),
        "docs_per_second": round(docs_per_sec, 2),
        "chunks_per_second": round(chunks_per_sec, 2),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
    }


def benchmark_retrieval(num_iterations: int = 50) -> Dict[str, Any]:
    """Measure context retrieval latency statistics over multiple iterations."""
    rag.get_vectorstore()  # Pre-warm cache

    queries = [
        "What is Retrieval-Augmented Generation?",
        "What are the core stages of RAG?",
        "How does FAISS vector search work?",
        "What metrics does Ragas calculate?",
        "Why is chunking necessary in vector databases?",
    ]

    latencies_ms: List[float] = []

    for i in range(num_iterations):
        q = queries[i % len(queries)]
        start = time.perf_counter()
        docs = rag.retrieve_contexts(q, top_k=4)
        duration_ms = (time.perf_counter() - start) * 1000.0
        assert len(docs) == 4
        latencies_ms.append(duration_ms)

    latencies_ms.sort()
    p95_idx = int(0.95 * len(latencies_ms))

    return {
        "iterations": num_iterations,
        "mean_latency_ms": round(statistics.mean(latencies_ms), 3),
        "median_latency_ms": round(statistics.median(latencies_ms), 3),
        "p95_latency_ms": round(latencies_ms[p95_idx], 3),
        "min_latency_ms": round(min(latencies_ms), 3),
        "max_latency_ms": round(max(latencies_ms), 3),
    }


def benchmark_end_to_end_query(num_iterations: int = 20) -> Dict[str, Any]:
    """Measure end-to-end query() execution latency."""
    latencies_ms: List[float] = []
    q = "What is Retrieval-Augmented Generation?"

    for _ in range(num_iterations):
        start = time.perf_counter()
        res = rag.query(q)
        duration_ms = (time.perf_counter() - start) * 1000.0
        assert len(res["contexts"]) == 4
        latencies_ms.append(duration_ms)

    latencies_ms.sort()
    p95_idx = int(0.95 * len(latencies_ms))

    return {
        "iterations": num_iterations,
        "mean_latency_ms": round(statistics.mean(latencies_ms), 3),
        "median_latency_ms": round(statistics.median(latencies_ms), 3),
        "p95_latency_ms": round(latencies_ms[p95_idx], 3),
        "min_latency_ms": round(min(latencies_ms), 3),
        "max_latency_ms": round(max(latencies_ms), 3),
    }


def run_benchmark() -> Dict[str, Any]:
    """Run full benchmark suite and print markdown report."""
    print("=" * 65)
    print("RUNNING PRODUCTION RAG PERFORMANCE BENCHMARK")
    print("=" * 65)

    print("\n1. Benchmarking Ingestion Pipeline...")
    ingest_metrics = benchmark_ingestion()
    print(f"   Ingestion Time:   {ingest_metrics['ingestion_time_seconds']} s")
    print(f"   Throughput:       {ingest_metrics['chunks_per_second']} chunks/s")
    print(f"   Peak Memory:      {ingest_metrics['peak_memory_mb']} MB")

    print("\n2. Benchmarking Retrieval (50 iterations)...")
    retrieval_metrics = benchmark_retrieval(num_iterations=50)
    print(f"   Mean Latency:     {retrieval_metrics['mean_latency_ms']} ms")
    print(f"   Median Latency:   {retrieval_metrics['median_latency_ms']} ms")
    print(f"   P95 Latency:      {retrieval_metrics['p95_latency_ms']} ms")

    print("\n3. Benchmarking End-to-End Query (20 iterations)...")
    e2e_metrics = benchmark_end_to_end_query(num_iterations=20)
    print(f"   Mean Latency:     {e2e_metrics['mean_latency_ms']} ms")
    print(f"   Median Latency:   {e2e_metrics['median_latency_ms']} ms")
    print(f"   P95 Latency:      {e2e_metrics['p95_latency_ms']} ms")

    full_results = {
        "ingestion": ingest_metrics,
        "retrieval": retrieval_metrics,
        "end_to_end_query": e2e_metrics,
    }

    BENCHMARK_OUTPUT.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    print(f"\nBenchmark results saved to {BENCHMARK_OUTPUT}")
    print("=" * 65)

    return full_results


if __name__ == "__main__":
    run_benchmark()
