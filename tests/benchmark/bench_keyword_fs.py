# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Micro-benchmarks for the SQLite FTS5 keyword sidecar (KeywordFS).

Covers indexing throughput, BM25 lookup latency, grep-recall speedup vs.
brute-force scan, and CJK tokenizer overhead.

Run directly:
    python tests/benchmark/bench_keyword_fs.py

This is a standalone script, not a pytest test (no test_ prefix) so it
won't be picked up by the normal test suite.
"""
from __future__ import annotations

import os
import random
import string
import sys
import time
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openviking.storage.keywordfs.keyword_fs import KeywordFS
from openviking.storage.keywordfs.config import KeywordConfig
from openviking.storage.keywordfs.tokenizer import tokenize

ACCOUNT = "bench_acct"
N_WORDS_PER_DOC = 200


def generate_doc(n_words=N_WORDS_PER_DOC, has_cjk=False):
    words = []
    for _ in range(n_words):
        wlen = random.randint(3, 10)
        word = "".join(random.choice(string.ascii_lowercase) for _ in range(wlen))
        words.append(word)
    if has_cjk:
        for _ in range(n_words // 5):
            cjk = "".join(
                chr(0x4E00 + random.randint(0, 0x400))
                for _ in range(random.randint(2, 6))
            )
            words.insert(random.randint(0, len(words)), cjk)
    return " ".join(words)


def pct(values, p):
    s = sorted(values)
    idx = int(len(s) * p // 100)
    return s[min(idx, len(s) - 1)]


def bench_indexing(kfs, n_docs, has_cjk=False):
    docs = [f"viking://docs/doc_{i:08d}" for i in range(n_docs)]
    contents = [generate_doc(has_cjk=has_cjk) for _ in range(n_docs)]
    raw_bytes = sum(len(c) for c in contents)

    # warm up
    kfs.upsert(
        ACCOUNT,
        "viking://warm",
        "warm up text here now start",
        level=2,
        context_type="file",
        owner_user_id="owner0",
    )
    kfs.delete(ACCOUNT, "viking://warm")

    t0 = time.perf_counter()
    for i in range(n_docs):
        kfs.upsert(
            ACCOUNT,
            docs[i],
            contents[i],
            level=2,
            context_type="file",
            owner_user_id="owner0",
        )
    t1 = time.perf_counter()
    elapsed = t1 - t0

    db_size = kfs.db_path(ACCOUNT).stat().st_size
    return {
        "n_docs": n_docs,
        "raw_mb": raw_bytes / 1024 / 1024,
        "elapsed_s": round(elapsed, 3),
        "docs_per_sec": round(n_docs / elapsed, 0),
        "mb_per_sec": round(raw_bytes / 1024 / 1024 / elapsed, 2),
        "db_mb": round(db_size / 1024 / 1024, 2),
        "ratio": round(db_size / raw_bytes, 3),
    }


def bench_lookup(kfs, n_queries=500):
    queries = []
    for _ in range(n_queries):
        n = random.randint(1, 3)
        q = " ".join(
            "".join(
                random.choice(string.ascii_lowercase)
                for _ in range(random.randint(3, 6))
            )
            for _ in range(n)
        )
        queries.append(q)

    times_ms = []
    for q in queries:
        t0 = time.perf_counter()
        kfs.lookup(ACCOUNT, q, limit=50)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    return {
        "n_queries": n_queries,
        "p50_ms": round(pct(times_ms, 50), 3),
        "p95_ms": round(pct(times_ms, 95), 3),
        "p99_ms": round(pct(times_ms, 99), 3),
        "max_ms": round(max(times_ms), 3),
        "mean_ms": round(sum(times_ms) / len(times_ms), 3),
    }


def bench_grep_vs_scan(kfs, n_docs, n_searches=30):
    queries = []
    for _ in range(n_searches):
        wlen = random.randint(4, 7)
        word = "".join(random.choice(string.ascii_lowercase) for _ in range(wlen))
        queries.append(word)

    # FTS5 recall path
    fts_times = []
    for q in queries:
        t0 = time.perf_counter()
        kfs.lookup(ACCOUNT, q, limit=1000)
        t1 = time.perf_counter()
        fts_times.append((t1 - t0) * 1000)

    # Brute-force full-table scan
    all_contents = []
    conn = kfs._conn(ACCOUNT)
    cur = conn.execute("SELECT uri, content FROM kf")
    for row in cur:
        all_contents.append(row)

    scan_times = []
    for q in queries:
        t0 = time.perf_counter()
        for _uri, content in all_contents:
            if q in content:
                pass
        t1 = time.perf_counter()
        scan_times.append((t1 - t0) * 1000)

    fts_mean = sum(fts_times) / len(fts_times)
    scan_mean = sum(scan_times) / len(scan_times)
    speedup = scan_mean / fts_mean if fts_mean > 0 else 0

    return {
        "n_docs": n_docs,
        "n_searches": n_searches,
        "fts_mean_ms": round(fts_mean, 3),
        "scan_mean_ms": round(scan_mean, 3),
        "speedup_x": round(speedup, 1),
    }


def main():
    random.seed(42)
    tmpdir = tempfile.mkdtemp(prefix="kvbench_")
    print(f"Temp dir: {tmpdir}")
    try:
        config = KeywordConfig(enabled=True)

        print("\n=== 1. Indexing Throughput ===")
        for n in [1000, 5000, 20000]:
            fresh = KeywordFS(os.path.join(tmpdir, f"_{n}"), config)
            r = bench_indexing(fresh, n, has_cjk=False)
            print(
                f"  {n:>6d} docs: {r['docs_per_sec']:,.0f} docs/s, "
                f"{r['mb_per_sec']:.2f} MB/s, "
                f"db={r['db_mb']:.2f}MB (ratio={r['ratio']:.3f}), "
                f"{r['elapsed_s']:.2f}s"
            )
            fresh.close()

        print("\n=== 2. BM25 Lookup Latency ===")
        for size in [5000, 20000]:
            kfs = KeywordFS(os.path.join(tmpdir, f"_{size}"), config)
            r = bench_lookup(kfs, n_queries=1000)
            print(
                f"  {size:>6} docs: p50={r['p50_ms']}ms  "
                f"p95={r['p95_ms']}ms  p99={r['p99_ms']}ms  "
                f"max={r['max_ms']}ms  mean={r['mean_ms']}ms"
            )
            kfs.close()

        print("\n=== 3. Grep Recall vs Brute-Force Scan ===")
        for n in [1000, 5000, 20000]:
            fresh = KeywordFS(os.path.join(tmpdir, f"_{n}"), config)
            r = bench_grep_vs_scan(fresh, n, n_searches=50)
            print(
                f"  {n:>6} docs: FTS5={r['fts_mean_ms']:.3f}ms  "
                f"scan={r['scan_mean_ms']:.1f}ms  "
                f"speedup={r['speedup_x']:.1f}x"
            )
            fresh.close()

        print("\n=== 4. CJK Tokenizer Overhead ===")
        sample = generate_doc(500, has_cjk=True)
        for mode_label, mode_fn in [
            ("char", lambda: tokenize(sample, tokenizer_mode="auto", cjk_mode="char")),
            ("bigram", lambda: tokenize(sample, tokenizer_mode="auto", cjk_mode="bigram")),
        ]:
            t0 = time.perf_counter()
            for _ in range(1000):
                mode_fn()
            t1 = time.perf_counter()
            print(
                f"  {mode_label:6s}: {(t1-t0)*1000:.1f}ms for 1000 x 500-word CJK docs, "
                f"{1000/(t1-t0):.0f} ops/s"
            )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
