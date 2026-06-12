"""
FILE: 03_hybrid_search.py
LESSON: Phase 1 - Lesson 9 - Semantic Search
TOPIC: Hybrid search — combining BM25 and dense vector search with RRF fusion

WHAT THIS FILE TEACHES:
  - Why neither BM25 nor dense search alone is optimal
  - Reciprocal Rank Fusion (RRF) — the standard fusion algorithm
  - Weighted score fusion — when you want explicit control
  - Cascade re-ranking — dense retrieves, BM25 re-ranks
  - How to measure if hybrid beats either individual method
  - When NOT to use hybrid (small corpora, pure prose, no exact terms)

CORE INSIGHT:
  BM25 + Dense search find DIFFERENT documents.
  RRF fuses their rankings WITHOUT requiring score normalization.
  In practice, hybrid search improves Recall@10 by 5-15% over the best single method.

INSTALL: pip install numpy
"""

import math
import hashlib
import re
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ─── Shared Helpers ───────────────────────────────────────────────────────────

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","to","of","and","or",
    "in","on","at","by","for","with","as","this","that","it","its","from",
    "into","has","have","had","will","can","should",
}

def tokenize(text: str) -> list[str]:
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]

def mock_embed(text: str, dims: int = 64) -> np.ndarray:
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.RandomState(seed)
    vec  = rng.randn(dims).astype(np.float32)
    return vec / np.linalg.norm(vec)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ─── Minimal BM25 (from Lesson 9 file 1) ─────────────────────────────────────

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1; self.b = b
        self.corpus = []; self.doc_ids = []
        self.avgdl = 0.0; self.N = 0
        self.idf_cache: dict = {}
        self.inv_index: dict = defaultdict(list)

    def fit(self, documents: list[str], doc_ids: list[str]):
        self.N = len(documents)
        self.doc_ids = doc_ids
        self.corpus  = [tokenize(d) for d in documents]
        self.avgdl   = sum(len(d) for d in self.corpus) / max(self.N, 1)
        doc_freq: dict = defaultdict(set)
        for idx, toks in enumerate(self.corpus):
            for t in set(toks):
                doc_freq[t].add(idx)
                self.inv_index[t].append(idx)
        for term, docs in doc_freq.items():
            df = len(docs)
            self.idf_cache[term] = math.log((self.N - df + 0.5)/(df + 0.5) + 1)

    def _score(self, idx: int, terms: list[str]) -> float:
        toks = self.corpus[idx]; dl = len(toks)
        s = 0.0
        for t in terms:
            tf  = toks.count(t)
            if tf == 0: continue
            idf = self.idf_cache.get(t, 0.0)
            ln  = 1 - self.b + self.b * dl / max(self.avgdl, 1)
            s  += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * ln)
        return s

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms: return []
        candidates = set()
        for t in terms: candidates.update(self.inv_index.get(t, []))
        scores = [(self.doc_ids[i], self._score(i, terms)) for i in candidates]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ─── Minimal Dense Index ──────────────────────────────────────────────────────

class DenseIndex:
    def __init__(self, dims: int):
        self.dims = dims
        self._mat: Optional[np.ndarray] = None
        self._ids: list[str] = []

    def add(self, ids: list[str], vectors: np.ndarray):
        self._mat = vectors if self._mat is None else np.vstack([self._mat, vectors])
        self._ids.extend(ids)

    def search(self, query: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        if self._mat is None: return []
        qn     = query / (np.linalg.norm(query) + 1e-10)
        scores = np.dot(self._mat, qn)
        top    = np.argsort(-scores)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in top]


# ─── Fusion Algorithms ────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists:   list[list[tuple[str, float]]],
    k:              int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) — combine multiple ranked result lists.

    FORMULA:
      RRF(d) = Σ_list  1 / (k + rank_of_d_in_list)

    WHERE:
      k = 60  (smoothing constant — prevents rank 1 from dominating)
      rank_of_d = 1-based position (1 = first result, 2 = second, etc.)

    WHY k=60:
      From Cormack et al. 2009 — empirically optimal across many IR benchmarks.
      k=0 would make rank-1 result infinitely better than rank-2 (unstable).
      k=60 means rank 1 → 1/61 ≈ 0.016, rank 61 → 1/121 ≈ 0.008 (2× less).
      Reasonable decay without extreme rank-1 bias.

    WHY ranks (not raw scores):
      Different retrieval systems use incomparable score scales:
      BM25 scores: 0 to ~20 (depend on corpus stats)
      Cosine scores: -1 to +1
      Cross-encoder scores: logits (can be any range)
      Combining raw scores requires NORMALIZATION → error-prone.
      Ranks are scale-invariant → no normalization needed.

    Args:
        result_lists: List of result lists, each [(doc_id, score), ...]
        k:            Smoothing constant (default 60).

    Returns:
        Combined list of (doc_id, rrf_score) sorted by RRF score descending.
    """

    rrf_scores: dict[str, float] = defaultdict(float)

    for result_list in result_lists:
        for rank, (doc_id, _) in enumerate(result_list, start=1):
            # WHY 1/(k + rank) not 1/rank:
            #   1/rank without k would make rank-1 give 1.0, rank-2 give 0.5 (halved).
            #   With k=60: rank-1 gives 1/61≈0.016, rank-2 gives 1/62≈0.016 (tiny diff).
            #   This makes RRF robust to small rank differences between systems.
            rrf_scores[doc_id] += 1.0 / (k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return sorted_results


def weighted_score_fusion(
    bm25_results:   list[tuple[str, float]],
    dense_results:  list[tuple[str, float]],
    alpha:          float = 0.5,
) -> list[tuple[str, float]]:
    """
    Weighted linear combination of normalized BM25 and dense scores.

    FORMULA: score(d) = alpha × dense_norm(d) + (1 - alpha) × bm25_norm(d)

    WHY normalize first:
      BM25 scores are in [0, ~20]. Dense scores are in [-1, 1].
      Without normalization, dense scores would be irrelevant compared to BM25.
      Min-max normalization scales both to [0, 1] for fair combination.

    WHY alpha is tunable:
      For corpora with many exact technical terms: lower alpha (more BM25).
      For semantic, prose-heavy corpora: higher alpha (more dense).
      Default alpha=0.5 is a neutral starting point.

    WEAKNESS:
      Min-max normalization is sensitive to outliers.
      A single very high BM25 score compresses all other scores toward 0.
      RRF doesn't have this problem → preferred in practice.

    Args:
        bm25_results:  [(doc_id, bm25_score), ...] sorted by score
        dense_results: [(doc_id, dense_score), ...] sorted by score
        alpha:         Weight for dense scores. (1-alpha) for BM25.
    """

    def normalize(results: list[tuple[str, float]]) -> dict[str, float]:
        """Min-max normalize scores to [0, 1]."""
        if not results: return {}
        scores = [s for _, s in results]
        mn, mx = min(scores), max(scores)
        if mx == mn: return {doc_id: 1.0 for doc_id, _ in results}
        return {doc_id: (s - mn) / (mx - mn) for doc_id, s in results}

    dense_norm = normalize(dense_results)
    bm25_norm  = normalize(bm25_results)

    all_doc_ids = set(dense_norm) | set(bm25_norm)

    combined = {}
    for doc_id in all_doc_ids:
        d_score = dense_norm.get(doc_id, 0.0)  # WHY 0.0: doc not in dense results
        b_score = bm25_norm.get(doc_id,  0.0)
        combined[doc_id] = alpha * d_score + (1 - alpha) * b_score

    return sorted(combined.items(), key=lambda x: -x[1])


def cascade_rerank(
    first_stage_results:  list[tuple[str, float]],
    second_stage_fn,
    top_k:                int = 5,
) -> list[tuple[str, float]]:
    """
    Two-stage retrieval: first stage gets candidates, second stage re-ranks.

    COMMON PATTERN:
      Stage 1: Dense search → top-100 candidates (high recall)
      Stage 2: Cross-encoder re-rank → top-5 (high precision)

    WHY this works:
      Dense search is cheap for large top-K (100 results at ~2ms).
      Cross-encoder re-ranker is expensive per-pair but only runs on 100 pairs.
      Cross-encoders are much more accurate than bi-encoder (dense) search
      because they see both query AND document together (not separate embeddings).

    In this file: second_stage_fn is BM25 re-ranking for simplicity.
    In production: second_stage_fn is a cross-encoder (Phase 4 of curriculum).

    Args:
        first_stage_results: Candidates from stage 1.
        second_stage_fn:     Function(doc_id) → re-rank score.
        top_k:               Final results to return.
    """

    reranked = [
        (doc_id, second_stage_fn(doc_id))
        for doc_id, _ in first_stage_results
    ]
    reranked.sort(key=lambda x: -x[1])
    return reranked[:top_k]


# ─── Recall Measurement ───────────────────────────────────────────────────────

def measure_recall(
    results:  list[tuple[str, float]],
    relevant: set[str],
    k:        int,
) -> float:
    """Recall@K: fraction of relevant docs found in top-K results."""
    top_k_ids = {doc_id for doc_id, _ in results[:k]}
    return len(top_k_ids & relevant) / max(len(relevant), 1)


# ─── Demo Corpus ──────────────────────────────────────────────────────────────

CORPUS = [
    # (doc_id, content)
    ("doc_001", "Cisco ACI uses Leaf-Spine topology. APIC manages fabric policy. EPG contracts control communication."),
    ("doc_002", "The APIC REST API uses JSON over HTTPS. Authentication via aaaLogin. Tenant objects in uni/tn-."),
    ("doc_003", "ReadyOps validates ACI changes in Production-Representative environment before Live Operations."),
    ("doc_004", "Cisco Hypershield uses eBPF for kernel-level policy enforcement without appliances."),
    ("doc_005", "ISE TrustSec assigns SGTs at authentication. SXP propagates SGTs to non-TrustSec devices."),
    ("doc_006", "VXLAN EVPN provides multi-tenant fabric using BGP on Nexus 9000 as control plane."),
    ("doc_007", "Nexus 9336C-FX2 is 36-port 100G switch supporting ACI and NX-OS modes."),
    ("doc_008", "Bug CSCvh23456 affects APIC 5.2(1g) causing contract deployment failures on leaf nodes."),
    ("doc_009", "ReadyOps agent classes: Health Posture, Validation, Operational, Stress Adversarial."),
    ("doc_010", "Cisco Intersight manages UCS, HyperFlex, and third-party infrastructure from the cloud."),
    ("doc_011", "SD-WAN provides centralized policy management and zero-touch provisioning for WAN."),
    ("doc_012", "Cisco SSE (Security Service Edge) delivers cloud-based security from Cisco's SASE platform."),
]

GROUND_TRUTH = {
    "How does ACI manage endpoint group communication?": {"doc_001", "doc_002"},
    "How does ReadyOps ensure production changes are safe?": {"doc_003", "doc_009"},
    "What is the bug CSCvh23456?": {"doc_008"},
    "How does Cisco enforce network policy and segmentation?": {"doc_001", "doc_004", "doc_005"},
    "What is the Nexus 9336C-FX2?": {"doc_007"},
}


def run_hybrid_demo():
    """
    Full hybrid search demo comparing BM25, Dense, and Hybrid on ground truth.
    """

    print("=" * 65)
    print("HYBRID SEARCH: BM25 + Dense + RRF Comparison")
    print("=" * 65)

    doc_ids   = [c[0] for c in CORPUS]
    doc_texts = [c[1] for c in CORPUS]

    # ── Build BM25 index ──────────────────────────────────────────────────────
    bm25 = BM25()
    bm25.fit(doc_texts, doc_ids)

    # ── Build dense index ─────────────────────────────────────────────────────
    dims        = 64
    doc_vectors = np.vstack([mock_embed(t, dims) for t in doc_texts])
    dense       = DenseIndex(dims=dims)
    dense.add(doc_ids, doc_vectors)

    # ── Evaluate each method on ground truth ──────────────────────────────────
    K = 5  # evaluate Recall@5

    bm25_recalls   = []
    dense_recalls  = []
    rrf_recalls    = []
    wsum_recalls   = []

    for query, relevant in GROUND_TRUTH.items():
        q_vec = mock_embed(query, dims)

        bm25_res   = bm25.search(query, top_k=20)
        dense_res  = dense.search(q_vec, top_k=20)
        rrf_res    = reciprocal_rank_fusion([bm25_res, dense_res], k=60)
        wsum_res   = weighted_score_fusion(bm25_res, dense_res, alpha=0.5)

        bm25_recalls.append(measure_recall(bm25_res,  relevant, K))
        dense_recalls.append(measure_recall(dense_res, relevant, K))
        rrf_recalls.append(measure_recall(rrf_res,   relevant, K))
        wsum_recalls.append(measure_recall(wsum_res,  relevant, K))

    avg = lambda lst: sum(lst)/len(lst)

    print(f"\n  Recall@{K} comparison ({len(GROUND_TRUTH)} queries):")
    print(f"\n  {'Method':<30} {'Recall@'+str(K):>10}  Verdict")
    print(f"  {'─'*30} {'─'*10}  {'─'*25}")

    methods = [
        ("BM25 only",             avg(bm25_recalls)),
        ("Dense only",            avg(dense_recalls)),
        ("Hybrid (RRF)",          avg(rrf_recalls)),
        ("Hybrid (Weighted Sum)", avg(wsum_recalls)),
    ]
    best_recall = max(r for _, r in methods)

    for name, recall in methods:
        verdict = "← BEST" if recall == best_recall else ""
        print(f"  {name:<30} {recall:>10.3f}  {verdict}")

    # ── Per-query breakdown ────────────────────────────────────────────────────
    print(f"\n  Per-query breakdown (Recall@{K}):")
    print(f"\n  {'Query':<45} {'BM25':>6} {'Dense':>7} {'RRF':>7}")
    print(f"  {'─'*45} {'─'*6} {'─'*7} {'─'*7}")

    for i, (query, relevant) in enumerate(GROUND_TRUTH.items()):
        q_short = query[:43] + ".." if len(query) > 43 else query
        print(
            f"  {q_short:<45} "
            f"{bm25_recalls[i]:>6.3f} {dense_recalls[i]:>7.3f} {rrf_recalls[i]:>7.3f}"
        )

    # ── Show the "bug ID" case ────────────────────────────────────────────────
    print(f"\n\n  CASE STUDY: Exact term query ('Bug CSCvh23456')")
    print(f"  {'─'*60}")
    bug_query    = "What is the bug CSCvh23456?"
    bug_relevant = {"doc_008"}
    qvec         = mock_embed(bug_query, dims)

    bm25_bug  = bm25.search(bug_query, top_k=5)
    dense_bug = dense.search(qvec, top_k=5)
    rrf_bug   = reciprocal_rank_fusion([bm25_bug, dense_bug])[:5]

    for name, results in [("BM25", bm25_bug), ("Dense", dense_bug), ("RRF", rrf_bug)]:
        found = next((True for doc_id,_ in results if doc_id in bug_relevant), False)
        rank  = next((i+1 for i,(doc_id,_) in enumerate(results) if doc_id in bug_relevant), None)
        print(f"  {name:<10}: doc_008 found={str(found):<6} rank={rank or 'NOT FOUND'}")

    print(f"""
  WHY BM25 WINS on "CSCvh23456":
    The bug ID "CSCvh23456" is a rare, exact token.
    Dense embedding has never seen this specific string in training → low similarity.
    BM25 looks for exact token matches → finds it immediately.
    Hybrid RRF combines both signals → if BM25 finds it, it appears in the fused list.
""")


def rrf_k_sensitivity():
    """
    Show how the k parameter affects RRF scores and rankings.
    """

    print("\n" + "=" * 65)
    print("RRF k PARAMETER SENSITIVITY")
    print("=" * 65)

    # Doc A: BM25 rank 1, Dense rank 5
    # Doc B: BM25 rank 3, Dense rank 1
    # Doc C: BM25 rank 2, Dense rank 2

    results_bm25  = [("A", 9.5), ("C", 7.0), ("B", 5.5), ("D", 3.0), ("E", 1.0)]
    results_dense = [("B", 0.95), ("C", 0.90), ("E", 0.80), ("D", 0.75), ("A", 0.60)]

    print(f"\n  BM25 ranking:  " + " > ".join(f"{doc}({score:.1f})" for doc, score in results_bm25))
    print(f"  Dense ranking: " + " > ".join(f"{doc}({score:.2f})" for doc, score in results_dense))

    print(f"\n  {'Doc':<8}", end="")
    for k in [1, 10, 60, 200]:
        print(f"  {'RRF(k='+str(k)+')':>12}", end="")
    print()
    print("  " + "─" * 58)

    for doc in ["A", "B", "C", "D", "E"]:
        print(f"  {doc:<8}", end="")
        for k in [1, 10, 60, 200]:
            rrf_result = reciprocal_rank_fusion([results_bm25, results_dense], k=k)
            score = next((s for d, s in rrf_result if d == doc), 0.0)
            print(f"  {score:>12.5f}", end="")
        print()

    print(f"""
  OBSERVATION:
    With k=1:  Rank-1 is much more valuable than rank-2 (high sensitivity).
    With k=60: Differences between ranks are smoothed (standard default).
    With k=200: All ranks contribute nearly equally (too flat).
    k=60 is the sweet spot — standard across all RAG implementations.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_hybrid_demo()
    rrf_k_sensitivity()
