"""
FILE: 05_embedding_quality_eval.py
LESSON: Phase 1 - Lesson 8 - Embeddings
TOPIC: Evaluating embedding quality — Precision@K, Recall@K, MRR, NDCG

WHAT THIS FILE TEACHES:
  - WHY you must evaluate embeddings on YOUR corpus (not just MTEB scores)
  - Precision@K: of the top-K results, how many are correct?
  - Recall@K: of all correct answers, how many did we find in top-K?
  - MRR (Mean Reciprocal Rank): how high does the FIRST correct result appear?
  - NDCG@K: normalized discounted cumulative gain — the gold standard metric
  - Building a ground-truth evaluation dataset
  - Evaluating two models side by side

NO API NEEDED: Uses mock embeddings to teach the metrics.

INSTALL: pip install numpy
"""

import math
import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import Optional


# ─── Mock Embedding (no API needed) ──────────────────────────────────────────

def mock_embed(text: str, dims: int = 64, quality: float = 1.0) -> np.ndarray:
    """
    Deterministic mock embedding.
    quality: 0.0-1.0. Higher quality = vectors that better reflect semantic relationships.
    Used to simulate "good model" vs "bad model" comparison.
    """
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.RandomState(seed)
    base = rng.randn(dims)

    if quality < 1.0:
        # Reduce quality: add noise to make the vector less semantically meaningful
        noise = np.random.RandomState(seed + 1).randn(dims) * (1 - quality)
        base  = base * quality + noise

    norm = np.linalg.norm(base)
    return base / norm if norm > 0 else base


# ─── Ground Truth Dataset ─────────────────────────────────────────────────────

@dataclass
class QueryGroundTruth:
    """
    One query + its relevant document IDs (ground truth for evaluation).
    """
    query:        str
    relevant_ids: list[str]    # All document IDs that correctly answer this query
    ideal_order:  list[str]    # Preferred ranking (for graded NDCG evaluation)


@dataclass
class Document:
    doc_id:  str
    content: str
    tags:    list[str] = field(default_factory=list)


# ─── Evaluation Dataset ───────────────────────────────────────────────────────

DOCUMENTS = [
    Document("doc_001", "Cisco ACI uses Leaf-Spine topology. APIC manages fabric policy.", ["aci", "networking"]),
    Document("doc_002", "APIC controller is the central management plane for ACI fabric.",  ["aci", "networking"]),
    Document("doc_003", "ACI contracts define which endpoint groups can communicate.",       ["aci", "security"]),
    Document("doc_004", "ReadyOps validates changes in Production-Representative before Live Operations.", ["readyops"]),
    Document("doc_005", "ReadyOps agent classes: Health & Posture, Validation, Operational, Stress.",      ["readyops"]),
    Document("doc_006", "Cisco Hypershield uses eBPF to enforce policy at the kernel level.",              ["security"]),
    Document("doc_007", "ISE TrustSec assigns SGTs at authentication for microsegmentation.",              ["security", "aci"]),
    Document("doc_008", "Nexus 9000 supports VXLAN EVPN for multi-tenant fabric.",                        ["networking"]),
    Document("doc_009", "SD-WAN provides centralized policy and zero-touch provisioning.",                 ["networking"]),
    Document("doc_010", "Cisco Intersight manages UCS, HyperFlex, and third-party infrastructure.",        ["compute"]),
    Document("doc_011", "The quick brown fox has nothing to do with networking.",                           ["unrelated"]),
    Document("doc_012", "My cat enjoys sitting on warm network switches.",                                  ["unrelated"]),
]

GROUND_TRUTH = [
    QueryGroundTruth(
        query        = "How does ACI manage network policy?",
        relevant_ids = ["doc_001", "doc_002", "doc_003"],
        ideal_order  = ["doc_001", "doc_002", "doc_003"],
    ),
    QueryGroundTruth(
        query        = "How does ReadyOps ensure production safety?",
        relevant_ids = ["doc_004", "doc_005"],
        ideal_order  = ["doc_004", "doc_005"],
    ),
    QueryGroundTruth(
        query        = "How does Cisco enforce microsegmentation?",
        relevant_ids = ["doc_006", "doc_007"],
        ideal_order  = ["doc_006", "doc_007"],
    ),
    QueryGroundTruth(
        query        = "What is the Leaf-Spine topology?",
        relevant_ids = ["doc_001"],
        ideal_order  = ["doc_001"],
    ),
]


# ─── Retrieval Simulation ─────────────────────────────────────────────────────

def retrieve(
    query:    str,
    docs:     list[Document],
    model_quality: float = 1.0,
    top_k:    int = 5,
) -> list[tuple[str, float]]:
    """
    Simulate retrieval by embedding query + all docs and ranking by similarity.

    Args:
        query:         Search query string.
        docs:          Corpus of documents.
        model_quality: 1.0 = perfect model, 0.3 = noisy/bad model.
        top_k:         Number of results to return.

    Returns:
        List of (doc_id, score) tuples, ranked by score descending.
    """

    query_vec = mock_embed(query, quality=model_quality)
    doc_vecs  = {d.doc_id: mock_embed(d.content, quality=model_quality) for d in docs}

    # Cosine similarity (unit vectors → dot product)
    scores = {
        doc_id: float(np.dot(query_vec, vec))
        for doc_id, vec in doc_vecs.items()
    }

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_k]


# ─── Evaluation Metrics ───────────────────────────────────────────────────────

def precision_at_k(
    retrieved: list[str],
    relevant:  set[str],
    k:         int,
) -> float:
    """
    Precision@K: what fraction of the TOP-K results are relevant?

    Formula: Precision@K = |retrieved[:K] ∩ relevant| / K

    WHY Precision@K matters:
      Measures the QUALITY of the results shown to the user.
      If K=5 and 4 of the 5 shown results are relevant → P@5 = 0.80.
      Users see all K results, so precision directly measures their experience.

    Example:
      Retrieved: [doc_001, doc_011, doc_002, doc_012, doc_003]  (top 5)
      Relevant:  {doc_001, doc_002, doc_003}
      P@3 = |{doc_001}| / 3 = 1/3 = 0.33  (only 1 of first 3 was relevant)
      P@5 = |{doc_001, doc_002, doc_003}| / 5 = 3/5 = 0.60
    """
    top_k_ids = retrieved[:k]
    hits      = sum(1 for doc_id in top_k_ids if doc_id in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(
    retrieved: list[str],
    relevant:  set[str],
    k:         int,
) -> float:
    """
    Recall@K: what fraction of ALL relevant documents were found in the top-K?

    Formula: Recall@K = |retrieved[:K] ∩ relevant| / |relevant|

    WHY Recall@K matters:
      Measures COMPLETENESS — did we find all the evidence the model needs?
      In RAG, if a query has 3 relevant chunks but we only retrieve 2,
      the answer may be incomplete. Recall@K reveals this gap.

    Example:
      Retrieved: [doc_001, doc_011, doc_002, doc_012, doc_003]
      Relevant:  {doc_001, doc_002, doc_003}  (3 total)
      R@3 = |{doc_001}| / 3 = 1/3 = 0.33
      R@5 = |{doc_001, doc_002, doc_003}| / 3 = 3/3 = 1.00  ← found all!
    """
    top_k_ids = retrieved[:k]
    hits      = sum(1 for doc_id in top_k_ids if doc_id in relevant)
    return hits / len(relevant) if relevant else 0.0


def reciprocal_rank(
    retrieved: list[str],
    relevant:  set[str],
) -> float:
    """
    Reciprocal Rank: 1/position of the FIRST correct result.

    Formula: RR = 1 / rank_of_first_correct

    WHY RR matters:
      For queries with ONE gold answer, RR measures how high that answer ranks.
      If it's at position 1 → RR = 1.0 (perfect).
      If it's at position 3 → RR = 0.33 (found but not prominent).
      If not found → RR = 0.0.

    MRR (Mean Reciprocal Rank) = average RR across all queries.

    WHY MRR is useful for RAG:
      In RAG we pass top-K chunks to the LLM. If the BEST chunk is at
      position K instead of position 1, the LLM is less likely to use it
      (Lost in the Middle). MRR rewards systems that put the BEST chunk first.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank   # WHY /rank: linear decay with position
    return 0.0


def ndcg_at_k(
    retrieved:   list[str],
    ideal_order: list[str],
    relevant:    set[str],
    k:           int,
) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain).
    The gold standard metric for ranked retrieval quality.

    COMPONENTS:
      1. Gain: did we retrieve a relevant document? 1 or 0.
      2. Discount: position penalty — gain at position i is divided by log2(i+1).
         → Relevant docs at position 1 score MUCH higher than at position 5.
      3. DCG: sum of discounted gains.
      4. IDCG: DCG for the IDEAL ranking (all relevant docs at top positions).
      5. NDCG = DCG / IDCG.  Range: [0, 1].

    WHY NDCG beats Precision@K and Recall@K:
      - Precision ignores ORDER (rank 1 same as rank K).
      - Recall ignores order too.
      - NDCG rewards finding relevant docs AND putting them HIGH in the ranking.
      - This directly models the Lost in the Middle problem:
        if doc_001 is the answer but ranked 5th, NDCG is penalized.

    WHY log2(i+1) as discount:
      The first position is 10× more valuable than the 10th position.
      log2(1+1)=1, log2(2+1)≈1.58, log2(5+1)≈2.58, log2(10+1)≈3.46.
      The denominator grows slowly → positions 1-3 matter most.
    """

    def gain(doc_id: str, position_in_ideal: dict[str, int]) -> float:
        """
        Graded relevance: 2 if at ideal position 1, 1 if relevant, 0 if not.

        WHY graded (not binary):
          Binary NDCG treats all relevant docs equally.
          Graded NDCG rewards getting the BEST doc at the top.
        """
        if doc_id not in relevant:
            return 0.0
        ideal_pos = position_in_ideal.get(doc_id, len(ideal_order))
        if ideal_pos == 0:   # most relevant
            return 2.0
        return 1.0

    ideal_positions = {doc_id: i for i, doc_id in enumerate(ideal_order)}

    # DCG: discounted cumulative gain for the actual retrieval
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        g = gain(doc_id, ideal_positions)
        dcg += g / math.log2(i + 1)   # WHY +1: avoids log2(1)=0 at position 0

    # IDCG: ideal DCG (all relevant at top positions)
    idcg = 0.0
    ideal_retrieved = [d for d in ideal_order if d in relevant]
    for i, doc_id in enumerate(ideal_retrieved[:k], start=1):
        g = gain(doc_id, ideal_positions)
        idcg += g / math.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg   # WHY normalize by IDCG: makes metric comparable across queries


# ─── Full Evaluation Run ──────────────────────────────────────────────────────

def evaluate_model(model_quality: float, model_name: str, k: int = 5) -> dict:
    """
    Evaluate a simulated model over all ground truth queries.
    Returns average metrics across all queries.
    """

    all_p_at_k = []
    all_r_at_k = []
    all_rr     = []
    all_ndcg   = []

    for qt in GROUND_TRUTH:
        results     = retrieve(qt.query, DOCUMENTS, model_quality=model_quality, top_k=k)
        retrieved   = [doc_id for doc_id, _ in results]
        relevant    = set(qt.relevant_ids)

        all_p_at_k.append(precision_at_k(retrieved, relevant, k))
        all_r_at_k.append(recall_at_k(retrieved, relevant, k))
        all_rr.append(reciprocal_rank(retrieved, relevant))
        all_ndcg.append(ndcg_at_k(retrieved, qt.ideal_order, relevant, k))

    return {
        "model":     model_name,
        "k":         k,
        "precision": sum(all_p_at_k) / len(all_p_at_k),
        "recall":    sum(all_r_at_k) / len(all_r_at_k),
        "mrr":       sum(all_rr)     / len(all_rr),
        "ndcg":      sum(all_ndcg)   / len(all_ndcg),
    }


def per_query_evaluation(model_quality: float, model_name: str, k: int = 5):
    """
    Show per-query breakdown to identify where a model struggles.
    """

    print(f"\n  Per-query breakdown [{model_name}]:")
    print(f"  {'Query':<45} {'P@K':>6} {'R@K':>6} {'MRR':>6} {'NDCG':>6}")
    print(f"  {'─'*45} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

    for qt in GROUND_TRUTH:
        results   = retrieve(qt.query, DOCUMENTS, model_quality=model_quality, top_k=k)
        retrieved = [doc_id for doc_id, _ in results]
        relevant  = set(qt.relevant_ids)

        p     = precision_at_k(retrieved, relevant, k)
        r     = recall_at_k(retrieved, relevant, k)
        rr    = reciprocal_rank(retrieved, relevant)
        ndcg  = ndcg_at_k(retrieved, qt.ideal_order, relevant, k)

        query_short = qt.query[:43] + ".." if len(qt.query) > 43 else qt.query
        print(f"  {query_short:<45} {p:>6.3f} {r:>6.3f} {rr:>6.3f} {ndcg:>6.3f}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("EMBEDDING QUALITY EVALUATION: Metrics and Comparison")
    print("=" * 65)

    K = 5   # Evaluate top-5 results (common for RAG)

    # ── Evaluate two simulated models ─────────────────────────────────────────
    models = [
        (0.85, "Good model (voyage-3 simulated)"),
        (0.45, "Weak model (ada-002 on technical text)"),
    ]

    results_all = []
    for quality, name in models:
        result = evaluate_model(quality, name, k=K)
        results_all.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n  SUMMARY METRICS (K={K})")
    print(f"\n  {'Model':<40} {'P@K':>7} {'R@K':>7} {'MRR':>7} {'NDCG':>7}")
    print(f"  {'─'*40} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    for r in results_all:
        print(
            f"  {r['model']:<40} "
            f"{r['precision']:>7.3f} {r['recall']:>7.3f} "
            f"{r['mrr']:>7.3f} {r['ndcg']:>7.3f}"
        )

    # ── Per-query breakdown ────────────────────────────────────────────────────
    for quality, name in models:
        per_query_evaluation(quality, name, k=K)

    # ── Metric interpretations ─────────────────────────────────────────────────
    print(f"\n\n  {'─'*65}")
    print(f"  HOW TO READ THESE METRICS:")
    print(f"""
  Precision@{K}: of the {K} chunks shown to the LLM, how many are relevant?
    >0.80  excellent — almost everything retrieved is useful
    0.60   acceptable — some noise but most results are on-topic
    <0.40  poor — more irrelevant chunks than relevant → hallucination risk

  Recall@{K}: of all relevant documents, what fraction did we find?
    >0.90  excellent — nearly all evidence retrieved
    0.70   acceptable — might miss some supporting details
    <0.50  poor — key evidence likely missing → incomplete answers

  MRR: how high does the BEST answer appear in the ranking?
    >0.90  the correct chunk is almost always position 1-2
    0.60   correct chunk typically at position 2-3
    <0.40  correct chunk often buried → Lost in the Middle risk

  NDCG@{K}: overall ranking quality including position weighting
    >0.85  excellent — right answers at the top, wrong answers at the bottom
    0.65   good — mostly correct ranking
    <0.50  poor — ranking does not reflect relevance
""")
