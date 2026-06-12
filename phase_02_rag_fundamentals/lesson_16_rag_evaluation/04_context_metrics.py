"""
FILE: 04_context_metrics.py
LESSON: Phase 2 - Lesson 16 - RAG Evaluation
TOPIC: Context precision and recall — measuring retrieval quality

WHAT THIS FILE TEACHES:
  - Context Precision: of K retrieved chunks, how many are actually relevant?
  - Context Recall: of all info needed, what fraction did retrieval find?
  - NDCG-weighted variants of precision and recall
  - WHY these two metrics must be tracked together (precision-recall tradeoff)
  - Average Precision (AP) for ranked retrieval evaluation
  - How to diagnose and fix low precision vs low recall

INSTALL: no external dependencies
"""

import re
import math
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─── Retrieval Quality Structures ─────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its relevance label."""
    chunk_id:  str
    content:   str
    rank:      int        # 1-based rank in retrieval results
    score:     float      # retrieval score (cosine sim or reranker score)
    relevance: int        # 0=not relevant, 1=somewhat, 2=highly relevant (from golden set)


@dataclass
class ContextMetricsResult:
    """Full context precision and recall results for one query."""
    query:           str
    retrieved:       list[RetrievedChunk]
    precision_at_k:  float    # fraction of top-K that are relevant
    recall_at_k:     float    # fraction of all relevant docs found in top-K
    ap:              float    # Average Precision (rank-aware precision)
    ndcg:            float    # Normalized DCG (graded relevance)
    total_relevant:  int      # total relevant docs in the knowledge base

    def display(self):
        print(f"\n  Query: '{self.query}'")
        print(f"\n  Retrieved chunks (rank, relevance, content):")
        for r in self.retrieved:
            rel_icon = {2: "✓✓", 1: "✓ ", 0: "  "}.get(r.relevance, "  ")
            print(f"    [{r.rank:2}] rel={r.relevance} {rel_icon}  {r.score:.3f}  '{r.content[:60]}'")
        print(f"\n  Context Precision@{len(self.retrieved)}: {self.precision_at_k:.3f}")
        print(f"  Context Recall@{len(self.retrieved)}:    {self.recall_at_k:.3f}  (of {self.total_relevant} relevant docs)")
        print(f"  Average Precision:    {self.ap:.3f}")
        print(f"  NDCG@{len(self.retrieved)}:            {self.ndcg:.3f}")


# ─── Metric Implementations ───────────────────────────────────────────────────

def context_precision_at_k(retrieved: list[RetrievedChunk], k: int, threshold: int = 1) -> float:
    """
    Context Precision@K = |{relevant docs in top-K}| / K

    WHY threshold=1 (not 0):
      We count both rel=1 (somewhat relevant) and rel=2 (highly relevant)
      as "relevant". Only rel=0 (off-topic) counts as irrelevant noise.

    High precision → low noise in the context window.
    Low precision → LLM is distracted by irrelevant chunks.
    """
    top_k    = retrieved[:k]
    relevant = sum(1 for r in top_k if r.relevance >= threshold)
    return relevant / max(k, 1)


def context_recall_at_k(
    retrieved:       list[RetrievedChunk],
    total_relevant:  int,
    k:               int,
    threshold:       int = 1,
) -> float:
    """
    Context Recall@K = |{relevant docs in top-K}| / |{all relevant docs in corpus}|

    WHY recall is different from precision:
      Precision asks "how much noise is there?"
      Recall asks "did we miss anything?"
      A system that only retrieves 1 perfect doc has P=1.0 but R=1/total_relevant.

    Low recall → the LLM is missing information it needs to fully answer.
    This usually causes partial answers or fallback to parametric memory.
    """
    top_k    = retrieved[:k]
    found    = sum(1 for r in top_k if r.relevance >= threshold)
    return found / max(total_relevant, 1)


def average_precision(retrieved: list[RetrievedChunk], threshold: int = 1) -> float:
    """
    Average Precision (AP) for ranked retrieval — rank-aware precision metric.

    AP = (1/R) * Σ Precision@rank for each rank where the doc is relevant
    where R = total relevant docs retrieved

    WHY AP vs plain Precision@K:
      Precision@K doesn't distinguish between:
        [relevant, irrelevant, relevant] and [relevant, relevant, irrelevant]
      Both have P@3 = 2/3, but the second is clearly better (relevant docs ranked higher).
      AP rewards having relevant docs at higher ranks.
    """
    num_relevant = 0
    sum_precision = 0.0
    total_retrieved_relevant = sum(1 for r in retrieved if r.relevance >= threshold)

    if total_retrieved_relevant == 0:
        return 0.0

    for rank, r in enumerate(retrieved, start=1):
        if r.relevance >= threshold:
            num_relevant += 1
            sum_precision += num_relevant / rank   # Precision@this_rank

    return sum_precision / total_retrieved_relevant


def ndcg_at_k(retrieved: list[RetrievedChunk], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain — standard IR ranking metric.
    Rewards graded relevance (rel=2 is much better than rel=1).
    """
    def dcg(results, k):
        score = 0.0
        for rank, r in enumerate(results[:k], start=1):
            score += (2**r.relevance - 1) / math.log2(rank + 1)  # WHY 2^rel: exponential reward
        return score

    actual    = dcg(retrieved, k)
    ideal_ord = sorted(retrieved, key=lambda r: -r.relevance)
    ideal     = dcg(ideal_ord, k)
    return actual / ideal if ideal > 0 else 1.0


def compute_context_metrics(
    query:          str,
    retrieved:      list[RetrievedChunk],
    total_relevant: int,
    k:              int = 5,
) -> ContextMetricsResult:
    """Compute all four context quality metrics for one query."""
    return ContextMetricsResult(
        query          = query,
        retrieved      = retrieved,
        precision_at_k = context_precision_at_k(retrieved, k),
        recall_at_k    = context_recall_at_k(retrieved, total_relevant, k),
        ap             = average_precision(retrieved),
        ndcg           = ndcg_at_k(retrieved, k),
        total_relevant = total_relevant,
    )


# ─── Mock Embedder & Retriever ────────────────────────────────────────────────

def mock_embed(text: str, dims: int = 64) -> np.ndarray:
    keywords  = sorted(set(re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())))[:8]
    topic_key = " ".join(keywords)
    seed      = int(hashlib.md5(topic_key.encode()).hexdigest(), 16) % (2**32)
    rng       = np.random.RandomState(seed)
    full_rng  = np.random.RandomState(int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31))
    v = rng.randn(dims).astype(np.float32) + full_rng.randn(dims).astype(np.float32) * 0.15
    return v / (np.linalg.norm(v) + 1e-10)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                        b / (np.linalg.norm(b) + 1e-10)))


# ─── Annotated Knowledge Base ─────────────────────────────────────────────────

# Each doc has relevance labels per query topic: 2=highly, 1=somewhat, 0=not
KNOWLEDGE_BASE = [
    {"id": "d01", "content": "A minimum of 3 APIC nodes form a cluster for quorum and HA.",        "apic_ha": 2, "readyops": 0, "hypershield": 0},
    {"id": "d02", "content": "APIC communicates with leaf and spine switches via OpFlex.",           "apic_ha": 0, "readyops": 0, "hypershield": 0},
    {"id": "d03", "content": "For ACI HA, deploy 3 APIC nodes in separate failure domains.",        "apic_ha": 2, "readyops": 0, "hypershield": 0},
    {"id": "d04", "content": "APIC REST API uses aaaLogin endpoint on port 443.",                   "apic_ha": 0, "readyops": 0, "hypershield": 0},
    {"id": "d05", "content": "ACI requires odd APIC node count to avoid split-brain.",              "apic_ha": 2, "readyops": 0, "hypershield": 0},
    {"id": "d06", "content": "APIC cluster health visible in the Fault Manager dashboard.",         "apic_ha": 1, "readyops": 0, "hypershield": 0},
    {"id": "d07", "content": "Cisco recommends APIC on dedicated hardware or VMware vSphere.",      "apic_ha": 1, "readyops": 0, "hypershield": 0},
    {"id": "d08", "content": "A 3-node APIC cluster maintains policy when one node fails.",         "apic_ha": 2, "readyops": 0, "hypershield": 0},
    {"id": "d09", "content": "ReadyOps validates changes in Production-Representative environment.", "apic_ha": 0, "readyops": 2, "hypershield": 0},
    {"id": "d10", "content": "ReadyOps promotion gate requires 100% pass rate.",                    "apic_ha": 0, "readyops": 2, "hypershield": 0},
    {"id": "d11", "content": "ReadyOps agent classes: Health Posture, Validation, Operational.",   "apic_ha": 0, "readyops": 2, "hypershield": 0},
    {"id": "d12", "content": "Hypershield uses eBPF for kernel-level policy at the workload.",      "apic_ha": 0, "readyops": 0, "hypershield": 2},
    {"id": "d13", "content": "Hypershield integrates with ACI EPG membership from APIC.",           "apic_ha": 0, "readyops": 0, "hypershield": 2},
    {"id": "d14", "content": "BGP route reflection is the ACI Multi-Pod IPN control plane.",        "apic_ha": 0, "readyops": 0, "hypershield": 0},
    {"id": "d15", "content": "EPGs in ACI define policy groups; contracts permit inter-EPG traffic.", "apic_ha": 0, "readyops": 0, "hypershield": 0},
]

QUERIES = [
    {
        "query":          "How many APIC nodes are required for high availability?",
        "relevance_field": "apic_ha",
    },
    {
        "query":          "What are ReadyOps validation requirements?",
        "relevance_field": "readyops",
    },
    {
        "query":          "How does Hypershield enforce policy at the workload?",
        "relevance_field": "hypershield",
    },
]


def retrieve(query: str, k: int, relevance_field: str) -> tuple[list[RetrievedChunk], int]:
    """Retrieve top-K chunks for a query and label them using the golden annotations."""
    q_vec  = mock_embed(query)
    scored = sorted(
        [(doc, cosine_sim(q_vec, mock_embed(doc["content"]))) for doc in KNOWLEDGE_BASE],
        key=lambda x: -x[1]
    )[:k]

    results = [
        RetrievedChunk(
            chunk_id  = doc["id"],
            content   = doc["content"],
            rank      = rank,
            score     = score,
            relevance = doc[relevance_field],
        )
        for rank, (doc, score) in enumerate(scored, start=1)
    ]
    total_relevant = sum(1 for doc in KNOWLEDGE_BASE if doc[relevance_field] >= 1)
    return results, total_relevant


def run_context_metrics_demo():
    """Compute and display context precision and recall for three queries."""

    print("=" * 70)
    print("CONTEXT PRECISION & RECALL: Measuring Retrieval Quality")
    print("=" * 70)

    k = 5
    agg_precision = agg_recall = agg_ap = agg_ndcg = 0.0

    for tc in QUERIES:
        retrieved, total_rel = retrieve(tc["query"], k=k, relevance_field=tc["relevance_field"])
        result = compute_context_metrics(tc["query"], retrieved, total_rel, k=k)
        result.display()
        agg_precision += result.precision_at_k
        agg_recall    += result.recall_at_k
        agg_ap        += result.ap
        agg_ndcg      += result.ndcg

    n = len(QUERIES)
    print(f"\n  {'═'*65}")
    print(f"  AGGREGATE OVER {n} QUERIES:")
    print(f"  Context Precision@{k}: {agg_precision/n:.3f}")
    print(f"  Context Recall@{k}:    {agg_recall/n:.3f}")
    print(f"  Mean Average Precision: {agg_ap/n:.3f}")
    print(f"  NDCG@{k}:               {agg_ndcg/n:.3f}")


def precision_recall_tradeoff():
    """Show the precision-recall tradeoff as K varies."""

    print("\n" + "=" * 70)
    print("PRECISION-RECALL TRADEOFF AS K VARIES")
    print("=" * 70)
    print(f"""
  SETUP: Query = "APIC HA requirements"
         There are 4 highly relevant docs + 2 somewhat relevant in the corpus.

  K   Precision@K   Recall@K   What happens
  ─   ───────────   ────────   ─────────────────────────────────────────
  1   1.00          0.20       Only 1 doc retrieved — likely relevant, misses 80%
  3   0.80          0.50       3 docs: 2-3 relevant — decent but still misses half
  5   0.60          0.75       5 docs: 3 relevant — good recall, some noise enters
  10  0.40          0.90       10 docs: 4 relevant — high recall, 60% noise
  15  0.27          0.95       15 docs: 4 relevant — near-full recall, 73% noise

  IMPLICATION:
    Increasing K always raises recall but always lowers precision.
    This is the fundamental precision-recall tradeoff in retrieval.

  HOW RERANKING HELPS:
    K=50 retrieval (high recall) → reranker top-5 (high precision)
    Gets the best of both: high recall entering reranker, high precision to LLM.

  PRODUCTION RECOMMENDATION:
    K=50–100 at retrieval stage  (ensures recall)
    M=3–7 after reranking         (ensures precision)
    Monitor both: if recall drops, increase K; if precision drops, tune reranker.

  DIAGNOSTIC TABLE:
  ┌──────────────────────┬───────────────────────────────────────────┐
  │ Symptom              │ Probable cause and fix                    │
  ├──────────────────────┼───────────────────────────────────────────┤
  │ Low precision, OK    │ Too many irrelevant docs in context.      │
  │ recall               │ Fix: add reranking or reduce K.           │
  ├──────────────────────┼───────────────────────────────────────────┤
  │ Low recall, OK       │ Relevant chunks not retrieved at all.     │
  │ precision            │ Fix: increase K, improve chunking,        │
  │                      │ add query rewriting (HyDE/multi-query).   │
  ├──────────────────────┼───────────────────────────────────────────┤
  │ Both low             │ Embedding model or chunking fundamentally  │
  │                      │ wrong for your domain. Consider fine-tune. │
  └──────────────────────┴───────────────────────────────────────────┘
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_context_metrics_demo()
    precision_recall_tradeoff()
