"""
FILE: 01_why_reranking.py
LESSON: Phase 2 - Lesson 15 - Reranking
TOPIC: Why retrieval ranking fails and where reranking fits in the pipeline

WHAT THIS FILE TEACHES:
  - The bi-encoder problem: independent encodings lose query-document interaction
  - Three concrete failure modes of pure vector similarity ranking
  - Position-recall degradation: rank 1 vs rank 10 quality difference
  - Why two-stage retrieval (retrieve-K, rerank-M) is the production standard
  - Cost and latency tradeoffs at each stage

INSTALL: no external dependencies
"""

import re
import math
import hashlib
import numpy as np
from dataclasses import dataclass


# ─── Mock Embedding ───────────────────────────────────────────────────────────

def mock_embed(text: str, dims: int = 64) -> np.ndarray:
    """
    Deterministic mock embedding.
    Texts that share keywords get similar embeddings (same topic seed).
    WHY topic-keyed seed: simulates the property that semantically related
    texts cluster together in real embedding spaces.
    """
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


# ─── Bi-Encoder Failure Modes ─────────────────────────────────────────────────

@dataclass
class FailureModeExample:
    mode:        str
    query:       str
    high_ranked: str   # document that scores high (wrong or suboptimal)
    low_ranked:  str   # document that should score higher (correct but lowly ranked)
    explanation: str


FAILURE_MODES = [
    FailureModeExample(
        mode        = "Semantic Overlap Trap",
        query       = "How many APIC nodes are required for high availability?",
        high_ranked = "APIC nodes are critical components of the Cisco ACI fabric management system. "
                      "APIC nodes communicate with leaf and spine switches. APIC is Cisco's policy controller.",
        low_ranked  = "A minimum of 3 APIC nodes form a cluster to maintain quorum and ensure HA.",
        explanation = "The high-ranked doc mentions 'APIC nodes' 3 times — high lexical/semantic overlap. "
                      "But it never answers the question. The low-ranked doc directly answers it. "
                      "Bi-encoder sees topic overlap; cross-encoder sees answer quality.",
    ),
    FailureModeExample(
        mode        = "Specificity Inversion",
        query       = "What is the maximum number of leaf switches in a single ACI pod?",
        high_ranked = "Cisco ACI supports leaf and spine switch topologies with CLOS fabric design. "
                      "Leaf switches connect to endpoints and spine switches provide fabric connectivity.",
        low_ranked  = "ACI 6.0 supports up to 200 leaf switches per pod in a VXLAN-based fabric.",
        explanation = "Generic doc about leaf/spine architecture scores higher because it matches "
                      "vocabulary broadly. Specific answer doc ('200 leaf switches') is buried lower. "
                      "Cross-encoder rewards specificity.",
    ),
    FailureModeExample(
        mode        = "Negation Blindness",
        query       = "Which versions of ACI are NOT affected by the APIC memory leak bug?",
        high_ranked = "The APIC memory leak issue in ACI 5.2(1g) and 5.2(2e) causes instability. "
                      "Affected versions should be patched immediately.",
        low_ranked  = "ACI 6.0 and later versions are unaffected by the APIC memory leak issue fixed in 5.2(3a).",
        explanation = "Query asks about UN-affected versions. Bi-encoder matches 'APIC memory leak' "
                      "and ranks the affected-versions doc high. Cross-encoder understands negation "
                      "and promotes the unaffected-versions doc.",
    ),
]


def demonstrate_failure_modes():
    """Show concrete ranking inversions that bi-encoder retrieval produces."""

    print("=" * 70)
    print("BI-ENCODER FAILURE MODES: Why Vector Similarity Is Not Enough")
    print("=" * 70)

    for i, fm in enumerate(FAILURE_MODES, 1):
        q_vec    = mock_embed(fm.query)
        high_sim = cosine_sim(q_vec, mock_embed(fm.high_ranked))
        low_sim  = cosine_sim(q_vec, mock_embed(fm.low_ranked))

        print(f"\n  FAILURE MODE {i}: {fm.mode}")
        print(f"  Query: '{fm.query[:80]}'")
        print()
        print(f"  Bi-encoder rank 1 (score={high_sim:.3f}):")
        print(f"    '{fm.high_ranked[:90]}...'")
        print(f"  Bi-encoder rank 2 (score={low_sim:.3f}):")
        print(f"    '{fm.low_ranked[:90]}...'")
        print()
        print(f"  WHY this is wrong: {fm.explanation}")
        print()
        print(f"  Cross-encoder fix: would see BOTH texts simultaneously and recognize")
        print(f"    that rank 2 actually answers the query — inverting the ranking.")


# ─── Retrieval Quality Degradation by Rank ────────────────────────────────────

ANNOTATED_CORPUS = [
    # (text, relevance_to_query)
    # Query: "APIC cluster minimum nodes for HA"
    # relevance: 2=highly relevant, 1=somewhat relevant, 0=not relevant
    ("A minimum of 3 APIC nodes form a cluster. With 3 nodes, one can fail while two maintain quorum.", 2),
    ("APIC communicates with leaf and spine switches using OpFlex policy protocol.", 0),
    ("For ACI HA, deploy 3 APIC nodes in separate failure domains or power domains.", 2),
    ("APIC nodes are distributed across the fabric. Each node runs Cisco NX-OS.", 1),
    ("The APIC GUI is available on port 443 using HTTPS. REST API uses the same port.", 0),
    ("High availability in ACI requires an odd number of APIC nodes to prevent split-brain.", 2),
    ("Cisco recommends deploying APIC on dedicated Cisco APIC hardware or VMware vSphere.", 1),
    ("BGP route reflection is used as the control plane for ACI Multi-Pod IPN.", 0),
    ("APIC cluster health can be monitored via the Fault Manager dashboard in the GUI.", 1),
    ("A 3-node APIC cluster maintains policy even when one physical APIC fails.", 2),
]


def measure_dcg(results: list[tuple[str, int]], k: int) -> float:
    """
    Discounted Cumulative Gain @K.
    WHY DCG: heavily penalizes highly relevant docs appearing at low ranks.
    DCG = Σ (relevance_i / log2(rank_i + 1)) for i in 1..K
    """
    dcg = 0.0
    for rank, (_, rel) in enumerate(results[:k], start=1):
        dcg += rel / math.log2(rank + 1)    # WHY log2: standard logarithmic discount
    return dcg


def ideal_dcg(results: list[tuple[str, int]], k: int) -> float:
    """IDCG: DCG of the ideal (perfectly sorted) ranking."""
    sorted_results = sorted(results, key=lambda x: -x[1])
    return measure_dcg(sorted_results, k)


def rank_quality_demo():
    """
    Show how DCG decreases as relevant documents appear at lower ranks.
    This motivates why we want the most relevant docs ranked highest.
    """

    print("\n" + "=" * 70)
    print("RANKING QUALITY: Why Rank Position Matters")
    print("=" * 70)

    query = "APIC cluster minimum nodes for HA"
    q_vec = mock_embed(query)

    # Bi-encoder retrieval
    scored = sorted(
        [(text, rel, cosine_sim(q_vec, mock_embed(text)))
         for text, rel in ANNOTATED_CORPUS],
        key=lambda x: -x[2]
    )

    print(f"\n  Query: '{query}'")
    print(f"\n  Bi-encoder ranking:")
    print(f"  {'Rank':<5} {'Score':>6}  {'Rel':>3}  Content")
    print(f"  {'─'*4} {'─'*6}  {'─'*3}  {'─'*55}")

    bi_results = []
    for rank, (text, rel, score) in enumerate(scored, 1):
        marker = "✓✓" if rel == 2 else ("✓" if rel == 1 else "  ")
        print(f"  {rank:<5} {score:>6.3f}  {rel:>3} {marker} '{text[:55]}'")
        bi_results.append((text, rel))

    # Metrics
    k = 5
    bi_dcg   = measure_dcg(bi_results, k)
    ideal    = ideal_dcg(bi_results, k)
    ndcg     = bi_dcg / ideal if ideal > 0 else 0.0

    print(f"\n  DCG@{k}:  {bi_dcg:.3f} (ideal: {ideal:.3f})")
    print(f"  NDCG@{k}: {ndcg:.3f}  (1.0 = perfect ranking)")
    print(f"\n  Observation:")
    print(f"    Highly relevant docs (rel=2) are scattered across ranks 1–{len(scored)}.")
    print(f"    A reranker would move all rel=2 docs to ranks 1–4, raising NDCG toward 1.0.")


# ─── Two-Stage Architecture Walkthrough ───────────────────────────────────────

def two_stage_architecture():
    """Walk through the pipeline logic and cost model for two-stage retrieval."""

    print("\n" + "=" * 70)
    print("TWO-STAGE RETRIEVAL ARCHITECTURE")
    print("=" * 70)

    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STAGE 1: RETRIEVAL (Bi-Encoder + ANN Index)                        │
  │                                                                     │
  │  Input:  Raw query                                                  │
  │  Process: Embed query → cosine similarity over HNSW index           │
  │  Output:  Top-K candidates (K = 50–200)                             │
  │  Latency: 5–50ms (pre-computed document embeddings)                 │
  │  Model:   Voyage AI / OpenAI Ada (bi-encoder, 768-dim)              │
  │  WHY:     Fast. Pre-computable. Scales to millions of documents.    │
  └─────────────────────────────────────────────────────────────────────┘
                             ↓ Top-K candidates
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STAGE 2: RERANKING (Cross-Encoder or LLM)                          │
  │                                                                     │
  │  Input:  [Query + Doc_1], [Query + Doc_2], ..., [Query + Doc_K]     │
  │  Process: K forward passes of cross-encoder (joint query-doc input) │
  │  Output:  Re-scored list → select top-M (M = 3–10)                 │
  │  Latency: 50–300ms (K forward passes, parallelizable)              │
  │  Model:   cross-encoder/ms-marco-MiniLM or Cohere Rerank            │
  │  WHY:     Precise. Sees query+doc together. Corrects stage-1 errors.│
  └─────────────────────────────────────────────────────────────────────┘
                             ↓ Top-M chunks
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STAGE 3: GENERATION (LLM)                                          │
  │                                                                     │
  │  Input:  System prompt + top-M chunks + user query                  │
  │  Output: Final answer                                               │
  │  Latency: 200–2000ms depending on output length                     │
  └─────────────────────────────────────────────────────────────────────┘

  PARAMETER GUIDANCE:
  ┌───────────┬────────────────┬──────────────────────────────────────┐
  │ Parameter │ Typical range  │ How to choose                        │
  ├───────────┼────────────────┼──────────────────────────────────────┤
  │ K (rerank │ 50–200         │ Start at 100. Increase if precision  │
  │ pool)     │                │ is low (relevant docs not in pool).  │
  ├───────────┼────────────────┼──────────────────────────────────────┤
  │ M (to LLM)│ 3–10           │ 3–5 for factual Q&A.                 │
  │           │                │ 5–10 for synthesis or comparison.    │
  │           │                │ Match to context window budget.      │
  ├───────────┼────────────────┼──────────────────────────────────────┤
  │ Reranker  │ cross-encoder  │ cross-encoder: fast, free to run     │
  │ type      │ or Cohere API  │ Cohere: managed, excellent quality   │
  │           │ or LLM judge   │ LLM judge: best quality, expensive   │
  └───────────┴────────────────┴──────────────────────────────────────┘

  COST MODEL (per query, 2026 pricing):
    Stage 1 (retrieval):  ~$0.0001 (embedding 1 query, ANN search)
    Stage 2 (reranking):  ~$0.001 (Cohere Rerank, 100 docs)
    Stage 3 (generation): ~$0.01  (Haiku, 200K context, short answer)
    ─────────────────────────────────────────
    Total:                ~$0.012 per query
    Reranking share:       ~8%
    Reranking quality gain: +15–30% NDCG@5 over no reranking

  CONCLUSION: Reranking is inexpensive relative to generation costs.
  The quality improvement almost always justifies the cost.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demonstrate_failure_modes()
    rank_quality_demo()
    two_stage_architecture()
