"""
FILE: 05_mini_project_reranking_pipeline.py
LESSON: Phase 2 - Lesson 15 - Reranking
TOPIC: Complete two-stage retrieval + reranking pipeline with evaluation

WHAT THIS FILE TEACHES:
  - Full pipeline: query → Stage 1 retrieve K → Stage 2 rerank → top-M to LLM
  - Evaluation: NDCG@5, MRR@5, Precision@5 before and after reranking
  - Reranker selection: auto-select based on query type and budget
  - Context assembly from reranked docs
  - WHY evaluation matters: without measurement you can't prove improvement

INSTALL: pip install numpy anthropic python-dotenv
"""

import os
import re
import math
import json
import hashlib
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    HAS_ANTHROPIC = False


# ─── Utilities ────────────────────────────────────────────────────────────────

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


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ─── Mock Reranker (cross-encoder simulation) ─────────────────────────────────

def rerank_score(query: str, document: str) -> float:
    """Mock cross-encoder/LLM-judge relevance score."""
    stop = {"what", "how", "does", "the", "and", "for", "with", "are", "is", "can"}
    q   = {t.lower() for t in re.findall(r"\b\w{3,}\b", query) if t.lower() not in stop}
    d   = {t.lower() for t in re.findall(r"\b\w{3,}\b", document) if t.lower() not in stop}
    term_ov  = len(q & d) / max(len(q), 1)
    semantic = cosine_sim(mock_embed(query), mock_embed(document))
    length_f = min(1.0, len(document) / 100.0)
    return float(np.clip(0.45 * semantic + 0.40 * term_ov + 0.15 * length_f, 0.0, 1.0))


# ─── Knowledge Base ───────────────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    {"id": "kb01", "content": "A minimum of 3 APIC nodes form a cluster for high availability quorum.", "topic": "apic-ha",       "relevance_map": {"apic ha": 2, "apic nodes": 2}},
    {"id": "kb02", "content": "APIC communicates with leaf and spine switches using OpFlex policy.",    "topic": "apic-opflex",   "relevance_map": {"apic ha": 0, "apic protocol": 1}},
    {"id": "kb03", "content": "For ACI HA, deploy 3 APIC nodes in separate physical failure domains.", "topic": "apic-ha",       "relevance_map": {"apic ha": 2, "apic nodes": 2}},
    {"id": "kb04", "content": "APIC REST API uses aaaLogin endpoint on port 443 over HTTPS.",          "topic": "apic-api",      "relevance_map": {"apic ha": 0, "apic api": 2}},
    {"id": "kb05", "content": "ACI requires odd APIC node count to avoid split-brain in HA.",          "topic": "apic-ha",       "relevance_map": {"apic ha": 2, "apic nodes": 2}},
    {"id": "kb06", "content": "The APIC cluster health is visible in the Fault Manager dashboard.",    "topic": "apic-ops",      "relevance_map": {"apic ha": 1, "apic monitor": 2}},
    {"id": "kb07", "content": "Cisco recommends APIC on dedicated hardware or VMware vSphere.",        "topic": "apic-deploy",   "relevance_map": {"apic ha": 1, "apic deploy": 2}},
    {"id": "kb08", "content": "BGP route reflection is the ACI Multi-Pod IPN control plane.",         "topic": "multipod",      "relevance_map": {"apic ha": 0, "multipod": 2}},
    {"id": "kb09", "content": "A 3-node APIC cluster maintains policy even when one node fails.",      "topic": "apic-ha",       "relevance_map": {"apic ha": 2, "apic nodes": 2}},
    {"id": "kb10", "content": "ReadyOps validates ACI changes in a Production-Representative env.",    "topic": "readyops",      "relevance_map": {"readyops": 2, "validation": 2}},
    {"id": "kb11", "content": "ReadyOps promotion gate requires 100% pass rate from all agents.",      "topic": "readyops",      "relevance_map": {"readyops": 2, "promotion": 2}},
    {"id": "kb12", "content": "ReadyOps agent classes: Health Posture, Validation, Operational, Stress.", "topic": "readyops",  "relevance_map": {"readyops": 2, "agents": 2}},
    {"id": "kb13", "content": "Hypershield uses eBPF for kernel-level policy at the workload.",        "topic": "hypershield",   "relevance_map": {"hypershield": 2, "ebpf": 2}},
    {"id": "kb14", "content": "Hypershield integrates with ACI EPG membership from APIC.",             "topic": "hypershield",   "relevance_map": {"hypershield": 2, "aci epg": 2}},
    {"id": "kb15", "content": "EPGs in ACI define policy groups. Contracts permit inter-EPG traffic.", "topic": "epg",           "relevance_map": {"epg": 2, "contract": 2}},
]


# ─── Evaluation Metrics ───────────────────────────────────────────────────────

def dcg_at_k(results: list[tuple[dict, int]], k: int) -> float:
    """
    Discounted Cumulative Gain.
    WHY: heavily penalizes highly relevant docs appearing at low ranks.
    DCG = Σ (2^rel - 1) / log2(rank + 1) for rank in 1..K
    WHY (2^rel - 1): the exponential form rewards rel=2 much more than rel=1.
    """
    dcg = 0.0
    for rank, (_, rel) in enumerate(results[:k], start=1):
        dcg += (2**rel - 1) / math.log2(rank + 1)
    return dcg


def ndcg_at_k(results: list[tuple[dict, int]], k: int) -> float:
    """NDCG: DCG normalized by the ideal DCG (perfect ranking)."""
    ideal  = sorted(results, key=lambda x: -x[1])
    idcg   = dcg_at_k(ideal, k)
    if idcg == 0:
        return 1.0
    return dcg_at_k(results, k) / idcg


def mrr_at_k(results: list[tuple[dict, int]], k: int) -> float:
    """
    Mean Reciprocal Rank.
    WHY MRR: 1.0 = relevant doc is #1. 0.5 = at #2. 0.33 = at #3.
    Strong indicator of whether the top result is actually useful.
    """
    for rank, (_, rel) in enumerate(results[:k], start=1):
        if rel >= 1:
            return 1.0 / rank
    return 0.0


def precision_at_k(results: list[tuple[dict, int]], k: int, threshold: int = 1) -> float:
    """Fraction of top-K results with relevance >= threshold."""
    hits = sum(1 for _, rel in results[:k] if rel >= threshold)
    return hits / min(k, len(results))


# ─── Two-Stage Pipeline ───────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Full pipeline result: retrieval + reranking + assembled context."""
    query:           str
    stage1_docs:     list[tuple[dict, float]]  # (doc, bi_score) — pre-reranking
    stage2_docs:     list[tuple[dict, float]]  # (doc, ce_score) — post-reranking
    context:         str
    context_tokens:  int

    # Evaluation scores (if relevance labels provided)
    ndcg_before:   Optional[float] = None
    ndcg_after:    Optional[float] = None
    mrr_before:    Optional[float] = None
    mrr_after:     Optional[float] = None
    prec_before:   Optional[float] = None
    prec_after:    Optional[float] = None

    def display(self):
        print(f"\n  Query: '{self.query}'")
        print(f"\n  Stage 1 (bi-encoder top-5):")
        for rank, (doc, score) in enumerate(self.stage1_docs[:5], 1):
            print(f"    [{rank}] {score:.3f}  '{doc['content'][:60]}'")
        print(f"\n  Stage 2 (reranked top-5):")
        for rank, (doc, score) in enumerate(self.stage2_docs[:5], 1):
            print(f"    [{rank}] {score:.4f}  '{doc['content'][:60]}'")
        if self.ndcg_before is not None:
            print(f"\n  NDCG@5:  {self.ndcg_before:.3f} → {self.ndcg_after:.3f}  "
                  f"(Δ{self.ndcg_after - self.ndcg_before:+.3f})")
            print(f"  MRR@5:   {self.mrr_before:.3f} → {self.mrr_after:.3f}  "
                  f"(Δ{self.mrr_after - self.mrr_before:+.3f})")
            print(f"  P@5:     {self.prec_before:.3f} → {self.prec_after:.3f}  "
                  f"(Δ{self.prec_after - self.prec_before:+.3f})")
        print(f"\n  Context assembled: {self.context_tokens} tokens")


class TwoStagePipeline:
    """
    Complete two-stage retrieval + reranking pipeline.

    Stage 1: Dense retrieval over KNOWLEDGE_BASE (bi-encoder + cosine sim).
    Stage 2: Cross-encoder (mock) reranking of top-K candidates.
    Output:  Top-M chunks assembled into context string for LLM.
    """

    def run(
        self,
        query:          str,
        stage1_k:       int = 10,    # retrieve this many candidates in stage 1
        stage2_m:       int = 5,     # keep this many after reranking
        relevance_map:  Optional[dict] = None,  # {doc_id: relevance} for eval
    ) -> PipelineResult:

        # Stage 1: dense retrieval
        q_vec   = mock_embed(query)
        stage1  = sorted(
            [(doc, cosine_sim(q_vec, mock_embed(doc["content"]))) for doc in KNOWLEDGE_BASE],
            key=lambda x: -x[1],
        )[:stage1_k]

        # Stage 2: cross-encoder reranking
        stage2_scored = sorted(
            [(doc, rerank_score(query, doc["content"])) for doc, _ in stage1],
            key=lambda x: -x[1],
        )[:stage2_m]

        # Assemble context
        context_parts = [
            f"[Chunk {i+1}] {doc['content']}"
            for i, (doc, _) in enumerate(stage2_scored)
        ]
        context = "\n\n".join(context_parts)

        result = PipelineResult(
            query          = query,
            stage1_docs    = stage1,
            stage2_docs    = stage2_scored,
            context        = context,
            context_tokens = approx_tokens(context),
        )

        # Evaluation (optional)
        if relevance_map:
            def labeled(docs):
                return [(doc, relevance_map.get(doc["id"], 0)) for doc, _ in docs]

            s1_labeled = labeled(stage1)
            s2_labeled = labeled(stage2_scored)

            result.ndcg_before = ndcg_at_k(s1_labeled, k=5)
            result.ndcg_after  = ndcg_at_k(s2_labeled, k=5)
            result.mrr_before  = mrr_at_k(s1_labeled,  k=5)
            result.mrr_after   = mrr_at_k(s2_labeled,  k=5)
            result.prec_before = precision_at_k(s1_labeled, k=5)
            result.prec_after  = precision_at_k(s2_labeled, k=5)

        return result


# ─── Evaluation Suite ─────────────────────────────────────────────────────────

EVAL_QUERIES = [
    {
        "query":         "How many APIC nodes are required for ACI high availability?",
        "relevance":     {"kb01": 2, "kb03": 2, "kb05": 2, "kb06": 1, "kb07": 1, "kb09": 2},
        "description":   "APIC HA node count",
    },
    {
        "query":         "What is the ReadyOps promotion gate requirement?",
        "relevance":     {"kb10": 1, "kb11": 2, "kb12": 1},
        "description":   "ReadyOps promotion gate",
    },
    {
        "query":         "How does Hypershield use eBPF to enforce policy?",
        "relevance":     {"kb13": 2, "kb14": 1},
        "description":   "Hypershield eBPF policy",
    },
    {
        "query":         "What is an EPG and how do contracts work in ACI?",
        "relevance":     {"kb15": 2, "kb14": 1},
        "description":   "EPG and contracts",
    },
]


def run_eval_suite():
    """Run evaluation suite: aggregate NDCG, MRR, P@5 before and after reranking."""

    print("=" * 70)
    print("TWO-STAGE PIPELINE EVALUATION SUITE")
    print("=" * 70)

    pipeline = TwoStagePipeline()
    totals   = defaultdict(float)
    n        = len(EVAL_QUERIES)

    for tc in EVAL_QUERIES:
        print(f"\n  ─── [{tc['description']}] ───")
        result = pipeline.run(
            query         = tc["query"],
            stage1_k      = 10,
            stage2_m      = 5,
            relevance_map = tc["relevance"],
        )
        result.display()
        totals["ndcg_b"] += result.ndcg_before
        totals["ndcg_a"] += result.ndcg_after
        totals["mrr_b"]  += result.mrr_before
        totals["mrr_a"]  += result.mrr_after
        totals["prec_b"] += result.prec_before
        totals["prec_a"] += result.prec_after

    print(f"\n  {'═'*65}")
    print(f"  AGGREGATE RESULTS ACROSS {n} QUERIES:")
    print(f"  Metric       Before    After    Delta")
    print(f"  ─────────    ──────    ─────    ─────")
    for metric, (kb, ka) in [
        ("NDCG@5",   ("ndcg_b", "ndcg_a")),
        ("MRR@5",    ("mrr_b",  "mrr_a")),
        ("P@5",      ("prec_b", "prec_a")),
    ]:
        b = totals[kb] / n
        a = totals[ka] / n
        print(f"  {metric:<12} {b:.3f}     {a:.3f}    {a-b:+.3f}")


def reranker_selection_guide():
    """Print decision guide for reranker selection in production."""

    print("\n" + "=" * 70)
    print("RERANKER SELECTION: Production Decision Guide")
    print("=" * 70)
    print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │  Is query volume > 500K/day AND latency SLA < 100ms?               │
  │  YES → Self-hosted cross-encoder (bge-reranker-large on GPU)       │
  │  NO  ↓                                                             │
  │                                                                    │
  │  Is explainability or custom scoring rubric required?              │
  │  YES → LLM-as-judge (Haiku pointwise, offline or low-QPS path)    │
  │  NO  ↓                                                             │
  │                                                                    │
  │  Is multilingual corpus or no GPU available?                       │
  │  YES → Cohere Rerank v3.5 (managed API, best default)              │
  │  NO  ↓                                                             │
  │                                                                    │
  │  DEFAULT: Cohere Rerank v3.5 (high quality, zero infra overhead)   │
  └────────────────────────────────────────────────────────────────────┘

  PARAMETER GUIDANCE FOR PRODUCTION:
  ┌───────────┬────────────┬───────────────────────────────────────────┐
  │ stage1_k  │ 50–150     │ Larger K = higher recall entering reranker│
  │           │            │ 100 is a safe default                     │
  ├───────────┼────────────┼───────────────────────────────────────────┤
  │ stage2_m  │ 3–10       │ 3–5 for factual Q&A                       │
  │           │            │ 5–10 for synthesis or comparison queries   │
  │           │            │ Keep M × avg_chunk_tokens < context budget │
  ├───────────┼────────────┼───────────────────────────────────────────┤
  │ threshold │ 0.05–0.25  │ Filter scores below threshold              │
  │           │            │ Always add fallback: if empty, use top-3   │
  └───────────┴────────────┴───────────────────────────────────────────┘

  PHASE 2 SUMMARY — what you have now:
    Lesson 13: Chunking — fixed, semantic, recursive, hierarchical
    Lesson 14: Query Rewriting — analysis, HyDE, multi-query, decomposition
    Lesson 15: Reranking — bi-encoder failure modes, cross-encoder, Cohere, LLM judge

  PHASE 2 REMAINING:
    Lesson 16: RAG Evaluation — RAGAS, faithfulness, answer relevance, context precision
    Lesson 17: Generation Patterns — citation-grounded prompts, chain-of-thought, refusal
    Lesson 18: Production RAG — monitoring, latency SLOs, cost attribution, deployment
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_eval_suite()
    reranker_selection_guide()
