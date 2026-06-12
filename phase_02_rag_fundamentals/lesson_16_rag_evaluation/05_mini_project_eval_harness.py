"""
FILE: 05_mini_project_eval_harness.py
LESSON: Phase 2 - Lesson 16 - RAG Evaluation
TOPIC: Complete RAG evaluation harness with golden dataset and all four metrics

WHAT THIS FILE TEACHES:
  - Building a golden dataset (query, reference answer, relevant chunk IDs)
  - Full eval loop: for each query → retrieve → generate → score all metrics
  - Aggregating RAGAS metrics across a query set
  - Regression detection: comparing two pipeline configurations
  - Generating an eval report with pass/fail per metric
  - WHY you run eval before AND after every pipeline change

INSTALL: pip install anthropic python-dotenv numpy
"""

import os
import re
import json
import math
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


# ─── Knowledge Base ───────────────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    {"id": "kb01", "content": "A minimum of 3 APIC nodes form a cluster for quorum and HA in Cisco ACI."},
    {"id": "kb02", "content": "APIC communicates with leaf and spine switches using OpFlex policy protocol."},
    {"id": "kb03", "content": "For ACI HA, deploy 3 APIC nodes in separate physical failure domains."},
    {"id": "kb04", "content": "APIC REST API exposes aaaLogin on port 443 over HTTPS for authentication."},
    {"id": "kb05", "content": "ACI requires an odd number of APIC nodes to avoid split-brain in HA scenarios."},
    {"id": "kb06", "content": "A 3-node APIC cluster maintains policy management even when one node fails."},
    {"id": "kb07", "content": "ReadyOps is Criterion Networks' continuous validation platform for infrastructure."},
    {"id": "kb08", "content": "ReadyOps validates changes in a Production-Representative environment before promotion."},
    {"id": "kb09", "content": "The ReadyOps promotion gate requires 100% pass rate from all Validation agents."},
    {"id": "kb10", "content": "ReadyOps agent classes: Health and Posture, Validation, Operational, Stress and Adversarial."},
    {"id": "kb11", "content": "Cisco Hypershield uses eBPF for kernel-level microsegmentation at the workload."},
    {"id": "kb12", "content": "Hypershield integrates with ACI EPG membership propagated via APIC for policy."},
    {"id": "kb13", "content": "EPGs in ACI define policy groups. Contracts govern inter-EPG traffic."},
    {"id": "kb14", "content": "ACI Multi-Pod extends the fabric across geographies using a VXLAN IPN."},
    {"id": "kb15", "content": "ISE TrustSec assigns Security Group Tags at authentication for microsegmentation."},
]


# ─── Golden Dataset ────────────────────────────────────────────────────────────

@dataclass
class GoldenQuery:
    """One entry in the evaluation golden dataset."""
    query:            str
    reference_answer: str         # ideal answer written by domain expert
    relevant_ids:     list[str]   # IDs of chunks needed to answer the question
    answer_claims:    list[str]   # atomic claims in the reference answer


GOLDEN_DATASET: list[GoldenQuery] = [
    GoldenQuery(
        query            = "How many APIC nodes are required for high availability?",
        reference_answer = "A minimum of 3 APIC nodes are required for high availability. "
                           "3 nodes prevent split-brain. One can fail while two maintain quorum.",
        relevant_ids     = ["kb01", "kb03", "kb05", "kb06"],
        answer_claims    = ["APIC requires 3 nodes for HA.",
                            "3 nodes prevent split-brain.",
                            "One node can fail while two maintain quorum."],
    ),
    GoldenQuery(
        query            = "What does the ReadyOps promotion gate do?",
        reference_answer = "The ReadyOps promotion gate blocks changes from reaching Live Operations "
                           "until all Validation agent tests pass at 100%.",
        relevant_ids     = ["kb08", "kb09"],
        answer_claims    = ["The promotion gate blocks changes until tests pass.",
                            "All Validation agents must pass at 100%."],
    ),
    GoldenQuery(
        query            = "How does Hypershield enforce policy without appliances?",
        reference_answer = "Hypershield uses eBPF to enforce policy at the kernel level within workloads. "
                           "It integrates with ACI EPG membership from APIC for consistent policy.",
        relevant_ids     = ["kb11", "kb12"],
        answer_claims    = ["Hypershield uses eBPF for policy enforcement.",
                            "Policy is enforced at the kernel level.",
                            "Hypershield integrates with ACI EPG membership."],
    ),
    GoldenQuery(
        query            = "What are the ReadyOps agent classes?",
        reference_answer = "ReadyOps has four agent classes: Health and Posture, Validation, "
                           "Operational, and Stress and Adversarial.",
        relevant_ids     = ["kb10"],
        answer_claims    = ["ReadyOps has four agent classes.",
                            "Classes include Health and Posture, Validation, Operational, and Stress."],
    ),
]


# ─── Inline Metric Functions ──────────────────────────────────────────────────

def _context_precision(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits  = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / max(k, 1)


def _context_recall(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits  = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / max(len(relevant_ids), 1)


def _lexical_faithfulness(claims: list[str], context: str) -> float:
    """Fast lexical proxy for faithfulness: how many claims have terms in context."""
    stop = {"what", "how", "does", "the", "and", "for", "with", "are", "is", "a", "an", "that"}
    supported = 0
    for claim in claims:
        c_tok = {t.lower() for t in re.findall(r"\b\w{3,}\b", claim) if t.lower() not in stop}
        k_tok = {t.lower() for t in re.findall(r"\b\w{3,}\b", context)}
        if not c_tok:
            supported += 1
            continue
        overlap = len(c_tok & k_tok) / len(c_tok)
        if overlap >= 0.4:
            supported += 1
    return supported / max(len(claims), 1)


def _answer_relevance_proxy(query: str, answer: str) -> float:
    """
    Fast proxy for answer relevance (no LLM call).
    Measures how much of the query's vocabulary is present in the answer.
    WHY: if the answer doesn't mention the query's key terms, it's likely evasive.
    """
    stop  = {"what", "how", "does", "the", "and", "for", "with", "are", "is", "an", "a"}
    q_tok = {t.lower() for t in re.findall(r"\b\w{3,}\b", query) if t.lower() not in stop}
    a_tok = {t.lower() for t in re.findall(r"\b\w{3,}\b", answer)}
    if not q_tok:
        return 1.0
    overlap = len(q_tok & a_tok) / len(q_tok)
    # Also measure semantic similarity
    semantic = cosine_sim(mock_embed(query), mock_embed(answer))
    return float(np.clip(0.5 * overlap + 0.5 * semantic, 0.0, 1.0))


# ─── Mock RAG Pipeline ────────────────────────────────────────────────────────

ANSWER_SYSTEM = """Answer the question using ONLY the provided context chunks.
Be specific and concise. Do not add information not in the context.
If the context does not contain the answer, say "Not found in provided context." """

ANSWER_USER = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

MOCK_ANSWERS = {
    "apic nodes":  "APIC requires a minimum of 3 nodes for high availability. "
                   "3 nodes prevent split-brain — one can fail while two maintain quorum.",
    "promotion gate": "The ReadyOps promotion gate blocks changes from reaching Live Operations "
                      "until all Validation agent tests pass at 100%.",
    "hypershield ebpf": "Hypershield uses eBPF for kernel-level policy enforcement at the workload. "
                        "It integrates with ACI EPG membership from APIC for policy consistency.",
    "readyops agent": "ReadyOps has four agent classes: Health and Posture, Validation, "
                      "Operational, and Stress and Adversarial.",
}


def retrieve_chunks(query: str, k: int) -> list[tuple[str, str, float]]:
    """Retrieve top-K chunks: (chunk_id, content, score)."""
    q_vec  = mock_embed(query)
    scored = sorted(
        [(doc["id"], doc["content"], cosine_sim(q_vec, mock_embed(doc["content"])))
         for doc in KNOWLEDGE_BASE],
        key=lambda x: -x[2],
    )
    return scored[:k]


def generate_answer(query: str, context: str) -> str:
    """Generate answer from context (LLM or mock)."""
    if HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 150,
            system     = ANSWER_SYSTEM,
            messages   = [{"role": "user", "content": ANSWER_USER.format(context=context, query=query)}],
        )
        return resp.content[0].text.strip()
    else:
        q_low = query.lower()
        return next(
            (ans for key, ans in MOCK_ANSWERS.items() if any(w in q_low for w in key.split())),
            "This information is not available in the provided context.",
        )


# ─── Evaluation Harness ───────────────────────────────────────────────────────

@dataclass
class QueryEvalResult:
    """Evaluation result for one query."""
    query:             str
    answer:            str
    retrieved_ids:     list[str]
    context_precision: float
    context_recall:    float
    faithfulness:      float
    answer_relevance:  float


@dataclass
class EvalReport:
    """Aggregated evaluation report across all golden queries."""
    pipeline_name:  str
    results:        list[QueryEvalResult]
    mean_precision: float
    mean_recall:    float
    mean_faith:     float
    mean_relevance: float

    # Pass/fail thresholds (production targets)
    PRECISION_THRESHOLD = 0.70
    RECALL_THRESHOLD    = 0.75
    FAITH_THRESHOLD     = 0.85
    RELEVANCE_THRESHOLD = 0.80

    def display(self):
        print(f"\n  Pipeline: {self.pipeline_name}")
        print(f"\n  Per-query breakdown:")
        print(f"  {'Query':<45} {'P@5':>5} {'R@5':>5} {'Faith':>6} {'Relev':>6}")
        print(f"  {'─'*44} {'─'*5} {'─'*5} {'─'*6} {'─'*6}")
        for r in self.results:
            print(f"  {r.query[:44]:<44} {r.context_precision:>5.2f} "
                  f"{r.context_recall:>5.2f} {r.faithfulness:>6.2f} {r.answer_relevance:>6.2f}")

        print(f"\n  ─── Aggregate Metrics ───")
        self._print_metric("Context Precision@5", self.mean_precision, self.PRECISION_THRESHOLD)
        self._print_metric("Context Recall@5",    self.mean_recall,    self.RECALL_THRESHOLD)
        self._print_metric("Faithfulness",         self.mean_faith,     self.FAITH_THRESHOLD)
        self._print_metric("Answer Relevance",     self.mean_relevance, self.RELEVANCE_THRESHOLD)

    def _print_metric(self, name: str, score: float, threshold: float):
        status = "PASS" if score >= threshold else "FAIL"
        bar    = "█" * int(score * 20)
        print(f"  {name:<22} {score:.3f} {bar:<20} [{status}] (threshold: {threshold:.2f})")


class EvalHarness:
    """
    Runs the full evaluation loop for a RAG pipeline.
    For each golden query: retrieve → generate → score all four metrics.
    """

    def run(
        self,
        golden_dataset: list[GoldenQuery],
        pipeline_name:  str = "baseline",
        k:              int = 5,
    ) -> EvalReport:

        results = []

        for gq in golden_dataset:
            # Retrieve
            chunks     = retrieve_chunks(gq.query, k=k)
            chunk_ids  = [cid for cid, _, _ in chunks]
            context    = "\n".join(f"[{i+1}] {text}" for i, (_, text, _) in enumerate(chunks))

            # Generate
            answer     = generate_answer(gq.query, context)

            # Score
            precision  = _context_precision(chunk_ids, gq.relevant_ids, k=k)
            recall     = _context_recall(chunk_ids, gq.relevant_ids, k=k)
            faith      = _lexical_faithfulness(gq.answer_claims, context)
            relevance  = _answer_relevance_proxy(gq.query, answer)

            results.append(QueryEvalResult(
                query             = gq.query,
                answer            = answer,
                retrieved_ids     = chunk_ids,
                context_precision = precision,
                context_recall    = recall,
                faithfulness      = faith,
                answer_relevance  = relevance,
            ))

        n = len(results)
        return EvalReport(
            pipeline_name  = pipeline_name,
            results        = results,
            mean_precision = sum(r.context_precision for r in results) / n,
            mean_recall    = sum(r.context_recall    for r in results) / n,
            mean_faith     = sum(r.faithfulness      for r in results) / n,
            mean_relevance = sum(r.answer_relevance  for r in results) / n,
        )


# ─── Regression Detection ─────────────────────────────────────────────────────

def compare_pipelines(report_a: EvalReport, report_b: EvalReport):
    """
    Compare two pipeline evaluation reports.
    Flags regressions (metrics that dropped by more than 0.03).
    WHY 0.03 threshold: small fluctuations (±0.02) can be noise; 0.03+ is meaningful.
    """
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPARISON: '{report_a.pipeline_name}' vs '{report_b.pipeline_name}'")
    print("=" * 70)
    print(f"\n  {'Metric':<22} {report_a.pipeline_name[:12]:>12} {report_b.pipeline_name[:12]:>12} {'Δ':>7}  Status")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*7}  {'─'*10}")

    metrics = [
        ("Context Precision",  report_a.mean_precision, report_b.mean_precision),
        ("Context Recall",     report_a.mean_recall,    report_b.mean_recall),
        ("Faithfulness",       report_a.mean_faith,     report_b.mean_faith),
        ("Answer Relevance",   report_a.mean_relevance, report_b.mean_relevance),
    ]

    for name, score_a, score_b in metrics:
        delta  = score_b - score_a
        status = "REGRESSION" if delta < -0.03 else ("IMPROVEMENT" if delta > 0.03 else "stable")
        print(f"  {name:<22} {score_a:>12.3f} {score_b:>12.3f} {delta:>+7.3f}  {status}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def run_eval_demo():
    print("=" * 70)
    print("RAG EVALUATION HARNESS: Full Golden Dataset Eval")
    print("=" * 70)

    harness = EvalHarness()

    # Run baseline (k=5)
    baseline = harness.run(GOLDEN_DATASET, pipeline_name="baseline-k5", k=5)
    print(f"\n  === BASELINE PIPELINE ===")
    baseline.display()

    # Simulate "improved" pipeline (k=10 — more recall)
    improved = harness.run(GOLDEN_DATASET, pipeline_name="improved-k10", k=10)
    print(f"\n  === IMPROVED PIPELINE (k=10) ===")
    improved.display()

    # Compare
    compare_pipelines(baseline, improved)

    print(f"""
  HOW TO USE THIS HARNESS IN DEVELOPMENT:
    1. Create a golden dataset from your domain (min 50 queries).
    2. Run harness on baseline → record scores.
    3. Make ONE change (chunk size, k, reranker, prompt).
    4. Re-run harness → compare with compare_pipelines().
    5. Keep changes that improve target metrics.
    6. Gate production releases: all metrics must pass threshold.

  GOLDEN DATASET CREATION TIPS:
    - Cover diverse query types: factual, procedural, comparative, edge cases.
    - Include "impossible" queries (not in KB) to test graceful refusal.
    - Include ambiguous queries to test disambiguation.
    - Include multi-hop queries to test decomposition.
    - Aim for 20% of queries that are hard/adversarial.

  PRODUCTION SIGN-OFF CHECKLIST:
    Context Precision@5  ≥ 0.70  {'✓ PASS' if baseline.mean_precision >= 0.70 else '✗ FAIL'}
    Context Recall@5     ≥ 0.75  {'✓ PASS' if baseline.mean_recall >= 0.75 else '✗ FAIL'}
    Faithfulness         ≥ 0.85  {'✓ PASS' if baseline.mean_faith >= 0.85 else '✗ FAIL'}
    Answer Relevance     ≥ 0.80  {'✓ PASS' if baseline.mean_relevance >= 0.80 else '✗ FAIL'}
""")


if __name__ == "__main__":
    run_eval_demo()
