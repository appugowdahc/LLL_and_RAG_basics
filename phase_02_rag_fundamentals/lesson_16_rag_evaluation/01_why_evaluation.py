"""
FILE: 01_why_evaluation.py
LESSON: Phase 2 - Lesson 16 - RAG Evaluation
TOPIC: Why evaluation is non-negotiable and what can fail silently

WHAT THIS FILE TEACHES:
  - Four silent failure modes in a RAG pipeline
  - What "eval-driven development" means for RAG
  - Metric landscape: retrieval metrics vs generation metrics
  - The RAGAS framework overview (Es et al., 2023)
  - WHY you need both automated metrics and human baselines

INSTALL: no external dependencies
"""

import re
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─── Silent Failure Taxonomy ──────────────────────────────────────────────────

@dataclass
class FailureMode:
    name:         str
    layer:        str    # retrieval / generation / both
    description:  str
    symptom:      str    # what the user sees
    metric:       str    # which RAGAS metric catches it
    example:      str


SILENT_FAILURES = [
    FailureMode(
        name        = "Topical Hit, Wrong Chunk",
        layer       = "retrieval",
        description = "The retriever finds the right section of the document "
                      "but ranks the wrong chunk first. The answer is in chunk 4 "
                      "but the pipeline only sends chunks 1–3 to the LLM.",
        symptom     = "LLM responds: 'The context does not contain this information.'",
        metric      = "Context Recall",
        example     = "Query: 'What is the max leaf count per ACI pod?'\n"
                      "Chunk 1: 'ACI pods are connected via IPN.' (retrieved #1)\n"
                      "Chunk 4: 'ACI 6.0 supports 200 leafs per pod.' (missed)",
    ),
    FailureMode(
        name        = "Fluent Hallucination",
        layer       = "generation",
        description = "The LLM produces a well-formed, confident-sounding answer "
                      "that goes beyond what the context states. The extra fact "
                      "is drawn from parametric (training) memory, not context.",
        symptom     = "Answer sounds correct but contains a wrong or unverifiable detail.",
        metric      = "Faithfulness",
        example     = "Context: 'APIC requires 3 nodes for HA.'\n"
                      "Answer:  'APIC requires 3 nodes for HA. It was first "
                      "introduced in ACI 1.0 in 2014.' ← fabricated claim",
    ),
    FailureMode(
        name        = "Evasive Answer",
        layer       = "generation",
        description = "The LLM has the correct chunks but refuses to commit to "
                      "a direct answer. It hedges excessively, restates the question, "
                      "or answers a slightly different question.",
        symptom     = "User gets a vague non-answer despite the information being in context.",
        metric      = "Answer Relevance",
        example     = "Query: 'How many APIC nodes are required?'\n"
                      "Answer: 'The number of APIC nodes depends on your "
                      "availability requirements and deployment scenario.' ← evasion",
    ),
    FailureMode(
        name        = "Noise Retrieval",
        layer       = "retrieval",
        description = "Too many off-topic chunks make it into the top-K, "
                      "diluting the relevant content. The LLM's attention is "
                      "split and it may weight irrelevant chunks too heavily.",
        symptom     = "Answers are partially correct but contain irrelevant details.",
        metric      = "Context Precision",
        example     = "Query: 'APIC HA requirements?'\n"
                      "Retrieved K=10: 3 relevant + 7 about BGP, spanning-tree, "
                      "ISE TrustSec — all ACI-related but not HA-related.",
    ),
]


def demonstrate_silent_failures():
    """Print each failure mode with symptoms and the metric that catches it."""

    print("=" * 70)
    print("SILENT FAILURE MODES IN RAG")
    print("=" * 70)
    print(f"""
  WHY 'SILENT':
    These failures don't crash the pipeline. The LLM always produces text.
    Without evaluation, they are invisible — the system appears to work
    while producing wrong, incomplete, or hallucinated answers in production.
""")

    for i, fm in enumerate(SILENT_FAILURES, 1):
        print(f"  FAILURE {i}: {fm.name}  [{fm.layer.upper()} layer]")
        print(f"  Description: {fm.description}")
        print(f"  Symptom:     {fm.symptom}")
        print(f"  Caught by:   {fm.metric}")
        print(f"  Example:\n    {fm.example.replace(chr(10), chr(10)+'    ')}")
        print()


# ─── Metric Landscape ─────────────────────────────────────────────────────────

def metric_landscape():
    """Map metrics to what they measure and which layer they cover."""

    print("=" * 70)
    print("METRIC LANDSCAPE: What Each Metric Measures")
    print("=" * 70)
    print(f"""
  RETRIEVAL METRICS (Stage 1 quality):
  ┌─────────────────────┬──────────────────────────────────────────────┐
  │ Metric              │ What it asks                                 │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ Context Precision   │ Of all retrieved chunks, what fraction       │
  │                     │ are actually relevant?                       │
  │                     │ HIGH = low noise in context                  │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ Context Recall      │ Of all needed information, what fraction     │
  │                     │ did retrieval find?                          │
  │                     │ HIGH = no critical chunks were missed        │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ MRR@K, NDCG@K       │ How high does the first relevant chunk rank? │
  │                     │ (From Lesson 15)                             │
  └─────────────────────┴──────────────────────────────────────────────┘

  GENERATION METRICS (LLM output quality):
  ┌─────────────────────┬──────────────────────────────────────────────┐
  │ Metric              │ What it asks                                 │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ Faithfulness        │ Is every claim in the answer supported       │
  │                     │ by retrieved context? (anti-hallucination)   │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ Answer Relevance    │ Does the answer address the question         │
  │                     │ that was asked?                              │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ Correctness         │ Is the answer factually correct?             │
  │                     │ (requires reference answer — expensive)      │
  └─────────────────────┴──────────────────────────────────────────────┘

  RAGAS METRIC ORIGIN:
    Es et al. (2023) — "RAGAS: Automated Evaluation of Retrieval
    Augmented Generation" — arXiv:2309.15217
    GitHub: explodinggradients/ragas

  WHICH METRICS TO TRACK IN PRODUCTION:
    Minimum viable eval set:
      Faithfulness         ← are we hallucinating?
      Context Recall       ← are we finding all the needed chunks?
    Full eval set adds:
      Answer Relevance     ← is the LLM actually answering?
      Context Precision    ← are we cluttering context with noise?
""")


# ─── Eval-Driven Development ──────────────────────────────────────────────────

def eval_driven_development():
    """Show the development cycle that uses metrics to guide improvements."""

    print("=" * 70)
    print("EVAL-DRIVEN DEVELOPMENT FOR RAG")
    print("=" * 70)
    print(f"""
  WORKFLOW:

    1. Build a golden dataset FIRST (50–200 (query, answer, chunks) triples)
       ↓
    2. Implement baseline pipeline (simple chunking, no reranking)
    ↓
    3. Run eval → record baseline metrics
    ↓
    4. Make ONE change (e.g., switch chunk size from 256 to 512 tokens)
    ↓
    5. Re-run eval → compare metrics vs baseline
    ↓
    6. Keep the change if metrics improve; revert if not
    ↓
    Repeat for each component: chunking → query rewriting → reranking → prompt

  WHY ONE CHANGE AT A TIME:
    If you change chunking + reranking simultaneously and metrics improve,
    you don't know which change helped (or if one helped and one hurt).
    Ablation discipline is mandatory.

  TYPICAL IMPROVEMENT SEQUENCE FOR A NEW RAG SYSTEM:
  ┌────────────────────────────────┬────────────────────────────────────┐
  │ Change                         │ Typical metric gain                │
  ├────────────────────────────────┼────────────────────────────────────┤
  │ Baseline (no RAG)              │ Faithfulness: ~0.40                │
  │ Add RAG (naive chunking)       │ Faithfulness: +0.25 → 0.65        │
  │ Semantic chunking              │ Context Recall: +0.10             │
  │ Add reranking (Cohere)         │ Context Precision: +0.15          │
  │ Query rewriting (HyDE)         │ Context Recall: +0.08             │
  │ Better prompt (strict grounded)│ Faithfulness: +0.15 → ~0.90      │
  └────────────────────────────────┴────────────────────────────────────┘

  TARGET METRICS FOR PRODUCTION SIGN-OFF:
    Faithfulness       ≥ 0.90
    Context Precision  ≥ 0.75
    Context Recall     ≥ 0.80
    Answer Relevance   ≥ 0.85

  IF ONE METRIC IS LOW:
    Low Faithfulness   → Fix the generation prompt (add strict grounding rule)
    Low Context Recall → Fix retrieval (more K, query rewriting, better chunking)
    Low Precision      → Fix reranking (add reranker, increase score threshold)
    Low Relevance      → Fix query understanding or LLM prompt structure
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demonstrate_silent_failures()
    metric_landscape()
    eval_driven_development()
