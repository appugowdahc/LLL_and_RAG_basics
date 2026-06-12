"""
FILE: 02_faithfulness.py
LESSON: Phase 2 - Lesson 16 - RAG Evaluation
TOPIC: Faithfulness — measuring whether the answer is grounded in context

WHAT THIS FILE TEACHES:
  - Faithfulness definition: claims supported by context / total claims
  - Claim extraction: breaking an answer into atomic verifiable statements
  - Claim verification: checking each claim against retrieved chunks
  - LLM-as-judge faithfulness (RAGAS approach)
  - Lexical proxy for fast offline faithfulness estimation
  - WHY faithfulness is the primary anti-hallucination metric in RAG

INSTALL: pip install anthropic python-dotenv
"""

import os
import re
import json
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    HAS_ANTHROPIC = False


# ─── Claim Extraction ─────────────────────────────────────────────────────────

CLAIM_EXTRACTION_SYSTEM = """You are a claim extractor for evaluating AI-generated answers.
Given an answer text, extract every atomic factual claim — statements that are individually verifiable.

Rules:
  - One claim per sentence or sub-clause.
  - Keep claims self-contained (no pronouns).
  - Strip hedging language ("I think", "likely") — extract the underlying claim.
  - Ignore meta-statements ("Based on the context...", "According to the document...").

Output ONLY a JSON array of strings, one string per claim. Example:
["Claim one.", "Claim two.", "Claim three."]"""

CLAIM_EXTRACTION_USER = "Answer text:\n{answer}\n\nExtract all atomic claims as a JSON array:"

MOCK_CLAIM_EXTRACTIONS = {
    "3 nodes apic ha":   ["APIC requires a minimum of 3 nodes for high availability.",
                          "With 3 APIC nodes, one node can fail without losing quorum."],
    "readyops validation":["ReadyOps validates changes in a Production-Representative environment.",
                           "ReadyOps requires a 100% pass rate for promotion.",
                           "ReadyOps agent classes include Health and Posture, Validation, and Stress."],
    "hypershield ebpf":  ["Hypershield uses eBPF for kernel-level policy enforcement.",
                          "eBPF does not require dedicated hardware appliances.",
                          "Hypershield integrates with ACI EPG membership."],
}


def extract_claims(answer: str) -> list[str]:
    """
    Break an answer into a list of atomic verifiable claims.

    WHY atomic claims:
      "APIC requires 3 nodes for HA and was released in ACI 1.0."
      → Two claims: (1) "APIC requires 3 nodes for HA" (verifiable),
                    (2) "Released in ACI 1.0" (potentially fabricated).
      If we only check the whole sentence, we miss that claim (2) is wrong.
      Atomization lets us score at claim granularity.
    """
    if HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 300,
            system     = CLAIM_EXTRACTION_SYSTEM,
            messages   = [{"role": "user", "content": CLAIM_EXTRACTION_USER.format(answer=answer)}],
        )
        text = resp.content[0].text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try line-by-line
            return [line.strip("- •").strip() for line in text.splitlines() if line.strip()]
    else:
        a_low = answer.lower()
        for key, claims in MOCK_CLAIM_EXTRACTIONS.items():
            if all(w in a_low for w in key.split()):
                return claims
        # Heuristic: split on period + space
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]


# ─── Claim Verification ───────────────────────────────────────────────────────

VERIFICATION_SYSTEM = """You are a fact-checker for AI-generated answers.
Given a claim and a context passage, determine whether the claim is directly supported,
contradicted, or not addressed by the context.

Output ONLY a JSON object in this exact format:
{"verdict": "supported"|"contradicted"|"not_in_context", "reason": "<one sentence>"}"""

VERIFICATION_USER = """Context:
{context}

Claim to verify: "{claim}"

Is this claim supported, contradicted, or not addressed by the context? Output JSON:"""


def _lexical_entailment(claim: str, context: str, threshold: float = 0.35) -> bool:
    """
    Fast lexical proxy for claim verification.
    WHY lexical: no LLM call needed. Good enough for claims with unique key terms.
    Returns True if enough of the claim's key terms appear in context.
    """
    stop = {"the", "is", "are", "was", "for", "and", "or", "in", "of", "to",
            "a", "an", "that", "this", "with", "by"}
    claim_tok   = {t.lower() for t in re.findall(r"\b\w{3,}\b", claim) if t.lower() not in stop}
    context_tok = {t.lower() for t in re.findall(r"\b\w{3,}\b", context)}
    if not claim_tok:
        return True
    overlap = len(claim_tok & context_tok) / len(claim_tok)
    return overlap >= threshold


def verify_claim(claim: str, context: str, use_llm: bool = True) -> tuple[str, str]:
    """
    Verify whether a single claim is supported by context.
    Returns (verdict, reason) where verdict in {"supported", "contradicted", "not_in_context"}.

    WHY two-stage verification:
      1. Lexical proxy first (fast, no LLM): if overlap is very high → supported.
      2. LLM judge second: for ambiguous cases where lexical fails.
    """
    # Fast path: lexical check first
    if _lexical_entailment(claim, context, threshold=0.60):
        return "supported", "Key terms from claim found in context."

    if use_llm and HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 80,
            system     = VERIFICATION_SYSTEM,
            messages   = [{"role": "user", "content": VERIFICATION_USER.format(
                context=context[:1000], claim=claim
            )}],
        )
        text = resp.content[0].text.strip()
        try:
            parsed = json.loads(text)
            return parsed["verdict"], parsed.get("reason", "")
        except (json.JSONDecodeError, KeyError):
            verdict = "supported" if "support" in text.lower() else "not_in_context"
            return verdict, text[:100]
    else:
        # Mock: use lexical with lower threshold
        supported = _lexical_entailment(claim, context, threshold=0.25)
        verdict   = "supported" if supported else "not_in_context"
        reason    = ("Claim terms found in context." if supported
                     else "Claim terms not found in context.")
        return verdict, reason


# ─── Faithfulness Scorer ──────────────────────────────────────────────────────

@dataclass
class ClaimVerdict:
    claim:   str
    verdict: str    # supported / contradicted / not_in_context
    reason:  str


@dataclass
class FaithfulnessResult:
    answer:        str
    context:       str
    claims:        list[ClaimVerdict]
    score:         float     # supported_count / total_claims
    supported:     int
    contradicted:  int
    not_in_context: int

    def display(self):
        print(f"\n  Faithfulness score: {self.score:.2f}")
        print(f"  Claims: {len(self.claims)} total  |  "
              f"{self.supported} supported  |  "
              f"{self.contradicted} contradicted  |  "
              f"{self.not_in_context} not in context")
        for cv in self.claims:
            icon = {"supported": "✓", "contradicted": "✗", "not_in_context": "?"}.get(cv.verdict, "?")
            print(f"  [{icon}] '{cv.claim[:70]}'")
            print(f"      → {cv.reason[:65]}")


def faithfulness_score(
    answer:  str,
    context: str,
) -> FaithfulnessResult:
    """
    Compute the faithfulness of an answer against its retrieved context.

    Algorithm (RAGAS-inspired):
      1. Extract atomic claims from answer.
      2. Verify each claim against context (lexical + LLM).
      3. faithfulness = supported_count / total_claims.

    WHY this catches hallucination:
      Parametric memory claims (things the LLM "knows" but weren't retrieved)
      will fail the context verification step — they are not_in_context.
      Faithfulness score directly measures how much the LLM is "going off-script."
    """
    claims = extract_claims(answer)
    verdicts = []
    supported = contradicted = not_in_context = 0

    for claim in claims:
        verdict, reason = verify_claim(claim, context)
        verdicts.append(ClaimVerdict(claim=claim, verdict=verdict, reason=reason))
        if verdict == "supported":
            supported += 1
        elif verdict == "contradicted":
            contradicted += 1
        else:
            not_in_context += 1

    total = len(claims)
    score = supported / total if total > 0 else 1.0  # no claims → trivially faithful

    return FaithfulnessResult(
        answer         = answer,
        context        = context,
        claims         = verdicts,
        score          = score,
        supported      = supported,
        contradicted   = contradicted,
        not_in_context = not_in_context,
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "description": "Fully grounded answer (should be ~1.0)",
        "context": (
            "A minimum of 3 APIC nodes are required for high availability in Cisco ACI. "
            "When one APIC node fails, the remaining two maintain quorum and continue "
            "managing fabric policy without interruption."
        ),
        "answer": (
            "APIC requires a minimum of 3 nodes for high availability. "
            "With 3 nodes, one can fail while the remaining two maintain quorum."
        ),
    },
    {
        "description": "Partially hallucinated answer (should be ~0.5)",
        "context": (
            "A minimum of 3 APIC nodes are required for high availability in Cisco ACI."
        ),
        "answer": (
            "APIC requires a minimum of 3 nodes for high availability. "
            "APIC was first introduced in ACI version 1.0 released in 2013."
        ),
    },
    {
        "description": "ReadyOps with extra claim (mixed)",
        "context": (
            "ReadyOps validates changes in a Production-Representative environment. "
            "The promotion gate requires 100% pass rate from all Validation agents."
        ),
        "answer": (
            "ReadyOps validates changes in a Production-Representative environment. "
            "The promotion gate requires 100% pass rate. "
            "ReadyOps uses machine learning to detect anomalies in real time."
        ),
    },
]


def run_faithfulness_demo():
    print("=" * 70)
    print("FAITHFULNESS SCORING: Claim-Level Answer Grounding")
    print("=" * 70)

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n  ─── Test {i}: {tc['description']} ───")
        print(f"  Context: '{tc['context'][:100]}...'")
        print(f"  Answer:  '{tc['answer'][:100]}...'")

        result = faithfulness_score(tc["answer"], tc["context"])
        result.display()

    print(f"""
  HOW TO INTERPRET:
    ≥ 0.90:  Production-ready faithfulness. LLM is well-grounded.
    0.70–0.89: Moderate hallucination. Strengthen grounding prompt.
    0.50–0.69: Significant fabrication. Audit retrieval + prompt.
    < 0.50:  Severe hallucination. Pipeline review needed.

  MOST COMMON ROOT CAUSES OF LOW FAITHFULNESS:
    1. Weak grounding prompt — LLM not told to ONLY use context.
    2. Missing context — relevant chunk not retrieved, LLM fills from memory.
    3. Short context — only 1–2 chunks, LLM extends beyond what's there.
    4. LLM model too strong parametric memory vs instruction following.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_faithfulness_demo()
