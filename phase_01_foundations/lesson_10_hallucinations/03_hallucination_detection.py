"""
FILE: 03_hallucination_detection.py
LESSON: Phase 1 - Lesson 10 - Hallucinations
TOPIC: Detection techniques — self-consistency, source attribution, entailment

WHAT THIS FILE TEACHES:
  - Source attribution via prompt engineering (cheapest, zero extra API calls)
  - Self-consistency check (sample N answers, measure divergence)
  - Simple lexical entailment check (sentence-level overlap proxy for NLI)
  - Claim extraction + grounding check
  - WHY each technique has its failure mode

INSTALL: pip install anthropic python-dotenv  (live demo; runs without key too)
"""

import os
import re
import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    HAS_ANTHROPIC = False


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class GroundingResult:
    """Result of checking whether an answer is grounded in context."""
    answer:             str
    context:            str
    grounded_sentences: list[str]
    ungrounded_sentences: list[str]
    grounding_score:    float          # 0.0–1.0: fraction of sentences grounded
    verdict:            str            # "grounded" | "partial" | "hallucinated"


@dataclass
class ConsistencyResult:
    """Result of self-consistency sampling."""
    answers:            list[str]
    agreement_score:    float          # 0.0 = all different, 1.0 = all same
    majority_answer:    Optional[str]
    verdict:            str


# ─── Technique 1: Source Attribution Prompting ───────────────────────────────

ATTRIBUTION_SYSTEM_PROMPT = """You are a precise technical assistant.
Answer ONLY from the provided context. For every sentence in your answer,
append a citation tag like [source:1] using the chunk number from the context.
If a claim is NOT supported by any chunk, write [source:unsupported].
If the context does not contain enough information, say so explicitly.
Never answer from prior knowledge — only from provided context."""

ATTRIBUTION_USER_TEMPLATE = """Context:
{context}

Question: {question}

Instructions:
- Answer in 3-5 sentences maximum.
- Each sentence must end with [source:N] citing the context chunk number, or [source:unsupported].
- Do not add information not found in the context.
"""


def build_attribution_prompt(question: str, chunks: list[str]) -> str:
    """
    Build the user message with numbered context chunks.
    WHY numbered: the model needs explicit identifiers to cite.
    Without numbers, it says "according to the context" — useless for validation.
    """
    context_text = "\n\n".join(
        f"[Chunk {i+1}]: {chunk}"
        for i, chunk in enumerate(chunks)
    )
    return ATTRIBUTION_USER_TEMPLATE.format(
        context  = context_text,
        question = question,
    )


def parse_citations(answer: str) -> dict[str, list[str]]:
    """
    Extract citation tags from an answer with [source:N] notation.

    Returns:
        Dict mapping citation label → list of sentences using that citation.
    """
    # WHY sentence splitting on ". " then filter: handles multi-sentence answers
    sentences  = re.split(r"(?<=[.!?])\s+", answer.strip())
    citations: dict[str, list[str]] = {}

    for sent in sentences:
        # WHY findall: a sentence might cite multiple chunks
        matches = re.findall(r"\[source:([^\]]+)\]", sent)
        for m in matches:
            citations.setdefault(m, []).append(sent)

    return citations


def analyze_attribution(answer: str) -> dict:
    """
    Given an attributed answer, compute the grounding breakdown.
    Returns counts of supported vs unsupported sentences.
    """
    citations     = parse_citations(answer)
    unsupported   = citations.get("unsupported", [])
    supported     = {k: v for k, v in citations.items() if k != "unsupported"}
    total_cited   = sum(len(v) for v in citations.values())
    supported_cnt = sum(len(v) for v in supported.values())

    return {
        "total_sentences":        total_cited,
        "supported_sentences":    supported_cnt,
        "unsupported_sentences":  len(unsupported),
        "grounding_rate":         supported_cnt / max(total_cited, 1),
        "cited_chunks":           list(supported.keys()),
        "unsupported_text":       unsupported,
    }


# ─── Technique 2: Lexical Entailment Proxy ───────────────────────────────────

def lexical_entailment_score(claim: str, context: str) -> float:
    """
    Approximate NLI entailment using word overlap (Jaccard similarity).

    WHY this is a PROXY, not real NLI:
      Real NLI uses a transformer (e.g., DeBERTa-large-mnli) to check whether
      the context semantically IMPLIES the claim. That requires a model call.
      Jaccard overlap is fast and dependency-free, but will miss paraphrases.
      e.g., "minimum 3 nodes" vs "at least three" → Jaccard ≈ 0 but entailment = true.

    In production: use a dedicated NLI model or an LLM-as-judge call.
    Here: Jaccard is a useful teaching approximation.

    Args:
        claim:   The sentence to check (from the LLM's answer).
        context: The source material to check against.

    Returns:
        0.0–1.0 overlap score. > 0.3 is a rough "grounded" threshold.
    """
    def tokenize(text: str) -> set[str]:
        # WHY lowercase: "APIC" and "apic" should match
        tokens = re.findall(r"\b\w+\b", text.lower())
        # WHY filter short: "a", "is", "the" add noise to overlap
        return {t for t in tokens if len(t) > 3}

    claim_tokens   = tokenize(claim)
    context_tokens = tokenize(context)

    if not claim_tokens:
        return 0.0

    intersection = claim_tokens & context_tokens
    union        = claim_tokens | context_tokens

    # WHY intersection/claim_tokens (not Jaccard):
    #   We care about "what fraction of the claim's key terms appear in context"
    #   not the symmetric Jaccard. Claim can be short; context is long.
    return len(intersection) / len(claim_tokens)


def check_answer_grounding(
    answer:    str,
    context:   str,
    threshold: float = 0.30,
) -> GroundingResult:
    """
    Check every sentence in the answer against the context for grounding.

    Args:
        answer:     LLM-generated answer to check.
        context:    Retrieved context that should ground the answer.
        threshold:  Minimum lexical overlap to count as "grounded".

    Returns:
        GroundingResult with per-sentence breakdown and overall verdict.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    grounded,  ungrounded = [], []

    for sent in sentences:
        score = lexical_entailment_score(sent, context)
        if score >= threshold:
            grounded.append(sent)
        else:
            ungrounded.append(sent)

    total         = len(sentences)
    grounding_pct = len(grounded) / max(total, 1)

    if grounding_pct >= 0.80:
        verdict = "grounded"
    elif grounding_pct >= 0.50:
        verdict = "partial"
    else:
        verdict = "hallucinated"

    return GroundingResult(
        answer               = answer,
        context              = context,
        grounded_sentences   = grounded,
        ungrounded_sentences = ungrounded,
        grounding_score      = grounding_pct,
        verdict              = verdict,
    )


# ─── Technique 3: Self-Consistency ───────────────────────────────────────────

def simulate_self_consistency(
    question: str,
    context:  str,
    n:        int = 5,
) -> ConsistencyResult:
    """
    Self-consistency: sample N answers at temp>0, check if they agree.

    WHY this works:
      If the model has strong grounding (e.g., context clearly states the answer),
      all N samples will converge on the same answer even at high temperature.
      If the model is uncertain, samples will diverge → hallucination signal.

    In production: run actual API calls with temperature=0.7 or higher.
    Here: we simulate different answer variants to demonstrate the concept.
    """

    # Simulate: this question has a clear answer in context → high consistency
    clear_answers = [
        "The APIC cluster requires a minimum of 3 nodes for high availability.",
        "ACI requires at least 3 APIC nodes for HA.",
        "For high availability, you need a minimum of 3 APIC nodes.",
        "The minimum APIC node count for HA is 3.",
        "Cisco ACI requires 3 APIC nodes minimum for high availability operation.",
    ]

    # Simulate: this question is ambiguous → low consistency
    uncertain_answers = [
        "The maximum is 180 leaf switches per pod.",
        "ACI supports up to 200 leaf switches per pod in version 6.0.",
        "The leaf count limit is 160 per pod.",
        "ACI can scale to 256 leaf switches in large deployments.",
        "Maximum leaf scale is approximately 200 per pod.",
    ]

    if "minimum" in question.lower() or "how many apic" in question.lower():
        answers = clear_answers[:n]
    else:
        answers = uncertain_answers[:n]

    # Extract the key numeric/specific token from each answer
    def extract_key_token(text: str) -> str:
        numbers = re.findall(r"\b\d+\b", text)
        return numbers[0] if numbers else text.split()[-1]

    keys   = [extract_key_token(a) for a in answers]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1

    majority_key   = max(counts, key=lambda x: counts[x])
    majority_count = counts[majority_key]
    agreement      = majority_count / n

    majority_ans = next((a for a in answers if extract_key_token(a) == majority_key), None)

    if agreement >= 0.80:
        verdict = "consistent (low hallucination risk)"
    elif agreement >= 0.60:
        verdict = "partial agreement (verify against source)"
    else:
        verdict = "inconsistent (high hallucination risk)"

    return ConsistencyResult(
        answers         = answers,
        agreement_score = agreement,
        majority_answer = majority_ans,
        verdict         = verdict,
    )


# ─── Demo: Run All Three Techniques ──────────────────────────────────────────

def run_detection_demo():
    """
    Demonstrate all three detection techniques on sample questions.
    """

    print("=" * 70)
    print("HALLUCINATION DETECTION: Three Techniques")
    print("=" * 70)

    context = (
        "The APIC cluster in Cisco ACI requires a minimum of 3 nodes "
        "for high availability. Maximum supported scale is 200 leaf switches "
        "per pod in ACI version 6.0. The APIC uses REST API over HTTPS on port 443. "
        "ACI fabric uses VXLAN as the overlay protocol."
    )

    # ── Technique 1: Attribution Analysis (simulated) ────────────────────────
    print("\n  TECHNIQUE 1: Source Attribution Analysis")
    print("  " + "─" * 60)

    grounded_answer = (
        "The APIC cluster requires a minimum of 3 nodes for HA. [source:1] "
        "The maximum leaf scale is 200 per pod. [source:1] "
        "APIC REST API operates on HTTPS port 443. [source:1]"
    )
    hallucinated_answer = (
        "The APIC cluster requires a minimum of 3 nodes for HA. [source:1] "
        "ACI supports Multi-Pod with up to 12 pods per deployment. [source:unsupported] "
        "The APIC login endpoint is /api/aaaLogin.json. [source:unsupported]"
    )

    for label, ans in [("Grounded answer", grounded_answer), ("Hallucinated answer", hallucinated_answer)]:
        result = analyze_attribution(ans)
        print(f"\n  [{label}]")
        print(f"    Grounding rate:        {result['grounding_rate']:.1%}")
        print(f"    Supported sentences:   {result['supported_sentences']}")
        print(f"    Unsupported sentences: {result['unsupported_sentences']}")
        if result["unsupported_text"]:
            for s in result["unsupported_text"]:
                print(f"    → UNGROUNDED: '{s[:70]}'")

    # ── Technique 2: Lexical Entailment ──────────────────────────────────────
    print("\n\n  TECHNIQUE 2: Lexical Entailment (word overlap proxy for NLI)")
    print("  " + "─" * 60)

    test_sentences = [
        ("The APIC cluster requires 3 nodes minimum.",           "grounded"),
        ("ACI uses VXLAN as its overlay encapsulation.",         "grounded"),
        ("ACI Multi-Pod requires a minimum of 2 remote pods.",   "hallucinated"),
        ("The default RADIUS port in ISE is 1812.",              "hallucinated"),
        ("APIC REST API uses HTTPS.",                            "grounded"),
    ]

    print(f"\n  {'Sentence':<55} {'Score':>6}  {'Verdict'}")
    print(f"  {'─'*55} {'─'*6}  {'─'*15}")
    for sent, expected in test_sentences:
        score   = lexical_entailment_score(sent, context)
        verdict = "grounded" if score >= 0.30 else "not grounded"
        match   = "✓" if verdict == expected else "✗"
        print(f"  {sent[:55]:<55} {score:>6.3f}  {verdict}  {match}")

    print(f"\n  NOTE: Jaccard fails on paraphrases and synonyms.")
    print(f"  Production: use DeBERTa-mnli or GPT-4 as NLI judge.")

    # ── Technique 3: Self-Consistency ────────────────────────────────────────
    print("\n\n  TECHNIQUE 3: Self-Consistency Sampling")
    print("  " + "─" * 60)

    questions = [
        "What is the minimum APIC node count for HA?",
        "What is the maximum leaf switch scale in ACI?",
    ]

    for q in questions:
        result = simulate_self_consistency(q, context, n=5)
        print(f"\n  Question: '{q}'")
        print(f"  Samples:")
        for i, ans in enumerate(result.answers, 1):
            print(f"    {i}. {ans}")
        print(f"  Agreement: {result.agreement_score:.0%}")
        print(f"  Verdict:   {result.verdict}")
        if result.majority_answer:
            print(f"  Best answer: '{result.majority_answer}'")


# ─── Full Grounding Check Demo ────────────────────────────────────────────────

def run_grounding_check_demo():
    """
    Show the sentence-level grounding check on good and bad answers.
    """

    print("\n" + "=" * 70)
    print("GROUNDING CHECK: Sentence-Level Hallucination Detection")
    print("=" * 70)

    context = (
        "ReadyOps is Criterion Networks' continuous validation platform. "
        "It operates across two isolated environments: Production-Representative "
        "and Live Operations. Changes are validated in Production-Representative "
        "before being promoted to Live Operations via a formal promotion gate. "
        "The promotion gate requires 100% validation pass rate."
    )

    good_answer = (
        "ReadyOps is a continuous validation platform by Criterion Networks. "
        "It uses two isolated environments: Production-Representative and Live Operations. "
        "Changes must pass a formal promotion gate before reaching Live Operations."
    )

    hallucinated_answer = (
        "ReadyOps is a continuous validation platform by Criterion Networks. "
        "It supports a 90% pass threshold for emergency changes. "
        "The platform integrates with Jira for ticket-driven promotion workflows. "
        "API access is available on port 8443."
    )

    for label, answer in [("Well-grounded answer", good_answer), ("Hallucinated answer", hallucinated_answer)]:
        result = check_answer_grounding(answer, context)
        print(f"\n  [{label}]")
        print(f"  Overall verdict: {result.verdict.upper()} (score={result.grounding_score:.1%})")
        print(f"  Grounded sentences   ({len(result.grounded_sentences)}):")
        for s in result.grounded_sentences:
            print(f"    ✓ {s}")
        print(f"  Ungrounded sentences ({len(result.ungrounded_sentences)}):")
        for s in result.ungrounded_sentences:
            print(f"    ✗ {s}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_detection_demo()
    run_grounding_check_demo()
