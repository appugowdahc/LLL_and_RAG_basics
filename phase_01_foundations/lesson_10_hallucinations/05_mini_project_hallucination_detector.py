"""
FILE: 05_mini_project_hallucination_detector.py
LESSON: Phase 1 - Lesson 10 - Hallucinations
TOPIC: Mini-project — Hallucination detector for RAG answers

WHAT THIS BUILDS:
  A post-generation hallucination checker that:
    1. Extracts atomic claims from an LLM answer
    2. Checks each claim against the retrieved context
    3. Assigns a faithfulness verdict to the full answer
    4. Outputs a structured report with evidence for each claim
    5. Supports optional live LLM grading (LLM-as-judge pattern)

  This is what you run AFTER getting the LLM answer, before showing it to the user.
  In a production RAG system, answers with faithfulness < 0.80 would be:
    - Flagged for human review, OR
    - Rejected and re-queried with a tighter prompt, OR
    - Shown to the user with a warning ("This answer may not be fully supported")

INSTALL: pip install anthropic python-dotenv
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    HAS_ANTHROPIC = False


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Claim:
    """One atomic claim extracted from an LLM answer."""
    text:            str
    overlap_score:   float          # lexical overlap with context (0.0–1.0)
    is_supported:    bool           # True if overlap >= threshold
    evidence:        Optional[str]  # best matching context sentence


@dataclass
class HallucinationReport:
    """Full hallucination audit for one answer."""
    question:           str
    answer:             str
    context:            str
    claims:             list[Claim]
    faithfulness:       float          # supported / total
    verdict:            str            # "pass" | "warn" | "fail"
    recommendation:     str            # action to take

    def display(self):
        """Print a structured audit report."""
        verdict_icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(self.verdict, "?")

        print(f"\n  {'─'*64}")
        print(f"  {verdict_icon} HALLUCINATION REPORT  [{self.verdict.upper()}]")
        print(f"  {'─'*64}")
        print(f"  Question:     {self.question[:60]}")
        print(f"  Faithfulness: {self.faithfulness:.0%}  ({len([c for c in self.claims if c.is_supported])}/{len(self.claims)} claims supported)")
        print(f"  Action:       {self.recommendation}")
        print(f"\n  Claims:")

        for i, claim in enumerate(self.claims, 1):
            icon = "✓" if claim.is_supported else "✗"
            print(f"\n    [{i}] {icon}  '{claim.text[:70]}'")
            print(f"         overlap={claim.overlap_score:.2f}  supported={claim.is_supported}")
            if claim.evidence:
                print(f"         evidence: '{claim.evidence[:65]}'")
            elif not claim.is_supported:
                print(f"         evidence: (no match found in context)")


# ─── Claim Extraction ─────────────────────────────────────────────────────────

def extract_claims(answer: str) -> list[str]:
    """
    Split an answer into atomic claims (sentences).

    WHY sentence-level granularity:
      Each sentence typically contains one factual claim.
      Sub-sentence claim extraction (e.g., "X does A, B, and C" → three claims)
      requires an LLM. Sentence splitting is a good approximation.

    In production: use an LLM to extract atomic claims with:
      "Break the following answer into a list of atomic factual claims, one per line."
    """
    # WHY strip tags: remove citation artifacts like [Chunk1] before scoring
    clean = re.sub(r"\[Chunk\d+\]|\[unsupported\]|\[source:[^\]]+\]", "", answer)

    # WHY sentence regex vs split("."): handles "v6.0" and "Dr. Smith" correctly
    sentences = re.split(r"(?<=[.!?])\s+", clean.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


# ─── Context Matching ─────────────────────────────────────────────────────────

def find_best_context_match(claim: str, context: str) -> tuple[float, Optional[str]]:
    """
    Find the most relevant context sentence for a given claim.
    Returns (overlap_score, matching_sentence).

    WHY sentence-by-sentence matching:
      Matching against the entire context rewards long claims (many tokens overlap).
      Matching against individual sentences is fairer and returns meaningful evidence.
    """

    def tokenize(text: str) -> set[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        return {t for t in tokens if len(t) > 3}

    claim_toks    = tokenize(claim)
    if not claim_toks:
        return 0.0, None

    # Split context into sentences
    ctx_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]

    best_score    = 0.0
    best_sentence = None

    for ctx_sent in ctx_sentences:
        ctx_toks  = tokenize(ctx_sent)
        if not ctx_toks:
            continue
        overlap   = len(claim_toks & ctx_toks) / len(claim_toks)
        if overlap > best_score:
            best_score    = overlap
            best_sentence = ctx_sent

    return best_score, best_sentence


# ─── Full Hallucination Detector ──────────────────────────────────────────────

class HallucinationDetector:
    """
    Post-generation hallucination checker.
    Call check() after every RAG answer before displaying to the user.
    """

    def __init__(self, support_threshold: float = 0.30, fail_threshold: float = 0.50):
        """
        Args:
            support_threshold: Overlap score for a claim to count as supported (default 0.30).
            fail_threshold:    If faithfulness < this, verdict = "fail" (default 0.50).
        """
        self.support_threshold = support_threshold
        self.fail_threshold    = fail_threshold

    def check(self, question: str, answer: str, context: str) -> HallucinationReport:
        """
        Run a full hallucination check on one answer.

        Args:
            question: The user's original question.
            answer:   The LLM's answer.
            context:  The retrieved context that was given to the LLM.

        Returns:
            HallucinationReport with per-claim breakdown and verdict.
        """
        claim_texts = extract_claims(answer)
        claims      = []

        for ct in claim_texts:
            score, evidence = find_best_context_match(ct, context)
            claims.append(Claim(
                text          = ct,
                overlap_score = score,
                is_supported  = score >= self.support_threshold,
                evidence      = evidence,
            ))

        total       = len(claims)
        supported   = sum(1 for c in claims if c.is_supported)
        faithfulness = supported / max(total, 1)

        # Verdict and recommendation
        if faithfulness >= 0.80:
            verdict        = "pass"
            recommendation = "Answer is well-grounded. Safe to show to user."
        elif faithfulness >= self.fail_threshold:
            verdict        = "warn"
            recommendation = "Answer is partially grounded. Review unsupported claims before use."
        else:
            verdict        = "fail"
            recommendation = "Answer has significant ungrounded claims. Re-query with tighter prompt or escalate."

        return HallucinationReport(
            question       = question,
            answer         = answer,
            context        = context,
            claims         = claims,
            faithfulness   = faithfulness,
            verdict        = verdict,
            recommendation = recommendation,
        )


# ─── LLM-as-Judge (optional) ──────────────────────────────────────────────────

LLM_JUDGE_PROMPT = """You are a strict faithfulness evaluator.

Given a CONTEXT and an ANSWER, evaluate whether each sentence in the ANSWER
is supported by the CONTEXT.

Respond with a JSON list where each item has:
  "sentence": <the sentence>,
  "supported": true or false,
  "reason": <one-line reason>

Only evaluate based on the CONTEXT. Do not use outside knowledge.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with the JSON list only. No other text."""


def llm_judge_check(question: str, answer: str, context: str) -> Optional[list[dict]]:
    """
    Use Claude as a faithfulness judge.
    More accurate than lexical overlap but requires an API call.

    WHY LLM-as-judge:
      Handles paraphrases, synonyms, and semantic equivalence that lexical
      overlap misses. Example: "at least 3 nodes" vs "minimum of 3 nodes"
      — lexical overlap is low but they're semantically equivalent.

    COST: ~500 tokens per call (Haiku = ~$0.0003 per check).
    """
    if not HAS_ANTHROPIC:
        return None

    import json
    client = anthropic.Anthropic()

    prompt = LLM_JUDGE_PROMPT.format(context=context, answer=answer)

    try:
        resp = client.messages.create(
            model      = "claude-haiku-4-5-20251001",   # WHY Haiku: fast and cheap for evaluation
            max_tokens = 600,
            messages   = [{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        # WHY extract from code block: model sometimes wraps JSON in ```json
        if "```" in text:
            text = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
            text = text.group(1) if text else text
        return json.loads(text)
    except Exception as e:
        print(f"  [LLM judge error: {e}]")
        return None


# ─── Test Suite ───────────────────────────────────────────────────────────────

def run_detector_suite():
    """
    Run the hallucination detector on a set of test cases.
    """

    detector = HallucinationDetector(support_threshold=0.28, fail_threshold=0.50)

    test_cases = [
        {
            "label":    "Fully grounded answer",
            "question": "What environments does ReadyOps use?",
            "context":  (
                "ReadyOps operates across two isolated environments: "
                "Production-Representative and Live Operations. "
                "Changes are validated in Production-Representative first. "
                "A 100% validation pass rate is required before promotion."
            ),
            "answer":   (
                "ReadyOps uses two isolated environments: Production-Representative "
                "and Live Operations. Changes are first validated in the "
                "Production-Representative environment. Promotion to Live Operations "
                "requires passing a 100% validation gate."
            ),
        },
        {
            "label":    "Partially hallucinated answer",
            "question": "How does ReadyOps handle failed validations?",
            "context":  (
                "ReadyOps agent classes include Validation agents that run pre-change tests. "
                "If tests fail, the promotion gate blocks the change from reaching Live Operations. "
                "All validation results are logged for audit."
            ),
            "answer":   (
                "ReadyOps runs Validation agents to test changes before promotion. "
                "If validation fails, the change is blocked from Live Operations. "
                "Failed validations trigger an automatic Jira ticket with the failure details. "
                "The team receives a Slack notification with the test report."
            ),
        },
        {
            "label":    "Heavily hallucinated answer",
            "question": "What is the APIC cluster minimum for ACI?",
            "context":  "ACI fabric uses VXLAN as the overlay. Leaf switches connect to endpoints.",
            "answer":   (
                "The APIC cluster requires a minimum of 3 nodes for high availability. "
                "For production deployments, 5 APIC nodes are recommended. "
                "APIC uses port 443 for REST API access. "
                "The cluster communicates over a dedicated management VLAN."
            ),
        },
    ]

    print("=" * 70)
    print("HALLUCINATION DETECTOR: Test Suite")
    print("=" * 70)

    for tc in test_cases:
        print(f"\n  ══ {tc['label']} ══")
        report = detector.check(tc["question"], tc["answer"], tc["context"])
        report.display()


# ─── Live API Demo ────────────────────────────────────────────────────────────

def live_detection_demo():
    """
    If API key is set: generate an answer with Claude then run the detector on it.
    """

    if not HAS_ANTHROPIC:
        print("\n  [Skipping live demo — ANTHROPIC_API_KEY not set]")
        return

    client   = anthropic.Anthropic()
    detector = HallucinationDetector()

    context = (
        "Cisco ACI uses a Leaf-Spine topology. The APIC cluster is the "
        "policy controller and requires a minimum of 3 nodes for HA. "
        "ACI 6.0 supports up to 200 leaf switches per pod. "
        "ACI uses VXLAN for the fabric overlay."
    )
    question = "What is the APIC cluster HA requirement and ACI scale limits?"

    system_prompt = (
        "Answer ONLY from the provided context. Do not add extra information."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    print("\n  Calling Claude (Haiku) with grounding context...")
    resp = client.messages.create(
        model      = "claude-haiku-4-5-20251001",
        max_tokens = 200,
        system     = system_prompt,
        messages   = [{"role": "user", "content": user_message}],
    )
    answer = resp.content[0].text

    print(f"\n  Question: {question}")
    print(f"  Answer:   {answer}")

    report = detector.check(question, answer, context)
    report.display()

    print("\n  Running LLM-as-judge check...")
    judge_results = llm_judge_check(question, answer, context)
    if judge_results:
        print(f"  LLM judge verdict (per sentence):")
        for item in judge_results:
            icon = "✓" if item.get("supported") else "✗"
            print(f"    {icon} '{item.get('sentence', '')[:60]}'")
            print(f"       reason: {item.get('reason', '')}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_detector_suite()
    live_detection_demo()
