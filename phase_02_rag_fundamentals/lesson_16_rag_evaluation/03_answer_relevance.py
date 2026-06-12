"""
FILE: 03_answer_relevance.py
LESSON: Phase 2 - Lesson 16 - RAG Evaluation
TOPIC: Answer relevance — does the answer address the question?

WHAT THIS FILE TEACHES:
  - RAGAS answer relevance: reverse question generation + embedding similarity
  - WHY you need answer relevance in addition to faithfulness
  - Detecting evasive answers, topic drift, and over-hedging
  - Lexical proxy for reference-free relevance estimation
  - Combining faithfulness + relevance into a single "generation quality" score

INSTALL: pip install anthropic python-dotenv numpy
"""

import os
import re
import json
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    HAS_ANTHROPIC = False


# ─── Embedding Utility ────────────────────────────────────────────────────────

def mock_embed(text: str, dims: int = 64) -> np.ndarray:
    """
    Deterministic mock embedding (keyword-seeded).
    WHY mock: answer relevance needs embeddings; this avoids Voyage AI API key.
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


# ─── Reverse Question Generator ───────────────────────────────────────────────

REVERSE_Q_SYSTEM = """You are a question generator.
Given an answer text, generate {n} questions that this answer would directly answer.
Each question should be self-contained and answerable from the given answer.
Output ONLY a JSON array of question strings:
["Question one?", "Question two?", ...]"""

REVERSE_Q_USER = """Answer:
{answer}

Generate {n} questions that this answer directly addresses (JSON array only):"""

MOCK_REVERSE_QUESTIONS = {
    "3 nodes apic":       [
        "How many APIC nodes are required for HA?",
        "What is the minimum APIC cluster size?",
        "What provides quorum in ACI APIC?",
    ],
    "readyops validates": [
        "What does ReadyOps validate?",
        "How does ReadyOps promote changes?",
        "What is the ReadyOps promotion gate requirement?",
    ],
    "hypershield ebpf":   [
        "How does Hypershield enforce policy?",
        "What technology does Hypershield use?",
        "How is Hypershield deployed at the workload?",
    ],
    "vague deployment":   [
        "What are some general deployment considerations?",
        "What should I consider for networking?",
    ],
}


def generate_reverse_questions(answer: str, n: int = 3) -> list[str]:
    """
    Generate N questions that the answer would answer (RAGAS approach).

    WHY reverse questions:
      Faithfulness asks: "Is the answer grounded?" (anti-hallucination)
      Answer relevance asks: "Does the answer address the question?" (anti-evasion)

      The reverse question trick is reference-free — you don't need a human-labeled
      "correct answer". Instead:
        1. Generate N questions Q' from the answer A.
        2. Embed each Q'.
        3. Measure cosine_sim(embed(Q'), embed(Q_original)).
        4. High similarity → A addresses Q_original.

      If A answers "What are network deployment considerations?" (vague)
      when Q was "How many APIC nodes for HA?", the reverse questions
      will be vague general questions — very different from Q_original.
      Similarity drops → low relevance score → evasion detected.
    """
    if HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 200,
            system     = REVERSE_Q_SYSTEM.format(n=n),
            messages   = [{"role": "user", "content": REVERSE_Q_USER.format(answer=answer, n=n)}],
        )
        text = resp.content[0].text.strip()
        try:
            return json.loads(text)[:n]
        except json.JSONDecodeError:
            lines = re.findall(r'"([^"]+\?)"', text)
            return lines[:n] if lines else [answer[:80] + "?"]
    else:
        a_low = answer.lower()
        for key, qs in MOCK_REVERSE_QUESTIONS.items():
            if all(w in a_low for w in key.split()):
                return qs[:n]
        # Generic fallback
        words = answer.split()[:5]
        return [f"What is {' '.join(words)}?"] * min(n, 1)


# ─── Answer Relevance Scorer ──────────────────────────────────────────────────

@dataclass
class AnswerRelevanceResult:
    """Full result of answer relevance scoring."""
    query:             str
    answer:            str
    reverse_questions: list[str]
    similarities:      list[float]
    score:             float       # mean cosine similarity of reverse Qs to original Q

    def display(self):
        print(f"\n  Query:   '{self.query}'")
        print(f"  Answer:  '{self.answer[:90]}'")
        print(f"\n  Reverse questions generated from answer:")
        for q, sim in zip(self.reverse_questions, self.similarities):
            bar = "█" * int(sim * 20)
            print(f"    [{sim:.3f}] {bar:<20}  '{q}'")
        print(f"\n  Answer Relevance Score: {self.score:.3f}")
        if self.score >= 0.85:
            print(f"  Verdict: RELEVANT — answer addresses the query well.")
        elif self.score >= 0.65:
            print(f"  Verdict: PARTIAL — answer is related but may miss the point.")
        else:
            print(f"  Verdict: IRRELEVANT — answer does not address the query.")


def answer_relevance_score(query: str, answer: str, n: int = 3) -> AnswerRelevanceResult:
    """
    Compute answer relevance using reverse question generation.

    Score = mean(cosine_sim(embed(reverse_Q_i), embed(original_Q))) for i in 1..N

    WHY N=3 (default):
      A single reverse question may be a fluke. Averaging over 3 reduces variance.
      More than 5 adds LLM cost without significant accuracy improvement.

    Score interpretation:
      0.90+: Excellent — answer directly targets the query.
      0.75–0.90: Good — answer is on-topic, may be slightly indirect.
      0.60–0.75: Mediocre — answer is tangentially related.
      < 0.60:  Poor — answer evades the question or is off-topic.
    """
    reverse_qs  = generate_reverse_questions(answer, n=n)
    q_vec       = mock_embed(query)
    sims        = [cosine_sim(q_vec, mock_embed(rq)) for rq in reverse_qs]
    mean_sim    = float(np.mean(sims)) if sims else 0.0

    return AnswerRelevanceResult(
        query             = query,
        answer            = answer,
        reverse_questions = reverse_qs,
        similarities      = sims,
        score             = mean_sim,
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "description": "Direct, specific answer (high relevance expected)",
        "query":  "How many APIC nodes are required for high availability?",
        "answer": "APIC requires a minimum of 3 nodes for high availability. "
                  "With 3 APIC nodes, one can fail while the remaining two maintain quorum.",
    },
    {
        "description": "Evasive / hedged answer (low relevance expected)",
        "query":  "How many APIC nodes are required for high availability?",
        "answer": "The number of APIC nodes required depends on various factors including "
                  "your availability requirements and deployment scenario. High availability "
                  "is important for production environments.",
    },
    {
        "description": "Topic drift — answered different question (low relevance)",
        "query":  "What is the ReadyOps promotion gate?",
        "answer": "ReadyOps is a continuous validation platform built by Criterion Networks "
                  "that validates network infrastructure changes. It uses multiple agent classes.",
    },
    {
        "description": "Good ReadyOps answer (high relevance expected)",
        "query":  "What is the ReadyOps promotion gate?",
        "answer": "The ReadyOps promotion gate blocks changes from reaching Live Operations "
                  "until all Validation agent tests pass at 100%. No change is promoted unless "
                  "the gate opens.",
    },
]


def run_relevance_demo():
    print("=" * 70)
    print("ANSWER RELEVANCE: Does the Answer Address the Question?")
    print("=" * 70)

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n  ─── Test {i}: {tc['description']} ───")
        result = answer_relevance_score(tc["query"], tc["answer"], n=3)
        result.display()


def combined_generation_quality():
    """Show how faithfulness and relevance together characterize generation quality."""

    print("\n" + "=" * 70)
    print("COMBINED GENERATION QUALITY MATRIX")
    print("=" * 70)
    print(f"""
  ┌──────────────────────────────────────────────────────────────────────┐
  │              HIGH Faithfulness          LOW Faithfulness             │
  │  ─────────────────────────────────────────────────────────────────  │
  │  HIGH     ✓ IDEAL: Answer is grounded  ✗ HALLUCINATED but relevant  │
  │  Relevance  and addresses the question.   Answer addresses Q but     │
  │             Ship it.                      adds made-up details.      │
  │                                           Fix: stronger grounding.   │
  │  ─────────────────────────────────────────────────────────────────  │
  │  LOW      ✗ EVASIVE: LLM hedges and     ✗ WORST CASE: Off-topic     │
  │  Relevance  avoids the question but        AND fabricated.            │
  │             stays within context.          Fix everything.            │
  │             Fix: prompt redesign.                                     │
  └──────────────────────────────────────────────────────────────────────┘

  COMBINED SCORE (simple harmonic mean, like F1):
    generation_quality = 2 * faithfulness * relevance / (faithfulness + relevance)

  EXAMPLE:
    faithfulness=0.90, relevance=0.90 → quality=0.90  ← excellent
    faithfulness=0.95, relevance=0.50 → quality=0.65  ← evasive, address prompt
    faithfulness=0.50, relevance=0.95 → quality=0.65  ← hallucinating, strengthen grounding
    faithfulness=0.50, relevance=0.50 → quality=0.50  ← broken pipeline
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_relevance_demo()
    combined_generation_quality()
