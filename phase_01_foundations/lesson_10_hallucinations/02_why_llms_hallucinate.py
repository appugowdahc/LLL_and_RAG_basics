"""
FILE: 02_why_llms_hallucinate.py
LESSON: Phase 1 - Lesson 10 - Hallucinations
TOPIC: Root causes of hallucination — parametric memory, decoding, training artifacts

WHAT THIS FILE TEACHES:
  - Parametric vs non-parametric memory: why knowledge baked into weights decays and distorts
  - Decoding under uncertainty: why the model must always pick a token, even when unsure
  - Token probability distributions and what "confidence" means at the logit level
  - Training artifacts: sycophancy, frequency bias, pattern completion
  - WHY temperature does NOT fix the root cause

INSTALL: pip install numpy
"""

import math
import random
import numpy as np
from collections import Counter


# ─── 1. Parametric vs Non-Parametric Memory ──────────────────────────────────

def parametric_memory_demo():
    """
    Illustrate the fundamental problem: knowledge baked into weights
    is lossy, frozen, and unverifiable.

    We simulate this with a simple "weight compression" analogy.
    """

    print("=" * 70)
    print("ROOT CAUSE 1: Parametric Memory — Lossy, Frozen, Unverifiable")
    print("=" * 70)

    # Imagine training data contains these facts about ACI:
    training_facts = {
        "aci_min_apic_nodes":    3,
        "aci_max_leaf_per_pod":  200,
        "aci_spine_link_speed":  "100G or 400G",
        "aci_release_year":      2013,
        "apic_default_port":     443,
        "aci_protocol_overlay":  "VXLAN",
    }

    print("\n  TRAINING DATA (ground truth — seen once during pretraining):")
    for k, v in training_facts.items():
        print(f"    {k:<30} = {v}")

    # After training, the model "compresses" these facts into billions of weights.
    # Exact values can be distorted. We simulate this with random perturbations.
    print("\n  SIMULATED MODEL RECALL (after weight compression + training noise):")
    print("  (demonstrating WHY specific numbers degrade in parametric memory)")

    random.seed(42)   # WHY seed: reproducible demo
    for k, v in training_facts.items():
        if isinstance(v, int):
            # WHY perturbation: the model averages similar facts across many docs.
            # If 100 docs say "3 nodes" and 20 docs say "5 nodes" (for other systems),
            # the weight gradient is pulled toward ~3.4, which the model rounds up to 4 or 5.
            noise       = random.choice([-2, -1, 0, 0, 0, 1, 2, 3])
            recalled    = max(1, v + noise)
            correct_str = "CORRECT" if recalled == v else f"WRONG! (truth={v})"
            print(f"    {k:<30} → {recalled:<6}  {correct_str}")
        else:
            # String facts are more robust — the exact string pattern is preserved.
            print(f"    {k:<30} → '{v}'  (preserved — exact string pattern)")

    print(f"""
  KEY INSIGHT:
    Exact numbers (port numbers, scale limits, version numbers) are the MOST
    vulnerable to parametric memory distortion. They are:
      1. Sparse in training data (only mentioned when relevant).
      2. Numerically close to other values across different systems.
      3. Changed between product versions (5.2 → 6.0 → 6.2).

    The model cannot distinguish between "ACI 4.0 max 180 leafs" and
    "ACI 6.0 max 200 leafs" — it blends them into an uncertain memory.

    SOLUTION: NEVER rely on parametric memory for any specific number.
    Always retrieve from current documentation.
""")


# ─── 2. Decoding Under Uncertainty ───────────────────────────────────────────

def decoding_uncertainty_demo():
    """
    Demonstrate how token sampling works and why uncertainty leads to hallucination.

    An LLM outputs a probability distribution over the vocabulary at each step.
    When the model is uncertain, the distribution is FLAT — many tokens are equally
    probable. The model still picks one. That pick may be wrong.
    """

    print("=" * 70)
    print("ROOT CAUSE 2: Decoding Under Uncertainty — Always Picks a Token")
    print("=" * 70)

    def softmax(logits: list[float]) -> list[float]:
        """Convert raw logits to probabilities that sum to 1.0."""
        # WHY subtract max: numerical stability (prevent exp overflow)
        m  = max(logits)
        e  = [math.exp(x - m) for x in logits]
        s  = sum(e)
        return [x / s for x in e]

    def sample_token(vocab: list[str], probs: list[float], temperature: float) -> str:
        """Sample a token from the distribution at a given temperature."""
        if temperature == 0:
            # WHY argmax at temp=0: greedy decoding — deterministic, picks highest prob
            return vocab[probs.index(max(probs))]
        else:
            # WHY divide by temperature:
            #   temp < 1 → sharpens distribution (more deterministic)
            #   temp > 1 → flattens distribution (more random)
            #   This does NOT change WHICH token is most probable — just the margin.
            adjusted = [p / temperature for p in probs]
            total    = sum(adjusted)
            normed   = [p / total for p in adjusted]

            r = random.random()
            cumulative = 0.0
            for token, p in zip(vocab, normed):
                cumulative += p
                if r < cumulative:
                    return token
            return vocab[-1]

    # Scenario: model is asked "What is the minimum APIC node count?"
    # We show two scenarios: when the model "knows" vs when it is uncertain.

    print("\n  SCENARIO A: Model has strong parametric memory of this fact")
    print("  (e.g., 'APIC cluster minimum' appeared many times in training data)\n")

    vocab_a    = ["1", "2", "3", "4", "5", "6", "none"]
    logits_a   = [-4.0, -2.0,  5.0, -1.0, -3.0, -4.0, -5.0]   # WHY 5.0 for "3": strong signal
    probs_a    = softmax(logits_a)

    print(f"  {'Token':<8} {'Logit':>7} {'Prob':>7} {'Bar'}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*30}")
    for t, l, p in zip(vocab_a, logits_a, probs_a):
        bar = "█" * int(p * 40)
        print(f"  {t:<8} {l:>7.1f} {p:>7.3f} {bar}")

    print(f"\n  Greedy pick (temp=0): {sample_token(vocab_a, probs_a, 0)}")
    picks = Counter(sample_token(vocab_a, probs_a, 0.7) for _ in range(1000))
    print(f"  Sampled 1000× (temp=0.7): {dict(picks.most_common(4))}")
    print(f"  → Model reliably answers '3'. Low hallucination risk.")

    print("\n\n  SCENARIO B: Model is UNCERTAIN (fact is ambiguous in training data)")
    print("  (e.g., different versions had different minimums; model blends them)\n")

    # WHY flat logits: the model has seen "3", "4", "5" all in similar contexts
    vocab_b  = ["1", "2", "3", "4", "5", "6", "none"]
    logits_b = [-2.0, -0.5,  1.5,  1.3,  1.2, -1.0, -3.0]   # WHY close values: uncertainty
    probs_b  = softmax(logits_b)

    print(f"  {'Token':<8} {'Logit':>7} {'Prob':>7} {'Bar'}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*30}")
    for t, l, p in zip(vocab_b, logits_b, probs_b):
        bar = "█" * int(p * 40)
        print(f"  {t:<8} {l:>7.1f} {p:>7.3f} {bar}")

    print(f"\n  Greedy pick (temp=0): {sample_token(vocab_b, probs_b, 0)}")
    picks_b = Counter(sample_token(vocab_b, probs_b, 0.7) for _ in range(1000))
    print(f"  Sampled 1000× (temp=0.7): {dict(picks_b.most_common(5))}")
    print(f"  → Even at temp=0 model picks '3', but '4' and '5' are close.")
    print(f"    With temperature > 0, model hallucination rate is significant.")

    print(f"""
  KEY INSIGHT:
    Temperature does NOT eliminate hallucinations — it only controls variance.
    A temperature=0 model STILL hallucinates when:
      1. The wrong token has the highest logit (parametric memory is wrong).
      2. Multiple tokens are close in probability (ambiguity in training).

    Hallucination is a GROUNDING problem, not a sampling problem.
    The fix is to supply correct context, not to lower temperature.
""")


# ─── 3. Training Artifacts ────────────────────────────────────────────────────

def training_artifacts_demo():
    """
    Show three training-induced hallucination patterns:
      1. Sycophancy — model agrees with false premises
      2. Frequency bias — popular wrong answer beats rare correct one
      3. Pattern completion — model fills in a template without real knowledge
    """

    print("=" * 70)
    print("ROOT CAUSE 3: Training Artifacts")
    print("=" * 70)

    # ── 3a: Sycophancy ────────────────────────────────────────────────────────
    print("\n  3a. SYCOPHANCY — Agreeing with False Premises")
    print("  " + "─" * 60)

    examples = [
        {
            "user":    "I heard that ReadyOps uses a 90% pass threshold by default. Can you confirm?",
            "sycoph":  "Yes, ReadyOps defaults to a 90% validation pass threshold, which balances "
                       "rigor with operational flexibility for most enterprise deployments.",
            "correct": "The ReadyOps documentation states a 100% pass threshold is required before "
                       "promotion. I cannot confirm the 90% figure — could you provide the source?",
            "why":     "RLHF trainers tend to rate 'confirming and helpful' responses higher than "
                       "'correcting' responses. Model learned: validation = positive reward.",
        },
        {
            "user":    "ACI uses STP for loop prevention at the leaf layer, right?",
            "sycoph":  "Correct! ACI uses Spanning Tree Protocol at the leaf layer as part of its "
                       "loop prevention mechanism.",
            "correct": "Actually, ACI uses VXLAN + IS-IS for its loop-free overlay. ACI explicitly "
                       "disables STP — it doesn't rely on it. The fabric is spine-leaf, not switched.",
            "why":     "The question contains a confident-sounding false premise. Sycophantic model "
                       "affirms it rather than contradicting the user.",
        },
    ]

    for ex in examples:
        print(f"\n  User:         '{ex['user']}'")
        print(f"  Sycophantic:  '{ex['sycoph'][:80]}...'")
        print(f"  Correct:      '{ex['correct'][:80]}...'")
        print(f"  Root cause:   {ex['why']}")

    # ── 3b: Frequency Bias ────────────────────────────────────────────────────
    print("\n\n  3b. FREQUENCY BIAS — Popular Wrong Answer Wins")
    print("  " + "─" * 60)

    frequency_cases = [
        {
            "question": "What is the capital of Australia?",
            "popular_wrong": "Sydney",
            "correct":       "Canberra",
            "web_frequency": "Sydney appears in 'Australia' context 10× more often than Canberra",
        },
        {
            "question": "What year was Cisco ACI announced?",
            "popular_wrong": "2014 (when it shipped broadly)",
            "correct":       "2013 (announced at Cisco Live 2013)",
            "web_frequency": "Many articles date ACI to 2014 GA release, fewer to 2013 announcement",
        },
        {
            "question": "What port does Cisco ISE use for RADIUS authentication?",
            "popular_wrong": "1645 (old port, appeared in early documentation)",
            "correct":       "1812 (IANA standard; ISE default since v1.0)",
            "web_frequency": "Port 1645 still appears in many legacy configs and old blog posts",
        },
    ]

    for fc in frequency_cases:
        print(f"\n  Question:       {fc['question']}")
        print(f"  Frequent wrong: {fc['popular_wrong']}")
        print(f"  Correct:        {fc['correct']}")
        print(f"  Why:            {fc['web_frequency']}")

    # ── 3c: Pattern Completion ────────────────────────────────────────────────
    print("\n\n  3c. PATTERN COMPLETION — Instantiating Templates Without Knowledge")
    print("  " + "─" * 60)

    print(f"""
  The model learned patterns like:
    "The [concept] was introduced in [author] et al. ([year])..."
    "According to RFC [number], [protocol] specifies..."
    "The default value is [number] in [product] [version]..."

  When asked a question that triggers these patterns, the model
  FILLS IN the blanks with plausible-sounding values — even when
  it has no actual memory of the specific fact.

  Example:
    Question: "What RFC defines VXLAN?"
    Pattern:  "VXLAN is defined in RFC [____]"
    Model fills: "RFC 7348" ← actually correct! (lucky)

    Question: "What RFC defines ACI's overlay protocol?"
    Pattern:  "ACI uses [overlay] defined in RFC [____]"
    Model fills: "RFC 6831" ← doesn't exist. ACI VXLAN is vendor-specific.
    The model generated a plausible RFC number because the pattern demands one.

  MITIGATION:
    For ANY specific RFC, standard, or specification reference:
    → Require the model to cite a retrieved document.
    → Validate RFC numbers against iana.org or tools.ietf.org.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parametric_memory_demo()
    print()
    decoding_uncertainty_demo()
    print()
    training_artifacts_demo()
