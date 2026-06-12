"""
FILE: 03_reasoning_failures.py
LESSON: Phase 1 - Lesson 11 - LLM Limitations
TOPIC: Reasoning failures — what LLMs cannot reliably do

WHAT THIS FILE TEACHES:
  - WHY LLMs are pattern matchers, not symbolic reasoners
  - Multi-hop arithmetic failures with infrastructure examples
  - Negation logic failures (common in "which devices are NOT affected?")
  - Constraint satisfaction failures (filter over multiple hard constraints)
  - Counting failures
  - What RAG does NOT fix (reasoning is model-side, not data-side)
  - What DOES fix reasoning: chain-of-thought, tool use, code interpreter

INSTALL: no external dependencies
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Optional


# ─── Reasoning Failure Taxonomy ───────────────────────────────────────────────

@dataclass
class ReasoningCase:
    """A labeled reasoning task with expected vs likely LLM behavior."""
    category:        str
    task:            str
    correct_answer:  Any
    llm_likely_does: str
    failure_mode:    str
    rag_fixes:       bool
    real_fix:        str


REASONING_CASES = [

    # ── Multi-hop arithmetic ───────────────────────────────────────────────────
    ReasoningCase(
        category       = "Multi-hop arithmetic",
        task           = (
            "Rack A: 24 ports at 40G. Rack B: 36 ports at 100G. "
            "Rack C: 12 ports at 400G. What is total bandwidth capacity in Tbps?"
        ),
        correct_answer = (24*40 + 36*100 + 12*400) / 1000,   # 7.36 Tbps
        llm_likely_does = "Performs steps but may mis-multiply or carry an error mid-chain.",
        failure_mode   = "Each arithmetic step has ~2-5% error rate; errors compound across steps.",
        rag_fixes      = False,
        real_fix       = "Use Python code interpreter (tool use) or structured step-by-step CoT.",
    ),

    # ── Negation logic ────────────────────────────────────────────────────────
    ReasoningCase(
        category       = "Negation logic",
        task           = (
            "The following devices run ACI 5.2(1g): leaf-101, leaf-102, spine-201. "
            "The following devices are NOT affected by CSCvh23456: leaf-103, spine-202. "
            "Which devices ARE affected?"
        ),
        correct_answer = ["leaf-101", "leaf-102", "spine-201"],
        llm_likely_does = "Often confuses the affected/unaffected sets; may list all devices.",
        failure_mode   = "Negation reasoning requires holding two sets and computing difference — LLMs underperform on NOT logic.",
        rag_fixes      = False,
        real_fix       = "Use Python set difference: affected = aci_5_2 - not_affected",
    ),

    # ── Constraint satisfaction ───────────────────────────────────────────────
    ReasoningCase(
        category       = "Constraint satisfaction",
        task           = (
            "From: [N9K-C9336C: VXLAN=YES, 100G=YES, price=$35K], "
            "[N9K-C93180: VXLAN=YES, 100G=YES, price=$22K], "
            "[N7K-C7009: VXLAN=NO, 100G=YES, price=$45K], "
            "[N9K-C93120: VXLAN=YES, 100G=NO, price=$18K]. "
            "Which switches support VXLAN AND 100G uplinks AND cost < $40K?"
        ),
        correct_answer = ["N9K-C9336C", "N9K-C93180"],
        llm_likely_does = "May include N7K-C7009 (satisfies price+100G but fails VXLAN) or miss one constraint.",
        failure_mode   = "Multi-constraint AND filtering is brittle; model may satisfy 2/3 constraints and still output.",
        rag_fixes      = False,
        real_fix       = "Metadata filtering in search engine (Lesson 9) or structured SQL query.",
    ),

    # ── Counting ─────────────────────────────────────────────────────────────
    ReasoningCase(
        category       = "Counting",
        task           = (
            "Count the total number of unique EPGs in this configuration: "
            "EPG-Web, EPG-DB, EPG-App, EPG-Web, EPG-Cache, EPG-DB, EPG-Monitor"
        ),
        correct_answer = 5,   # unique: Web, DB, App, Cache, Monitor
        llm_likely_does = "May answer 7 (total) or 5 (unique) — inconsistent across runs.",
        failure_mode   = "LLMs do not count by enumeration; they estimate from pattern. Duplicate detection is unreliable.",
        rag_fixes      = False,
        real_fix       = "len(set(['EPG-Web','EPG-DB','EPG-App','EPG-Cache','EPG-Monitor']))",
    ),

    # ── Formal logic ─────────────────────────────────────────────────────────
    ReasoningCase(
        category       = "Formal logic (syllogism)",
        task           = (
            "Premise 1: All ACI fabrics use VXLAN for the overlay. "
            "Premise 2: Fabric-X uses 802.1Q trunking (no VXLAN). "
            "Conclusion: Is Fabric-X an ACI fabric?"
        ),
        correct_answer = False,
        llm_likely_does = "Often answers 'Yes, Fabric-X could be an ACI fabric in a mixed mode...' — hedges instead of applying the syllogism.",
        failure_mode   = "LLMs generate plausible-sounding qualifications rather than apply strict logical rules.",
        rag_fixes      = False,
        real_fix       = "Rule engine or explicit boolean check in code.",
    ),

    # ── Multi-hop fact retrieval ───────────────────────────────────────────────
    ReasoningCase(
        category       = "Multi-hop fact retrieval",
        task           = (
            "What switch model is used in customer ACME's spine layer, "
            "and what is its maximum fabric uplink speed?"
        ),
        correct_answer = "Requires: (1) find ACME's spine model from config, (2) look up that model's spec.",
        llm_likely_does = "Hallucinates a switch model + speed without retrieving actual config.",
        failure_mode   = "Two-hop: first hop retrieves customer config, second hop retrieves spec. Both hops require retrieval.",
        rag_fixes      = True,   # WHY True: RAG + multi-step retrieval (agentic RAG) can handle this
        real_fix       = "Agentic RAG: step 1 = retrieve config, step 2 = retrieve spec for found model.",
    ),
]


# ─── Demonstration ────────────────────────────────────────────────────────────

def display_reasoning_failures():
    """Print all reasoning failure cases with analysis."""

    print("=" * 72)
    print("REASONING FAILURES: What LLMs Cannot Reliably Do")
    print("=" * 72)

    for case in REASONING_CASES:
        rag_label = "YES" if case.rag_fixes else "NO"
        print(f"\n  ── {case.category.upper()} ──")
        print(f"  Task:          {case.task[:80]}...")
        print(f"  Correct answer: {case.correct_answer}")
        print(f"  LLM behavior:  {case.llm_likely_does}")
        print(f"  Failure mode:  {case.failure_mode}")
        print(f"  RAG fixes this: {rag_label}")
        print(f"  Real fix:      {case.real_fix}")


# ─── Arithmetic Verification ──────────────────────────────────────────────────

def verify_arithmetic():
    """
    Show WHY arithmetic should be delegated to code.
    Demonstrate the step-by-step error accumulation problem.
    """

    print("\n" + "=" * 72)
    print("ARITHMETIC: Why Code Beats LLM for Calculations")
    print("=" * 72)

    # The task
    racks = [
        {"name": "Rack A", "ports": 24,  "speed_gbps": 40},
        {"name": "Rack B", "ports": 36,  "speed_gbps": 100},
        {"name": "Rack C", "ports": 12,  "speed_gbps": 400},
    ]

    print("\n  TASK: Compute total fabric bandwidth from rack specifications")
    print(f"\n  {'Rack':<10} {'Ports':>7} {'Speed (Gbps)':>14} {'Capacity (Gbps)':>17}")
    print(f"  {'─'*10} {'─'*7} {'─'*14} {'─'*17}")

    total_gbps = 0
    for rack in racks:
        capacity = rack["ports"] * rack["speed_gbps"]
        total_gbps += capacity
        print(f"  {rack['name']:<10} {rack['ports']:>7} {rack['speed_gbps']:>14} {capacity:>17,}")

    print(f"  {'TOTAL':<10} {'':>7} {'':>14} {total_gbps:>17,}")
    print(f"\n  Total capacity: {total_gbps:,} Gbps = {total_gbps/1000:.2f} Tbps")
    print(f"\n  Python code (always correct):")
    print(f"    racks = [(24, 40), (36, 100), (12, 400)]")
    print(f"    total = sum(p * s for p, s in racks)  # → {sum(p*s for p,s in [(24,40),(36,100),(12,400)]):,} Gbps")

    # Simulate LLM arithmetic error rate
    print(f"\n  LLM arithmetic error simulation:")
    print(f"  (Each step has ~3% error rate; errors compound)")

    random_import = __import__("random")
    random_import.seed(42)

    error_rate = 0.03
    runs = 20
    correct = 0

    for _ in range(runs):
        # Simulate each multiplication step with error probability
        result = 0
        for rack in racks:
            step = rack["ports"] * rack["speed_gbps"]
            if random_import.random() < error_rate:
                step = step + random_import.choice([-rack["speed_gbps"], rack["speed_gbps"],
                                                     -rack["ports"], rack["ports"]])
            result += step

        if result == total_gbps:
            correct += 1

    print(f"  Simulated 20 LLM runs: {correct}/20 correct = {correct/runs:.0%} accuracy")
    print(f"  (Real LLMs perform better on simple arithmetic but degrade on long chains)")
    print(f"\n  RULE: For ANY calculation that matters, use tool_use + Python code.")


# ─── Negation Failure Demo ────────────────────────────────────────────────────

def negation_failure_demo():
    """
    Demonstrate the negation reasoning problem with a set operation.
    Shows how Python code trivially solves what LLMs struggle with.
    """

    print("\n" + "=" * 72)
    print("NEGATION LOGIC: Set Operations LLMs Bungle, Python Solves Trivially")
    print("=" * 72)

    # All devices running ACI 5.2(1g)
    running_5_2_1g = {"leaf-101", "leaf-102", "leaf-103", "spine-201", "spine-202"}

    # Devices confirmed NOT affected by the bug (e.g., already patched)
    not_affected   = {"leaf-103", "spine-202"}

    # Question: Which devices ARE still at risk?
    at_risk = running_5_2_1g - not_affected

    print(f"\n  Data:")
    print(f"    Devices running ACI 5.2(1g):  {sorted(running_5_2_1g)}")
    print(f"    Devices NOT affected (patched): {sorted(not_affected)}")
    print(f"\n  Question: Which devices ARE at risk from CSCvh23456?")
    print(f"\n  Python answer (set difference): {sorted(at_risk)}")
    print(f"\n  Common LLM errors on this task:")
    print(f"    ✗ Lists ALL devices running 5.2(1g) (ignores NOT affected set)")
    print(f"    ✗ Lists only the NOT affected devices (inverts the logic)")
    print(f"    ✗ Lists all 5 correctly but adds '...and others may also be affected'")
    print(f"\n  WHY: Negation requires maintaining two sets and computing their difference.")
    print(f"  LLMs are optimized for generation, not set algebra.")
    print(f"\n  MITIGATION: When query contains 'NOT affected', 'EXCEPT', 'EXCLUDING',")
    print(f"  route to structured filtering (metadata filter or SQL) not free-form LLM.")


# ─── Chain-of-Thought Improvement ────────────────────────────────────────────

def chain_of_thought_explanation():
    """
    Explain why chain-of-thought helps some reasoning failures but not all.
    """

    print("\n" + "=" * 72)
    print("CHAIN-OF-THOUGHT: What It Fixes and What It Doesn't")
    print("=" * 72)

    print(f"""
  CHAIN-OF-THOUGHT (CoT) prompting:
    Instruction: "Think step by step before giving your final answer."

  WHY it helps:
    Forces the model to emit intermediate reasoning tokens.
    Those tokens become context for subsequent tokens → reduces drift.
    Works well for: simple arithmetic, logical deduction, structured comparison.

  DOESN'T HELP WHEN:
    1. Arithmetic chains are long: errors still compound even step-by-step.
    2. Negation: CoT helps but doesn't eliminate negation blindness.
    3. Data doesn't exist in context: CoT can't generate facts it doesn't have.
    4. Hallucination mid-chain: model can hallucinate a "step" that looks valid.

  COMPARISON:

  Task: "What is 24×40 + 36×100 + 12×400?"

  Without CoT:
    → "The total is 6,960 Gbps" (may be wrong, no audit trail)

  With CoT:
    → "Step 1: 24 × 40 = 960. Step 2: 36 × 100 = 3,600. Step 3: 12 × 400 = 4,800.
       Total: 960 + 3,600 + 4,800 = 9,360 Gbps" (wrong: 960+3600+4800 = 9360, correct is 7360)
    → CoT shows the error: Step 3 + Sum are wrong. But the model still made an error.

  With Tool Use (Python):
    → "Let me calculate: result = 24*40 + 36*100 + 12*400 = 7360" (always correct)

  RULE:
    CoT: use for multi-step reasoning where intermediate steps are verifiable.
    Tool use: use for arithmetic, logic, constraint satisfaction, counting.
    Both: required for multi-hop retrieval (retrieve → reason → retrieve again).
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    display_reasoning_failures()
    verify_arithmetic()
    negation_failure_demo()
    chain_of_thought_explanation()
