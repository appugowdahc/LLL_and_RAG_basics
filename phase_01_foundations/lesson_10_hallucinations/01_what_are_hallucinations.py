"""
FILE: 01_what_are_hallucinations.py
LESSON: Phase 1 - Lesson 10 - Hallucinations
TOPIC: What hallucinations are, the three types, and why they matter

WHAT THIS FILE TEACHES:
  - Definition of an LLM hallucination
  - Three-type taxonomy: intrinsic, extrinsic, confabulation
  - Real examples from infrastructure/networking domain
  - Why hallucinations are dangerous in enterprise RAG
  - WHY confident-sounding wrong answers are more dangerous than uncertain ones

INSTALL: pip install anthropic python-dotenv  (live demo; runs without API key too)
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ─── Hallucination Taxonomy ───────────────────────────────────────────────────

@dataclass
class HallucinationExample:
    """
    A labeled example of an LLM hallucination.
    Each instance shows the context, a hallucinated output, and the correct answer.
    """
    hallucination_type: str          # "intrinsic" | "extrinsic" | "confabulation"
    context:            Optional[str]
    question:           str
    hallucinated_answer: str
    correct_answer:      str
    why_it_happens:      str
    danger_level:        str         # "high" | "medium" | "low"


HALLUCINATION_EXAMPLES = [

    # ── Intrinsic Hallucination ────────────────────────────────────────────────
    # Model is given the correct context but contradicts it.
    HallucinationExample(
        hallucination_type  = "intrinsic",
        context             = (
            "The APIC cluster in Cisco ACI requires a minimum of 3 nodes "
            "for high availability. Maximum supported scale is 200 leaf switches "
            "per pod in ACI version 6.0."
        ),
        question            = "How many APIC nodes are required for ACI HA?",
        hallucinated_answer = "For high availability, Cisco ACI requires a minimum of 5 APIC nodes.",
        correct_answer      = "3 APIC nodes (as stated in the provided context).",
        why_it_happens      = (
            "The model's parametric memory (from training) contains '5' as a "
            "commonly associated number with HA configurations. It over-rides "
            "the retrieved context — a 'Lost in the Middle' failure mode."
        ),
        danger_level        = "high",
    ),

    # ── Extrinsic Hallucination ───────────────────────────────────────────────
    # Model adds correct-sounding but ungrounded information.
    HallucinationExample(
        hallucination_type  = "extrinsic",
        context             = (
            "Cisco ISE uses RADIUS for network access authentication. "
            "TrustSec assigns Security Group Tags (SGTs) at authentication time."
        ),
        question            = "What protocols does ISE use for authentication?",
        hallucinated_answer = (
            "ISE uses RADIUS for authentication. It also supports TACACS+ for "
            "device administration, 802.1X for wired/wireless access, and MAB "
            "(MAC Authentication Bypass) for non-supplicant devices. "
            "The default RADIUS port is 1812."
        ),
        correct_answer      = "RADIUS (as stated in context). The rest is ungrounded by the provided context.",
        why_it_happens      = (
            "ISE does support TACACS+, 802.1X, and MAB in reality — the model "
            "has this in parametric memory and adds it as context-enrichment. "
            "The extra info may be correct but is not attributable to the source. "
            "In regulated environments, every claim needs a citation."
        ),
        danger_level        = "medium",
    ),

    # ── Confabulation ─────────────────────────────────────────────────────────
    # Model fabricates entities (papers, people, specs) that don't exist.
    HallucinationExample(
        hallucination_type  = "confabulation",
        context             = None,   # no context provided — pure parametric query
        question            = "What paper introduced the ACI fabric architecture?",
        hallucinated_answer = (
            "The ACI fabric architecture was formally introduced in: "
            "Zhang, L., Patel, R., & Kim, S. (2013). 'Application Centric "
            "Infrastructure: A New Paradigm for Data Center Networking.' "
            "IEEE Transactions on Network and Service Management, 10(4), 312-326."
        ),
        correct_answer      = (
            "No specific academic paper introduced ACI — it is a Cisco product "
            "described in Cisco documentation and whitepapers, not a peer-reviewed "
            "publication. The cited paper does not exist."
        ),
        why_it_happens      = (
            "LLMs learned the citation format (Author, Year, Title, Journal) as a "
            "pattern from millions of papers in training data. When asked for a "
            "paper, the model instantiates the pattern with plausible-sounding "
            "names and numbers — pure pattern completion, no real memory."
        ),
        danger_level        = "high",
    ),

    # ── Sycophantic Hallucination (special case of extrinsic) ─────────────────
    HallucinationExample(
        hallucination_type  = "extrinsic",
        context             = "ReadyOps requires all validation gates to pass before promotion.",
        question            = "I read that ReadyOps allows 80% pass rate for emergency changes, right?",
        hallucinated_answer = (
            "Yes, ReadyOps does support an emergency bypass mode where changes "
            "can be promoted with an 80% validation pass rate when flagged as "
            "P1 incidents. This is documented in the ReadyOps operations guide."
        ),
        correct_answer      = (
            "The context says 100% pass rate is required. The 80% figure was "
            "introduced by the user in the question — the model validated a false premise."
        ),
        why_it_happens      = (
            "SYCOPHANCY: RLHF training rewarded outputs that make users feel "
            "validated. Model learned to agree with user-stated 'facts' rather "
            "than contradict them — even when those facts are wrong."
        ),
        danger_level        = "high",
    ),
]


# ─── Display and Analysis ─────────────────────────────────────────────────────

def display_examples():
    """Walk through each hallucination example with diagnosis."""

    print("=" * 70)
    print("HALLUCINATION TAXONOMY: Types with Real Infrastructure Examples")
    print("=" * 70)

    type_colors = {
        "intrinsic":     "[INTRINSIC]    ",
        "extrinsic":     "[EXTRINSIC]    ",
        "confabulation": "[CONFABULATION]",
    }

    for i, ex in enumerate(HALLUCINATION_EXAMPLES, 1):
        label = type_colors[ex.hallucination_type]
        print(f"\n{'─'*70}")
        print(f"  Example {i}: {label} (danger={ex.danger_level.upper()})")
        print(f"{'─'*70}")

        if ex.context:
            print(f"\n  CONTEXT PROVIDED TO MODEL:")
            # WHY wrap at 65 chars: readable terminal output
            for line in _wrap(ex.context, 65):
                print(f"    {line}")

        print(f"\n  QUESTION:")
        print(f"    {ex.question}")

        print(f"\n  HALLUCINATED ANSWER:")
        for line in _wrap(ex.hallucinated_answer, 65):
            print(f"    {line}")

        print(f"\n  CORRECT ANSWER:")
        for line in _wrap(ex.correct_answer, 65):
            print(f"    {line}")

        print(f"\n  WHY IT HAPPENS:")
        for line in _wrap(ex.why_it_happens, 65):
            print(f"    {line}")


def _wrap(text: str, width: int) -> list[str]:
    """Simple word-wrap for terminal display."""
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    return lines


# ─── Why Confident-Sounding Errors Are Worse ──────────────────────────────────

def confidence_danger_analysis():
    """
    Demonstrate why LLM hallucinations are uniquely dangerous:
    they are delivered with high confidence and fluent language.
    """

    print("\n" + "=" * 70)
    print("WHY CONFIDENT HALLUCINATIONS ARE MORE DANGEROUS THAN UNCERTAIN ONES")
    print("=" * 70)

    scenarios = [
        {
            "label":    "Human Expert (uncertain)",
            "answer":   "I'm not sure about the exact scale limits for ACI 6.0 — "
                        "let me check the data sheet before giving you a number.",
            "danger":   "LOW — uncertainty signals the user to verify independently.",
        },
        {
            "label":    "LLM (hallucinating, confident)",
            "answer":   "ACI 6.0 supports a maximum of 180 leaf switches per pod "
                        "and 12 pods per Multi-Pod deployment, for a total fabric "
                        "scale of 2,160 leaf switches.",
            "danger":   "HIGH — fluent, specific, authoritative tone. "
                        "User copies this into a design doc without checking.",
        },
        {
            "label":    "RAG-grounded LLM (with citation)",
            "answer":   "According to the ACI 6.0 data sheet (doc_006): "
                        "ACI 6.0 supports 200 leaf switches per pod. "
                        "I did not find Multi-Pod scale limits in the retrieved documents.",
            "danger":   "LOW — answer is cited; scope limits are explicit.",
        },
    ]

    print(f"\n  {'Scenario':<40} {'Danger':<8}")
    print(f"  {'─'*40} {'─'*8}")

    for s in scenarios:
        print(f"\n  [{s['label']}]")
        for line in _wrap(s["answer"], 64):
            print(f"    {line}")
        print(f"  → Danger: {s['danger']}")

    print(f"""
  KEY INSIGHT:
    The danger of LLM hallucinations is NOT that they occur —
    it's that they are indistinguishable from correct answers
    to a non-expert user.

    A human saying "I think it's 180..." signals uncertainty.
    An LLM saying "ACI 6.0 supports 180 leaf switches per pod"
    sounds like it read the data sheet.

    This is why RAG + citations is not optional in enterprise use —
    it transforms unverifiable assertions into auditable, citable claims.
""")


# ─── Hallucination Rate by Task ───────────────────────────────────────────────

def hallucination_rates_by_task():
    """
    Show which task types are high vs low hallucination risk.
    This guides where RAG mitigation is most critical.
    """

    print("=" * 70)
    print("HALLUCINATION RISK BY TASK TYPE")
    print("=" * 70)

    tasks = [
        # (task, risk, why, mitigation)
        ("Summarize a provided document",    "LOW",  "Context is provided; model paraphrases.",                "None needed; add faithfulness check"),
        ("Explain a general concept",        "LOW",  "General knowledge; verifiable against many sources.",   "Spot-check against known facts"),
        ("Retrieve a specific version/date", "HIGH", "Precise numbers are poorly memorized.",                 "Always retrieve; never rely on memory"),
        ("Cite a paper or specification",    "HIGH", "Model generates plausible-looking citations.",           "Require retrieved source; validate DOI"),
        ("Describe a CLI command or API",    "HIGH", "Commands evolve; old syntax persists in training data.", "Retrieve from vendor docs"),
        ("Answer about current events",      "HIGH", "Training cutoff; post-cutoff events are unknown.",      "Real-time retrieval required"),
        ("Write code from a description",    "MED",  "Logic is checkable; API details may be wrong.",         "Test execution; retrieve API docs"),
        ("Translate text",                   "LOW",  "Linguistic transformation; meaning is stable.",          "Low RAG value"),
    ]

    print(f"\n  {'Task':<42} {'Risk':<8} {'Mitigation'}")
    print(f"  {'─'*42} {'─'*8} {'─'*30}")

    for task, risk, why, mitigation in tasks:
        print(f"  {task:<42} {risk:<8} {mitigation}")

    print(f"""
  RULE OF THUMB for Criterion Networks RAG:
    ANY query about: version numbers, port numbers, CLI syntax, scale limits,
    configuration parameters, bug IDs, CVE details, protocol specs —
    MUST go through retrieval. Never trust parametric memory for specifics.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    display_examples()
    confidence_danger_analysis()
    hallucination_rates_by_task()
