"""
FILE: 01_knowledge_cutoff.py
LESSON: Phase 1 - Lesson 11 - LLM Limitations
TOPIC: Knowledge cutoff — what the model doesn't know and why

WHAT THIS FILE TEACHES:
  - What a training cutoff means in practice
  - The lag between training cutoff and model release (typically 6–12 months)
  - Categories of knowledge that are ALWAYS missing from LLMs
  - How to detect when a query requires post-cutoff knowledge
  - Routing strategy: which queries need retrieval vs which can use parametric memory
  - WHY internal documentation is permanently missing regardless of cutoff

INSTALL: no external dependencies
"""

import re
from dataclasses import dataclass
from typing import Optional
from datetime import date


# ─── Model Cutoff Reference ───────────────────────────────────────────────────

# Approximate training cutoffs for major models (as of mid-2025).
# WHY store this: when building a RAG router, you need to know whether a query
# is likely to touch post-cutoff knowledge. If query mentions a date or version
# that is past the model's cutoff, always route to retrieval.
MODEL_CUTOFFS = {
    "claude-sonnet-4-6":         date(2025, 8, 1),
    "claude-opus-4-8":           date(2025, 8, 1),
    "gpt-4o":                    date(2024, 4, 1),
    "gpt-4-turbo":               date(2023, 12, 1),
    "gemini-1.5-pro":            date(2024, 5, 1),
    "llama-3.1-405b":            date(2023, 12, 1),
}

# Release lag: typical gap between training cutoff and public availability
# WHY this matters: even "new" models may be trained on data 6-12 months old
TYPICAL_RELEASE_LAG_MONTHS = 6


# ─── Knowledge Category Analysis ──────────────────────────────────────────────

@dataclass
class KnowledgeCategory:
    """Classification of knowledge by cutoff sensitivity."""
    name:            str
    cutoff_risk:     str   # "high" | "medium" | "low"
    always_missing:  bool  # True if RAG is always required regardless of cutoff
    example:         str
    rag_recommendation: str


KNOWLEDGE_CATEGORIES = [

    KnowledgeCategory(
        name            = "Current software versions",
        cutoff_risk     = "high",
        always_missing  = False,
        example         = "What is the latest ACI version? What features are in ISE 3.4?",
        rag_recommendation = "Always retrieve from vendor release notes or documentation portal.",
    ),

    KnowledgeCategory(
        name            = "Security advisories (CVEs)",
        cutoff_risk     = "high",
        always_missing  = False,
        example         = "Is my ACI version affected by CVE-2025-12345?",
        rag_recommendation = "Retrieve from Cisco PSIRT or NVD. Never rely on parametric memory for CVEs.",
    ),

    KnowledgeCategory(
        name            = "Internal documentation",
        cutoff_risk     = "high",
        always_missing  = True,  # WHY always_missing: internal docs never in training data
        example         = "What is our ACI naming convention? What does our change window SLA say?",
        rag_recommendation = "Index internal docs into private knowledge base. No alternative.",
    ),

    KnowledgeCategory(
        name            = "Customer/tenant configuration",
        cutoff_risk     = "high",
        always_missing  = True,
        example         = "Does tenant ACME have contract between EPG-Web and EPG-DB?",
        rag_recommendation = "Retrieve from live APIC API or configuration snapshot database.",
    ),

    KnowledgeCategory(
        name            = "Product EOL and pricing",
        cutoff_risk     = "medium",
        always_missing  = False,
        example         = "Is Nexus 7000 still sold? What is the list price for N9K-C9336C?",
        rag_recommendation = "Retrieve from vendor portal. Prices change; EOL dates are updated.",
    ),

    KnowledgeCategory(
        name            = "Incident and post-mortem history",
        cutoff_risk     = "high",
        always_missing  = True,
        example         = "What was the root cause of last Tuesday's fabric outage?",
        rag_recommendation = "Index incident tickets and post-mortem reports into private KB.",
    ),

    KnowledgeCategory(
        name            = "Well-established networking concepts",
        cutoff_risk     = "low",
        always_missing  = False,
        example         = "What is BGP route reflection? How does OSPF elect a DR?",
        rag_recommendation = "Parametric memory usually sufficient. Add retrieval for org-specific configs.",
    ),

    KnowledgeCategory(
        name            = "General Python/coding patterns",
        cutoff_risk     = "low",
        always_missing  = False,
        example         = "How do I parse JSON in Python? What is a decorator?",
        rag_recommendation = "Parametric memory sufficient for stable language features.",
    ),
]


def display_knowledge_categories():
    """Print the knowledge category risk table."""

    print("=" * 72)
    print("KNOWLEDGE CUTOFF RISK BY CATEGORY")
    print("=" * 72)

    print(f"\n  {'Category':<35} {'Risk':<8} {'Always missing?':<17} {'RAG needed?'}")
    print(f"  {'─'*35} {'─'*8} {'─'*17} {'─'*11}")

    for kc in KNOWLEDGE_CATEGORIES:
        always = "YES — internal"  if kc.always_missing else "No"
        rag    = "ALWAYS"          if kc.always_missing or kc.cutoff_risk == "high" else \
                 "Sometimes"       if kc.cutoff_risk == "medium" else "Rarely"
        print(f"  {kc.name:<35} {kc.cutoff_risk:<8} {always:<17} {rag}")

    print(f"""
  RULE:
    "Always missing" = the information was never in any public training corpus.
    Even if the model's cutoff were yesterday, it still couldn't answer
    questions about your organization's private data. RAG is the only path.
""")


# ─── Cutoff Detection ─────────────────────────────────────────────────────────

def detect_cutoff_risk(
    query:         str,
    model:         str = "claude-sonnet-4-6",
    today:         date = date.today(),
) -> dict:
    """
    Heuristic detector: does this query likely require post-cutoff knowledge?

    Signals:
      1. Query contains a recent year or "latest", "new", "current", "2025".
      2. Query mentions a specific version number.
      3. Query mentions CVE or advisory ID.
      4. Query contains organization-internal terms (heuristic: first-person org language).

    Returns:
        dict with risk_level, signals found, and recommendation.
    """

    cutoff = MODEL_CUTOFFS.get(model, date(2024, 1, 1))
    q_low  = query.lower()

    signals = []

    # Signal 1: Recent year or "latest"/"current"
    years = re.findall(r"\b(202[3-9]|203\d)\b", query)
    for y in years:
        if date(int(y), 1, 1) > cutoff:
            signals.append(f"mentions year {y} (after cutoff {cutoff.year})")

    if any(w in q_low for w in ["latest", "current", "newest", "today", "now", "recent"]):
        signals.append("contains recency keyword (latest/current/now)")

    # Signal 2: Specific version numbers (e.g., "ACI 6.2", "ISE 3.4")
    versions = re.findall(r"\b\d+\.\d+(?:\.\d+)?\b", query)
    for v in versions:
        signals.append(f"contains version number {v} (may post-date training)")

    # Signal 3: CVE or advisory ID
    if re.search(r"CVE-\d{4}-\d+", query, re.IGNORECASE):
        signals.append("contains CVE ID (security advisory — always retrieve)")

    if re.search(r"CSC[a-z]{2}\d+", query, re.IGNORECASE):
        signals.append("contains Cisco bug ID (always retrieve from PSIRT)")

    # Signal 4: Internal/private data indicators
    internal_keywords = ["our", "we ", "my org", "my company", "internal", "runbook",
                         "ticket", "incident", "sla", "customer", "tenant"]
    for kw in internal_keywords:
        if kw in q_low:
            signals.append(f"private data signal: '{kw}' (always retrieve)")
            break

    # Determine risk level
    if any("always retrieve" in s for s in signals):
        risk = "critical"
    elif len(signals) >= 2:
        risk = "high"
    elif len(signals) == 1:
        risk = "medium"
    else:
        risk = "low"

    recommendation = {
        "critical": "MUST retrieve. Do not use parametric memory at all.",
        "high":     "Strongly prefer retrieval. Flag if no relevant docs found.",
        "medium":   "Retrieve and verify. Model answer may be stale.",
        "low":      "Parametric memory likely sufficient. Optional retrieval.",
    }[risk]

    return {
        "query":          query,
        "model_cutoff":   cutoff.isoformat(),
        "risk_level":     risk,
        "signals":        signals,
        "recommendation": recommendation,
    }


def run_cutoff_detection_demo():
    """
    Run the cutoff detector on a set of representative queries.
    """

    print("=" * 72)
    print("KNOWLEDGE CUTOFF DETECTOR: Query Risk Analysis")
    print("=" * 72)

    queries = [
        "What is the BGP route reflection process?",
        "What is the latest version of Cisco ACI?",
        "Is my ACI 6.2 affected by CVE-2025-12345?",
        "What does our change management SLA specify?",
        "How do I configure an EPG contract in ACI?",
        "What new features were added to ISE in 2025?",
        "Does our tenant ACME have a web-to-DB contract?",
        "What is the difference between VXLAN and NVGRE?",
        "Fix the CSCvh23456 bug we found in APIC 5.2(1g)",
    ]

    for q in queries:
        result = detect_cutoff_risk(q)
        risk   = result["risk_level"].upper()
        print(f"\n  [{risk:<8}] {q}")
        for sig in result["signals"]:
            print(f"             • {sig}")
        print(f"             → {result['recommendation']}")


# ─── Staleness Impact Analysis ────────────────────────────────────────────────

def staleness_impact_analysis():
    """
    Show what happens when an LLM answers from stale parametric memory.
    These are realistic examples of staleness-induced errors.
    """

    print("\n" + "=" * 72)
    print("STALENESS IMPACT: What Goes Wrong With Outdated Parametric Memory")
    print("=" * 72)

    cases = [
        {
            "scenario":    "Version recommendation",
            "query":       "What ACI version should I deploy for a new fabric?",
            "stale_ans":   "Deploy ACI 5.2 — it's the current stable release.",
            "reality":     "ACI 6.0 and 6.1 have been released; 5.2 is now legacy.",
            "consequence": "Customer deploys an older, unsupported version.",
        },
        {
            "scenario":    "CVE risk assessment",
            "query":       "Is our APIC 5.2(1g) instance at risk from recent vulnerabilities?",
            "stale_ans":   "No known critical CVEs affect APIC 5.2(1g) at this time.",
            "reality":     "CVE-2024-XXXXX was published after training cutoff, affecting 5.2(1g).",
            "consequence": "Security team believes they are safe; vulnerability goes unpatched.",
        },
        {
            "scenario":    "Feature availability",
            "query":       "Does ACI support microsegmentation at the workload level?",
            "stale_ans":   "ACI supports microsegmentation via EPG policy, but workload-level "
                           "enforcement requires dedicated perimeter tools.",
            "reality":     "Cisco Hypershield + ACI integration (released 2024) enables "
                           "kernel-level workload policy via eBPF.",
            "consequence": "Architect designs an unnecessarily complex solution using legacy tooling.",
        },
        {
            "scenario":    "End-of-life status",
            "query":       "Can I still purchase Nexus 7000 switches?",
            "stale_ans":   "Yes, Nexus 7000 is available and widely deployed in enterprise DCs.",
            "reality":     "Nexus 7000 reached end-of-sale and end-of-software-maintenance.",
            "consequence": "Procurement team attempts to buy EOL hardware; quote fails.",
        },
    ]

    for c in cases:
        print(f"\n  SCENARIO: {c['scenario']}")
        print(f"  Query:       '{c['query']}'")
        print(f"  Stale model: '{c['stale_ans'][:80]}'")
        print(f"  Reality:     {c['reality']}")
        print(f"  Risk:        {c['consequence']}")

    print(f"""
  MITIGATION PATTERN (applies to all scenarios):
    1. Run cutoff_risk detector on the query.
    2. If risk = 'high' or 'critical', always retrieve from the knowledge base.
    3. Include date metadata in retrieved chunks: "Updated: 2025-06-01".
    4. Instruct the model: "If your training data and the retrieved document differ,
       trust the retrieved document."
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    display_knowledge_categories()
    run_cutoff_detection_demo()
    staleness_impact_analysis()
