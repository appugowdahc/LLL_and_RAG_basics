"""
FILE: 01_query_analysis.py
LESSON: Phase 2 - Lesson 14 - Query Understanding and Rewriting
TOPIC: Query analysis — classify queries before deciding how to rewrite them

WHAT THIS FILE TEACHES:
  - Query type classification (factual, procedural, comparative, multi-hop)
  - Complexity scoring: how much does this query need rewriting?
  - Vocabulary mismatch detection
  - Exact-term detection (queries that should NOT be rewritten)
  - Intent extraction: what kind of answer does the user need?
  - WHY you must analyze before rewriting — not all queries benefit

INSTALL: no external dependencies
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ─── Query Classification ─────────────────────────────────────────────────────

QUERY_TYPES = {
    "factual": {
        "description": "Requests a specific fact, value, or definition.",
        "examples":    ["What is the minimum APIC node count?", "What port does APIC use?"],
        "signals":     [r"\bwhat is\b", r"\bhow many\b", r"\bwhat are the\b", r"\bdefine\b"],
        "best_rewrite": "HyDE — hypothetical answer embeds better than short factual query",
    },
    "procedural": {
        "description": "Asks how to perform a task or configure something.",
        "examples":    ["How do I configure an EPG contract?", "How to set up APIC HA?"],
        "signals":     [r"\bhow (do|to|can)\b", r"\bsteps? to\b", r"\bprocedure\b", r"\bconfigure\b"],
        "best_rewrite": "HyDE + expansion — procedural docs use step-specific vocab",
    },
    "comparative": {
        "description": "Asks to compare two or more things.",
        "examples":    ["What is the difference between ACI and NX-OS?", "Compare BM25 and dense search"],
        "signals":     [r"\bdifference between\b", r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b", r"\bpros and cons\b"],
        "best_rewrite": "Sub-question decomposition — one retrieval per subject being compared",
    },
    "multi_hop": {
        "description": "Requires chaining multiple facts to reach an answer.",
        "examples":    ["What switch model is in ACME's spine layer and what is its max port speed?"],
        "signals":     [r"\band (what|how|why)\b", r"\bwhich\b.{0,30}\band\b", r"\bthen\b"],
        "best_rewrite": "Sub-question decomposition — each hop is a separate retrieval",
    },
    "exact_term": {
        "description": "Contains an exact identifier (bug ID, CVE, model number, port).",
        "examples":    ["CSCvh23456 workaround", "CVE-2024-12345", "Nexus 9336C-FX2 specs"],
        "signals":     [r"CVE-\d{4}-\d+", r"CSC[a-z]{2}\d+", r"\b\d+\.\d+\.\d+\b",
                        r"\bN[0-9]K-[A-Z0-9-]+\b"],
        "best_rewrite": "NONE — exact terms should reach BM25 unmodified",
    },
    "negation": {
        "description": "Asks about what is NOT true, or what should be excluded.",
        "examples":    ["Which devices are NOT affected by CSCvh23456?"],
        "signals":     [r"\bnot\s+\w+\b", r"\bexcept\b", r"\bexcluding\b", r"\bnone\b"],
        "best_rewrite": "Route to structured filter — negation logic is fragile in LLMs",
    },
    "general": {
        "description": "General knowledge or conceptual question.",
        "examples":    ["What is BGP route reflection?", "Explain VXLAN encapsulation"],
        "signals":     [r"\bexplain\b", r"\bwhat is a\b", r"\bhow does .{0,20} work\b"],
        "best_rewrite": "Optional HyDE — parametric memory may already be sufficient",
    },
}


@dataclass
class QueryAnalysis:
    """Complete analysis of a single user query."""
    query:              str
    query_type:         str
    complexity_score:   float         # 0.0 = simple, 1.0 = very complex
    needs_rewrite:      bool
    recommended_technique: str
    signals_found:      list[str]
    sub_questions:      list[str]     # detected sub-questions if multi-part
    exact_terms:        list[str]     # detected exact identifiers
    word_count:         int


def analyze_query(query: str) -> QueryAnalysis:
    """
    Classify a query and recommend a rewriting strategy.

    Algorithm:
      1. Detect exact terms (skip rewriting for these).
      2. Score complexity (multiple ? marks, "and", multi-hop signals).
      3. Match against type signals (precedence order matters).
      4. Recommend technique based on type + complexity.
    """
    q_low     = query.lower()
    words     = query.split()

    # ── Exact term detection ──────────────────────────────────────────────────
    exact_terms = []
    for pattern in QUERY_TYPES["exact_term"]["signals"]:
        matches = re.findall(pattern, query, re.IGNORECASE)
        exact_terms.extend(matches)

    # ── Type detection (order matters: exact > negation > comparative > ...) ──
    type_order = ["exact_term", "negation", "comparative", "multi_hop", "procedural", "factual", "general"]
    detected_type = "general"

    for qtype in type_order:
        for pattern in QUERY_TYPES[qtype]["signals"]:
            if re.search(pattern, q_low):
                detected_type = qtype
                break
        if detected_type != "general" or qtype == "general":
            break

    # ── Complexity scoring ────────────────────────────────────────────────────
    # WHY these signals: each adds independent information demand
    complexity = 0.0
    if len(words) <= 4:                              complexity += 0.3   # too short
    if len(words) >= 20:                             complexity += 0.2   # very long
    if q_low.count(" and ") >= 2:                   complexity += 0.3   # multi-part
    if re.search(r"\band\s+what\b|\band\s+how\b", q_low): complexity += 0.4
    if re.search(r"\?.*\?", query):                 complexity += 0.4   # multiple questions
    if detected_type == "multi_hop":                complexity += 0.3
    if detected_type == "comparative":              complexity += 0.2
    complexity = min(1.0, complexity)

    # ── Sub-question detection ────────────────────────────────────────────────
    sub_questions = []
    if "?" in query:
        parts = [p.strip() + "?" for p in query.rstrip("?").split("?") if p.strip()]
        if len(parts) > 1:
            sub_questions = parts
    elif " and " in q_low and detected_type in ("multi_hop", "comparative"):
        # Heuristic: split on " and " for comparative/multi-hop queries
        parts = re.split(r"\s+and\s+", query, maxsplit=1)
        if len(parts) == 2:
            sub_questions = [parts[0].strip() + "?", "And " + parts[1].strip() + "?"]

    # ── Rewrite decision ──────────────────────────────────────────────────────
    needs_rewrite    = detected_type not in ("exact_term", "negation") and complexity >= 0.1
    recommended      = QUERY_TYPES[detected_type]["best_rewrite"]

    # ── Signals found ─────────────────────────────────────────────────────────
    signals = []
    for pattern in QUERY_TYPES[detected_type]["signals"]:
        m = re.search(pattern, q_low)
        if m:
            signals.append(f"'{pattern}' matched '{m.group(0)}'")

    return QueryAnalysis(
        query                 = query,
        query_type            = detected_type,
        complexity_score      = complexity,
        needs_rewrite         = needs_rewrite,
        recommended_technique = recommended,
        signals_found         = signals[:3],
        sub_questions         = sub_questions,
        exact_terms           = exact_terms,
        word_count            = len(words),
    )


# ─── Vocabulary Gap Detector ──────────────────────────────────────────────────

def vocabulary_gap_score(query: str, corpus_vocab: set[str]) -> float:
    """
    Estimate how well the query's vocabulary matches the indexed corpus.

    A high gap score → high vocabulary mismatch → retrieval will suffer
    → rewriting / expansion is beneficial.

    Args:
        query:        User query string.
        corpus_vocab: Set of tokens in the indexed corpus.

    Returns:
        0.0 = full overlap (no rewriting needed)
        1.0 = no overlap (rewriting is critical)
    """
    query_tokens = {t.lower() for t in re.findall(r"\b\w{3,}\b", query)}
    stop         = {"what", "how", "does", "the", "and", "for", "with", "are", "can", "will"}
    query_tokens -= stop

    if not query_tokens:
        return 0.5   # can't assess

    overlap = sum(1 for t in query_tokens if t in corpus_vocab)
    return 1.0 - (overlap / len(query_tokens))   # WHY invert: 0 = good, 1 = bad


# ─── Demo ─────────────────────────────────────────────────────────────────────

def run_analysis_demo():
    """
    Classify and analyze a diverse set of queries.
    """

    print("=" * 70)
    print("QUERY ANALYSIS: Classification and Rewrite Recommendations")
    print("=" * 70)

    queries = [
        "What is the minimum APIC node count for high availability?",
        "How do I configure an EPG contract in Cisco ACI?",
        "What is the difference between ACI and traditional NX-OS switching?",
        "What switch model does ACME use in the spine layer and what is its max speed?",
        "CSCvh23456 workaround for APIC 5.2(1g)",
        "CVE-2024-12345 affected versions",
        "Which devices are NOT affected by the bug?",
        "What is BGP?",
        "How does ReadyOps validate ACI changes and what agent classes does it use?",
        "APIC HA?",
    ]

    for q in queries:
        analysis = analyze_query(q)
        needs    = "YES" if analysis.needs_rewrite else "NO"
        print(f"\n  Query: '{q}'")
        print(f"  Type:       {analysis.query_type:<15} Complexity: {analysis.complexity_score:.2f}  Rewrite: {needs}")
        print(f"  Technique:  {analysis.recommended_technique}")
        if analysis.exact_terms:
            print(f"  Exact terms found: {analysis.exact_terms}")
        if analysis.sub_questions:
            print(f"  Sub-questions: {analysis.sub_questions}")


def vocabulary_gap_demo():
    """
    Show vocabulary gap detection with a sample corpus vocabulary.
    """

    print("\n" + "=" * 70)
    print("VOCABULARY GAP: Does the Query Match the Corpus?")
    print("=" * 70)

    # Simulated corpus vocabulary (key terms from indexed ACI docs)
    corpus_vocab = {
        "apic", "cluster", "nodes", "high", "availability", "quorum",
        "leaf", "spine", "topology", "fabric", "vxlan", "epg", "contract",
        "tenant", "policy", "readyops", "validation", "production",
        "representative", "operations", "promotion", "gate", "agent",
        "hypershield", "ebpf", "ise", "trustsec", "sgt",
    }

    test_queries = [
        ("Low gap (matches corpus)",    "What is the APIC cluster HA requirement?"),
        ("Medium gap (partial match)",  "How many controller nodes do I need?"),
        ("High gap (synonym mismatch)", "What is the minimum controller size for redundancy?"),
        ("Exact term (no gap risk)",    "CSCvh23456 contract deployment failure"),
    ]

    print(f"\n  {'Gap':>6}  {'Query'}")
    print(f"  {'─'*6}  {'─'*55}")
    for label, q in test_queries:
        gap = vocabulary_gap_score(q, corpus_vocab)
        print(f"  {gap:>6.2f}  '{q}'  [{label}]")

    print(f"""
  INTERPRETATION:
    Gap 0.0–0.3: Query vocab matches corpus → standard retrieval
    Gap 0.3–0.6: Partial match → consider HyDE or multi-query
    Gap 0.6–1.0: Poor match → multi-query expansion strongly recommended

  EXAMPLE:
    "minimum controller size for redundancy" (gap=0.75)
    → Corpus uses "APIC", "nodes", "HA" — not "controller" or "redundancy"
    → Without rewriting, embedding similarity is low → wrong chunks retrieved
    → Multi-query: add "APIC cluster minimum nodes high availability"
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_analysis_demo()
    vocabulary_gap_demo()
