"""
FILE: 05_mini_project_query_router.py
LESSON: Phase 1 - Lesson 11 - LLM Limitations
TOPIC: Mini-project — Query router that classifies limitations and routes accordingly

WHAT THIS BUILDS:
  A query router that:
    1. Analyzes each incoming query for LLM limitation signals
    2. Classifies the query into one or more limitation categories
    3. Recommends a routing strategy: parametric | static-rag | realtime-rag | tool | escalate
    4. Generates a retrieval plan (which indexes to search, which APIs to call)
    5. Shows why routing decisions matter for answer quality

  In a production RAG system, this runs BEFORE retrieval to decide:
    - Should we retrieve at all? (low-risk queries may not need it)
    - Which knowledge base to search? (public docs vs private vs real-time)
    - Should we delegate to a tool? (arithmetic, filtering)
    - Should we escalate to human? (no indexed data, high-risk decisions)

INSTALL: no external dependencies
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ─── Limitation Categories ────────────────────────────────────────────────────

LIMITATION_SIGNALS = {
    "knowledge_cutoff": {
        "patterns": [
            r"\b(latest|current|newest|recent|now|today)\b",
            r"\b202[4-9]\b|\b203\d\b",          # recent years
            r"CVE-\d{4}-\d+",                    # CVE IDs
            r"CSC[a-z]{2}\d+",                   # Cisco bug IDs
            r"\bv?\d+\.\d+(?:\.\d+)?\b",         # version numbers
        ],
        "description": "Query involves potentially post-cutoff or rapidly-changing information.",
        "route":       "static-rag",
    },
    "private_data": {
        "patterns": [
            r"\b(our|my|we|internal|runbook|sla|customer|tenant|incident|ticket)\b",
            r"\b(acme|customer\s+\w+|client)\b",
            r"\b(naming convention|change window|escalation|on-?call)\b",
            r"\b(post-?mortem|outage|root cause)\b",
        ],
        "description": "Query requires private or organizational knowledge not in training data.",
        "route":       "static-rag",
    },
    "realtime_data": {
        "patterns": [
            r"\b(current(ly)?|right now|live|active|status|fault|alert|alarm)\b",
            r"\b(up|down|reachable|unreachable|connected|disconnected)\b",
            r"\b(metric|latency|throughput|utilization|cpu|memory)\b",
            r"\b(open ticket|active incident|current change)\b",
        ],
        "description": "Query requires live system state from real-time APIs.",
        "route":       "realtime-rag",
    },
    "arithmetic": {
        "patterns": [
            r"\b(calculate|compute|total|sum|how many|count|average|percentage|ratio)\b",
            r"\b\d+\s*[×x\*]\s*\d+",             # multiplication expression
            r"\b\d+\s*[+\-]\s*\d+",               # arithmetic expression
            r"\b(bandwidth|capacity|throughput)\s+\w+\s+\d+",
        ],
        "description": "Query requires arithmetic computation — delegate to code tool.",
        "route":       "tool",
    },
    "negation_filter": {
        "patterns": [
            r"\bnot\s+(affected|running|using|configured|in)\b",
            r"\b(except|excluding|without|other than)\b",
            r"\bwhich\s+\w+\s+are\s+not\b",
            r"\b(exclude|filter out|remove)\b",
        ],
        "description": "Query contains negation/exclusion logic — route to structured filter.",
        "route":       "tool",
    },
    "constraint_filter": {
        "patterns": [
            r"\b(support(?:s)?\s+\w+\s+and)\b",
            r"\b(price|cost|budget)\s*[<>≤≥]\s*\$?\d+",
            r"\ball\s+\w+\s+that\s+(have|support|are)\b",
            r"\bwhich\s+\w+\s+(have|support|include)\s+both\b",
        ],
        "description": "Query requires multi-constraint filtering — use metadata filter or SQL.",
        "route":       "tool",
    },
    "general_knowledge": {
        "patterns": [
            r"\bwhat\s+is\s+(bgp|ospf|vxlan|vlan|tcp|udp|http|dns|dhcp|nat)\b",
            r"\bhow\s+does\s+(spanning tree|routing|switching|encapsulation)\b",
            r"\bexplain\s+(the\s+)?(concept|protocol|algorithm|mechanism)\b",
        ],
        "description": "General networking or computing concept — parametric memory is likely sufficient.",
        "route":       "parametric",
    },
}


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class LimitationSignal:
    """A detected limitation signal in the query."""
    category:    str
    pattern:     str
    match_text:  str
    description: str


@dataclass
class RoutingDecision:
    """The result of query routing analysis."""
    query:               str
    signals:             list[LimitationSignal]
    primary_route:       str          # "parametric" | "static-rag" | "realtime-rag" | "tool" | "hybrid" | "escalate"
    retrieval_plan:      list[str]    # which indexes/APIs to query
    tool_plan:           list[str]    # which tools/code to run
    confidence:          str          # "high" | "medium" | "low"
    explanation:         str

    def display(self):
        print(f"\n  ──────────────────────────────────────────────────────────────")
        print(f"  QUERY: '{self.query}'")
        print(f"  Route: {self.primary_route.upper()} (confidence: {self.confidence})")
        print(f"  Why:   {self.explanation}")

        if self.signals:
            print(f"  Signals detected:")
            for sig in self.signals:
                print(f"    [{sig.category:<20}] matched '{sig.match_text}' → {sig.description[:50]}")

        if self.retrieval_plan:
            print(f"  Retrieval plan:")
            for step in self.retrieval_plan:
                print(f"    → {step}")

        if self.tool_plan:
            print(f"  Tool plan:")
            for step in self.tool_plan:
                print(f"    → {step}")


# ─── Query Router ─────────────────────────────────────────────────────────────

class QueryRouter:
    """
    Routes incoming queries to the appropriate handling strategy
    based on detected LLM limitation signals.
    """

    # Priority order: higher index = override lower
    ROUTE_PRIORITY = {
        "parametric":   0,
        "static-rag":   1,
        "realtime-rag": 2,
        "tool":         2,
        "hybrid":       3,
        "escalate":     4,
    }

    def route(self, query: str) -> RoutingDecision:
        """
        Analyze a query and return a routing decision.
        """
        q_low   = query.lower()
        signals = self._detect_signals(q_low, query)
        route   = self._decide_route(signals)
        plan    = self._build_plan(query, signals, route)

        return RoutingDecision(
            query          = query,
            signals        = signals,
            primary_route  = route,
            retrieval_plan = plan["retrieval"],
            tool_plan      = plan["tools"],
            confidence     = plan["confidence"],
            explanation    = plan["explanation"],
        )

    def _detect_signals(self, q_low: str, original: str) -> list[LimitationSignal]:
        """
        Scan the query for limitation signals using regex patterns.
        Returns one signal per matched pattern.
        """
        found = []
        for category, config in LIMITATION_SIGNALS.items():
            for pattern in config["patterns"]:
                match = re.search(pattern, q_low, re.IGNORECASE)
                if match:
                    found.append(LimitationSignal(
                        category    = category,
                        pattern     = pattern,
                        match_text  = match.group(0),
                        description = config["description"],
                    ))
                    break   # WHY break: one signal per category is enough
        return found

    def _decide_route(self, signals: list[LimitationSignal]) -> str:
        """
        Choose the primary route based on detected signals.
        Multiple signals → combine routes (hybrid).
        """
        routes = {sig.category: LIMITATION_SIGNALS[sig.category]["route"]
                  for sig in signals}

        if not routes:
            return "parametric"

        unique_routes = set(routes.values())

        # If only tool signals, route to tool
        if unique_routes == {"tool"}:
            return "tool"

        # If both static-rag and realtime-rag needed, it's hybrid
        if "static-rag" in unique_routes and "realtime-rag" in unique_routes:
            return "hybrid"

        # If any realtime needed
        if "realtime-rag" in unique_routes:
            return "realtime-rag" if "tool" not in unique_routes else "hybrid"

        # If only static-rag (or + tool), still route to static-rag
        if "static-rag" in unique_routes:
            return "static-rag"

        return "parametric"

    def _build_plan(self, query: str, signals: list[LimitationSignal], route: str) -> dict:
        """
        Build a specific retrieval and tool execution plan.
        """
        retrieval = []
        tools     = []
        q_low     = query.lower()

        cats = {s.category for s in signals}

        # Retrieval steps
        if route in ("static-rag", "hybrid"):
            if "private_data" in cats or "our" in q_low or "internal" in q_low:
                retrieval.append("Search: private knowledge base (runbooks, configs, incidents)")
            if "knowledge_cutoff" in cats:
                retrieval.append("Search: public product documentation (with date >= 2024-01-01 filter)")
            if not retrieval:
                retrieval.append("Search: primary knowledge base (keyword + semantic hybrid)")

        if route in ("realtime-rag", "hybrid"):
            if any(w in q_low for w in ["fault", "alert", "alarm", "active"]):
                retrieval.append("Live API: APIC /api/node/class/faultSummary")
            if any(w in q_low for w in ["up", "down", "status", "reachable"]):
                retrieval.append("Live API: APIC /api/node/class/topSystem (node health)")
            if any(w in q_low for w in ["ticket", "incident"]):
                retrieval.append("Live API: ServiceNow incident query")
            if not [r for r in retrieval if "Live API" in r]:
                retrieval.append("Live API: relevant system data feed")

        # Tool steps
        if "arithmetic" in cats:
            tools.append("Code tool: Python arithmetic computation")
        if "negation_filter" in cats:
            tools.append("Code tool: Python set difference for exclusion filter")
        if "constraint_filter" in cats:
            tools.append("Code tool or DB: Multi-constraint filter query")

        # Confidence
        if len(signals) == 0:
            confidence  = "high"
            explanation = "No limitation signals — parametric memory likely sufficient."
        elif len(signals) == 1:
            confidence  = "high"
            explanation = f"Single signal detected ({signals[0].category}). Routing is clear."
        elif len(signals) <= 3:
            confidence  = "medium"
            explanation = f"Multiple signals ({', '.join(cats)}). Plan covers all detected needs."
        else:
            confidence  = "low"
            explanation = f"Complex query with {len(signals)} signals. Human review recommended."

        return {"retrieval": retrieval, "tools": tools, "confidence": confidence, "explanation": explanation}


# ─── Demo ─────────────────────────────────────────────────────────────────────

def run_router_demo():
    """
    Route a set of representative enterprise queries.
    """

    print("=" * 72)
    print("QUERY ROUTER: Limitation-Aware Routing for RAG Systems")
    print("=" * 72)

    router  = QueryRouter()

    queries = [
        # General knowledge — parametric ok
        "What is the BGP route reflection mechanism?",

        # Knowledge cutoff — need static RAG
        "What is the latest Cisco ACI version and its key features?",

        # Private data — need private KB
        "What does our change management SLA specify for P1 incidents?",

        # Real-time — need live API
        "Are there any active faults on leaf-101 right now?",

        # Arithmetic — need code tool
        "If we have 24 ports at 40G and 36 ports at 100G, what is the total bandwidth in Tbps?",

        # Negation filter — need code tool
        "Which devices running APIC 5.2(1g) are NOT yet patched for CSCvh23456?",

        # Multi-signal hybrid
        "What active incidents are we tracking for the ACI 6.0 fabric, and what does our P1 runbook say about escalation?",

        # Constraint filter
        "Which switch models support VXLAN and 100G uplinks and cost less than $40K?",

        # CVE — cutoff + potentially real-time
        "Is our current APIC version affected by CVE-2025-12345?",
    ]

    for q in queries:
        decision = router.route(q)
        decision.display()


# ─── Routing Decision Matrix ──────────────────────────────────────────────────

def routing_decision_matrix():
    """
    Print a summary decision matrix for quick reference.
    """

    print("\n" + "=" * 72)
    print("ROUTING DECISION MATRIX: Quick Reference")
    print("=" * 72)

    print(f"""
  Signal detected          → Route to
  ─────────────────────────────────────────────────────────────────
  "latest", "current",     → static-rag (with date filter)
  version number, CVE ID

  "our", "internal",       → static-rag (private knowledge base)
  "runbook", "SLA"

  "active fault", "status  → realtime-rag (live system API)
  right now", "live"

  "calculate", arithmetic  → tool (Python code interpreter)
  expression

  "NOT affected",          → tool (set difference in code)
  "excluding", "except"

  Multiple signals         → hybrid (retrieval + tool as needed)

  None of the above        → parametric (model answers from memory)
                             (add optional RAG for high-stakes domains)

  ─────────────────────────────────────────────────────────────────
  ESCALATION TRIGGERS (no automated answer):
    - Query about specific customer data and KB is empty
    - CVE query with no indexed PSIRT data
    - Operational action (e.g., "delete the EPG") — require human approval
    - Conflicting information in multiple retrieved chunks
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_router_demo()
    routing_decision_matrix()
