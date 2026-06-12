"""
FILE: 04_window_filling_strategies.py
LESSON: Phase 1 - Lesson 6 - Context Window
TOPIC: Context window filling strategies — how to allocate the token budget
       across retrieved chunks, history, and other components.

WHAT THIS FILE TEACHES:
  - Four production strategies for filling the context window:
    1. Fixed-K (simple, predictable)
    2. Dynamic token-budget (efficient, variable)
    3. Score-threshold (quality-gated, may return 0 chunks)
    4. Hierarchical (multi-tier budget isolation)
  - How to implement each strategy in code
  - Tradeoffs: cost, quality, predictability, edge case safety
  - Which strategy to use when

INSTALL:
  pip install anthropic python-dotenv tiktoken
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens_local(text: str) -> int:
        return len(_enc.encode(text))
    HAS_TIKTOKEN = True
except ImportError:
    def count_tokens_local(text: str) -> int:
        return int(len(text.split()) / 0.75)
    HAS_TIKTOKEN = False


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A retrieved document chunk from a vector database.
    """
    chunk_id:  str
    source:    str
    content:   str
    score:     float    # similarity score from vector search (0-1)
    tier:      str = "general"  # for hierarchical: "core" | "supporting" | "general"

    @property
    def token_count(self) -> int:
        return count_tokens_local(self.content)

    def to_context_string(self) -> str:
        """Format chunk for insertion into RAG prompt."""
        return (
            f"[{self.chunk_id}] Source: {self.source}  |  Score: {self.score:.2f}\n"
            f"{self.content}"
        )


@dataclass
class FillingResult:
    """
    Result of a context window filling operation.
    Tracks what was included and what was excluded.
    """
    selected_chunks:  list[Chunk]
    excluded_chunks:  list[Chunk]
    total_tokens:     int
    budget_used:      int           # token budget allocated for docs
    strategy_name:    str
    notes:            list[str] = field(default_factory=list)

    @property
    def utilization_pct(self) -> float:
        return self.total_tokens / max(self.budget_used, 1) * 100

    @property
    def avg_score(self) -> float:
        if not self.selected_chunks:
            return 0.0
        return sum(c.score for c in self.selected_chunks) / len(self.selected_chunks)

    def display(self):
        print(f"\n  Strategy: {self.strategy_name}")
        print(f"  Selected: {len(self.selected_chunks)} chunks  |  "
              f"Excluded: {len(self.excluded_chunks)} chunks")
        print(f"  Tokens: {self.total_tokens:,} / {self.budget_used:,} "
              f"({self.utilization_pct:.1f}% of doc budget)")
        print(f"  Avg relevance score: {self.avg_score:.3f}")

        if self.selected_chunks:
            print(f"  {'ID':<12} {'Score':>6} {'Tokens':>7} {'Source'}")
            for c in self.selected_chunks:
                print(f"    {c.chunk_id:<10} {c.score:>6.3f} {c.token_count:>6}  {c.source[:35]}")

        if self.excluded_chunks:
            excluded_ids = [c.chunk_id for c in self.excluded_chunks]
            print(f"  Excluded: {excluded_ids}")

        for note in self.notes:
            print(f"  ⚠ {note}")


# ─── Strategy 1: Fixed-K ──────────────────────────────────────────────────────

def strategy_fixed_k(
    chunks:       list[Chunk],
    k:            int = 5,
    order:        str = "relevance",
) -> FillingResult:
    """
    Always select exactly K chunks (or fewer if fewer exist).
    Order by relevance score before selecting.

    WHY to use this:
      - Predictable cost: cost per query is always the same.
      - Simple to reason about and debug.
      - Good starting point when building a new RAG system.

    WHY NOT to use this for production:
      - Simple queries might only need 1 chunk → wastes 4 chunks worth of tokens.
      - Complex queries might need 10 chunks → K=5 may miss critical context.
      - Same token cost whether the query is easy or hard.

    Args:
        chunks:  All retrieved chunks, any order.
        k:       Number of chunks to select.
        order:   How to sort before selecting: "relevance" or "recency"

    Returns:
        FillingResult with exactly min(k, len(chunks)) chunks.
    """

    if order == "relevance":
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
    else:
        sorted_chunks = chunks

    selected = sorted_chunks[:k]
    excluded = sorted_chunks[k:]

    total_tokens = sum(c.token_count for c in selected)
    # WHY 200 × k for budget estimate:
    #   Fixed-K assumes each chunk is ~200 tokens. This is just for display.
    budget_used  = 200 * k

    result = FillingResult(
        selected_chunks=selected,
        excluded_chunks=excluded,
        total_tokens=total_tokens,
        budget_used=budget_used,
        strategy_name=f"Fixed-K (k={k}, order={order})",
    )

    if len(chunks) < k:
        result.notes.append(
            f"Only {len(chunks)} chunks available — could not fill K={k}"
        )

    return result


# ─── Strategy 2: Dynamic Token Budget ────────────────────────────────────────

def strategy_dynamic_budget(
    chunks:          list[Chunk],
    token_budget:    int = 60_000,
    min_score:       float = 0.0,
    ordering:        str = "lost_in_middle",
) -> FillingResult:
    """
    Fill up to token_budget tokens with the highest-scoring chunks.
    Stop adding chunks when the next chunk would overflow the budget.

    WHY to use this:
      - Uses the full available budget when needed.
      - Short queries use fewer tokens automatically (cost scales with complexity).
      - Naturally accommodates variable-size chunks.

    WHY it can be tricky:
      - Response cost varies per query (harder to budget).
      - A very long chunk might be excluded even though shorter ones fit.
        (Consider sorting chunks by score THEN token_count for edge cases.)

    Args:
        chunks:       All retrieved chunks.
        token_budget: Max tokens to use for retrieved content.
        min_score:    Skip chunks below this score (quality gate).
        ordering:     "relevance" | "lost_in_middle" — final ordering of selected chunks.

    Returns:
        FillingResult with as many chunks as fit within budget.
    """

    # Sort by score descending to greedily pick the best-scoring chunks first
    candidates = sorted(chunks, key=lambda c: c.score, reverse=True)

    selected    = []
    total_toks  = 0

    for chunk in candidates:
        if chunk.score < min_score:
            continue  # skip below quality gate

        chunk_toks = chunk.token_count

        if total_toks + chunk_toks > token_budget:
            # Skip this chunk — would overflow budget
            # NOTE: Don't break here — a SMALLER future chunk might still fit
            continue

        selected.append(chunk)
        total_toks += chunk_toks

    # Apply final ordering
    if ordering == "lost_in_middle":
        # Best chunks at START and END — mitigation for Lost in Middle problem
        n = len(selected)
        top_half    = selected[:(n + 1) // 2]
        bottom_half = list(reversed(selected[(n + 1) // 2:]))
        selected    = top_half + bottom_half
    # else: keep relevance order (already sorted)

    excluded = [c for c in candidates if c not in selected]

    return FillingResult(
        selected_chunks=selected,
        excluded_chunks=excluded,
        total_tokens=total_toks,
        budget_used=token_budget,
        strategy_name=f"Dynamic Budget ({token_budget:,} token budget, ordering={ordering})",
    )


# ─── Strategy 3: Score Threshold ─────────────────────────────────────────────

def strategy_score_threshold(
    chunks:          list[Chunk],
    threshold:       float = 0.70,
    token_budget:    int   = 60_000,
    fallback_top_k:  int   = 1,
) -> FillingResult:
    """
    Only inject chunks with similarity score >= threshold.
    If NO chunks meet threshold, fall back to top-K to avoid empty context.

    WHY to use this:
      - Prevents low-quality chunks from polluting the context.
      - If all retrieved chunks are irrelevant, NOTHING is injected →
        model says "NOT IN CONTEXT" instead of hallucinating from bad chunks.
      - Best for systems where the knowledge base is incomplete (not all topics covered).

    WHY to add a fallback:
      - Without fallback: if threshold is too high, model has ZERO context →
        always answers "NOT IN CONTEXT" even when a borderline chunk would help.
      - Fallback_top_k=1: inject at least the best chunk no matter what.

    Args:
        chunks:         All retrieved chunks.
        threshold:      Minimum score to include a chunk.
        token_budget:   Max tokens for docs.
        fallback_top_k: If 0 chunks pass threshold, use top K instead.

    Returns:
        FillingResult with quality-filtered chunks.
    """

    # Quality gate
    above_threshold = [c for c in chunks if c.score >= threshold]
    notes = []

    if not above_threshold:
        # Fallback: use top-K
        sorted_all    = sorted(chunks, key=lambda c: c.score, reverse=True)
        above_threshold = sorted_all[:fallback_top_k]
        notes.append(
            f"No chunks met threshold={threshold:.2f}. "
            f"Fell back to top-{fallback_top_k} chunk(s) "
            f"(score={above_threshold[0].score:.2f})."
        )

    # Apply dynamic budget within threshold-passing chunks
    result = strategy_dynamic_budget(
        chunks=above_threshold,
        token_budget=token_budget,
    )

    excluded_below = [c for c in chunks if c not in above_threshold]
    result.excluded_chunks.extend(excluded_below)

    result.strategy_name = (
        f"Score Threshold (threshold={threshold:.2f}, fallback_k={fallback_top_k})"
    )
    result.notes.extend(notes)

    return result


# ─── Strategy 4: Hierarchical Budget ─────────────────────────────────────────

def strategy_hierarchical(
    chunks:          list[Chunk],
    tier_budgets:    Optional[dict] = None,
) -> FillingResult:
    """
    Allocate separate token budgets to different chunk tiers.
    Each tier is filled independently — one tier can't steal budget from another.

    WHY to use this:
      - In enterprise RAG, some docs are AUTHORITATIVE (policy docs, specs)
        and others are SUPPORTING (examples, tutorials).
      - Fixed-K and dynamic budget treat all chunks equally.
      - Hierarchical ensures core authoritative content is ALWAYS included,
        even if it fills the budget before supporting chunks can be added.

    Tiers: chunk.tier should be "core", "supporting", or "general".
      core:        Policy documents, official specs → always include first
      supporting:  Examples, tutorials, FAQs → include after core
      general:     Background, context → include if budget remains

    Args:
        chunks:       All retrieved chunks.
        tier_budgets: Dict mapping tier name → token budget.
                      Default: core=30K, supporting=20K, general=10K

    Returns:
        FillingResult with tier-isolated budget allocation.
    """

    if tier_budgets is None:
        tier_budgets = {
            "core":       30_000,
            "supporting": 20_000,
            "general":    10_000,
        }

    selected_all  = []
    excluded_all  = []
    notes         = []

    for tier, budget in tier_budgets.items():
        tier_chunks = [c for c in chunks if c.tier == tier]

        if not tier_chunks:
            notes.append(f"No chunks with tier='{tier}' — budget unused.")
            continue

        # Fill this tier's budget independently
        tier_result  = strategy_dynamic_budget(
            chunks=tier_chunks,
            token_budget=budget,
        )

        selected_all.extend(tier_result.selected_chunks)
        excluded_all.extend(tier_result.excluded_chunks)

    total_budget = sum(tier_budgets.values())
    total_tokens = sum(c.token_count for c in selected_all)

    return FillingResult(
        selected_chunks=selected_all,
        excluded_chunks=excluded_all,
        total_tokens=total_tokens,
        budget_used=total_budget,
        strategy_name=(
            f"Hierarchical (core={tier_budgets.get('core',0)/1000:.0f}K, "
            f"supporting={tier_budgets.get('supporting',0)/1000:.0f}K, "
            f"general={tier_budgets.get('general',0)/1000:.0f}K)"
        ),
        notes=notes,
    )


# ─── Strategy Comparison ─────────────────────────────────────────────────────

def compare_all_strategies(chunks: list[Chunk]):
    """
    Run all four strategies on the same chunk set and compare results.
    """

    strategies = [
        strategy_fixed_k(chunks, k=3),
        strategy_dynamic_budget(chunks, token_budget=600),   # small budget for demo
        strategy_score_threshold(chunks, threshold=0.75),
        strategy_hierarchical(chunks, tier_budgets={"core": 300, "supporting": 200, "general": 100}),
    ]

    print("\n" + "=" * 65)
    print("STRATEGY COMPARISON: Same chunks, 4 strategies")
    print("=" * 65)

    for result in strategies:
        result.display()

    print("\n  RECOMMENDATION MATRIX:")
    print(f"  {'Situation':<45} {'Strategy'}")
    print(f"  {'─'*45} {'─'*20}")
    recs = [
        ("Starting a new RAG system",                      "Fixed-K (k=5)"),
        ("Cost scales with query complexity",               "Dynamic Budget"),
        ("Knowledge base is incomplete",                    "Score Threshold"),
        ("Mix of authoritative + background docs",          "Hierarchical"),
        ("Queries vary from simple to complex",             "Dynamic Budget"),
        ("Compliance / audit requirements",                 "Score Threshold + Verbatim"),
    ]
    for situation, strategy in recs:
        print(f"  {situation:<45} {strategy}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Build a diverse set of sample chunks with different tiers and scores
    sample_chunks = [
        Chunk("doc_001", "aci_policy_spec.pdf, p.4",         "Cisco ACI (Application Centric Infrastructure) uses a policy-driven model. The APIC controller manages the entire fabric. EPGs communicate through contracts.", 0.94, tier="core"),
        Chunk("doc_002", "readyops_spec.pdf, p.1",           "ReadyOps performs continuous validation across Live Operations and Production-Representative environments. Changes require formal promotion with validation gate sign-off.", 0.91, tier="core"),
        Chunk("doc_003", "hypershield_overview.pdf, p.2",    "Cisco Hypershield is an AI-native security architecture using eBPF to enforce policy at the kernel level without dedicated appliances. Supports autonomous segmentation.", 0.87, tier="core"),
        Chunk("doc_004", "aci_deployment_guide.pdf, p.12",   "To deploy ACI: 1) Configure APIC cluster. 2) Discover switches. 3) Configure fabric access policies. 4) Create tenants, VRFs, BDs, EPGs.", 0.80, tier="supporting"),
        Chunk("doc_005", "readyops_agent_classes.pdf, p.5",  "ReadyOps agent classes: Health & Posture monitors ongoing health. Validation runs test suites. Operational executes changes. Stress & Adversarial tests resilience.", 0.77, tier="supporting"),
        Chunk("doc_006", "aci_troubleshooting.pdf, p.23",    "Common ACI issues: APIC unreachable (check OOB management). EPG connectivity failures (check contracts and filters). Spine discovery issues (check ISIS adjacency).", 0.68, tier="supporting"),
        Chunk("doc_007", "cisco_infrastructure_101.pdf",     "Cisco is a leading networking and security vendor. Products span switching, routing, wireless, security, and data center. Cisco partners include VARs, SIs, and MSPs.", 0.42, tier="general"),
        Chunk("doc_008", "sdn_background.pdf, p.1",          "Software-defined networking separates the control plane from the data plane. SDN enables programmable network infrastructure through centralized control.", 0.38, tier="general"),
    ]

    compare_all_strategies(sample_chunks)
