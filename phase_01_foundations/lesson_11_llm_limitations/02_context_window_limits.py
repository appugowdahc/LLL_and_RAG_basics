"""
FILE: 02_context_window_limits.py
LESSON: Phase 1 - Lesson 11 - LLM Limitations
TOPIC: Context window — capacity, cost, attention degradation, and practical limits

WHAT THIS FILE TEACHES:
  - Context window sizes for current models
  - Why large context ≠ equal attention (Lost in the Middle revisited quantitatively)
  - Token cost as a function of context size
  - How to compute effective usable context for a RAG system
  - Why 3-5 focused chunks beats 50 noisy ones
  - WHY context window is a budget, not a free resource

INSTALL: no external dependencies (tiktoken optional for accurate counts)
"""

import math
from dataclasses import dataclass
from typing import Optional


# ─── Model Context Windows and Pricing ───────────────────────────────────────

@dataclass
class ModelSpec:
    """Specification for a single LLM."""
    name:              str
    context_tokens:    int           # total context window
    input_price_per_M: float         # $ per 1M input tokens
    output_price_per_M: float        # $ per 1M output tokens
    cache_read_per_M:  Optional[float] = None   # prompt cache read discount


MODEL_SPECS = [
    ModelSpec("claude-sonnet-4-6",     200_000, 3.00,   15.00,  0.30),
    ModelSpec("claude-haiku-4-5",      200_000, 0.80,    4.00,  0.08),
    ModelSpec("claude-opus-4-8",       200_000, 15.00,  75.00,  1.50),
    ModelSpec("gpt-4o",                128_000, 2.50,   10.00,  1.25),
    ModelSpec("gpt-4o-mini",           128_000, 0.15,    0.60,  0.075),
    ModelSpec("gemini-1.5-pro",      1_000_000, 1.25,    5.00,  None),
    ModelSpec("gemini-1.5-flash",    1_000_000, 0.075,  0.30,  None),
]


def context_cost_table():
    """
    Print a table of context window cost at various fill levels.
    WHY this matters: context is not free. A 200K context at $3/M = $0.60 per call.
    At 1000 queries/day that's $600/day for input tokens alone.
    """

    print("=" * 72)
    print("CONTEXT WINDOW COST: Why Context is a Budget, Not a Free Resource")
    print("=" * 72)

    fill_levels = [1_000, 10_000, 50_000, 100_000, 200_000]

    for spec in MODEL_SPECS:
        print(f"\n  Model: {spec.name} ({spec.context_tokens:,} token window)")
        print(f"  {'Context fill':>15} {'$/call':>10} {'$/day @1K queries':>20}")
        print(f"  {'─'*15} {'─'*10} {'─'*20}")

        for fill in fill_levels:
            if fill > spec.context_tokens:
                continue
            cost_per_call = fill / 1_000_000 * spec.input_price_per_M
            cost_per_day  = cost_per_call * 1_000
            print(f"  {fill:>15,} {cost_per_call:>10.4f} {cost_per_day:>20.2f}")

    print(f"""
  INSIGHT:
    Claude Sonnet at 200K context = $0.60 per call.
    At 1,000 queries/day = $600/day = $18,000/month just for input tokens.
    RAG reduces this by sending only the relevant 3-5 chunks (~3,000 tokens)
    instead of the entire knowledge base.
    Savings: 200K → 3K tokens = 98.5% reduction in input cost.
""")


# ─── Attention Degradation Model ─────────────────────────────────────────────

def attention_by_position(position_fraction: float) -> float:
    """
    Approximate attention weight as a function of relative position in context.
    Based on Liu et al. (2023) "Lost in the Middle" findings.

    position_fraction: 0.0 = start of context, 1.0 = end of context.

    The paper showed:
      - Strong primacy effect (start of context has high recall).
      - Strong recency effect (end of context has high recall).
      - "U-shaped" degradation — middle content is least attended.
      - At 30 documents, items in position 6-20 have 55-65% recall vs 80%+ at edges.

    WHY this matters for RAG:
      If you put 20 chunks in the context, the most relevant chunk
      should be at position 0 or position 19, not position 10.
    """
    # WHY U-shape formula: peak at 0.0 and 1.0, trough at 0.5
    # A simple approximation: score = base + amplitude * cos(2π * position)
    # where base=0.60, amplitude=0.20 gives ~0.80 at edges and ~0.40 in middle.
    base      = 0.60
    amplitude = 0.20
    return base + amplitude * math.cos(2 * math.pi * position_fraction)


def attention_degradation_demo():
    """
    Visualize attention degradation and compute expected recall for
    different chunk counts and orderings.
    """

    print("=" * 72)
    print("ATTENTION DEGRADATION: Lost in the Middle Effect")
    print("=" * 72)

    print("\n  Attention weight by position in context (approximate):")
    print(f"\n  {'Position':<12} {'Attention':>10}  {'Bar'}")
    print(f"  {'─'*12} {'─'*10}  {'─'*30}")

    positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for p in positions:
        a   = attention_by_position(p)
        bar = "█" * int(a * 30)
        label = "START" if p == 0.0 else ("END" if p == 1.0 else "middle" if abs(p - 0.5) < 0.05 else "")
        print(f"  {p:>8.0%}      {a:>10.3f}  {bar}  {label}")

    # Compute expected recall for different top-K retrieval strategies
    print(f"\n  Expected recall vs number of retrieved chunks (if answer is in random position):")
    print(f"\n  {'Chunks retrieved':<20} {'Expected recall':>16} {'Recommendation'}")
    print(f"  {'─'*20} {'─'*16} {'─'*35}")

    for k in [1, 3, 5, 10, 20, 50]:
        # Expected attention = average over all positions if answer at random position
        positions_k = [i / max(k - 1, 1) for i in range(k)]
        avg_attention = sum(attention_by_position(p) for p in positions_k) / k

        if k <= 3:
            rec = "Ideal for high-precision use cases"
        elif k <= 5:
            rec = "Good balance (recommended default)"
        elif k <= 10:
            rec = "Use only when recall is critical"
        else:
            rec = "Avoid — attention dilution is severe"

        print(f"  {k:<20} {avg_attention:>16.3f}  {rec}")

    print(f"""
  PRACTICAL RULE:
    With a 200K token window you CAN fit 500 chunks.
    You SHOULD NOT. Use top-5 to top-10.
    Better: use a two-stage approach:
      Stage 1: Retrieve top-20 (high recall).
      Stage 2: Re-rank top-20, keep only top-5 (high precision).
    This is the "retrieve-then-rerank" pattern (Lesson 13).
""")


# ─── Effective Context Budget ─────────────────────────────────────────────────

@dataclass
class ContextBudget:
    """
    Breakdown of how context tokens are allocated in a RAG call.
    WHY this dataclass: makes token budget decisions explicit and auditable.
    """
    model:              str
    total_window:       int
    system_tokens:      int    # fixed cost: system prompt
    history_tokens:     int    # variable: conversation history
    query_tokens:       int    # per-query cost
    chunk_tokens_each:  int    # average tokens per retrieved chunk
    output_reserve:     int    # reserved for the model's answer

    @property
    def fixed_overhead(self) -> int:
        """Tokens consumed before any chunks are added."""
        return self.system_tokens + self.history_tokens + self.query_tokens + self.output_reserve

    @property
    def available_for_chunks(self) -> int:
        """Remaining tokens after overhead."""
        return self.total_window - self.fixed_overhead

    @property
    def max_chunks(self) -> int:
        """Maximum chunks that fit within budget."""
        return max(0, self.available_for_chunks // self.chunk_tokens_each)

    def display(self):
        total = self.total_window
        bar_scale = total / 40   # 40-char bar

        print(f"  Model:               {self.model}")
        print(f"  Total window:        {total:,} tokens")
        print(f"  ─── Budget breakdown ───────────────────────────────────")

        components = [
            ("System prompt",    self.system_tokens,    "▓"),
            ("History",          self.history_tokens,   "▒"),
            ("Query",            self.query_tokens,     "░"),
            ("Output reserve",   self.output_reserve,   "·"),
            ("Chunks (avail.)",  self.available_for_chunks, "█"),
        ]

        for name, tokens, char in components:
            bar_len = max(1, int(tokens / bar_scale))
            bar     = char * bar_len
            pct     = tokens / total * 100
            print(f"  {name:<20} {tokens:>8,}  ({pct:>4.1f}%)  {bar}")

        print(f"  ─────────────────────────────────────────────────────────")
        print(f"  Max chunks (@ {self.chunk_tokens_each} tok each): {self.max_chunks}")
        print(f"  Recommended usage: top-{min(self.max_chunks, 5)} (attention quality)")


def context_budget_demo():
    """
    Show how the context budget breaks down for different RAG configurations.
    """

    print("=" * 72)
    print("EFFECTIVE CONTEXT BUDGET: What's Left for Chunks After Overhead")
    print("=" * 72)

    configs = [
        ContextBudget(
            model             = "claude-sonnet-4-6 (minimal RAG)",
            total_window      = 200_000,
            system_tokens     = 300,
            history_tokens    = 0,
            query_tokens      = 50,
            chunk_tokens_each = 400,
            output_reserve    = 1_000,
        ),
        ContextBudget(
            model             = "claude-sonnet-4-6 (multi-turn RAG)",
            total_window      = 200_000,
            system_tokens     = 500,
            history_tokens    = 8_000,
            query_tokens      = 100,
            chunk_tokens_each = 400,
            output_reserve    = 2_000,
        ),
        ContextBudget(
            model             = "gpt-4o (128K window)",
            total_window      = 128_000,
            system_tokens     = 500,
            history_tokens    = 4_000,
            query_tokens      = 80,
            chunk_tokens_each = 400,
            output_reserve    = 1_500,
        ),
    ]

    for cfg in configs:
        print()
        cfg.display()

    print(f"""
  DESIGN RULE:
    Never fill the entire context window with retrieved chunks.
    Reserve headroom for:
      - Conversation history growth (multi-turn)
      - Longer-than-average queries
      - Safety margin (overlong chunks, prompt templates)

    Target: use no more than 70% of the token budget for chunks.
    This leaves room for history without context overflow mid-conversation.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    context_cost_table()
    print()
    attention_degradation_demo()
    print()
    context_budget_demo()
