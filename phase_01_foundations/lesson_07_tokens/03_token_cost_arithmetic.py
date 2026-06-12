"""
FILE: 03_token_cost_arithmetic.py
LESSON: Phase 1 - Lesson 7 - Tokens Deep-Dive
TOPIC: Token cost arithmetic — calculate, project, and optimize API costs

WHAT THIS FILE TEACHES:
  - The exact cost formula for every Claude model
  - How input vs output token costs differ (output is 5× more expensive)
  - Prompt Caching: when it saves money, when it doesn't
  - Monthly cost projections for different RAG architectures
  - The break-even analysis: when to add compression or caching
  - Model routing: when Haiku pays for itself vs Sonnet

WHY TOKEN COSTS MATTER NOW (not later):
  A poorly designed RAG system that reaches 10,000 queries/day
  can cost $500-$2,000/month more than an optimized one.
  Understanding costs at query design time is 10× cheaper than
  optimizing retroactively on a deployed system.

INSTALL:
  pip install anthropic python-dotenv
"""

from dataclasses import dataclass
import math


# ─── Pricing Table (per million tokens) ──────────────────────────────────────

# WHY a dict of dicts:
#   Single source of truth. When Anthropic updates pricing,
#   update once here — all calculations update automatically.
#   In production: load this from a config file or environment variables.

PRICING = {
    "claude-haiku-4-5-20251001": {
        "label":       "Haiku 4.5",
        "input":        0.80,
        "output":       4.00,
        "cache_write":  1.00,
        "cache_read":   0.08,
        "context_k":    200,   # context window in K tokens
    },
    "claude-sonnet-4-6": {
        "label":       "Sonnet 4.6",
        "input":        3.00,
        "output":      15.00,
        "cache_write":  3.75,
        "cache_read":   0.30,
        "context_k":    200,
    },
    "claude-opus-4-8": {
        "label":       "Opus 4.8",
        "input":       15.00,
        "output":      75.00,
        "cache_write": 18.75,
        "cache_read":   1.50,
        "context_k":    200,
    },
}

MIN_CACHE_TOKENS = 1024   # WHY 1024: Anthropic's minimum for a cache block to activate


# ─── Core Cost Functions ──────────────────────────────────────────────────────

def cost_per_query(
    model:                str,
    input_tokens:         int,
    output_tokens:        int,
    cache_write_tokens:   int = 0,
    cache_read_tokens:    int = 0,
) -> float:
    """
    Calculate USD cost for a single API call.

    Token types:
      - Regular input:   tokens the model processes for the first time
      - Cache write:     tokens written to the prompt cache (one-time)
      - Cache read:      tokens served from prompt cache (per call benefit)
      - Output:          tokens the model generates

    WHY output costs 5× input:
      Output generation requires autoregressive decoding: one forward pass
      per output token. Input tokens are processed in ONE forward pass (parallel).
      Generating 100 tokens = 100 serial forward passes.
      Processing 100 input tokens = 1 parallel pass.

    Args:
        model:              Model ID string.
        input_tokens:       Regular (non-cached) input tokens.
        output_tokens:      Generated output tokens.
        cache_write_tokens: Tokens stored in prompt cache this call.
        cache_read_tokens:  Tokens served from cache this call.
    """

    p  = PRICING[model]
    M  = 1_000_000   # per-million divisor

    # WHY (input - cache_write - cache_read) for regular input:
    #   cache_write and cache_read tokens are billed separately.
    #   They must NOT also be billed as regular input.
    regular_input = max(0, input_tokens - cache_write_tokens - cache_read_tokens)

    return (
        regular_input       / M * p["input"]
        + output_tokens     / M * p["output"]
        + cache_write_tokens / M * p["cache_write"]
        + cache_read_tokens  / M * p["cache_read"]
    )


def monthly_cost(
    model:               str,
    queries_per_day:     int,
    input_tokens:        int,
    output_tokens:       int,
    cache_write_tokens:  int = 0,
    cache_read_tokens:   int = 0,
) -> dict:
    """
    Project monthly cost for a RAG system at a given query volume.

    Args:
        queries_per_day:  Average queries per day (assume uniform load).
        input/output tokens: Per-query token counts.
        cache_*_tokens:   Per-query cache token counts.

    Returns:
        Dict with daily, monthly, and per-1K-query costs.
    """

    daily = queries_per_day * cost_per_query(
        model, input_tokens, output_tokens,
        cache_write_tokens, cache_read_tokens
    )

    return {
        "model":            PRICING[model]["label"],
        "queries_per_day":  queries_per_day,
        "cost_per_query":   cost_per_query(model, input_tokens, output_tokens,
                                           cache_write_tokens, cache_read_tokens),
        "daily_cost":       daily,
        "monthly_cost":     daily * 30,
        "per_1k_queries":   cost_per_query(model, input_tokens, output_tokens,
                                           cache_write_tokens, cache_read_tokens) * 1_000,
    }


# ─── Scenario 1: RAG Architecture Comparison ─────────────────────────────────

def compare_rag_architectures():
    """
    Compare total monthly costs across different RAG system designs.
    Shows WHY architectural choices (chunk count, compression, caching)
    have large financial impact at scale.
    """

    MODEL         = "claude-sonnet-4-6"
    QUERIES_DAY   = 5_000    # medium enterprise load
    OUTPUT_TOKS   = 400      # typical detailed response

    # WHY specific token counts for each architecture:
    #   These represent realistic production RAG prompt sizes
    architectures = [
        {
            "name":              "Naive RAG (10 chunks, no cache)",
            "input_tokens":      15_000,   # system(400) + 10×chunks(1,200 ea) + hist(200) + query(200)
            "cache_write":       0,
            "cache_read":        0,
        },
        {
            "name":              "Optimized RAG (5 chunks, no cache)",
            "input_tokens":      7_000,    # fewer, better-ranked chunks
            "cache_write":       0,
            "cache_read":        0,
        },
        {
            "name":              "Cached System Prompt (10 chunks)",
            "input_tokens":      15_000,
            "cache_write":       400,      # system prompt cached on first call
            "cache_read":        400,      # all subsequent calls read from cache
        },
        {
            "name":              "Compressed Chunks (10→5 chunks via LLM)",
            "input_tokens":      7_000,    # compression halved doc tokens
            "cache_write":       400,
            "cache_read":        400,
        },
        {
            "name":              "Full Optimization (compressed + cached)",
            "input_tokens":      5_500,    # minimal prompt
            "cache_write":       400,
            "cache_read":        400,
        },
    ]

    print("=" * 70)
    print(f"RAG ARCHITECTURE COST COMPARISON  —  {QUERIES_DAY:,} queries/day  —  {MODEL}")
    print("=" * 70)
    print(f"\n  {'Architecture':<45} {'$/query':>9} {'$/day':>8} {'$/month':>10}")
    print(f"  {'─'*45} {'─'*9} {'─'*8} {'─'*10}")

    baseline_monthly = None

    for arch in architectures:
        result = monthly_cost(
            model              = MODEL,
            queries_per_day    = QUERIES_DAY,
            input_tokens       = arch["input_tokens"],
            output_tokens      = OUTPUT_TOKS,
            cache_write_tokens = arch["cache_write"],
            cache_read_tokens  = arch["cache_read"],
        )

        if baseline_monthly is None:
            baseline_monthly = result["monthly_cost"]

        savings = baseline_monthly - result["monthly_cost"]
        savings_str = f"  save ${savings:,.0f}/mo" if savings > 0 else ""

        print(
            f"  {arch['name']:<45} "
            f"${result['cost_per_query']:>7.4f} "
            f"${result['daily_cost']:>6.2f} "
            f"${result['monthly_cost']:>8.2f}"
            f"{savings_str}"
        )

    print(f"\n  OUTPUT: {OUTPUT_TOKS} tokens × ${PRICING[MODEL]['output']}/1M = "
          f"${OUTPUT_TOKS/1_000_000*PRICING[MODEL]['output']:.4f}/query output cost")


# ─── Scenario 2: Model Routing Break-Even ────────────────────────────────────

def model_routing_analysis():
    """
    Calculate when routing simple queries to Haiku instead of Sonnet pays off.

    KEY INSIGHT:
      If a Haiku classifier call costs less than the Sonnet savings it enables,
      routing is financially justified even if Haiku is sometimes wrong.
    """

    print("\n" + "=" * 70)
    print("MODEL ROUTING: When does Haiku classification pay for itself?")
    print("=" * 70)

    # Haiku as a classifier: tiny input+output
    CLASSIFIER_INPUT  = 200   # the query text
    CLASSIFIER_OUTPUT =  20   # "simple" or "complex" label

    # The actual task
    TASK_INPUT_TOKS  = 8_000
    TASK_OUTPUT_TOKS = 400

    haiku_task_cost  = cost_per_query("claude-haiku-4-5-20251001",  TASK_INPUT_TOKS, TASK_OUTPUT_TOKS)
    sonnet_task_cost = cost_per_query("claude-sonnet-4-6",          TASK_INPUT_TOKS, TASK_OUTPUT_TOKS)

    haiku_classify_cost = cost_per_query("claude-haiku-4-5-20251001", CLASSIFIER_INPUT, CLASSIFIER_OUTPUT)

    savings_per_routed_query = sonnet_task_cost - haiku_task_cost - haiku_classify_cost

    print(f"\n  Per-query cost breakdown:")
    print(f"    Haiku task cost:       ${haiku_task_cost:.5f}")
    print(f"    Sonnet task cost:      ${sonnet_task_cost:.5f}")
    print(f"    Haiku classify cost:   ${haiku_classify_cost:.5f}")
    print(f"    Savings per routing:   ${savings_per_routed_query:.5f}  "
          f"({'✓ routing saves money' if savings_per_routed_query > 0 else '✗ routing costs more'})")

    print(f"\n  AT SCALE (10,000 queries/day, 60% routable to Haiku):")
    routable_per_day = 10_000 * 0.60
    daily_savings    = routable_per_day * savings_per_routed_query
    monthly_savings  = daily_savings * 30

    print(f"    Routable queries/day:  {routable_per_day:,.0f}")
    print(f"    Daily savings:         ${daily_savings:,.2f}")
    print(f"    Monthly savings:       ${monthly_savings:,.2f}")


# ─── Scenario 3: Prompt Cache Break-Even ─────────────────────────────────────

def cache_break_even_analysis():
    """
    Calculate when prompt caching saves money.

    CRITICAL RULE:
      Caching a block costs cache_write on the FIRST call.
      Every SUBSEQUENT call within 5 minutes reads at cache_read rate.
      Break-even: how many calls does it take to recover the write premium?

    Anthropic cache_write = input × 1.25 (25% premium).
    cache_read  = input × 0.10 (90% discount).
    Break-even  = after just 2 reads, you're already saving money.
    """

    print("\n" + "=" * 70)
    print("PROMPT CACHE BREAK-EVEN ANALYSIS")
    print("=" * 70)

    MODEL = "claude-sonnet-4-6"
    p     = PRICING[MODEL]
    M     = 1_000_000

    cache_block_sizes = [
        ("System prompt",   500),
        ("System + docs",   5_000),
        ("Full base prompt", 20_000),
        ("Large doc block",  50_000),
    ]

    for name, block_tokens in cache_block_sizes:
        if block_tokens < MIN_CACHE_TOKENS:
            print(f"\n  {name} ({block_tokens:,} tok): TOO SMALL — min cache block = {MIN_CACHE_TOKENS:,} tokens")
            continue

        write_cost    = block_tokens / M * p["cache_write"]
        read_cost     = block_tokens / M * p["cache_read"]
        regular_cost  = block_tokens / M * p["input"]

        write_premium = write_cost - regular_cost      # Extra cost on first call
        read_savings  = regular_cost - read_cost       # Savings per cached call

        # WHY ceil: need at least this many reads to break even
        if read_savings <= 0:
            break_even_calls = float("inf")
        else:
            break_even_calls = math.ceil(write_premium / read_savings)

        print(f"\n  {name} ({block_tokens:,} tokens):")
        print(f"    Regular input cost:   ${regular_cost:.5f}")
        print(f"    Cache write cost:     ${write_cost:.5f}  (+${write_premium:.5f} premium)")
        print(f"    Cache read cost:      ${read_cost:.5f}  (-${read_savings:.5f} savings)")
        print(f"    Break-even at:        {break_even_calls} reads after the write")
        print(f"    → {'ALWAYS USE CACHE' if break_even_calls <= 2 else 'Cache if >'+str(break_even_calls)+' reads expected'}")


# ─── Scenario 4: Input vs Output Cost Ratio ──────────────────────────────────

def input_output_ratio():
    """
    Show WHY concise output is as important as efficient input.
    Many developers focus only on input token count, ignoring that
    output tokens cost 5× more per token.
    """

    print("\n" + "=" * 70)
    print("INPUT vs OUTPUT COST: Why verbose responses are expensive")
    print("=" * 70)

    MODEL = "claude-sonnet-4-6"
    p     = PRICING[MODEL]
    M     = 1_000_000

    INPUT_TOKS = 8_000   # fixed

    output_sizes = [
        ("One-liner answer",      50),
        ("Brief paragraph",      150),
        ("Typical response",     400),
        ("Detailed response",    800),
        ("Full report",        2_000),
        ("Exhaustive analysis", 8_000),
    ]

    print(f"\n  Fixed input: {INPUT_TOKS:,} tokens  |  Model: {MODEL}")
    print(f"\n  {'Output type':<25} {'Output toks':>12} {'Input $':>9} {'Output $':>9} {'Total $':>9} {'Output%':>8}")
    print(f"  {'─'*25} {'─'*12} {'─'*9} {'─'*9} {'─'*9} {'─'*8}")

    for name, output_toks in output_sizes:
        input_cost  = INPUT_TOKS    / M * p["input"]
        output_cost = output_toks   / M * p["output"]
        total       = input_cost + output_cost
        output_pct  = output_cost / total * 100

        print(
            f"  {name:<25} {output_toks:>12,} "
            f"${input_cost:>7.5f} ${output_cost:>7.5f} ${total:>7.5f} "
            f"{output_pct:>7.1f}%"
        )

    print(f"""
  KEY TAKEAWAY:
    At 400 output tokens, output cost = {400/M*p["output"]/(INPUT_TOKS/M*p["input"]+400/M*p["output"])*100:.0f}% of total cost.
    At 2,000 output tokens, output cost dominates.

  PRODUCTION RULE:
    Always set max_tokens to the minimum your use case needs.
    "Summarize in 2 sentences" is 5× cheaper than "Explain in detail."
    In RAG: direct factual answers (short) >> exhaustive explanations (expensive).
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 70)
    print("TOKEN COST ARITHMETIC: Plan your RAG budget before you build")
    print("=" * 70)

    compare_rag_architectures()
    model_routing_analysis()
    cache_break_even_analysis()
    input_output_ratio()
