"""
FILE: 06_mini_project_token_budget_planner.py
LESSON: Phase 1 - Lesson 7 - Tokens Deep-Dive
TOPIC: Mini-Project — Full token budget planner for a RAG deployment

WHAT THIS PROJECT BUILDS:
  A self-contained token budget planner that:
    1. Profiles your actual document corpus (measures real token counts by type)
    2. Models context window allocation (system / docs / history / query / output)
    3. Projects monthly API costs at your expected query volume
    4. Shows how Prompt Caching changes the economics
    5. Recommends optimal chunk size for your corpus
    6. Validates a real API call against the budget plan

TIES TOGETHER:
  - Lesson 1:  API token counting
  - Lesson 6:  Context window anatomy, prompt caching
  - Lesson 7:  BPE, content type profiles, cost arithmetic

NO EXTERNAL CORPUS NEEDED:
  The planner works with synthetic samples of different content types.
  In production: replace CORPUS_SAMPLES with your actual document chunks.

INSTALL:
  pip install anthropic python-dotenv tiktoken
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def tok_count(text: str) -> int:
        return len(_enc.encode(text))
    HAS_TIKTOKEN = True
except ImportError:
    def tok_count(text: str) -> int:
        return int(len(text.split()) / 0.75)
    HAS_TIKTOKEN = False


# ─── Pricing (per 1M tokens) ─────────────────────────────────────────────────

PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output":  4.00, "cache_write": 1.00, "cache_read": 0.08},
    "claude-sonnet-4-6":         {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "claude-opus-4-8":           {"input":15.00, "output": 75.00, "cache_write":18.75, "cache_read": 1.50},
}


# ─── Corpus Profile ───────────────────────────────────────────────────────────

@dataclass
class CorpusProfile:
    """
    Describes the token characteristics of a document corpus.
    Measured from representative samples.
    """
    name:             str
    avg_chars_per_doc: float
    avg_tokens_per_doc: float
    avg_chunk_tokens:  float
    chars_per_token:   float    # derived from above
    content_types:     dict     # {"prose": 0.5, "code": 0.3, "yaml": 0.2}
    language:          str = "English"
    language_multiplier: float = 1.0  # 1.0 = same as English; 2.0 = 2× more tokens


def profile_corpus_samples(samples: dict[str, list[str]]) -> CorpusProfile:
    """
    Measure actual token statistics from representative corpus samples.

    Args:
        samples: Dict mapping content_type → list of sample texts.
                 e.g. {"prose": ["...", "..."], "code": ["..."]}

    Returns:
        CorpusProfile with measured averages.
    """

    all_tokens  = []
    all_chars   = []
    type_counts = {}

    for content_type, texts in samples.items():
        type_tokens = [tok_count(t) for t in texts]
        type_chars  = [len(t) for t in texts]

        all_tokens.extend(type_tokens)
        all_chars.extend(type_chars)

        type_counts[content_type] = len(texts)

    if not all_tokens:
        raise ValueError("No samples provided")

    total_samples = len(all_tokens)
    avg_tokens    = sum(all_tokens) / total_samples
    avg_chars     = sum(all_chars)  / total_samples

    # Content type proportions
    content_types = {
        ct: count / total_samples
        for ct, count in type_counts.items()
    }

    return CorpusProfile(
        name              = "Measured Corpus",
        avg_chars_per_doc = avg_chars,
        avg_tokens_per_doc= avg_tokens,
        avg_chunk_tokens  = avg_tokens,
        chars_per_token   = avg_chars / max(avg_tokens, 1),
        content_types     = content_types,
    )


# ─── Budget Plan ──────────────────────────────────────────────────────────────

@dataclass
class RAGBudgetPlan:
    """
    Full token budget plan for a RAG deployment.
    Covers context window allocation and monthly cost projections.
    """

    # Deployment parameters
    model:              str
    context_window:     int   = 200_000
    queries_per_day:    int   = 1_000
    expected_output:    int   = 400

    # Per-query allocations (tokens)
    system_tokens:      int   = 500
    few_shot_tokens:    int   = 0
    doc_budget_tokens:  int   = 60_000
    history_tokens:     int   = 4_000
    query_tokens:       int   = 200
    output_reserve:     int   = 30_000

    # Prompt caching
    use_cache:          bool  = True
    cacheable_tokens:   int   = 0     # tokens in static cacheable block

    # Corpus profile
    avg_chunk_tokens:   int   = 300   # from corpus profile

    def max_chunks(self) -> int:
        """Max chunks that fit in doc_budget at avg_chunk_tokens each."""
        return max(1, self.doc_budget_tokens // self.avg_chunk_tokens)

    def total_input_per_query(self) -> int:
        return (
            self.system_tokens
            + self.few_shot_tokens
            + self.doc_budget_tokens
            + self.history_tokens
            + self.query_tokens
        )

    def context_utilization(self) -> float:
        return self.total_input_per_query() / self.context_window * 100

    def is_safe(self) -> bool:
        """True if total tokens is within 85% of context window."""
        return self.total_input_per_query() <= self.context_window * 0.85

    def cost_per_query(self, use_cache_read: bool = False) -> float:
        """
        Cost for one query.

        Args:
            use_cache_read: If True, cacheable_tokens are billed at cache_read rate.
                            First call (cache write) bills at cache_write rate.
        """
        p = PRICING[self.model]
        M = 1_000_000

        total_input = self.total_input_per_query()

        if self.use_cache and self.cacheable_tokens >= 1024:
            if use_cache_read:
                regular_input = total_input - self.cacheable_tokens
                return (
                    regular_input         / M * p["input"]
                    + self.expected_output / M * p["output"]
                    + self.cacheable_tokens / M * p["cache_read"]
                )
            else:
                regular_input = total_input - self.cacheable_tokens
                return (
                    regular_input         / M * p["input"]
                    + self.expected_output / M * p["output"]
                    + self.cacheable_tokens / M * p["cache_write"]
                )

        return (
            total_input          / M * p["input"]
            + self.expected_output / M * p["output"]
        )

    def monthly_cost(self) -> float:
        """
        Estimate monthly cost.
        Assumes first call per day writes cache, remaining reads from cache.
        """
        queries = self.queries_per_day * 30

        if self.use_cache and self.cacheable_tokens >= 1024:
            # First call per day: cache write
            write_calls = 30   # one cache write per day (5-min TTL refresh)
            read_calls  = queries - write_calls

            write_cost = write_calls * self.cost_per_query(use_cache_read=False)
            read_cost  = max(0, read_calls) * self.cost_per_query(use_cache_read=True)
            return write_cost + read_cost

        return queries * self.cost_per_query(use_cache_read=False)

    def display(self):
        """Print the full budget plan."""

        print(f"\n  MODEL: {self.model}")
        print(f"  Context window: {self.context_window:,} tokens")
        print()

        # Context allocation bar
        total = self.total_input_per_query()
        bar_w = 50

        allocs = [
            ("System",       self.system_tokens,     "▓"),
            ("Few-shot",     self.few_shot_tokens,   "░"),
            ("Documents",    self.doc_budget_tokens, "█"),
            ("History",      self.history_tokens,    "▒"),
            ("Query",        self.query_tokens,       "▓"),
        ]

        print(f"  CONTEXT WINDOW ALLOCATION  ({total:,} / {self.context_window:,} tokens, "
              f"{self.context_utilization():.1f}%)")
        print()

        for name, toks, bar_char in allocs:
            if toks == 0:
                continue
            pct     = toks / self.context_window * 100
            bar_len = max(1, int(pct / 100 * bar_w))
            bar     = bar_char * bar_len
            print(f"    {name:<15} {toks:>8,} toks  ({pct:5.1f}%)  {bar}")

        # Show unused
        unused = self.context_window - total
        unused_pct = unused / self.context_window * 100
        print(f"    {'Free headroom':<15} {unused:>8,} toks  ({unused_pct:5.1f}%)")
        print()

        # Metrics
        print(f"  Max chunks/query:    {self.max_chunks():,} "
              f"(at {self.avg_chunk_tokens} tok/chunk avg)")
        print(f"  Context safe:        {'✓ YES' if self.is_safe() else '✗ NO — EXCEEDS 85%'}")
        print(f"  Output reserve:      {self.output_reserve:,} tokens")

        print()
        print(f"  COST PROJECTIONS  —  {self.queries_per_day:,} queries/day")

        cost_nocache = self.cost_per_query(use_cache_read=False)
        cost_cached  = self.cost_per_query(use_cache_read=True) if self.use_cache and self.cacheable_tokens >= 1024 else cost_nocache
        monthly      = self.monthly_cost()

        print(f"    Cost/query (no cache):   ${cost_nocache:.5f}")
        if self.use_cache and self.cacheable_tokens >= 1024:
            print(f"    Cost/query (cache read): ${cost_cached:.5f}  "
                  f"({(1-cost_cached/cost_nocache)*100:.1f}% cheaper)")
        print(f"    Estimated monthly:       ${monthly:,.2f}")
        print(f"    Estimated annual:        ${monthly*12:,.2f}")

    def compare_with_config(self, other: "RAGBudgetPlan", other_label: str = "Alternative"):
        """Compare this plan with another configuration."""

        self_monthly  = self.monthly_cost()
        other_monthly = other.monthly_cost()
        diff          = self_monthly - other_monthly

        print(f"\n  COMPARISON: Current vs {other_label}")
        print(f"    Current monthly:   ${self_monthly:,.2f}")
        print(f"    {other_label:18} ${other_monthly:,.2f}")
        print(f"    Difference:        ${abs(diff):,.2f}/mo "
              f"({'savings' if diff > 0 else 'extra cost'})")
        print(f"    Annual impact:     ${abs(diff)*12:,.2f}")


# ─── Validation: Test the Budget Against Real API ─────────────────────────────

def validate_budget_with_real_call(plan: RAGBudgetPlan):
    """
    Make a real API call using the budget plan parameters.
    Verify that actual token usage aligns with plan estimates.
    """

    print("\n" + "─" * 60)
    print("LIVE VALIDATION: Real API call vs budget plan")
    print("─" * 60)

    # Build a test prompt that uses the plan's token allocations
    system_prompt = (
        "You are a precise enterprise knowledge assistant for Criterion Networks. "
        "Answer using ONLY the provided context. Cite [Doc N]. "
        "If not in context, respond: NOT IN PROVIDED CONTEXT."
    )

    # Synthetic retrieved docs (simulating doc_budget_tokens usage)
    # We use ~300 tokens per doc — 3 docs = ~900 tokens of doc content
    retrieved_docs = (
        "[Doc 1] Source: readyops_spec.pdf | Score: 0.93\n"
        "ReadyOps performs continuous validation across two environments: "
        "Live Operations and Production-Representative. The Production-Representative "
        "environment can be a digital twin, physical lab, or hybrid. Operational changes "
        "execute in Live Operations ONLY after formal promotion with validation gate sign-off.\n\n"

        "[Doc 2] Source: readyops_agents.pdf | Score: 0.87\n"
        "ReadyOps agent classes: Health & Posture monitors ongoing health. "
        "Validation runs automated test suites. Operational executes approved changes. "
        "Stress & Adversarial tests resilience. All agents share one intent model "
        "across both environments.\n\n"

        "[Doc 3] Source: criterion_overview.pdf | Score: 0.79\n"
        "Criterion Networks is a Cisco Premier Advisor and MINT Partner. "
        "The company provides infrastructure lifecycle coverage: PoV, Validation, "
        "Production, and Operations. The tagline is: Validate Before You Operate. "
        "Validate While You Operate."
    )

    user_message = (
        f"CONTEXT DOCUMENTS:\n{retrieved_docs}\n\n"
        f"QUESTION: How does ReadyOps ensure production safety, and what environments does it use?"
    )

    # ── Pre-call token count ──────────────────────────────────────────────────
    estimated_tokens = (
        tok_count(system_prompt)
        + tok_count(user_message)
    )

    # Verify with API count
    api_count = client.messages.count_tokens(
        model  = plan.model,
        system = system_prompt,
        messages = [{"role": "user", "content": user_message}],
    ).input_tokens

    print(f"\n  Token count comparison:")
    print(f"    tiktoken estimate:  {estimated_tokens:,}")
    print(f"    API actual count:   {api_count:,}")
    accuracy = (1 - abs(api_count - estimated_tokens) / api_count) * 100
    print(f"    Accuracy:           {accuracy:.1f}%  "
          f"({'✓ within 5%' if accuracy >= 95 else '⚠ >5% deviation'})")

    # ── Real API call ─────────────────────────────────────────────────────────
    print(f"\n  Making live API call...")

    start = time.perf_counter()
    first_tok_time = None
    response_text = ""

    # Build system with optional cache_control
    if plan.use_cache and tok_count(system_prompt) >= 1024:
        # WHY list format for system with cache_control:
        #   When using cache_control, system must be a list of content blocks.
        system_param = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
    else:
        system_param = system_prompt

    with client.messages.stream(
        model       = plan.model,
        max_tokens  = plan.expected_output,
        temperature = 0,
        system      = system_param,
        messages    = [{"role": "user", "content": user_message}],
    ) as stream:
        for delta in stream.text_stream:
            if first_tok_time is None:
                first_tok_time = time.perf_counter()
            response_text += delta
            print(delta, end="", flush=True)

        final_msg = stream.get_final_message()
        usage     = final_msg.usage

    print()

    total_ms = int((time.perf_counter() - start) * 1000)
    ttft_ms  = int((first_tok_time - start) * 1000) if first_tok_time else 0

    # ── Cost reconciliation ────────────────────────────────────────────────────
    p = PRICING[plan.model]
    M = 1_000_000

    actual_input  = usage.input_tokens
    actual_output = usage.output_tokens
    cache_created = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read    = getattr(usage, "cache_read_input_tokens",     0) or 0

    regular_input = max(0, actual_input - cache_created - cache_read)
    actual_cost   = (
        regular_input   / M * p["input"]
        + actual_output / M * p["output"]
        + cache_created  / M * p["cache_write"]
        + cache_read     / M * p["cache_read"]
    )

    plan_estimate = plan.cost_per_query(use_cache_read=(cache_read > 0))

    print(f"\n  ACTUAL vs PLANNED:")
    print(f"    Input tokens:  actual={actual_input:,}  vs plan≈{plan.total_input_per_query():,}")
    print(f"    Output tokens: actual={actual_output:,}  vs plan≈{plan.expected_output}")
    print(f"    Cache created: {cache_created:,}")
    print(f"    Cache read:    {cache_read:,}")
    print(f"    Actual cost:   ${actual_cost:.5f}  vs plan≈${plan_estimate:.5f}")
    print(f"    Latency:       TTFT={ttft_ms}ms  Total={total_ms}ms")


# ─── Run the Full Planner ─────────────────────────────────────────────────────

def run_planner():
    """
    Build a full budget plan and compare configurations.
    """

    print("=" * 65)
    print("TOKEN BUDGET PLANNER: Criterion Networks RAG Deployment")
    print("=" * 65)

    # ── Profile a representative corpus ──────────────────────────────────────
    # In production: replace with your actual document samples

    corpus_samples = {
        "prose": [
            "Cisco ACI uses a Leaf-Spine topology. The APIC controller manages all fabric policy. EPGs communicate via contracts.",
            "ReadyOps performs continuous validation across Live Operations and Production-Representative environments. Changes require formal promotion.",
            "Cisco Hypershield embeds security in the network fabric using eBPF. It provides autonomous microsegmentation without dedicated appliances.",
        ],
        "config": [
            "tenant:\n  name: criterion\n  vrf:\n    name: prod-vrf\n    enforcement: enforced\n  bridge_domain:\n    name: prod-bd",
            '{"dn":"uni/tn-criterion","name":"criterion","pcEnfPref":"enforced","fvCtx":{"name":"prod-vrf"}}',
        ],
        "code": [
            "def validate_aci_config(apic, tenant, vrf):\n    result = apic.get_tenant(tenant)\n    if not result:\n        raise ValueError(f'Tenant {tenant} not found')\n    return apic.validate_vrf(tenant, vrf)",
        ],
    }

    profile = profile_corpus_samples(corpus_samples)

    print(f"\n  CORPUS PROFILE (measured from {sum(len(v) for v in corpus_samples.values())} samples):")
    print(f"    Avg tokens/chunk:  {profile.avg_chunk_tokens:.0f}")
    print(f"    Chars per token:   {profile.chars_per_token:.2f}")
    print(f"    Content mix:       {', '.join(f'{k}: {v:.0%}' for k,v in profile.content_types.items())}")

    # ── Plan A: Baseline (no cache, 10 chunks) ────────────────────────────────
    plan_a = RAGBudgetPlan(
        model             = "claude-sonnet-4-6",
        queries_per_day   = 3_000,
        expected_output   = 400,
        system_tokens     = 400,
        doc_budget_tokens = 10_000,   # 10 chunks × 1,000 tokens each
        history_tokens    = 2_000,
        query_tokens      = 200,
        use_cache         = False,
        avg_chunk_tokens  = int(profile.avg_chunk_tokens),
    )

    print("\n" + "─" * 60)
    print("PLAN A: Baseline (no caching, large chunks)")
    plan_a.display()

    # ── Plan B: Optimized (with cache, smaller chunks) ────────────────────────
    plan_b = RAGBudgetPlan(
        model              = "claude-sonnet-4-6",
        queries_per_day    = 3_000,
        expected_output    = 400,
        system_tokens      = 400,
        doc_budget_tokens  = 5_000,   # 5 smaller, more precise chunks
        history_tokens     = 2_000,
        query_tokens       = 200,
        use_cache          = True,
        cacheable_tokens   = 400,     # system prompt cached
        avg_chunk_tokens   = int(profile.avg_chunk_tokens),
    )

    print("\n" + "─" * 60)
    print("PLAN B: Optimized (prompt caching + smaller chunk budget)")
    plan_b.display()

    plan_a.compare_with_config(plan_b, "Plan B (optimized)")

    # ── Live validation ────────────────────────────────────────────────────────
    validate_budget_with_real_call(plan_b)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_planner()
