"""
FILE: 06_mini_project_context_optimizer.py
LESSON: Phase 1 - Lesson 6 - Context Window
TOPIC: Mini-Project — Full Context Window Optimizer for Production RAG

WHAT THIS PROJECT BUILDS:
  A production-ready RAG pipeline that:
    1. Accepts retrieved chunks + user query
    2. Chooses the optimal filling strategy based on query characteristics
    3. Applies compression where needed
    4. Orders chunks to mitigate "Lost in the Middle"
    5. Applies Anthropic Prompt Caching to static content
    6. Calls the API and returns response with full metrics

ARCHITECTURE:
  ┌─────────────────────────────────────────────────────────┐
  │  QUERY ANALYZER         (classify query type + intent)  │
  │  STRATEGY SELECTOR      (choose filling strategy)       │
  │  CHUNK COMPRESSOR       (compress if needed)            │
  │  CONTEXT ASSEMBLER      (order + budget + cache marks)  │
  │  CLAUDE API CALL        (with Prompt Caching)           │
  │  RESPONSE ANALYZER      (citations, grounding, cost)    │
  └─────────────────────────────────────────────────────────┘

TIES TOGETHER:
  - Lesson 1:  API setup, token counting, streaming
  - Lesson 2:  Model routing by query type
  - Lesson 4:  Prompt Caching with cache_control
  - Lesson 6:  All four filling strategies + compression

INSTALL:
  pip install anthropic python-dotenv tiktoken
"""

import os
import time
import json
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
except ImportError:
    def count_tokens_local(text: str) -> int:
        return int(len(text.split()) / 0.75)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:  str
    source:    str
    content:   str
    score:     float
    tier:      str = "general"   # "core" | "supporting" | "general"

    @property
    def token_count(self) -> int:
        return count_tokens_local(self.content)


@dataclass
class OptimizationConfig:
    """Controls how the context optimizer behaves."""
    model:              str   = "claude-sonnet-4-6"
    doc_token_budget:   int   = 60_000    # max tokens for retrieved docs
    history_budget:     int   = 4_000     # max tokens for conversation history
    score_threshold:    float = 0.65      # min score to include chunk
    max_chunks:         int   = 10        # hard cap on number of chunks
    compress_chunks:    bool  = False     # apply LLM compression to chunks
    use_prompt_cache:   bool  = True      # apply Anthropic Prompt Caching to system prompt
    ordering:           str   = "lost_in_middle"   # "relevance" | "lost_in_middle"
    fallback_top_k:     int   = 2         # if 0 chunks pass threshold, use top-K


@dataclass
class OptimizationReport:
    """What the optimizer did — returned with every RAG call."""
    strategy_used:       str
    chunks_retrieved:    int
    chunks_selected:     int
    chunks_excluded:     int
    total_input_tokens:  int
    output_tokens:       int
    cache_created_tokens: int
    cache_read_tokens:   int
    estimated_cost_usd:  float
    latency_ms:          int
    ttft_ms:             int
    citations_found:     list[str]
    grounding_pct:       float
    warnings:            list[str] = field(default_factory=list)

    def display(self):
        print("\n" + "═" * 60)
        print("OPTIMIZATION REPORT")
        print("═" * 60)
        print(f"  Strategy:          {self.strategy_used}")
        print(f"  Chunks:            {self.chunks_retrieved} retrieved → "
              f"{self.chunks_selected} selected, {self.chunks_excluded} excluded")
        print(f"  Input tokens:      {self.total_input_tokens:,}")
        print(f"  Output tokens:     {self.output_tokens:,}")
        print(f"  Prompt cache:      {self.cache_created_tokens:,} created, "
              f"{self.cache_read_tokens:,} read")
        print(f"  Est. cost:         ${self.estimated_cost_usd:.5f}")
        print(f"  Latency:           TTFT={self.ttft_ms}ms  Total={self.latency_ms}ms")
        print(f"  Citations found:   {self.citations_found or 'none'}")
        print(f"  Grounding:         {self.grounding_pct:.0f}% of sentences cite sources")

        if self.warnings:
            print(f"\n  Warnings:")
            for w in self.warnings:
                print(f"    ⚠ {w}")
        else:
            print(f"\n  ✓ No warnings")


# ─── Cost Calculator ──────────────────────────────────────────────────────────

# Anthropic pricing (per million tokens) — update as pricing changes
PRICING = {
    "claude-sonnet-4-6": {
        "input":         3.00,
        "output":       15.00,
        "cache_write":   3.75,
        "cache_read":    0.30,   # WHY 90% cheaper: cache reads avoid full recomputation
    },
    "claude-haiku-4-5-20251001": {
        "input":         0.80,
        "output":        4.00,
        "cache_write":   1.00,
        "cache_read":    0.08,
    },
}

def estimate_cost(
    model:                str,
    input_tokens:         int,
    output_tokens:        int,
    cache_created_tokens: int = 0,
    cache_read_tokens:    int = 0,
) -> float:
    """Calculate estimated API cost in USD."""
    p = PRICING.get(model, PRICING["claude-sonnet-4-6"])
    M = 1_000_000

    # WHY subtract cache tokens from regular input:
    #   cache_write tokens are charged at cache_write rate (slightly higher than input).
    #   cache_read tokens are charged at cache_read rate (much cheaper).
    #   Only the REMAINING non-cached input uses the standard input rate.
    regular_input = max(0, input_tokens - cache_created_tokens - cache_read_tokens)

    cost = (
        regular_input         / M * p["input"]
        + output_tokens       / M * p["output"]
        + cache_created_tokens / M * p["cache_write"]
        + cache_read_tokens    / M * p["cache_read"]
    )
    return cost


# ─── Context Window Optimizer ─────────────────────────────────────────────────

class ContextWindowOptimizer:
    """
    Production-ready context window optimizer.
    Selects the best filling strategy, applies compression,
    orders chunks, and assembles the final API messages.
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def _select_strategy(self, chunks: list[RetrievedChunk], query: str) -> str:
        """
        Auto-select filling strategy based on query and chunk characteristics.

        WHY rules-based not LLM-based:
          Strategy selection must be FAST (no extra API call) and
          DETERMINISTIC (same input → same strategy).
          An LLM-based classifier would add latency and cost.
        """

        avg_score    = sum(c.score for c in chunks) / max(len(chunks), 1)
        has_tiers    = any(c.tier != "general" for c in chunks)
        low_quality  = avg_score < self.config.score_threshold
        many_chunks  = len(chunks) > self.config.max_chunks

        # WHY this priority order:
        #   1. If we have tiered docs, hierarchical always wins
        #      (tiering encodes explicit priority from the caller).
        #   2. If all scores are low, use threshold to filter aggressively.
        #   3. If we have many chunks, dynamic budget handles the overflow.
        #   4. Default: relevance order, take top-K.
        if has_tiers:
            return "hierarchical"
        elif low_quality:
            return "threshold"
        elif many_chunks:
            return "dynamic"
        else:
            return "fixed_k"

    def _apply_strategy(
        self,
        strategy:  str,
        chunks:    list[RetrievedChunk],
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """
        Apply the selected strategy, return (selected, excluded).
        """

        if strategy == "fixed_k":
            sorted_c = sorted(chunks, key=lambda c: c.score, reverse=True)
            k        = min(len(sorted_c), self.config.max_chunks)
            return sorted_c[:k], sorted_c[k:]

        elif strategy == "dynamic":
            selected, excluded = [], []
            running  = 0
            for c in sorted(chunks, key=lambda c: c.score, reverse=True):
                if running + c.token_count <= self.config.doc_token_budget:
                    selected.append(c)
                    running += c.token_count
                else:
                    excluded.append(c)
            return selected, excluded

        elif strategy == "threshold":
            above    = [c for c in chunks if c.score >= self.config.score_threshold]
            below    = [c for c in chunks if c.score <  self.config.score_threshold]

            if not above:
                # Fallback
                sorted_all = sorted(chunks, key=lambda c: c.score, reverse=True)
                above      = sorted_all[:self.config.fallback_top_k]
                below      = sorted_all[self.config.fallback_top_k:]

            return above, below

        elif strategy == "hierarchical":
            tiers    = {"core": [], "supporting": [], "general": []}
            budgets  = {
                "core":       int(self.config.doc_token_budget * 0.50),
                "supporting": int(self.config.doc_token_budget * 0.33),
                "general":    int(self.config.doc_token_budget * 0.17),
            }

            selected_all = []
            excluded_all = []

            for chunk in chunks:
                tiers.setdefault(chunk.tier, []).append(chunk)

            for tier_name, tier_chunks in tiers.items():
                budget   = budgets.get(tier_name, 0)
                running  = 0
                for c in sorted(tier_chunks, key=lambda c: c.score, reverse=True):
                    if running + c.token_count <= budget:
                        selected_all.append(c)
                        running += c.token_count
                    else:
                        excluded_all.append(c)

            return selected_all, excluded_all

        return chunks, []

    def _order_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Order selected chunks to mitigate Lost in the Middle."""

        if self.config.ordering == "lost_in_middle":
            sorted_c    = sorted(chunks, key=lambda c: c.score, reverse=True)
            n           = len(sorted_c)
            top_half    = sorted_c[:(n + 1) // 2]
            bottom_half = list(reversed(sorted_c[(n + 1) // 2:]))
            return top_half + bottom_half

        # Default: relevance order
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def _build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        """Build the formatted context block for the RAG prompt."""

        lines = ["CONTEXT DOCUMENTS:", "─" * 50]
        for c in chunks:
            lines.append(
                f"[{c.chunk_id}] Source: {c.source}  "
                f"Relevance: {c.score:.2f}  Tier: {c.tier}"
            )
            lines.append(c.content)
            lines.append("")

        return "\n".join(lines)

    def optimize_and_call(
        self,
        query:             str,
        chunks:            list[RetrievedChunk],
        conversation_hist: list[dict] = None,
        system_prefix:     str = "",
    ) -> tuple[str, OptimizationReport]:
        """
        Full optimization pipeline + API call.

        Args:
            query:             The user's current question.
            chunks:            All retrieved chunks from the vector DB.
            conversation_hist: Previous turns [{"role": ..., "content": ...}].
            system_prefix:     Additional system prompt content to prepend.

        Returns:
            (response_text, report)
        """

        conversation_hist = conversation_hist or []
        warnings          = []

        # ── 1. Select and Apply Strategy ─────────────────────────────────────
        strategy = self._select_strategy(chunks, query)
        selected, excluded = self._apply_strategy(strategy, chunks)

        if not selected:
            return "NO RELEVANT CONTEXT FOUND", OptimizationReport(
                strategy_used=strategy,
                chunks_retrieved=len(chunks),
                chunks_selected=0,
                chunks_excluded=len(chunks),
                total_input_tokens=0,
                output_tokens=0,
                cache_created_tokens=0,
                cache_read_tokens=0,
                estimated_cost_usd=0.0,
                latency_ms=0,
                ttft_ms=0,
                citations_found=[],
                grounding_pct=0.0,
                warnings=["No chunks selected — all excluded by strategy"],
            )

        # ── 2. Order Chunks ───────────────────────────────────────────────────
        selected = self._order_chunks(selected)

        # ── 3. Build System Prompt (static, cacheable) ─────────────────────────
        system_grounding = (
            "You are a precise technical knowledge assistant.\n"
            "RULES:\n"
            "1. Answer using ONLY the provided CONTEXT DOCUMENTS.\n"
            "2. Cite every claim with [chunk_id].\n"
            "3. If not in context, respond: NOT IN PROVIDED CONTEXT\n"
            "4. Never add general knowledge or assumptions.\n"
        )

        if system_prefix:
            system_content = system_prefix + "\n\n" + system_grounding
        else:
            system_content = system_grounding

        # ── 4. Build Messages ──────────────────────────────────────────────────
        context_block = self._build_context_block(selected)
        user_content  = f"{context_block}\n\nQUESTION: {query}"

        messages = list(conversation_hist) + [
            {"role": "user", "content": user_content}
        ]

        # ── 5. Pre-call Token Count ────────────────────────────────────────────
        # WHY always count before calling:
        #   Prevents expensive failed calls. The optimizer may have
        #   estimated budgets incorrectly due to tiktoken vs API discrepancy.
        token_check = client.messages.count_tokens(
            model=self.config.model,
            system=system_content,
            messages=messages,
        )

        SAFE_LIMIT = 170_000
        if token_check.input_tokens > SAFE_LIMIT:
            warnings.append(
                f"Input ({token_check.input_tokens:,}) exceeds safe limit ({SAFE_LIMIT:,}). "
                f"Reducing chunk count."
            )
            # Emergency trim: drop lowest-scoring chunks until safe
            while selected and token_check.input_tokens > SAFE_LIMIT:
                selected.pop()   # remove last (lowest relevance after ordering)
                context_block = self._build_context_block(selected)
                user_content  = f"{context_block}\n\nQUESTION: {query}"
                messages[-1]["content"] = user_content
                token_check   = client.messages.count_tokens(
                    model=self.config.model,
                    system=system_content,
                    messages=messages,
                )

        # ── 6. Build system with optional cache_control ───────────────────────
        if self.config.use_prompt_cache:
            # WHY cache_control on system:
            #   The system prompt (grounding rules) never changes across queries.
            #   Cache it → pay for computation only once per 5-minute window.
            system_param = [
                {
                    "type":          "text",
                    "text":          system_content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_param = system_content

        # ── 7. API Call with Streaming ────────────────────────────────────────
        start_time       = time.perf_counter()
        first_token_time = None
        full_response    = ""

        print("\n" + "─" * 60)
        print(f"QUERY: {query}")
        print("─" * 60)
        print("RESPONSE: ", end="", flush=True)

        final_usage = None

        with client.messages.stream(
            model         = self.config.model,
            max_tokens    = 500,
            temperature   = 0,
            system        = system_param,
            messages      = messages,
        ) as stream:
            for delta in stream.text_stream:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_response += delta
                print(delta, end="", flush=True)

            # WHY get_final_message() after stream:
            #   Usage statistics (including cache tokens) are only available
            #   in the final message object, not during streaming.
            final_msg   = stream.get_final_message()
            final_usage = final_msg.usage

        print()

        total_ms = int((time.perf_counter() - start_time) * 1000)
        ttft_ms  = int((first_token_time - start_time) * 1000) if first_token_time else 0

        # ── 8. Analyze Response ───────────────────────────────────────────────

        citations = [
            c.chunk_id
            for c in selected
            if f"[{c.chunk_id}]" in full_response
        ]

        sentences        = [s for s in full_response.split(".") if len(s.strip()) > 5]
        cited_sentences  = [
            s for s in sentences
            if any(f"[{cid}]" in s for cid in citations)
        ]
        grounding_pct    = len(cited_sentences) / max(len(sentences), 1) * 100

        # ── 9. Cost Calculation ────────────────────────────────────────────────
        input_toks         = final_usage.input_tokens if final_usage else token_check.input_tokens
        output_toks        = final_usage.output_tokens if final_usage else 0
        cache_created      = getattr(final_usage, "cache_creation_input_tokens", 0) or 0
        cache_read         = getattr(final_usage, "cache_read_input_tokens",     0) or 0

        cost = estimate_cost(
            model                = self.config.model,
            input_tokens         = input_toks,
            output_tokens        = output_toks,
            cache_created_tokens = cache_created,
            cache_read_tokens    = cache_read,
        )

        report = OptimizationReport(
            strategy_used        = strategy,
            chunks_retrieved     = len(chunks),
            chunks_selected      = len(selected),
            chunks_excluded      = len(excluded),
            total_input_tokens   = input_toks,
            output_tokens        = output_toks,
            cache_created_tokens = cache_created,
            cache_read_tokens    = cache_read,
            estimated_cost_usd   = cost,
            latency_ms           = total_ms,
            ttft_ms              = ttft_ms,
            citations_found      = citations,
            grounding_pct        = grounding_pct,
            warnings             = warnings,
        )

        return full_response, report


# ─── Demo ─────────────────────────────────────────────────────────────────────

def run_demo():
    """
    End-to-end demo using the ContextWindowOptimizer on a realistic RAG scenario.
    """

    # Simulated vector DB results
    # WHY mixed tiers and scores:
    #   Realistic retrieval includes a mix of high/low relevance docs
    #   and documents with different importance levels.
    chunks = [
        RetrievedChunk(
            "chunk_001", "readyops_spec.pdf, p.1", tier="core", score=0.93,
            content=(
                "ReadyOps is Criterion Networks' continuous validation platform. "
                "It operates across two deliberately isolated environments: "
                "Live Operations and Production-Representative. "
                "The Production-Representative environment can be a digital twin, "
                "physical lab, or hybrid. Operational changes execute in Live Operations "
                "ONLY after validation and formal promotion from the Production-Representative."
            )
        ),
        RetrievedChunk(
            "chunk_002", "readyops_spec.pdf, p.4", tier="core", score=0.88,
            content=(
                "ReadyOps agent classes:\n"
                "Health & Posture: Continuous monitoring of network health and compliance posture.\n"
                "Validation: Automated test suites against the Production-Representative environment.\n"
                "Operational: Executes approved changes with full audit logging.\n"
                "Stress & Adversarial: Tests resilience under failure and attack conditions."
            )
        ),
        RetrievedChunk(
            "chunk_003", "aci_guide.pdf, p.12", tier="supporting", score=0.79,
            content=(
                "Cisco ACI uses Leaf-Spine topology. APIC manages fabric policy centrally. "
                "EPGs communicate through contracts. ReadyOps can deploy a digital twin "
                "of an ACI fabric using Cisco Nexus Dashboard's simulation mode, "
                "allowing validation of ACI policy changes before pushing to production."
            )
        ),
        RetrievedChunk(
            "chunk_004", "readyops_spec.pdf, p.8", tier="supporting", score=0.74,
            content=(
                "Validation gates in ReadyOps: A change must pass three gates before promotion: "
                "1) Configuration validation (no syntax errors, policy conflicts). "
                "2) Functional validation (traffic flows as expected in digital twin). "
                "3) Performance validation (no latency regression above 5% threshold)."
            )
        ),
        RetrievedChunk(
            "chunk_005", "cisco_intersight.pdf, p.3", tier="general", score=0.55,
            content=(
                "Cisco Intersight is a cloud-based infrastructure management platform. "
                "It provides unified management for UCS, HyperFlex, and third-party infrastructure. "
                "Intersight integrates with Kubernetes, VMware, and public cloud providers."
            )
        ),
        RetrievedChunk(
            "chunk_006", "general_sdn_background.pdf", tier="general", score=0.38,
            content=(
                "Software-defined networking separates control and data planes. "
                "SDN enables programmable networks through centralized control. "
                "OpenFlow was an early SDN protocol. Modern SDN uses REST APIs and YANG models."
            )
        ),
    ]

    config = OptimizationConfig(
        model            = "claude-sonnet-4-6",
        doc_token_budget = 8_000,    # small for demo
        score_threshold  = 0.60,
        max_chunks       = 5,
        use_prompt_cache = True,
        ordering         = "lost_in_middle",
        fallback_top_k   = 2,
    )

    optimizer = ContextWindowOptimizer(config)

    print("=" * 60)
    print("CONTEXT WINDOW OPTIMIZER: End-to-End Demo")
    print("=" * 60)
    print(f"  Config: strategy=auto, budget={config.doc_token_budget:,}tk, "
          f"threshold={config.score_threshold}, ordering={config.ordering}")
    print(f"  Chunks available: {len(chunks)}")

    query = (
        "How does ReadyOps ensure that only validated changes reach "
        "production, and what validation gates must be passed?"
    )

    response, report = optimizer.optimize_and_call(
        query  = query,
        chunks = chunks,
    )

    report.display()

    # Second call — should get cache read tokens
    print("\n" + "─" * 60)
    print("SECOND CALL (testing Prompt Cache hit)...")
    print("─" * 60)

    query2 = "Which ReadyOps agent class handles continuous health monitoring?"

    response2, report2 = optimizer.optimize_and_call(
        query  = query2,
        chunks = chunks,
    )

    report2.display()

    if report2.cache_read_tokens > 0:
        print(f"\n  ✓ Prompt Cache HIT — {report2.cache_read_tokens:,} tokens read from cache")
        savings = report2.cache_read_tokens / 1_000_000 * (
            PRICING["claude-sonnet-4-6"]["input"] - PRICING["claude-sonnet-4-6"]["cache_read"]
        )
        print(f"  ✓ Cache saved: ${savings:.5f} on this call")
    else:
        print(f"\n  Cache MISS — system prompt not yet in cache (may need >1024 tokens to cache)")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_demo()
