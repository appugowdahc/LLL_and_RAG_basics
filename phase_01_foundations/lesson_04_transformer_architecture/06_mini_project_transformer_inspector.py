"""
FILE: 06_mini_project_transformer_inspector.py
LESSON: Phase 1 - Lesson 4 - Transformer Architecture
TOPIC: Mini-Project — Full Transformer Inspector + RAG Feasibility Analyzer

WHAT THIS FILE TEACHES:
  - Ties together ALL of Lesson 4 into one tool
  - Inputs: any transformer config
  - Outputs: complete analysis:
      - parameter breakdown
      - memory requirements
      - KV cache at different context lengths
      - RAG feasibility (how many chunks fit, what GPU you need)
      - Cost estimation for API-based vs self-hosted deployment

USE THIS AS A DECISION TOOL:
  Before building a RAG system, run this inspector on your target model.
  It answers: Can I self-host this? What's the max context? What's the cost?
"""

import os
import math
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# ─── Infrastructure Catalog ───────────────────────────────────────────────────

GPU_CATALOG = [
    {"name": "RTX 4090",      "vram_gb": 24,  "tflops_fp16": 82.6,  "monthly_cost": 0},
    {"name": "A10G",          "vram_gb": 24,  "tflops_fp16": 31.2,  "monthly_cost": 0.75},
    {"name": "A100 40GB",     "vram_gb": 40,  "tflops_fp16": 312,   "monthly_cost": 2.21},
    {"name": "A100 80GB",     "vram_gb": 80,  "tflops_fp16": 312,   "monthly_cost": 3.67},
    {"name": "H100 SXM",      "vram_gb": 80,  "tflops_fp16": 989,   "monthly_cost": 8.00},
    {"name": "H100 NVL",      "vram_gb": 94,  "tflops_fp16": 835,   "monthly_cost": 9.00},
    {"name": "H200",          "vram_gb": 141, "tflops_fp16": 1979,  "monthly_cost": 12.00},
]

# API pricing per million tokens (input / output)
API_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00},
    "claude-opus-4-8":           {"input": 15.00, "output": 75.00},
}


@dataclass
class RAGScenario:
    """Describes a RAG deployment scenario for feasibility analysis."""
    name:             str
    chunk_size_tokens: int     # tokens per retrieved chunk
    top_k_chunks:     int      # number of chunks retrieved per query
    system_prompt_tokens: int  # size of system prompt
    query_tokens:     int      # average query length
    output_tokens:    int      # average response length
    daily_queries:    int      # queries per day
    model_name:       str      # API model or local model name


def analyze_rag_context_budget(scenario: RAGScenario, context_window: int) -> dict:
    """
    Analyze whether a RAG scenario fits within a model's context window.
    Returns budget breakdown and feasibility assessment.
    """

    # Total tokens needed per call
    context_tokens  = scenario.top_k_chunks * scenario.chunk_size_tokens
    input_tokens    = (
        scenario.system_prompt_tokens
        + context_tokens
        + scenario.query_tokens
    )
    total_tokens    = input_tokens + scenario.output_tokens

    fits            = total_tokens <= context_window
    utilization_pct = (total_tokens / context_window) * 100

    # Maximum chunks that fit given the context window
    reserved_tokens = (
        scenario.system_prompt_tokens
        + scenario.query_tokens
        + scenario.output_tokens
        + 100  # safety buffer
    )
    max_chunks = max(0, (context_window - reserved_tokens) // scenario.chunk_size_tokens)

    return {
        "system_prompt_tokens":  scenario.system_prompt_tokens,
        "context_tokens":        context_tokens,
        "query_tokens":          scenario.query_tokens,
        "total_input_tokens":    input_tokens,
        "output_tokens":         scenario.output_tokens,
        "total_tokens":          total_tokens,
        "context_window":        context_window,
        "fits":                  fits,
        "utilization_pct":       round(utilization_pct, 1),
        "max_chunks_possible":   max_chunks,
    }


def estimate_api_cost(scenario: RAGScenario, budget: dict) -> dict:
    """
    Estimate daily and monthly API cost for a RAG scenario.
    """

    if scenario.model_name not in API_PRICING:
        return {"error": f"Unknown model: {scenario.model_name}"}

    pricing = API_PRICING[scenario.model_name]

    cost_per_call = (
        budget["total_input_tokens"]  * pricing["input"]  / 1_000_000
        + scenario.output_tokens       * pricing["output"] / 1_000_000
    )
    daily_cost    = cost_per_call * scenario.daily_queries
    monthly_cost  = daily_cost * 30

    return {
        "cost_per_call_usd":  round(cost_per_call, 6),
        "daily_cost_usd":     round(daily_cost, 2),
        "monthly_cost_usd":   round(monthly_cost, 2),
        "annual_cost_usd":    round(daily_cost * 365, 2),
    }


def recommend_gpu(model_params_B: float, context_window_tokens: int) -> list[dict]:
    """
    Recommend GPUs for self-hosting a model with a given context window.
    """

    # Model weights: params × 2 bytes (fp16)
    model_gb = model_params_B * 2

    # KV cache at max context (approximate for 32 layers, 8 kv heads, d_head=128)
    kv_per_token_bytes = 2 * 32 * 8 * 128 * 2  # K+V, layers, kv_heads, d_head, fp16
    kv_cache_gb = (context_window_tokens * kv_per_token_bytes) / (1024 ** 3)

    # Activations and overhead: ~20% of model size
    overhead_gb = model_gb * 0.20

    total_required_gb = model_gb + kv_cache_gb + overhead_gb

    recommendations = []
    for gpu in GPU_CATALOG:
        if gpu["vram_gb"] >= total_required_gb:
            recommendations.append({
                **gpu,
                "memory_utilization_pct": round(total_required_gb / gpu["vram_gb"] * 100, 1),
            })

    return {
        "model_weights_gb":    round(model_gb, 1),
        "kv_cache_gb":         round(kv_cache_gb, 1),
        "overhead_gb":         round(overhead_gb, 1),
        "total_required_gb":   round(total_required_gb, 1),
        "compatible_gpus":     recommendations[:3],  # top 3 cheapest that fit
    }


def full_transformer_inspection(
    model_name:      str,
    params_B:        float,
    context_window:  int,
    scenario:        RAGScenario,
):
    """
    Full inspection: combine architecture analysis + RAG feasibility + cost.
    """

    print("\n" + "═"*65)
    print(f" TRANSFORMER INSPECTOR: {model_name}")
    print("═"*65)

    # ── Architecture Summary ───────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  ARCHITECTURE")
    print(f"{'─'*65}")
    print(f"  Parameters:      {params_B:.1f}B")
    print(f"  Context Window:  {context_window:,} tokens  (~{context_window*0.75:.0f} words)")
    print(f"  fp16 Weights:    {params_B * 2:.1f} GB")

    # ── RAG Budget Analysis ────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  RAG SCENARIO: {scenario.name}")
    print(f"{'─'*65}")

    budget = analyze_rag_context_budget(scenario, context_window)

    status = "✓ FITS" if budget["fits"] else "✗ OVERFLOW"
    print(f"  Status:            {status}")
    print(f"  Context window:    {budget['context_window']:>8,} tokens")
    print(f"  ─ System prompt:   {budget['system_prompt_tokens']:>8,} tokens")
    print(f"  ─ Retrieved docs:  {budget['context_tokens']:>8,} tokens  "
          f"({scenario.top_k_chunks} chunks × {scenario.chunk_size_tokens})")
    print(f"  ─ User query:      {budget['query_tokens']:>8,} tokens")
    print(f"  ─ Output reserved: {budget['output_tokens']:>8,} tokens")
    print(f"  ─────────────────────────────")
    print(f"    Total used:      {budget['total_tokens']:>8,} tokens  "
          f"({budget['utilization_pct']}% of window)")
    print(f"  Max chunks possible: {budget['max_chunks_possible']}")

    if not budget["fits"]:
        overflow = budget["total_tokens"] - budget["context_window"]
        print(f"\n  ⚠ OVERFLOW by {overflow:,} tokens!")
        print(f"  FIX OPTIONS:")
        print(f"    1. Reduce chunk size to ~{scenario.chunk_size_tokens * 0.7:.0f} tokens")
        print(f"    2. Reduce top_k to {max(1, budget['max_chunks_possible'])}")
        print(f"    3. Use a model with larger context window")

    # ── Cost Analysis ─────────────────────────────────────────────────────────
    if scenario.model_name in API_PRICING:
        print(f"\n{'─'*65}")
        print(f"  API COST ANALYSIS ({scenario.model_name})")
        print(f"{'─'*65}")

        costs = estimate_api_cost(scenario, budget)
        print(f"  Cost per call:   ${costs['cost_per_call_usd']:.6f}")
        print(f"  Daily queries:   {scenario.daily_queries:,}")
        print(f"  Daily cost:      ${costs['daily_cost_usd']:,.2f}")
        print(f"  Monthly cost:    ${costs['monthly_cost_usd']:,.2f}")
        print(f"  Annual cost:     ${costs['annual_cost_usd']:,.2f}")

    # ── GPU Recommendation ────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  GPU RECOMMENDATION (self-hosting, fp16)")
    print(f"{'─'*65}")

    gpu_info = recommend_gpu(params_B, context_window)
    print(f"  Memory breakdown:")
    print(f"    Model weights:   {gpu_info['model_weights_gb']:>6.1f} GB")
    print(f"    KV cache (max):  {gpu_info['kv_cache_gb']:>6.1f} GB")
    print(f"    Overhead (20%):  {gpu_info['overhead_gb']:>6.1f} GB")
    print(f"    Total required:  {gpu_info['total_required_gb']:>6.1f} GB")

    if gpu_info["compatible_gpus"]:
        print(f"\n  Compatible GPUs (cheapest first):")
        for gpu in gpu_info["compatible_gpus"]:
            cost_str = f"${gpu['monthly_cost']:.2f}/hr" if gpu["monthly_cost"] > 0 else "owned"
            print(f"    {gpu['name']:<15} {gpu['vram_gb']:>4}GB VRAM  "
                  f"  {gpu['memory_utilization_pct']:>5.1f}% utilization  "
                  f"  {cost_str}")
    else:
        print(f"  No single GPU fits. Needs multi-GPU or model quantization.")

    # ── Final Recommendation ──────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  RECOMMENDATION")
    print(f"{'─'*65}")

    if scenario.daily_queries < 1000:
        print(f"  Low volume (<1K queries/day): Use API. Infrastructure overhead not worth it.")
    elif scenario.daily_queries < 50000:
        print(f"  Medium volume (1K-50K/day): Evaluate API vs self-host based on monthly cost.")
    else:
        print(f"  High volume (>50K/day): Self-hosting likely more cost-effective.")

    if budget["fits"] and budget["utilization_pct"] < 70:
        print(f"  Context window has good headroom ({100-budget['utilization_pct']:.0f}% free).")
        print(f"  → Can increase top_k to {budget['max_chunks_possible']} chunks for better recall.")
    elif budget["fits"]:
        print(f"  Context window is tight ({budget['utilization_pct']}% used). Monitor carefully.")
    else:
        print(f"  Context window is INSUFFICIENT for this scenario. Redesign required.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Scenario 1: Startup RAG with Claude Sonnet API
    startup_scenario = RAGScenario(
        name="Startup Knowledge Base",
        chunk_size_tokens=512,
        top_k_chunks=5,
        system_prompt_tokens=300,
        query_tokens=50,
        output_tokens=300,
        daily_queries=500,
        model_name="claude-sonnet-4-6",
    )

    full_transformer_inspection(
        model_name="Claude Sonnet 4.6 (API)",
        params_B=70.0,
        context_window=200_000,
        scenario=startup_scenario,
    )

    # Scenario 2: Enterprise RAG — high volume, self-hosted Llama-3-70B
    enterprise_scenario = RAGScenario(
        name="Enterprise Legal Document Search",
        chunk_size_tokens=512,
        top_k_chunks=10,
        system_prompt_tokens=500,
        query_tokens=100,
        output_tokens=500,
        daily_queries=50_000,
        model_name="claude-sonnet-4-6",  # comparison price
    )

    full_transformer_inspection(
        model_name="Llama-3-70B (self-hosted)",
        params_B=70.0,
        context_window=131_072,
        scenario=enterprise_scenario,
    )

    # Scenario 3: Small model, aggressive RAG
    aggressive_scenario = RAGScenario(
        name="Overloaded Context (test overflow)",
        chunk_size_tokens=1000,
        top_k_chunks=20,
        system_prompt_tokens=500,
        query_tokens=200,
        output_tokens=800,
        daily_queries=10_000,
        model_name="claude-haiku-4-5-20251001",
    )

    full_transformer_inspection(
        model_name="Claude Haiku 4.5 (API)",
        params_B=20.0,
        context_window=200_000,
        scenario=aggressive_scenario,
    )
