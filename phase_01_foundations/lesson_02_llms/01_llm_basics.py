"""
FILE: 01_llm_basics.py
LESSON: Phase 1 - Lesson 2 - What are LLMs?
TOPIC: Explore LLM metadata, model comparison, and capabilities

WHAT THIS FILE TEACHES:
  - How to list and compare available Claude models
  - What model metadata tells you (context window, capabilities)
  - How the same question gets different quality answers from different models
  - How to programmatically choose the right model for the right task

KEY CONCEPT:
  An LLM is NOT a database — it doesn't "store" facts like a key-value store.
  It stores STATISTICAL PATTERNS in billions of floating-point weights.
  When you ask a question, it generates a plausible continuation — not a lookup.
  This is why it can be wrong even when confident.
"""

import os
import time
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# ─── Model Registry ───────────────────────────────────────────────────────────

# WHY a model registry dict:
#   In production, your RAG system will route different task types
#   to different models (cheapest model that meets quality bar).
#   A registry makes this routing logic readable and maintainable.
MODEL_REGISTRY = {
    "claude-haiku-4-5-20251001": {
        "display_name":     "Claude Haiku 4.5",
        "context_window":   200_000,   # max tokens in context (prompt + response)
        "max_output":         8_192,   # max tokens it can generate per response
        "input_cost_per_m":    0.80,   # USD per million input tokens
        "output_cost_per_m":   4.00,   # USD per million output tokens
        "best_for": [
            "Simple Q&A",
            "Classification",
            "Keyword extraction",
            "Fast responses where cost matters",
        ],
        "avoid_for": [
            "Complex multi-step reasoning",
            "Nuanced writing",
            "Long document summarization",
        ],
    },
    "claude-sonnet-4-6": {
        "display_name":     "Claude Sonnet 4.6",
        "context_window":   200_000,
        "max_output":       16_000,
        "input_cost_per_m":   3.00,
        "output_cost_per_m": 15.00,
        "best_for": [
            "RAG Q&A (best balance of cost + quality)",
            "Summarization",
            "Code generation",
            "Analysis and reasoning",
            "Most production RAG workloads",
        ],
        "avoid_for": [
            "Extremely simple tasks where Haiku is sufficient",
        ],
    },
    "claude-opus-4-8": {
        "display_name":     "Claude Opus 4.8",
        "context_window":   200_000,
        "max_output":       32_000,
        "input_cost_per_m":  15.00,
        "output_cost_per_m": 75.00,
        "best_for": [
            "Complex reasoning and planning",
            "Agentic RAG (multi-step agents)",
            "High-stakes document analysis",
            "Architecture-level system design",
        ],
        "avoid_for": [
            "High-volume low-stakes tasks (too expensive)",
            "Simple retrieval-augmented Q&A",
        ],
    },
}


def print_model_comparison():
    """
    Print a side-by-side comparison of all available models.

    WHY compare models:
      In RAG systems, choosing the wrong model tier is a common mistake.
      Using Opus for every call can cost 5-19x more than Haiku/Sonnet.
      Understanding the trade-offs is essential for production design.
    """

    print("\n" + "="*70)
    print("  LLM MODEL COMPARISON — Claude Family (2025)")
    print("="*70)

    # Header row
    print(f"\n{'Model':<28} {'Context':<10} {'In $/M':<10} {'Out $/M':<10}")
    print("─"*60)

    for model_id, info in MODEL_REGISTRY.items():
        print(
            f"{info['display_name']:<28}"          # Model name, left-aligned
            f"{info['context_window']:>8,} tk"      # Context window with comma separator
            f"  ${info['input_cost_per_m']:<8.2f}"  # Input cost
            f"  ${info['output_cost_per_m']:<8.2f}" # Output cost
        )

    print("\n--- Best For ---")
    for model_id, info in MODEL_REGISTRY.items():
        print(f"\n{info['display_name']}:")
        for use_case in info["best_for"]:
            print(f"  ✓ {use_case}")

    print("\n--- Avoid For ---")
    for model_id, info in MODEL_REGISTRY.items():
        print(f"\n{info['display_name']}:")
        for avoid in info["avoid_for"]:
            print(f"  ✗ {avoid}")


def compare_model_responses(question: str):
    """
    Ask the SAME question to all three Claude models.
    Compare:
      - Response quality
      - Response latency
      - Token usage and cost

    WHY this demo:
      Many engineers default to the largest model "to be safe".
      This demo shows that Haiku often handles simple questions just as well
      as Sonnet/Opus at a fraction of the cost.
      For RAG: use the smallest model that meets your quality bar.
    """

    print(f"\n{'='*65}")
    print(f"QUESTION: {question}")
    print(f"{'='*65}")

    for model_id, info in MODEL_REGISTRY.items():
        print(f"\n--- {info['display_name']} ---")

        # WHY time.perf_counter():
        #   Measures wall-clock time of the API call.
        #   perf_counter() is more precise than time.time() for short durations.
        #   Latency matters in RAG — users feel >2 second delays.
        start_time = time.perf_counter()

        response = client.messages.create(
            model=model_id,
            max_tokens=256,

            # WHY temperature=0 for comparison:
            #   We want to compare model CAPABILITY, not randomness.
            #   At temp=0, both runs of the same model give identical output,
            #   isolating the variable to model quality alone.
            temperature=0,

            messages=[{"role": "user", "content": question}]
        )

        elapsed = time.perf_counter() - start_time  # seconds

        answer        = response.content[0].text.strip()
        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # Calculate actual cost of this one call
        cost = (
            (input_tokens  / 1_000_000) * info["input_cost_per_m"] +
            (output_tokens / 1_000_000) * info["output_cost_per_m"]
        )

        print(f"Answer:  {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"Latency: {elapsed:.2f}s | "
              f"Tokens: in={input_tokens}, out={output_tokens} | "
              f"Cost: ${cost:.6f}")


def demonstrate_llm_is_not_a_database():
    """
    CRITICAL CONCEPT: LLMs store patterns, not facts.
    This demo shows the difference between:
      - Questions LLMs answer well (patterns in training data)
      - Questions LLMs fail on (facts requiring lookup/retrieval)

    WHY this matters for RAG:
      Understanding LLM failure modes is the entire motivation for RAG.
      Every failure type shown here has a specific RAG mitigation.
    """

    test_cases = [
        {
            "category": "SUCCEEDS: General knowledge (in training data)",
            "question":  "What is the capital of France?",
        },
        {
            "category": "SUCCEEDS: Reasoning (pattern in training data)",
            "question":  "If all A are B, and all B are C, are all A also C?",
        },
        {
            "category": "FAILS: Recent event (after training cutoff)",
            "question":  "What was the most downloaded app on the App Store last week?",
        },
        {
            "category": "FAILS/HALLUCINATES: Private data (not in training data)",
            "question":  "What does our company's Q3 2025 financial report say about EBITDA?",
        },
        {
            "category": "FAILS/HALLUCINATES: Precise current fact",
            "question":  "What is the exact current price of NVIDIA stock right now?",
        },
    ]

    print("\n" + "="*65)
    print("LLM IS NOT A DATABASE — Failure Mode Demonstration")
    print("Using claude-sonnet-4-6 | temperature=0")
    print("="*65)

    for case in test_cases:
        print(f"\n[{case['category']}]")
        print(f"Q: {case['question']}")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            temperature=0,
            messages=[{"role": "user", "content": case["question"]}]
        )

        answer = response.content[0].text.strip()
        print(f"A: {answer[:300]}{'...' if len(answer) > 300 else ''}")

    print("\n" + "─"*65)
    print("INSIGHT: The 'FAILS' cases are EXACTLY what RAG solves.")
    print("  - Recent events     → retrieve from live data sources")
    print("  - Private data      → retrieve from internal vector database")
    print("  - Precise facts     → retrieve from authoritative sources + cite")
    print("─"*65)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Demo 1: Model comparison table
    print_model_comparison()

    # Demo 2: Same question, three models
    compare_model_responses(
        "Explain what a vector database is in 2 sentences."
    )

    # Demo 3: LLM failure modes (motivation for RAG)
    demonstrate_llm_is_not_a_database()
