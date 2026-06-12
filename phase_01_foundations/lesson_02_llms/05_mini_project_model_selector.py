"""
FILE: 05_mini_project_model_selector.py
LESSON: Phase 1 - Lesson 2 - What are LLMs?
TOPIC: Mini-Project — Intelligent Model Selector + RAG-Ready Task Router

WHAT THIS FILE TEACHES:
  - How to classify incoming queries and route them to the right model
  - Why model selection is a cost optimization strategy in production RAG
  - How to build a reusable router that every future lesson will extend
  - Structured output in a real pipeline (query classification → model selection)

PRODUCTION INSIGHT:
  A naive RAG system sends EVERY query to the most expensive model (Opus).
  A smart RAG system classifies the query first, then routes:
    - Simple factual Q&A     → Haiku   (cheap, fast)
    - Summarization/Analysis → Sonnet  (balanced)
    - Complex reasoning/Agents → Opus  (powerful)

  At 10,000 queries/day:
    All Opus:   $0.09 × 10,000 = $900/day
    Smart route: ~$0.004 average × 10,000 = $40/day  (95% savings)
"""

import os
import json
import time
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# ─── Data Structures ──────────────────────────────────────────────────────────

# WHY @dataclass:
#   Creates a clean immutable value object with auto-generated __repr__.
#   Better than a plain dict — fields are named, typed, and self-documenting.
@dataclass
class TaskClassification:
    """Result of classifying a user query."""
    task_type:       str    # "simple_qa", "analysis", "complex_reasoning", "creative"
    complexity:      str    # "low", "medium", "high"
    recommended_model: str  # model id
    reasoning:       str    # why this classification was made
    estimated_output_tokens: int  # expected response length


@dataclass
class ModelResponse:
    """Result of a model call with full metadata."""
    answer:        str
    model_used:    str
    task_type:     str
    input_tokens:  int
    output_tokens: int
    latency_ms:    float
    cost_usd:      float


# ─── Cost Table ───────────────────────────────────────────────────────────────

# WHY define cost per model:
#   The router needs to report cost for each routed call.
#   Centralizing pricing here makes updates trivial.
MODEL_COSTS = {
    "claude-haiku-4-5-20251001": {"in": 0.80,  "out": 4.00},   # per million tokens
    "claude-sonnet-4-6":         {"in": 3.00,  "out": 15.00},
    "claude-opus-4-8":           {"in": 15.00, "out": 75.00},
}

# WHY define routing rules:
#   The classifier's output (task_type) maps to a specific model.
#   This table makes routing logic explicit and easy to tune.
ROUTING_TABLE = {
    "simple_qa":        "claude-haiku-4-5-20251001",
    "extraction":       "claude-haiku-4-5-20251001",
    "summarization":    "claude-sonnet-4-6",
    "analysis":         "claude-sonnet-4-6",
    "complex_reasoning":"claude-opus-4-8",
    "creative":         "claude-sonnet-4-6",
}


# ─── Query Classifier ─────────────────────────────────────────────────────────

def classify_query(query: str) -> TaskClassification:
    """
    Use an LLM to classify a user query and determine the optimal model.

    WHY use an LLM to classify queries:
      Hand-written rules ("if 'summarize' in query → sonnet") are brittle.
      An LLM classifier handles nuanced cases:
        "What is 2+2?" → simple_qa (haiku)
        "What is the strategic impact of quantum computing on RSA encryption?" → analysis (sonnet)

    WHY use HAIKU for the classifier itself:
      The classification task is simple → use the cheapest model.
      The savings from smart routing far outweigh the classifier's cost.

    Returns:
        TaskClassification with recommended model and reasoning.
    """

    # WHY tool_use for classification:
    #   We need structured output to programmatically extract task_type.
    #   Tool use guarantees the output matches our schema.
    tools = [
        {
            "name": "classify_task",
            "description": "Classify a user query to determine optimal LLM model",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": ["simple_qa", "extraction", "summarization",
                                 "analysis", "complex_reasoning", "creative"],
                        "description": "The category of task the query requires"
                    },
                    "complexity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "How complex the reasoning required is"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of the classification"
                    },
                    "estimated_output_tokens": {
                        "type": "integer",
                        "description": "Estimated number of output tokens needed (50-2000)"
                    }
                },
                "required": ["task_type", "complexity", "reasoning", "estimated_output_tokens"]
            }
        }
    ]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Cheapest model for cheap classification task
        max_tokens=200,
        temperature=0,    # Deterministic classification — no creativity needed
        tool_choice={"type": "tool", "name": "classify_task"},
        tools=tools,
        messages=[
            {
                "role": "user",
                "content": f"Classify this query for LLM routing:\n\n{query}"
            }
        ]
    )

    # Extract the structured result from the tool_use content block
    result = response.content[0].input

    # Look up the recommended model from the routing table
    recommended = ROUTING_TABLE.get(result["task_type"], "claude-sonnet-4-6")

    return TaskClassification(
        task_type=result["task_type"],
        complexity=result["complexity"],
        reasoning=result["reasoning"],
        recommended_model=recommended,
        estimated_output_tokens=result["estimated_output_tokens"]
    )


# ─── Task Router ──────────────────────────────────────────────────────────────

def route_and_execute(query: str, verbose: bool = True) -> ModelResponse:
    """
    Classify the query → route to optimal model → execute → return full metadata.

    This is the CORE pattern of a production RAG router.
    In a real system, this sits in front of your retriever:
      1. Classify query
      2. Route to model
      3. Retrieve (Phase 3-7 will build this)
      4. Generate

    Args:
        query:   The user's question.
        verbose: If True, print classification details.

    Returns:
        ModelResponse with answer + cost/latency metadata.
    """

    # Step 1: Classify
    classification = classify_query(query)

    if verbose:
        print(f"\n  Classification:")
        print(f"    Task type:   {classification.task_type}")
        print(f"    Complexity:  {classification.complexity}")
        print(f"    Reasoning:   {classification.reasoning}")
        print(f"    Routed to:   {classification.recommended_model}")

    # Step 2: Execute with the recommended model
    start = time.perf_counter()

    response = client.messages.create(
        model=classification.recommended_model,
        max_tokens=classification.estimated_output_tokens + 100,  # +100 buffer
        temperature=0,
        messages=[{"role": "user", "content": query}]
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Step 3: Calculate cost
    costs = MODEL_COSTS[classification.recommended_model]
    cost = (
        (response.usage.input_tokens  / 1_000_000) * costs["in"] +
        (response.usage.output_tokens / 1_000_000) * costs["out"]
    )

    return ModelResponse(
        answer=response.content[0].text.strip(),
        model_used=classification.recommended_model,
        task_type=classification.task_type,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=round(elapsed_ms, 1),
        cost_usd=round(cost, 6),
    )


# ─── Session Cost Comparator ──────────────────────────────────────────────────

def compare_routing_vs_always_opus(queries: list[str]):
    """
    Run all queries through the smart router AND through Opus directly.
    Compare total cost and show the savings.

    WHY this comparison:
      Demonstrates the economic case for model routing.
      In production: the savings justify building a routing layer.
    """

    print("\n" + "="*60)
    print("COST COMPARISON: Smart Routing vs Always Opus")
    print("="*60)

    smart_total_cost = 0.0
    opus_total_cost  = 0.0

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}]: {query[:70]}{'...' if len(query)>70 else ''}")

        # Smart routing
        smart_result = route_and_execute(query, verbose=True)
        smart_total_cost += smart_result.cost_usd

        # What Opus would cost for the same query
        opus_resp = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": query}]
        )
        opus_costs = MODEL_COSTS["claude-opus-4-8"]
        opus_cost = (
            (opus_resp.usage.input_tokens  / 1_000_000) * opus_costs["in"] +
            (opus_resp.usage.output_tokens / 1_000_000) * opus_costs["out"]
        )
        opus_total_cost += opus_cost

        print(f"\n  Smart router cost: ${smart_result.cost_usd:.6f} ({smart_result.model_used})")
        print(f"  Opus always cost:  ${opus_cost:.6f} (claude-opus-4-8)")
        savings = ((opus_cost - smart_result.cost_usd) / opus_cost * 100) if opus_cost > 0 else 0
        print(f"  Savings this query: {savings:.1f}%")

    print("\n" + "="*60)
    print(f"TOTAL across {len(queries)} queries:")
    print(f"  Smart router: ${smart_total_cost:.6f}")
    print(f"  Always Opus:  ${opus_total_cost:.6f}")
    total_savings = ((opus_total_cost - smart_total_cost) / opus_total_cost * 100) if opus_total_cost > 0 else 0
    print(f"  Total savings: {total_savings:.1f}%")
    print(f"\n  At 10,000 queries/day:")
    daily_smart = smart_total_cost / len(queries) * 10_000
    daily_opus  = opus_total_cost  / len(queries) * 10_000
    print(f"  Smart router: ${daily_smart:.2f}/day")
    print(f"  Always Opus:  ${daily_opus:.2f}/day")
    print(f"  Annual savings: ${(daily_opus - daily_smart) * 365:,.0f}/year")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Diverse set of queries to demonstrate routing
    test_queries = [
        "What is a vector database?",                                      # simple_qa → haiku
        "What is the capital of Japan?",                                   # simple_qa → haiku
        "Summarize the key differences between HNSW and IVF indexing.",    # summarization → sonnet
        "Analyze the trade-offs between sparse and dense retrieval in RAG systems and recommend an architecture for a legal document search system with 10M documents.", # analysis → sonnet
        "Design a multi-agent RAG system that can autonomously decompose complex research questions, retrieve from multiple specialized corpora, cross-validate findings, and synthesize a cited report.", # complex → opus
    ]

    compare_routing_vs_always_opus(test_queries)
