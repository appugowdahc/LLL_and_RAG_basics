"""
FILE: 04_llm_capabilities_demo.py
LESSON: Phase 1 - Lesson 2 - What are LLMs?
TOPIC: LLM Capabilities — Zero-shot, Few-shot, Chain-of-Thought, Structured Output

WHAT THIS FILE TEACHES:
  - Zero-shot prompting: ask without examples
  - Few-shot prompting: teach by example in the prompt
  - Chain-of-Thought (CoT): make the model reason step-by-step
  - Structured output: force JSON/schema output (critical for RAG pipelines)
  - System prompts: define the model's role and behavior

WHY THESE MATTER FOR RAG:
  - Structured output: RAG pipelines need JSON responses to parse programmatically
  - Few-shot: teach the model the exact output format you need
  - CoT: improves accuracy for multi-hop reasoning over retrieved documents
  - System prompts: define the "grounding" rules (answer only from context)
"""

import os
import json
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# ─── 1. Zero-Shot Prompting ───────────────────────────────────────────────────

def zero_shot_demo():
    """
    Zero-shot: ask the model to perform a task with NO examples.
    The model uses patterns learned during pre-training.

    WHY zero-shot works:
      LLMs have seen millions of examples of every task during pre-training.
      "Classify sentiment" doesn't need examples — the model already "knows"
      what sentiment classification means from training data.
    """

    print("\n" + "="*60)
    print("1. ZERO-SHOT PROMPTING")
    print("="*60)

    tasks = [
        {
            "task":   "Sentiment classification",
            "prompt": "Classify the sentiment of this review as POSITIVE, NEGATIVE, or NEUTRAL.\n\nReview: 'The product arrived on time and works exactly as described.'\n\nSentiment:"
        },
        {
            "task":   "Language detection",
            "prompt": "Detect the language of this text. Reply with just the language name.\n\nText: 'Bonjour, comment allez-vous?'\n\nLanguage:"
        },
        {
            "task":   "Entity extraction",
            "prompt": "Extract all company names from this text. Reply with a comma-separated list.\n\nText: 'Criterion Networks partners with Cisco, Microsoft, and AWS to deliver enterprise solutions.'\n\nCompanies:"
        },
    ]

    for case in tasks:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            temperature=0,  # Deterministic for classification tasks
            messages=[{"role": "user", "content": case["prompt"]}]
        )

        print(f"\n  Task: {case['task']}")
        print(f"  Output: {response.content[0].text.strip()}")


# ─── 2. Few-Shot Prompting ────────────────────────────────────────────────────

def few_shot_demo():
    """
    Few-shot: provide 2-5 (input, output) examples in the prompt.
    The model learns the PATTERN from your examples and applies it.

    WHY few-shot matters for RAG:
      When you need a very specific output format, few-shot examples
      are more reliable than lengthy instructions alone.
      Example: "Extract entities as JSON" + 2 examples → consistent JSON output.
    """

    print("\n" + "="*60)
    print("2. FEW-SHOT PROMPTING")
    print("="*60)

    # Task: Extract product info from unstructured text in a specific format
    # WHY few-shot here:
    #   The exact JSON key names and structure are arbitrary — you can't describe
    #   them purely in natural language as reliably as showing examples.
    few_shot_prompt = """Extract product information from text. Output as: NAME | PRICE | CATEGORY

Examples:
Text: "The Sony WH-1000XM5 headphones are available for $299.99 in the electronics section."
Output: Sony WH-1000XM5 | $299.99 | Electronics

Text: "Pick up a bag of organic coffee beans for just $14.50 in our grocery aisle."
Output: Organic Coffee Beans | $14.50 | Grocery

Text: "The ergonomic office chair Model X200 retails at $549 and is found in our furniture department."
Output: Model X200 Office Chair | $549 | Furniture

Now extract from:
Text: "Our bestselling laptop stand, the ProDesk Elite, is priced at $79.99 in the accessories section."
Output:"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=50,
        temperature=0,
        messages=[{"role": "user", "content": few_shot_prompt}]
    )

    print(f"\n  Result: {response.content[0].text.strip()}")
    print(f"\n  WHY IT WORKS: The model learned NAME | PRICE | CATEGORY format from examples.")
    print(f"  No format description needed — examples teach better than instructions.")


# ─── 3. Chain-of-Thought (CoT) ────────────────────────────────────────────────

def chain_of_thought_demo():
    """
    Chain-of-Thought: ask the model to reason step-by-step before answering.
    The phrase "Let's think step by step" or "Show your reasoning" triggers CoT.

    WHY CoT improves accuracy:
      LLMs generate tokens autoregressively — the reasoning steps become
      "working memory" that the model can reference when generating the final answer.
      Without CoT, the model tries to jump from question to answer in one step,
      increasing the chance of error for multi-step problems.

    WHY CoT matters for RAG:
      Multi-hop queries ("Find documents about X, then use those to answer Y")
      require multi-step reasoning. CoT improves accuracy on these queries.
    """

    print("\n" + "="*60)
    print("3. CHAIN-OF-THOUGHT PROMPTING")
    print("="*60)

    problem = """A RAG system processes 10,000 user queries per day.
Each query retrieves 5 documents averaging 500 tokens each.
The system prompt uses 200 tokens.
The user query averages 50 tokens.
The response averages 300 tokens.

Claude Sonnet costs $3/million input tokens and $15/million output tokens.

What is the total daily cost of running this RAG system?"""

    # WITHOUT CoT
    print("\n  Without Chain-of-Thought:")
    response_no_cot = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": problem + "\n\nAnswer:"}]
    )
    print(f"  {response_no_cot.content[0].text.strip()}")

    # WITH CoT
    # WHY "Think step by step":
    #   This phrase activates the model's learned CoT behavior.
    #   The model will break the problem into sub-steps before computing the final answer.
    print("\n  With Chain-of-Thought ('Think step by step'):")
    response_cot = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        temperature=0,
        messages=[{
            "role": "user",
            "content": problem + "\n\nThink step by step, then give the final answer."
        }]
    )
    print(f"  {response_cot.content[0].text.strip()}")


# ─── 4. Structured Output (JSON) ─────────────────────────────────────────────

def structured_output_demo():
    """
    Force the LLM to output valid JSON that can be parsed programmatically.

    WHY structured output is CRITICAL for RAG:
      In a RAG pipeline, the LLM response is not shown to a human directly.
      It's PARSED by downstream code:
        - Query rewriting → expects JSON with "rewritten_queries": [...]
        - Relevance scoring → expects {"score": 0.87, "reasoning": "..."}
        - Entity extraction → expects {"entities": [...], "relations": [...]}
      If the LLM outputs free text instead of JSON, the pipeline crashes.

    APPROACHES (in order of reliability):
      1. Instruct to output JSON (works 90% of time)
      2. Show JSON example in few-shot (works 95%)
      3. Use Anthropic's tool_use for guaranteed structure (works 99.9%)
    """

    print("\n" + "="*60)
    print("4. STRUCTURED OUTPUT")
    print("="*60)

    # RAG use case: Query Analysis
    # Before retrieving, a RAG system analyzes the query to:
    #   - Classify query type (factual, analytical, procedural)
    #   - Extract key entities to search for
    #   - Generate sub-queries for multi-hop retrieval
    user_query = "What are the differences between Pinecone, Weaviate, and Qdrant for production RAG?"

    # ── Approach 1: Instruct to output JSON ──────────────────────────────────
    print("\n  Approach 1: Instruct JSON output")

    json_instruction_prompt = f"""Analyze this user query for a RAG system.
Output a JSON object with this EXACT structure:
{{
  "query_type": "factual|analytical|procedural|comparative",
  "entities": ["list of key technical terms to search for"],
  "sub_queries": ["list of 2-3 simpler sub-questions to answer"],
  "complexity": "simple|moderate|complex"
}}

User query: {user_query}

Output ONLY the JSON, no other text:"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": json_instruction_prompt}]
    )

    raw_output = response.content[0].text.strip()

    # WHY try/except json.loads():
    #   Even with instructions, LLMs occasionally add "```json" wrappers
    #   or extra explanation text. Always validate JSON in production.
    try:
        # WHY strip markdown fences:
        #   Models sometimes wrap JSON in ```json ... ``` code blocks.
        #   Strip these before parsing.
        clean = raw_output.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        print(f"  ✓ Valid JSON parsed successfully")
        print(f"  Query type:  {parsed.get('query_type')}")
        print(f"  Entities:    {parsed.get('entities')}")
        print(f"  Sub-queries: {parsed.get('sub_queries')}")
        print(f"  Complexity:  {parsed.get('complexity')}")
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parse failed: {e}")
        print(f"  Raw output: {raw_output}")

    # ── Approach 2: Tool Use (Guaranteed Structure) ───────────────────────────
    # WHY tool_use:
    #   When you define a "tool" (function schema) and ask the model to call it,
    #   Anthropic GUARANTEES the output matches your schema.
    #   The model CANNOT output free text when forced to call a tool.
    #   This is the production-safe way to get structured outputs.
    print("\n  Approach 2: Tool Use (Guaranteed valid JSON schema)")

    # WHY tools=[]:
    #   We define a fake "tool" (function) that the model must "call".
    #   The tool's input_schema defines exactly what JSON keys are required.
    #   The model must output JSON matching this schema — or the API rejects it.
    tools = [
        {
            "name": "analyze_query",
            "description": "Analyze a user query for RAG retrieval planning",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["factual", "analytical", "procedural", "comparative"],
                        "description": "The type of query"
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key technical terms/entities to search for"
                    },
                    "sub_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Simpler sub-questions to answer independently"
                    },
                    "complexity": {
                        "type": "string",
                        "enum": ["simple", "moderate", "complex"]
                    }
                },
                "required": ["query_type", "entities", "sub_queries", "complexity"]
            }
        }
    ]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        temperature=0,

        # WHY tool_choice={"type": "tool", "name": "analyze_query"}:
        #   Forces the model to call THIS specific tool.
        #   Without this, the model might respond in text instead of calling the tool.
        tool_choice={"type": "tool", "name": "analyze_query"},

        tools=tools,
        messages=[{"role": "user", "content": f"Analyze this query: {user_query}"}]
    )

    # WHY response.content[0].input (not .text):
    #   When a tool is called, the content block type is "tool_use", not "text".
    #   The .input attribute contains the validated JSON matching our schema.
    tool_result = response.content[0].input

    print(f"  ✓ Guaranteed valid structure (tool_use)")
    print(f"  Query type:  {tool_result['query_type']}")
    print(f"  Entities:    {tool_result['entities']}")
    print(f"  Sub-queries: {tool_result['sub_queries']}")
    print(f"  Complexity:  {tool_result['complexity']}")
    print(f"\n  → Use tool_use in production RAG pipelines for reliable JSON.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    zero_shot_demo()
    few_shot_demo()
    chain_of_thought_demo()
    structured_output_demo()
