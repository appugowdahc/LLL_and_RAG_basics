"""
FILE: 03_token_counter.py
LESSON: Phase 1 - Lesson 1 - What is Generative AI?
TOPIC: Tokens — What they are, why they matter, and how to count/estimate cost

WHAT THIS FILE TEACHES:
  - What a token is (subword unit, NOT a word)
  - Why tokens matter for: context windows, cost, latency
  - How to count tokens BEFORE making an API call (cost estimation)
  - How to estimate cost for different workloads

CONCEPT: What is a Token?
─────────────────────────
LLMs don't read characters or words — they read TOKENS.
A tokenizer splits text into subword chunks:

  "Hello world"     → ["Hello", " world"]             = 2 tokens
  "tokenization"    → ["token", "ization"]             = 2 tokens
  "Supercalifrag.." → ["Super", "cali", "frag"...]     = many tokens
  "a"               → ["a"]                            = 1 token
  " "               → [" "]                            = 1 token

Rule of thumb: 1 token ≈ 0.75 words ≈ 4 characters (English)
               100 tokens ≈ 75 words

Why subwords?
  - Handles rare/unknown words by splitting them
  - Vocabulary stays manageable (~50k-100k tokens)
  - Works across languages without language-specific rules

CONCEPT: Why Tokens Matter for RAG
────────────────────────────────────
Context Window = max tokens the model can process at once
  (prompt + response combined)

In RAG, your prompt = question + retrieved documents + instructions
If retrieved documents are too long → they exceed the context window
→ This drives chunking strategy (Phase 5)

Cost:
  Input tokens:  $3 per million  (claude-sonnet-4-6 approximate)
  Output tokens: $15 per million (claude-sonnet-4-6 approximate)

  1000 calls × 1000 input tokens  = $3.00
  1000 calls × 500  output tokens = $7.50
  Total for 1000 calls:             $10.50
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# ─── Pricing Table (approximate, check docs for current pricing) ──────────────

# WHY a pricing dict:
#   Centralizing pricing makes it easy to update when Anthropic changes rates.
#   In production this would come from a config file or environment variable.
MODEL_PRICING = {
    "claude-opus-4-8": {
        "input_per_million":  15.00,   # dollars per 1M input tokens
        "output_per_million": 75.00,   # dollars per 1M output tokens
    },
    "claude-sonnet-4-6": {
        "input_per_million":   3.00,
        "output_per_million": 15.00,
    },
    "claude-haiku-4-5-20251001": {
        "input_per_million":   0.80,
        "output_per_million":  4.00,
    },
}


def count_tokens(text: str, model: str = "claude-sonnet-4-6") -> int:
    """
    Count tokens in a text string WITHOUT making a generation call.

    WHY count tokens before calling:
      - Validate prompt fits within context window
      - Estimate cost before committing
      - Enforce input limits in production APIs
      - Avoid 'context_length_exceeded' errors

    Args:
        text:  The text to count tokens for.
        model: The Claude model (tokenizer varies slightly by model).

    Returns:
        Number of tokens in the text.
    """

    # WHY client.messages.count_tokens():
    #   Anthropic provides a dedicated endpoint to count tokens.
    #   This does NOT run inference — it only runs the tokenizer.
    #   Much cheaper than a full generation call.
    #   Use this in production to validate inputs before expensive calls.
    result = client.messages.count_tokens(
        model=model,
        messages=[
            {"role": "user", "content": text}
        ]
    )

    # WHY result.input_tokens:
    #   The count_tokens response has the same usage structure as messages.create.
    #   input_tokens = number of tokens in the messages you provided.
    return result.input_tokens


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-sonnet-4-6"
) -> dict:
    """
    Estimate the cost of a single API call.

    Args:
        input_tokens:  Number of tokens in the prompt.
        output_tokens: Expected number of tokens in the response.
        model:         The Claude model to price for.

    Returns:
        dict with input_cost, output_cost, total_cost (in USD)
    """

    # WHY .get() with a fallback:
    #   If an unknown model is passed, we don't crash — we return zeros.
    #   In production: raise an exception instead, so bad configs are caught early.
    pricing = MODEL_PRICING.get(model, {"input_per_million": 0, "output_per_million": 0})

    # Cost formula: (tokens / 1,000,000) × price_per_million
    input_cost  = (input_tokens  / 1_000_000) * pricing["input_per_million"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]
    total_cost  = input_cost + output_cost

    return {
        "input_cost_usd":  round(input_cost,  6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd":  round(total_cost,  6),
    }


def analyze_text_tokenization(text: str):
    """
    Analyze how a piece of text tokenizes and what it costs.

    This is useful for:
    - Understanding why certain prompts are more expensive
    - Optimizing prompt length in production
    - Teaching token intuition
    """

    print(f"\n{'─'*55}")
    print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"{'─'*55}")

    # Character and word counts (for comparison with token count)
    char_count = len(text)
    word_count = len(text.split())

    # WHY count tokens for all 3 models:
    #   The same text may tokenize slightly differently across models.
    #   Demonstrating this builds awareness that token counts are model-specific.
    for model in MODEL_PRICING:
        token_count = count_tokens(text, model)
        tokens_per_word = round(token_count / word_count, 2) if word_count > 0 else 0

        print(f"\n  Model: {model}")
        print(f"  Characters: {char_count}  |  Words: {word_count}  |  Tokens: {token_count}")
        print(f"  Tokens/word ratio: {tokens_per_word}  (rule of thumb: ~1.33)")

        # Cost for 1000 calls with this input and 500 output tokens
        cost = estimate_cost(token_count, 500, model)
        cost_1000_calls = round(cost["total_cost_usd"] * 1000, 4)
        print(f"  Cost per call: ${cost['total_cost_usd']:.6f}  |  Cost × 1000 calls: ${cost_1000_calls}")


def rag_context_window_demo():
    """
    Demonstrate how RAG prompts consume the context window.

    In RAG, prompt = system instruction + retrieved docs + user question
    Understanding this breakdown helps you design chunking strategy.
    """

    print("\n" + "="*55)
    print("RAG PROMPT ANATOMY: Token Breakdown")
    print("="*55)

    system_instruction = """You are a helpful enterprise knowledge assistant.
Answer questions using ONLY the provided context documents.
If the answer is not in the context, respond with: "I don't have that information."
Always cite which document your answer comes from.
Be concise and professional."""

    retrieved_doc_1 = """[Document 1 - Source: company_policy.pdf, Page 3]
The vacation policy at Criterion Networks allows full-time employees to accrue
2.5 vacation days per month, up to a maximum of 30 days. Unused vacation days
may be carried over to the following year, but cannot exceed the 30-day cap.
Employees must submit vacation requests at least 2 weeks in advance via the HR portal."""

    retrieved_doc_2 = """[Document 2 - Source: employee_handbook.pdf, Page 12]
Remote work policy: Employees may work remotely up to 3 days per week with
manager approval. Core hours are 10am-3pm in the employee's local timezone.
All remote employees must have a stable internet connection and a dedicated
workspace free from distractions during client calls."""

    user_question = "How many vacation days do I get per month and can I carry them over?"

    # Build the full RAG prompt (what gets sent to the LLM)
    full_prompt = f"""System: {system_instruction}

Context Documents:
{retrieved_doc_1}

{retrieved_doc_2}

User Question: {user_question}

Answer:"""

    # WHY break down token counts by component:
    #   In production you want to know WHICH part of your prompt is expensive.
    #   System prompts, retrieved docs, and user questions all have different
    #   optimization strategies when you need to reduce context length.
    components = {
        "System instruction":  system_instruction,
        "Retrieved document 1": retrieved_doc_1,
        "Retrieved document 2": retrieved_doc_2,
        "User question":        user_question,
        "FULL PROMPT (total)":  full_prompt,
    }

    MODEL = "claude-sonnet-4-6"
    CONTEXT_WINDOW = 200_000  # Claude Sonnet's context window in tokens

    total_input_tokens = 0
    for component_name, component_text in components.items():
        tokens = count_tokens(component_text, MODEL)
        pct_of_context = round((tokens / CONTEXT_WINDOW) * 100, 4)
        if component_name == "FULL PROMPT (total)":
            total_input_tokens = tokens
            print(f"\n  {'─'*50}")
        print(f"  {component_name:<25}: {tokens:>5} tokens  ({pct_of_context}% of context window)")

    # Show remaining context window available for response
    # WHY this matters:
    #   If input_tokens + expected_output_tokens > context_window → API error.
    #   In production: enforce a budget per component so you never hit this.
    remaining = CONTEXT_WINDOW - total_input_tokens
    print(f"\n  Context window:   {CONTEXT_WINDOW:>7,} tokens")
    print(f"  Used by prompt:   {total_input_tokens:>7,} tokens")
    print(f"  Available for LLM response: {remaining:>7,} tokens")

    cost = estimate_cost(total_input_tokens, 300, MODEL)
    print(f"\n  Estimated cost per RAG call: ${cost['total_cost_usd']:.6f}")
    print(f"  Estimated cost × 10,000 daily calls: ${cost['total_cost_usd'] * 10000:.4f}/day")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("="*55)
    print("DEMO 1: Token counts for different text types")
    print("="*55)

    sample_texts = [
        "Hello world",                        # Simple, few tokens
        "Retrieval-Augmented Generation",     # Technical term — may split into subwords
        "supercalifragilisticexpialidocious",  # Long rare word — many subwords
        "def calculate_cosine_similarity(vec_a, vec_b):  # compute similarity",  # code
        "私はAIエンジニアです。",              # Non-English — typically more tokens per word
    ]

    for text in sample_texts:
        analyze_text_tokenization(text)

    print("\n")
    print("="*55)
    print("DEMO 2: RAG Context Window Breakdown")
    print("="*55)
    rag_context_window_demo()
