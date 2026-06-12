"""
FILE: 03_context_window_limits.py
LESSON: Phase 1 - Lesson 2 - What are LLMs?
TOPIC: Context Window — The fundamental constraint driving all of RAG

WHAT THIS FILE TEACHES:
  - What the context window is and why it's a hard limit
  - How to detect when input exceeds the context window
  - How context window usage breaks down in a RAG prompt
  - Why context window constraints drive chunking strategy (Phase 5)

CONCEPT: Context Window
────────────────────────
The context window is the MAXIMUM NUMBER OF TOKENS the model can
"see" at once during a single inference call.

This includes:
  - System prompt
  - All previous messages (conversation history)
  - Retrieved documents (in RAG)
  - The current user message
  - The model's response (counts against the window too)

Formula:
  total_tokens = system_tokens + history_tokens + retrieved_tokens
               + user_query_tokens + response_tokens
               ≤ context_window_size

If total_tokens > context_window → API ERROR (context_length_exceeded)

WHY THIS DRIVES RAG ARCHITECTURE:
  A legal document = 500,000 words = ~667,000 tokens
  Claude's context window = 200,000 tokens
  → The document CANNOT fit in one call
  → Solution: Chunk the document → embed chunks → retrieve relevant chunks only
  → Only the TOP-K relevant chunks (e.g. 5 × 500 tokens = 2,500 tokens) are injected
  → This is the entire motivation for Phase 5 (Chunking) and Phase 3 (Vector DBs)
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# ─── Context Window Sizes ─────────────────────────────────────────────────────

# WHY store these locally:
#   The API doesn't dynamically return context window sizes per model.
#   You need to know these to build guard rails in production.
#   Always check Anthropic docs for updates — these change with new model releases.
CONTEXT_WINDOWS = {
    "claude-haiku-4-5-20251001": 200_000,
    "claude-sonnet-4-6":         200_000,
    "claude-opus-4-8":           200_000,
}

# Context window budget allocation for a typical RAG system
# WHY allocate percentages:
#   You can't use 100% of the context for retrieved documents.
#   System prompts, user queries, and response space also consume tokens.
#   Define explicit budgets per component so you never hit the hard limit.
RAG_CONTEXT_BUDGET = {
    "system_prompt":     0.05,   # 5%  of context window
    "retrieved_docs":    0.60,   # 60% of context window (the main payload)
    "conversation_hist": 0.15,   # 15% for multi-turn history
    "user_query":        0.05,   # 5%  for the current question
    "response_reserve":  0.15,   # 15% reserved for the LLM's response
}
# Sum = 100% → never exceed the context window


def calculate_rag_token_budget(model: str = "claude-sonnet-4-6") -> dict:
    """
    Calculate how many tokens are available for each RAG component.

    Args:
        model: The Claude model to calculate budgets for.

    Returns:
        dict mapping component name → token budget
    """

    # WHY .get() with a default:
    #   If an unknown model is passed, fall back to a conservative 100k window.
    #   Production code should raise an error here instead — fail fast on bad config.
    window = CONTEXT_WINDOWS.get(model, 100_000)

    budget = {}
    for component, fraction in RAG_CONTEXT_BUDGET.items():
        # WHY int():
        #   Tokens are integers. Fractional tokens don't exist.
        #   int() truncates (floor), which is safe — always stay under budget.
        budget[component] = int(window * fraction)

    budget["total_context_window"] = window
    budget["model"] = model
    return budget


def demonstrate_context_budget():
    """
    Print the token budget for a RAG system using Claude Sonnet.
    Shows engineers exactly how to allocate context window space.
    """

    print("\n" + "="*60)
    print("RAG TOKEN BUDGET ALLOCATION")
    print("="*60)

    for model in CONTEXT_WINDOWS:
        budget = calculate_rag_token_budget(model)

        print(f"\nModel: {model}")
        print(f"Context Window: {budget['total_context_window']:,} tokens")
        print(f"{'─'*45}")

        for component, token_count in budget.items():
            if component in ("total_context_window", "model"):
                continue
            fraction = RAG_CONTEXT_BUDGET[component]
            print(f"  {component:<22}: {token_count:>7,} tokens  ({fraction*100:.0f}%)")


def simulate_context_overflow():
    """
    Demonstrate what happens when you try to send MORE tokens
    than the model's context window allows.

    WHY simulate this:
      Context overflow is a silent failure in many RAG systems.
      The API throws an error — but if you don't handle it,
      your RAG system crashes under heavy load.
      This demo shows the error and the correct handling pattern.
    """

    print("\n" + "="*60)
    print("CONTEXT WINDOW OVERFLOW SIMULATION")
    print("="*60)

    # WHY count tokens BEFORE calling:
    #   Never send a request you know will fail.
    #   count_tokens() is cheap (no inference) — always validate first.
    test_inputs = [
        {
            "label": "NORMAL: Fits in context window",
            # ~25 tokens
            "text": "What is a Large Language Model? Explain briefly."
        },
        {
            "label": "LARGE: Simulate a big document",
            # Repeat a sentence to create a large input (~5,000 tokens)
            "text": "This is a sentence from a very large document. " * 400
        },
        {
            "label": "EXTREME: Simulate exceeding context window",
            # Repeat to create ~250,000 tokens (exceeds 200k window)
            "text": "Simulating a very large corpus of text data. " * 10_000
        },
    ]

    MODEL = "claude-sonnet-4-6"
    WINDOW = CONTEXT_WINDOWS[MODEL]

    for case in test_inputs:
        print(f"\n  [{case['label']}]")

        # Step 1: Count tokens BEFORE sending (cheap, no inference)
        token_count = client.messages.count_tokens(
            model=MODEL,
            messages=[{"role": "user", "content": case["text"]}]
        ).input_tokens

        print(f"  Token count: {token_count:,} / {WINDOW:,} window")
        print(f"  Usage: {token_count/WINDOW*100:.1f}% of context window")

        # Step 2: Check if it fits BEFORE calling the expensive API
        # WHY this check:
        #   If we skip this and call messages.create() with too many tokens,
        #   we get an API error AND waste the latency of the failed call.
        #   Checking first is both cheaper and gives a better error message.
        if token_count > WINDOW:
            print(f"  ⛔ BLOCKED: Input exceeds context window ({token_count:,} > {WINDOW:,})")
            print(f"  RAG FIX: Chunk the document and retrieve only top-K relevant chunks.")
            continue

        # Step 3: Only call if it fits
        # WHY max_tokens=50:
        #   We only want to confirm the call works — not generate a full response.
        #   Small max_tokens = fast, cheap validation call.
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=50,
                temperature=0,
                messages=[{"role": "user", "content": case["text"]}]
            )
            print(f"  ✓ SUCCEEDED: Response received ({response.usage.output_tokens} output tokens)")

        # WHY catch BadRequestError:
        #   Anthropic raises anthropic.BadRequestError when context is exceeded.
        #   This is different from RateLimitError or AuthenticationError.
        #   Specific catches let you handle each failure mode appropriately.
        except anthropic.BadRequestError as e:
            print(f"  ✗ API ERROR: {e}")
            print(f"  RAG FIX: This is exactly why chunking + retrieval exists.")


def show_conversation_history_cost():
    """
    Demonstrate how multi-turn conversation history consumes context window.

    WHY this matters for RAG chatbots:
      Every message you add to the history costs input tokens.
      A 100-turn conversation might use 50,000 tokens just for history —
      leaving only 150,000 tokens for retrieved documents.
      Production RAG chatbots use sliding window history or summarization.
    """

    print("\n" + "="*60)
    print("CONVERSATION HISTORY: Token Cost Growth")
    print("="*60)

    MODEL  = "claude-sonnet-4-6"
    WINDOW = CONTEXT_WINDOWS[MODEL]

    # Simulate a growing conversation
    # WHY a list of dicts:
    #   The messages API requires conversation history as a list of role/content dicts.
    #   Appending to this list simulates a real multi-turn session.
    conversation_history = []

    turns = [
        ("user",      "What is RAG in AI?"),
        ("assistant", "RAG stands for Retrieval-Augmented Generation. It enhances LLMs by retrieving relevant documents before generating answers."),
        ("user",      "How does it differ from fine-tuning?"),
        ("assistant", "Fine-tuning bakes knowledge into model weights permanently. RAG retrieves knowledge dynamically at inference time, making it updatable without retraining."),
        ("user",      "What are the main components of a RAG system?"),
        ("assistant", "Three main components: (1) Document ingestion and chunking, (2) Embedding and vector storage, (3) Retrieval and generation pipeline."),
        ("user",      "What vector databases should I use?"),
        ("assistant", "For production: Pinecone (managed, scalable), Weaviate (open-source, rich features), Qdrant (high performance). For development: ChromaDB (lightweight), FAISS (in-memory)."),
        ("user",      "How do I evaluate my RAG system?"),
    ]

    for i, (role, content) in enumerate(turns):
        # Append this turn to the running conversation
        conversation_history.append({"role": role, "content": content})

        # Count current total tokens in the conversation
        token_count = client.messages.count_tokens(
            model=MODEL,
            messages=conversation_history
        ).input_tokens

        pct_used = token_count / WINDOW * 100

        # WHY print after EACH turn:
        #   Shows cumulative growth — engineers see how quickly history consumes context.
        print(f"\n  Turn {i+1:2d} | Role: {role:<10} | "
              f"Cumulative tokens: {token_count:>6,} / {WINDOW:,} ({pct_used:.3f}%)")

    print(f"\n  After {len(turns)} turns: {token_count:,} tokens used for history alone.")
    remaining_for_docs = WINDOW - token_count - int(WINDOW * 0.15)  # reserve 15% for response
    print(f"  Tokens remaining for retrieved documents: {remaining_for_docs:,}")
    print(f"\n  PRODUCTION SOLUTION:")
    print(f"  → Keep only last N turns (sliding window)")
    print(f"  → Or summarize old history to compress it")
    print(f"  → Both are implemented in Phase 10 (Agentic RAG Memory)")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demonstrate_context_budget()
    simulate_context_overflow()
    show_conversation_history_cost()
