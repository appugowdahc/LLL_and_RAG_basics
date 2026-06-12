"""
FILE: 04_autoregressive_decoding.py
LESSON: Phase 1 - Lesson 3 - How LLMs Work Internally
TOPIC: Autoregressive Decoding — Token-by-token generation + Streaming

WHAT THIS FILE TEACHES:
  - What autoregressive decoding means step-by-step
  - How sampling strategies work (greedy, top-k, top-p / nucleus)
  - How to implement STREAMING output (token-by-token in real-time)
  - Why streaming improves perceived latency in RAG applications
  - How to track time-to-first-token (TTFT) — key production metric

CONCEPT: Autoregressive Decoding
──────────────────────────────────
The model generates ONE token at a time. After generating each token,
that token is appended to the input and the model runs again:

  Input:   "The capital of France is"
  Step 1:  model(["The", "capital", "of", "France", "is"])
           → probs: {"Paris": 0.94, "Lyon": 0.01, ...}
           → sample: "Paris"

  Step 2:  model(["The", "capital", "of", "France", "is", "Paris"])
           → probs: {".": 0.78, ",": 0.12, " and": 0.05}
           → sample: "."

  Step 3:  model([..., "Paris", "."])
           → probs: {"<|endoftext|>": 0.91, ...}
           → stop (end-of-text token)

  Final: "The capital of France is Paris."

WHY THIS IS SLOW:
  Each step requires a FULL forward pass through ALL transformer layers.
  100 output tokens = 100 forward passes.
  This is why long responses take longer to generate than short ones.

IMPLICATIONS FOR RAG:
  - Shorter responses → lower latency → better UX
  - Streaming shows tokens as they generate → feels faster to users
  - Time-to-first-token (TTFT) matters more than total generation time
  - In RAG: short, precise answers are better than verbose ones

INSTALL:
  pip install anthropic python-dotenv
"""

import os
import time
import sys
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# ─── Greedy vs Sampling ───────────────────────────────────────────────────────

def explain_sampling_strategies():
    """
    Demonstrate different token sampling strategies conceptually.

    SAMPLING STRATEGIES:
    ──────────────────────
    1. GREEDY (temperature=0):
       Always pick the highest probability token.
       → Deterministic, but can get stuck in repetitive loops.
       → Use for: extraction, classification, factual Q&A in RAG.

    2. TOP-K SAMPLING:
       Only sample from the top-K highest probability tokens.
       Set the rest to 0. Then sample from this truncated distribution.
       → Prevents very unlikely tokens, allows some creativity.
       → K=50 is a common default.

    3. TOP-P (NUCLEUS) SAMPLING:
       Keep the smallest set of tokens whose cumulative probability ≥ p.
       → Dynamic: if one token has 95% probability (clear answer),
         only that token is in the nucleus.
       → If probabilities are spread out, include more tokens.
       → p=0.9 is a common default for creative tasks.

    4. TEMPERATURE + TOP-P (most common in practice):
       Apply temperature first (adjust distribution),
       then apply top-p (truncate unlikely tokens).
       → Claude uses this combination.

    MATH EXAMPLE:
      Raw logits: [Paris: 12.4, Lyon: 3.2, Nice: 2.1, Berlin: 1.8]

      After softmax (temperature=1.0):
        Paris: 0.940, Lyon: 0.030, Nice: 0.020, Berlin: 0.010

      After softmax (temperature=0.5) — sharper:
        Paris: 0.997, Lyon: 0.002, Nice: 0.001, Berlin: 0.000

      After softmax (temperature=2.0) — flatter:
        Paris: 0.680, Lyon: 0.140, Nice: 0.120, Berlin: 0.060

      Top-K(2): Only sample from [Paris, Lyon]
      Top-P(0.95): [Paris(0.94), Lyon(0.03)] — 0.94+0.03=0.97 ≥ 0.95
    """

    print("="*65)
    print("SAMPLING STRATEGIES (Conceptual)")
    print("="*65)

    # Simulate a token probability distribution
    vocab_sample = {
        "Paris":   0.940,
        "Lyon":    0.030,
        "Nice":    0.020,
        "Berlin":  0.008,
        "Madrid":  0.001,
        "Tokyo":   0.001,
    }

    print(f"\n  Context: 'The capital of France is ___'")
    print(f"\n  Token probabilities (temperature=1.0):")
    for token, prob in sorted(vocab_sample.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 50)
        print(f"    {token:<10}: {prob:.3f} {bar}")

    # Greedy
    greedy = max(vocab_sample, key=lambda t: vocab_sample[t])
    print(f"\n  GREEDY (temperature=0): Always picks '{greedy}' (prob={vocab_sample[greedy]})")

    # Top-K (K=2)
    top_k = sorted(vocab_sample.items(), key=lambda x: -x[1])[:2]
    print(f"\n  TOP-K (K=2): Sample only from {[t for t,p in top_k]}")
    total = sum(p for _, p in top_k)
    for t, p in top_k:
        renorm = p / total
        print(f"    {t:<10}: renormalized prob = {renorm:.3f}")

    # Top-P (P=0.95)
    sorted_tokens = sorted(vocab_sample.items(), key=lambda x: -x[1])
    nucleus = []
    cumprob = 0.0
    for token, prob in sorted_tokens:
        nucleus.append((token, prob))
        cumprob += prob
        if cumprob >= 0.95:
            break
    print(f"\n  TOP-P (nucleus, p=0.95): Include tokens until cumulative prob ≥ 0.95")
    print(f"  Nucleus: {[t for t,p in nucleus]}")
    print(f"  Cumulative prob: {sum(p for _,p in nucleus):.3f}")


# ─── Streaming Implementation ─────────────────────────────────────────────────

def stream_rag_response(query: str, context: str):
    """
    Stream a RAG response token-by-token to the terminal.

    WHY STREAMING IN RAG:
      Without streaming: user waits for the FULL response (2-10 seconds).
      With streaming:    user sees the first token in ~0.3-0.5 seconds (TTFT).
      The total time is the same — but perceived latency is dramatically better.

      TTFT (Time-to-First-Token) is the key UX metric for RAG applications.
      Target: TTFT < 500ms for good UX.

    HOW STREAMING WORKS:
      The API sends tokens one-by-one as server-sent events (SSE).
      Each event contains the next token text delta.
      We print each delta immediately (no buffering).

    Args:
        query:   The user's question.
        context: Retrieved document context (simulating RAG retrieval).
    """

    # Build the RAG prompt — same pattern as we'll use throughout the course
    rag_prompt = f"""Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}

Answer:"""

    print(f"\n{'='*65}")
    print(f"STREAMING RAG RESPONSE")
    print(f"Query: {query}")
    print(f"{'='*65}")
    print(f"\nAssistant: ", end="", flush=True)

    # Metrics tracking
    # WHY track these separately:
    #   TTFT (time to first token) is a key production metric.
    #   Total latency is less important than when the user SEES something.
    start_time       = time.perf_counter()
    first_token_time = None
    token_count      = 0
    total_text       = ""

    # ── Streaming API Call ────────────────────────────────────────────────────
    # WHY client.messages.stream() vs client.messages.create():
    #   .create() blocks until the FULL response is ready, then returns it.
    #   .stream() returns a context manager that yields events as they arrive.
    #   Use .stream() for all user-facing RAG interfaces.
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=300,
        temperature=0,  # Deterministic for RAG grounding
        messages=[{"role": "user", "content": rag_prompt}]
    ) as stream:

        # WHY stream.text_stream:
        #   Yields the TEXT DELTA of each new event (just the new characters).
        #   Other event types (message_start, content_block_start, etc.) are
        #   filtered out — we only get the text increments.
        for text_delta in stream.text_stream:

            # Record time of FIRST token
            if first_token_time is None:
                first_token_time = time.perf_counter()

            # WHY print with end="" and flush=True:
            #   end=""  → don't add newline after each delta (tokens are mid-word)
            #   flush=True → force immediate display (bypass Python's output buffer)
            #   Without flush=True, Python might batch multiple deltas before printing.
            print(text_delta, end="", flush=True)

            total_text  += text_delta
            token_count += 1  # approximate count (delta ≠ always one token)

    # Final newline after streaming completes
    print()

    # ── Latency Report ────────────────────────────────────────────────────────
    total_time   = time.perf_counter() - start_time
    ttft         = (first_token_time - start_time) if first_token_time else 0
    throughput   = token_count / total_time if total_time > 0 else 0

    print(f"\n  --- Streaming Metrics ---")
    print(f"  TTFT (Time-to-First-Token): {ttft*1000:.0f}ms")
    print(f"  Total latency:              {total_time*1000:.0f}ms")
    print(f"  Approximate throughput:     {throughput:.0f} tokens/sec")
    print(f"  Response length:            {len(total_text)} chars")


def compare_streaming_vs_blocking(query: str, context: str):
    """
    Compare the user experience of streaming vs blocking (non-streaming).

    BLOCKING: waits N seconds, then dumps entire response at once.
    STREAMING: shows first character in ~300ms, response builds in real-time.

    Same total wall clock time — but PERCEIVED latency is very different.
    """

    print(f"\n{'='*65}")
    print(f"COMPARISON: Blocking vs Streaming")
    print(f"{'='*65}")

    rag_prompt = f"""Context: {context}\n\nQuestion: {query}\nAnswer:"""

    # ── Blocking (no streaming) ───────────────────────────────────────────────
    print(f"\n[1] BLOCKING (messages.create):")
    print(f"    Waiting for full response", end="", flush=True)

    start = time.perf_counter()

    # Simulate waiting with dots (shows how blocking feels to the user)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        temperature=0,
        messages=[{"role": "user", "content": rag_prompt}]
    )

    elapsed = time.perf_counter() - start
    print(f" ({elapsed:.1f}s wait)")
    print(f"    Response: {response.content[0].text.strip()[:100]}...")
    print(f"    TTFT: {elapsed:.1f}s (entire wait before ANY text appears)")

    # ── Streaming ─────────────────────────────────────────────────────────────
    print(f"\n[2] STREAMING (messages.stream):")
    print(f"    Response: ", end="", flush=True)

    start = time.perf_counter()
    first_token = None

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=150,
        temperature=0,
        messages=[{"role": "user", "content": rag_prompt}]
    ) as stream:
        for delta in stream.text_stream:
            if first_token is None:
                first_token = time.perf_counter()
            print(delta, end="", flush=True)

    total_elapsed = time.perf_counter() - start
    ttft = (first_token - start) if first_token else 0

    print()
    print(f"    TTFT: {ttft*1000:.0f}ms  (first char appears immediately)")
    print(f"    Total: {total_elapsed:.1f}s (same total time, but FEELS faster)")

    print(f"\n  INSIGHT: Total time is nearly identical.")
    print(f"  But streaming feels 10x more responsive because the user sees output immediately.")
    print(f"  ALL production RAG UIs should use streaming.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Sample retrieved context (simulating what a vector DB would return in Phase 3)
    retrieved_context = """
[Source: rag_guide.pdf, Page 1]
Retrieval-Augmented Generation (RAG) is an AI architecture that improves LLM responses
by retrieving relevant documents from an external knowledge base before generating an answer.
RAG was introduced in a 2020 paper by Lewis et al. at Facebook AI Research.

[Source: rag_guide.pdf, Page 2]
The main components of a RAG system are: (1) Document ingestion — parsing and chunking
source documents; (2) Embedding — converting chunks to vector representations;
(3) Vector storage — indexing vectors in a database like Pinecone or Weaviate;
(4) Retrieval — finding relevant chunks at query time; (5) Generation — prompting
the LLM with retrieved context to produce a grounded answer.
"""

    user_query = "What are the main components of a RAG system?"

    # Demo 1: Sampling strategies
    explain_sampling_strategies()

    # Demo 2: Streaming RAG response
    stream_rag_response(user_query, retrieved_context)

    # Demo 3: Streaming vs Blocking comparison
    compare_streaming_vs_blocking(
        "What was RAG originally proposed for?",
        retrieved_context
    )
