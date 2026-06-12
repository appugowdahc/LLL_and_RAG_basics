"""
FILE: 05_kv_cache_demo.py
LESSON: Phase 1 - Lesson 4 - Transformer Architecture
TOPIC: KV Cache — The performance mechanism that makes RAG practical

WHAT THIS FILE TEACHES:
  - What the KV Cache is and why it exists
  - Why WITHOUT it, RAG with long contexts would be impossibly slow
  - How to calculate KV cache memory for any model + context length
  - Why Anthropic's Prompt Caching (disk-level cache) is different and saves money
  - How to use Anthropic's cache_control to cache retrieved documents

TWO TYPES OF CACHING IN RAG:
  ┌───────────────────────────────────────────────────────────────┐
  │ 1. KV CACHE (GPU memory, per-call, automatic)                  │
  │    Saves recomputing attention for previously seen tokens.     │
  │    Active during ONE generation call.                          │
  │    Freed when the call ends.                                   │
  │                                                                │
  │ 2. ANTHROPIC PROMPT CACHE (disk, cross-call, explicit)         │
  │    Saves re-encoding the same prompt prefix across MULTIPLE     │
  │    API calls. Active across calls (up to 5 minutes TTL).       │
  │    Dramatically reduces cost for repeated contexts.            │
  └───────────────────────────────────────────────────────────────┘

WHY KV CACHE IS CRITICAL FOR RAG:
  A RAG call with 10 retrieved chunks × 500 tokens each = 5,000 input tokens.
  At each generation step, WITHOUT KV cache, the model would re-process
  all 5,000 input tokens PLUS all generated output tokens so far.
  With KV cache: input tokens are processed ONCE. Only new output tokens run
  the full forward pass. Inference becomes O(output_tokens) not O(total²).

INSTALL:
  pip install anthropic python-dotenv
"""

import os
import time
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# ─── KV Cache Math ────────────────────────────────────────────────────────────

def calculate_kv_cache_size(
    d_model:    int,
    n_layers:   int,
    n_kv_heads: int,
    seq_len:    int,
    batch_size: int = 1,
    dtype_bytes: int = 2,   # fp16
) -> dict:
    """
    Calculate the KV cache memory requirement for a given model and context.

    KV CACHE FORMULA:
      For each transformer layer, we cache K and V matrices for all positions.

      K: [batch_size × n_kv_heads × seq_len × d_k]
      V: [batch_size × n_kv_heads × seq_len × d_k]

      Where d_k = d_model / n_heads  (but we use n_kv_heads for GQA models)

      Total KV cache = 2 × n_layers × batch_size × n_kv_heads × seq_len × d_k × dtype_bytes

    Args:
        d_model:     Model hidden size.
        n_layers:    Number of transformer layers.
        n_kv_heads:  Number of KV attention heads (< n_heads for GQA).
        seq_len:     Sequence length (context + output so far).
        batch_size:  Number of concurrent requests (1 for single call).
        dtype_bytes: Bytes per float (2 for fp16, 4 for fp32).

    Returns:
        dict with KV cache size in bytes, MB, GB.
    """

    # For GQA: d_head = d_model / n_total_heads (NOT n_kv_heads)
    # Here we compute it assuming n_kv_heads maps to a specific dimension
    # For simplicity: d_head = d_model / n_kv_heads * reduction_factor
    # Standard assumption: d_head = 128 (common for most models)
    d_head = 128  # Standard across Llama-3, Mistral, GPT-4 (estimated)

    # WHY ×2: one K matrix and one V matrix per layer
    kv_cache_elements = (
        2           # K and V
        * n_layers
        * batch_size
        * n_kv_heads
        * seq_len
        * d_head
    )

    kv_cache_bytes = kv_cache_elements * dtype_bytes
    kv_cache_mb    = kv_cache_bytes / (1024 ** 2)
    kv_cache_gb    = kv_cache_bytes / (1024 ** 3)

    return {
        "model_d_model":   d_model,
        "n_layers":        n_layers,
        "n_kv_heads":      n_kv_heads,
        "seq_len":         seq_len,
        "batch_size":      batch_size,
        "d_head":          d_head,
        "kv_cache_bytes":  kv_cache_bytes,
        "kv_cache_mb":     round(kv_cache_mb, 2),
        "kv_cache_gb":     round(kv_cache_gb, 3),
    }


def show_kv_cache_rag_impact():
    """
    Show how KV cache size grows with RAG context length.

    In RAG, your prompt = system_prompt + retrieved_chunks + query.
    Longer retrieved context → larger KV cache.
    This is the memory cost you pay for better retrieval (more chunks).
    """

    print("=" * 65)
    print("KV CACHE SIZE vs RAG CONTEXT LENGTH")
    print("Model: Llama-3-8B  (n_layers=32, n_kv_heads=8)")
    print("=" * 65)

    # Llama-3-8B config
    MODEL_CONFIG = {
        "d_model":    4096,
        "n_layers":   32,
        "n_kv_heads": 8,   # GQA: only 8 KV heads out of 32 query heads
    }

    context_scenarios = [
        (512,    "Short: system prompt + 1 chunk + query"),
        (1024,   "Medium: system prompt + 2 chunks + query"),
        (2048,   "Standard RAG: system + 4 chunks + query"),
        (4096,   "Extended: system + 8 chunks + query"),
        (8192,   "Large: system + 16 chunks + query"),
        (16384,  "Very large: 32+ chunks"),
        (32768,  "Massive: 64+ chunks (32K context)"),
        (131072, "Max Llama-3 context: 128K tokens"),
    ]

    print(f"\n  {'Context (tokens)':<20} {'KV Cache (MB)':<18} {'KV Cache (GB)':<15} Scenario")
    print(f"  {'─'*20} {'─'*18} {'─'*15} {'─'*35}")

    for seq_len, scenario in context_scenarios:
        info = calculate_kv_cache_size(seq_len=seq_len, **MODEL_CONFIG)
        print(
            f"  {seq_len:<20,}"
            f"  {info['kv_cache_mb']:<16.1f}"
            f"  {info['kv_cache_gb']:<13.3f}"
            f"  {scenario}"
        )

    print(f"\n  WHY THIS MATTERS:")
    print(f"  The GPU must hold: model weights + KV cache + activations in VRAM.")
    print(f"  Llama-3-8B weights ≈ 16 GB (fp16).")
    print(f"  At 128K context: weights (16 GB) + KV cache (16 GB) = 32 GB total.")
    print(f"  → Barely fits in an A100-40GB with careful memory management.")
    print(f"  → For 70B models, long RAG contexts require multi-GPU setups.")


def demonstrate_without_vs_with_kv_cache():
    """
    Conceptually demonstrate the speedup from KV caching.

    We can't directly observe KV cache behavior via the API,
    but we can reason about it with math and measure actual latency.
    """

    print("\n" + "=" * 65)
    print("WITHOUT vs WITH KV CACHE: Computation Analysis")
    print("=" * 65)

    # Scenario: RAG with 2,000 input tokens, generating 200 output tokens
    n_input  = 2000
    n_output = 200
    n_layers = 32
    d        = 4096

    print(f"\n  Scenario: n_input={n_input} tokens, n_output={n_output} tokens")
    print(f"  Model: d_model={d}, n_layers={n_layers}")

    # Without KV cache: at each output step i, process (n_input + i) tokens
    # Total FLOPs ∝ sum of (n_input + i)² for i in 0..n_output-1
    # Approximation: n_output × (n_input + n_output/2)²
    flops_no_cache = sum(
        (n_input + i) ** 2 for i in range(n_output)
    )

    # With KV cache: at each output step i, process ONLY the new token
    # Previous tokens' K,V are cached. Each step = 1 token × sequence context
    # FLOPs ∝ n_output × n_input (context for attention) + O(d²) for FFN
    flops_with_cache = n_input * n_output  # attention part only

    speedup = flops_no_cache / max(flops_with_cache, 1)

    print(f"\n  Attention FLOPs estimate:")
    print(f"    WITHOUT KV cache: {flops_no_cache:>15,}")
    print(f"    WITH KV cache:    {flops_with_cache:>15,}")
    print(f"    Speedup:          {speedup:>15,.1f}×")

    print(f"\n  Memory trade-off:")
    print(f"    KV cache stores K,V for all {n_input} input tokens.")
    print(f"    At d_head=128, 32 layers, 8 KV heads:")
    info = calculate_kv_cache_size(4096, 32, 8, n_input)
    print(f"    Cache size: {info['kv_cache_mb']:.1f} MB (pays for {speedup:.0f}× speedup)")


# ─── Anthropic Prompt Caching ─────────────────────────────────────────────────

def demonstrate_anthropic_prompt_caching():
    """
    Demonstrate Anthropic's Prompt Caching — cross-call input token caching.

    HOW IT WORKS:
      Anthropic caches the computed internal states of your prompt prefix.
      If the SAME prefix appears in the next call (within 5 minutes),
      those tokens are served from cache — you pay 10% of normal input price.

    RAG USE CASE:
      Your RAG system prompt + retrieved documents are often IDENTICAL
      across multiple user queries in a session.
      Cache the system prompt + docs → save 90% on those input tokens.

    COST IMPACT:
      Normal input: $3.00 / million tokens (Claude Sonnet)
      Cached input: $0.30 / million tokens (10% of normal)
      Cache write:  $3.75 / million tokens (one-time cost to warm cache)

      For 100 queries using the same 5,000-token document context:
        Without caching: 100 × 5,000 × $3/M    = $1.50
        With caching:    100 × 5,000 × $0.30/M  = $0.15  (90% savings!)
    """

    print("\n" + "=" * 65)
    print("ANTHROPIC PROMPT CACHING (Cross-Call Cache)")
    print("=" * 65)

    # A large system context that would be repeated across many RAG calls
    # In production: this would be your retrieved document chunks
    large_static_context = """
You are an expert RAG engineer assistant. Below is a comprehensive reference guide
on RAG system design that you should use to answer questions.

[DOCUMENT 1: RAG Architecture Overview]
Retrieval-Augmented Generation (RAG) is a framework that enhances large language
model responses by dynamically retrieving relevant information from an external
knowledge base at inference time. Unlike fine-tuning, RAG keeps the model weights
frozen and instead augments the prompt with retrieved context.

The core RAG pipeline consists of two phases:
INDEXING PHASE: Documents are loaded, parsed, split into chunks, embedded using
an embedding model, and stored in a vector database alongside their metadata.

RETRIEVAL PHASE: At query time, the user's question is embedded, a similarity
search retrieves the top-k most relevant chunks, those chunks are formatted into
a context window, and the LLM generates an answer grounded in that context.

[DOCUMENT 2: Vector Database Selection Guide]
When selecting a vector database for production RAG, consider these factors:
- Pinecone: Fully managed, auto-scaling, excellent for startups. Pay-as-you-go.
- Weaviate: Open-source with cloud option. Rich filtering. GraphQL API.
- Qdrant: High performance, written in Rust. Excellent for high-throughput.
- Milvus: Designed for massive scale (billions of vectors). Self-hosted.
- ChromaDB: Lightweight, local-first. Perfect for development and prototyping.

[DOCUMENT 3: Chunking Strategies]
The choice of chunking strategy significantly affects RAG retrieval quality:
Fixed-size chunking: Simple, predictable. Risk of cutting context mid-sentence.
Semantic chunking: Splits at natural boundaries. Better semantic coherence.
Hierarchical chunking: Parent/child chunks. Retrieves small chunks, returns parents.
""" * 3  # Repeat to make it larger (simulating a real large document corpus)

    user_questions = [
        "What is the difference between indexing and retrieval phases in RAG?",
        "Which vector database should I use for a startup?",
        "What chunking strategy preserves the most context?",
    ]

    # ── First Call: Warm the Cache ────────────────────────────────────────────
    print(f"\n  [CALL 1]: Cache WRITE (first time — cache miss)")
    print(f"  Context size: ~{len(large_static_context.split()):,} words")

    start = time.perf_counter()
    response1 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        temperature=0,
        system=[
            {
                "type": "text",
                "text": large_static_context,
                # WHY cache_control:
                #   This tells Anthropic: "cache this prefix for the next 5 minutes."
                #   The FIRST call with this marker pays the cache WRITE price.
                #   Subsequent calls with the same prefix pay the cache READ price.
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": "Answer questions using only the context above."
            }
        ],
        messages=[{"role": "user", "content": user_questions[0]}]
    )
    latency1 = (time.perf_counter() - start) * 1000

    usage1 = response1.usage
    # WHY check cache_creation_input_tokens:
    #   If > 0: this call WROTE tokens to the cache (cache miss).
    #   If cache_read_input_tokens > 0: tokens were read from cache (cache hit).
    cache_created = getattr(usage1, 'cache_creation_input_tokens', 0)
    cache_read    = getattr(usage1, 'cache_read_input_tokens', 0)

    print(f"  Latency:               {latency1:.0f}ms")
    print(f"  Input tokens:          {usage1.input_tokens}")
    print(f"  Cache WRITE tokens:    {cache_created}")
    print(f"  Cache READ tokens:     {cache_read}")
    print(f"  Answer: {response1.content[0].text.strip()[:80]}...")

    # ── Subsequent Calls: Cache Hits ──────────────────────────────────────────
    for i, question in enumerate(user_questions[1:], 2):
        print(f"\n  [CALL {i}]: Cache READ (same context — cache hit)")

        start = time.perf_counter()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": large_static_context,
                    "cache_control": {"type": "ephemeral"}  # Same cache marker
                },
                {
                    "type": "text",
                    "text": "Answer questions using only the context above."
                }
            ],
            messages=[{"role": "user", "content": question}]
        )
        latency = (time.perf_counter() - start) * 1000

        usage      = response.usage
        cache_read = getattr(usage, 'cache_read_input_tokens', 0)
        print(f"  Latency:               {latency:.0f}ms")
        print(f"  Cache READ tokens:     {cache_read}  ← paying 10% price on these!")
        print(f"  Answer: {response.content[0].text.strip()[:80]}...")

    # Cost analysis
    context_tokens = getattr(response1.usage, 'cache_creation_input_tokens', 2000)
    n_calls = 100
    cost_no_cache   = n_calls * context_tokens * 3.00 / 1_000_000
    cost_with_cache = (
        context_tokens * 3.75 / 1_000_000        +  # one cache write
        (n_calls - 1) * context_tokens * 0.30 / 1_000_000  # 99 cache reads
    )

    print(f"\n  COST ANALYSIS ({n_calls} queries, {context_tokens} context tokens):")
    print(f"    Without prompt caching: ${cost_no_cache:.4f}")
    print(f"    With prompt caching:    ${cost_with_cache:.4f}")
    print(f"    Savings:                {(cost_no_cache-cost_with_cache)/cost_no_cache*100:.0f}%")
    print(f"\n  RULE: Cache any static context shared across multiple RAG calls.")
    print(f"  Best candidates: system prompts, base knowledge, static documents.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    show_kv_cache_rag_impact()
    demonstrate_without_vs_with_kv_cache()
    demonstrate_anthropic_prompt_caching()
