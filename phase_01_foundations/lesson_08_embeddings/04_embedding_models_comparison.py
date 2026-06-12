"""
FILE: 04_embedding_models_comparison.py
LESSON: Phase 1 - Lesson 8 - Embeddings
TOPIC: Embedding model landscape — comparing models by capability, cost, and use case

WHAT THIS FILE TEACHES:
  - The key embedding models available in 2025
  - MTEB (Massive Text Embedding Benchmark) — the standard evaluation framework
  - Dimensions, max tokens, and cost per model
  - When to use asymmetric (Voyage) vs symmetric models
  - Code-specialized vs general-purpose models
  - How to pick the right model for your RAG use case

NO API NEEDED: This file is pure analysis — no API calls required.

INSTALL: none (pure Python)
"""


# ─── Model Registry ───────────────────────────────────────────────────────────

# WHY a list of dicts (not classes):
#   Model specs are tabular data — quick to update when providers release new models.
#   In production, load this from a config file or a provider's /models endpoint.

EMBEDDING_MODELS = [
    # ── Voyage AI (Anthropic recommended) ─────────────────────────────────────
    {
        "model_id":       "voyage-3",
        "provider":       "Voyage AI",
        "dimensions":     1024,
        "max_tokens":     32000,
        "price_per_1m":   0.06,   # $ per 1M tokens
        "mteb_retrieval": 59.1,   # MTEB Retrieval average (higher = better)
        "asymmetric":     True,   # supports document vs query input_type
        "multilingual":   False,  # primarily English
        "code_optimized": False,
        "best_for":       ["General RAG", "Enterprise docs", "Long documents"],
        "avoid_for":      ["Multilingual corpora", "Code-heavy repos"],
        "notes":          "Anthropic's recommended model. Best retrieval accuracy for English.",
    },
    {
        "model_id":       "voyage-3-lite",
        "provider":       "Voyage AI",
        "dimensions":     512,
        "max_tokens":     32000,
        "price_per_1m":   0.02,
        "mteb_retrieval": 57.4,
        "asymmetric":     True,
        "multilingual":   False,
        "code_optimized": False,
        "best_for":       ["High-volume RAG", "Cost-sensitive deployments", "Real-time search"],
        "avoid_for":      ["Maximum accuracy requirements"],
        "notes":          "3× cheaper than voyage-3, ~3% lower quality. Use for high-volume.",
    },
    {
        "model_id":       "voyage-code-3",
        "provider":       "Voyage AI",
        "dimensions":     1024,
        "max_tokens":     32000,
        "price_per_1m":   0.06,
        "mteb_retrieval": 61.0,   # Estimated on code retrieval benchmarks
        "asymmetric":     True,
        "multilingual":   False,
        "code_optimized": True,
        "best_for":       ["Code search", "Technical docs with code", "API documentation"],
        "avoid_for":      ["Pure prose documents", "Multilingual content"],
        "notes":          "Trained on code corpora. Best for repos, API docs, configs.",
    },
    {
        "model_id":       "voyage-multilingual-2",
        "provider":       "Voyage AI",
        "dimensions":     1024,
        "max_tokens":     32000,
        "price_per_1m":   0.06,
        "mteb_retrieval": 56.5,
        "asymmetric":     True,
        "multilingual":   True,
        "code_optimized": False,
        "best_for":       ["Multilingual enterprise docs", "Global knowledge bases"],
        "avoid_for":      ["English-only corpora (voyage-3 is better)"],
        "notes":          "Supports 35+ languages. Use when corpus has non-English content.",
    },

    # ── OpenAI ────────────────────────────────────────────────────────────────
    {
        "model_id":       "text-embedding-3-large",
        "provider":       "OpenAI",
        "dimensions":     3072,
        "max_tokens":     8191,
        "price_per_1m":   0.13,
        "mteb_retrieval": 55.7,
        "asymmetric":     False,   # symmetric — same encoding for docs and queries
        "multilingual":   True,
        "code_optimized": False,
        "best_for":       ["OpenAI-stack systems", "Existing GPT-4 deployments"],
        "avoid_for":      ["High-volume (expensive)", "Long documents (8K limit)"],
        "notes":          "Supports Matryoshka truncation to 256, 512, 1024, 1536 dims.",
    },
    {
        "model_id":       "text-embedding-3-small",
        "provider":       "OpenAI",
        "dimensions":     1536,
        "max_tokens":     8191,
        "price_per_1m":   0.02,
        "mteb_retrieval": 51.7,
        "asymmetric":     False,
        "multilingual":   True,
        "code_optimized": False,
        "best_for":       ["OpenAI stack, cost-sensitive"],
        "avoid_for":      ["Long documents", "Maximum retrieval quality"],
        "notes":          "Budget option in OpenAI stack. Much lower quality than voyage-3.",
    },

    # ── Cohere ────────────────────────────────────────────────────────────────
    {
        "model_id":       "embed-multilingual-v3.0",
        "provider":       "Cohere",
        "dimensions":     1024,
        "max_tokens":     512,    # WHY short: Cohere's embed-v3 has short context
        "price_per_1m":   0.10,
        "mteb_retrieval": 57.1,
        "asymmetric":     True,
        "multilingual":   True,
        "code_optimized": False,
        "best_for":       ["Multilingual", "Short documents", "Cohere-stack deployments"],
        "avoid_for":      ["Long documents (512 token limit)"],
        "notes":          "Best multilingual option after voyage-multilingual.",
    },

    # ── Open Source ───────────────────────────────────────────────────────────
    {
        "model_id":       "BAAI/bge-m3",
        "provider":       "BAAI (self-hosted)",
        "dimensions":     1024,
        "max_tokens":     8192,
        "price_per_1m":   0.00,   # free (compute cost only)
        "mteb_retrieval": 57.3,
        "asymmetric":     True,
        "multilingual":   True,
        "code_optimized": False,
        "best_for":       ["Privacy-sensitive deployments", "Air-gapped environments", "No API cost"],
        "avoid_for":      ["Cloud-only infrastructure", "Lowest latency requirements"],
        "notes":          "Self-hosted via sentence-transformers. State-of-art open source.",
    },
    {
        "model_id":       "nomic-embed-text-v1.5",
        "provider":       "Nomic (self-hosted/API)",
        "dimensions":     768,
        "max_tokens":     8192,
        "price_per_1m":   0.00,
        "mteb_retrieval": 53.8,
        "asymmetric":     False,
        "multilingual":   False,
        "code_optimized": False,
        "best_for":       ["Local development", "Ollama integration", "Free embedding"],
        "avoid_for":      ["Production with accuracy requirements"],
        "notes":          "Runs via Ollama locally. Good for dev without API keys.",
    },
]


# ─── Model Comparison Display ─────────────────────────────────────────────────

def show_model_comparison():
    """
    Full model comparison table ranked by MTEB retrieval score.
    """

    print("=" * 90)
    print("EMBEDDING MODEL LANDSCAPE 2025 — ranked by MTEB Retrieval score")
    print("=" * 90)

    sorted_models = sorted(EMBEDDING_MODELS, key=lambda m: -m["mteb_retrieval"])

    print(f"\n  {'Model':<35} {'Dims':>5} {'MaxTok':>7} {'$/1M':>6} "
          f"{'MTEB':>6} {'Asym':>5} {'Multi':>6} {'Code':>5}")
    print(f"  {'─'*35} {'─'*5} {'─'*7} {'─'*6} {'─'*6} {'─'*5} {'─'*6} {'─'*5}")

    for m in sorted_models:
        asym  = "✓" if m["asymmetric"]     else "✗"
        multi = "✓" if m["multilingual"]   else "✗"
        code  = "✓" if m["code_optimized"] else "✗"
        price = f"${m['price_per_1m']:.2f}" if m["price_per_1m"] > 0 else "FREE"

        model_str = f"{m['provider']}/{m['model_id']}"[:35]

        print(
            f"  {model_str:<35} {m['dimensions']:>5} "
            f"{m['max_tokens']:>7,} {price:>6} "
            f"{m['mteb_retrieval']:>6.1f} {asym:>5} {multi:>6} {code:>5}"
        )

    print(f"\n  MTEB = Massive Text Embedding Benchmark (Retrieval subset, higher = better)")
    print(f"  Asym = asymmetric encoding (separate document/query input_type)")


def show_model_recommendations():
    """
    Decision guide: which model to use in which scenario.
    """

    print("\n" + "=" * 65)
    print("MODEL SELECTION GUIDE")
    print("=" * 65)

    scenarios = [
        {
            "situation": "Default enterprise RAG (English docs)",
            "recommendation": "voyage-3",
            "why": "Best MTEB retrieval score, supports long docs (32K tokens), "
                   "Anthropic-recommended, asymmetric encoding",
        },
        {
            "situation": "High-volume RAG (>10M queries/month)",
            "recommendation": "voyage-3-lite",
            "why": "3× cheaper than voyage-3 with only ~3% quality loss. "
                   "At scale, this saves thousands of dollars monthly.",
        },
        {
            "situation": "RAG over code repositories or API docs",
            "recommendation": "voyage-code-3",
            "why": "Trained on code corpora. Understands function signatures, "
                   "class names, and code patterns better than general models.",
        },
        {
            "situation": "Multilingual corpus (Chinese, Japanese, Arabic, etc.)",
            "recommendation": "voyage-multilingual-2",
            "why": "Trained across 35+ languages. Superior to voyage-3 for "
                   "non-English retrieval tasks.",
        },
        {
            "situation": "Air-gapped / privacy-sensitive deployment",
            "recommendation": "BAAI/bge-m3",
            "why": "State-of-art open source, runs locally via sentence-transformers. "
                   "No API calls → no data leaves the network.",
        },
        {
            "situation": "Local development without API keys",
            "recommendation": "nomic-embed-text-v1.5 via Ollama",
            "why": "Run locally with Ollama. No API key needed. "
                   "Switch to voyage-3 before production.",
        },
    ]

    for s in scenarios:
        print(f"\n  SITUATION: {s['situation']}")
        print(f"  RECOMMENDATION: {s['recommendation']}")
        print(f"  WHY: {s['why']}")


def cost_comparison_at_scale():
    """
    Compare embedding cost across models for a real RAG corpus.
    """

    print("\n" + "=" * 65)
    print("COST COMPARISON AT SCALE")
    print("=" * 65)

    corpus_sizes = [
        ("Small (startup)",         10_000,  5_000_000),   # 10K docs, 5M total tokens
        ("Medium (SMB)",           100_000, 50_000_000),   # 100K docs, 50M tokens
        ("Large (enterprise)",   1_000_000,500_000_000),   # 1M docs, 500M tokens
    ]

    print()
    for corpus_name, n_docs, total_tokens in corpus_sizes:
        print(f"  [{corpus_name}: {n_docs:,} docs, {total_tokens/1_000_000:.0f}M tokens]")
        print(f"  {'Model':<35} {'Cost to embed':>15} {'Storage':>12}")
        print(f"  {'─'*35} {'─'*15} {'─'*12}")

        for m in sorted(EMBEDDING_MODELS, key=lambda m: m["price_per_1m"]):
            cost_usd      = total_tokens / 1_000_000 * m["price_per_1m"]
            storage_gb    = n_docs * m["dimensions"] * 4 / (1024**3)  # float32
            cost_str      = f"${cost_usd:,.2f}" if cost_usd > 0 else "FREE"
            storage_str   = f"{storage_gb:.2f} GB"

            model_str = f"{m['provider']}/{m['model_id']}"[:35]
            print(f"  {model_str:<35} {cost_str:>15} {storage_str:>12}")
        print()


def asymmetric_vs_symmetric_explained():
    """
    Deep explanation of asymmetric embedding — the most commonly misunderstood concept.
    """

    print("\n" + "=" * 65)
    print("ASYMMETRIC vs SYMMETRIC EMBEDDING — Critical for RAG")
    print("=" * 65)

    print(f"""
  SYMMETRIC EMBEDDING (e.g., OpenAI ada-002):
    Both documents AND queries use the same encoding function.
    Doc:   embed("ACI uses Leaf-Spine topology")   → vector_doc
    Query: embed("How does ACI work?")             → vector_query
    The model asks: "Do these two texts MEAN THE SAME THING?"

    Problem: a short, incomplete query never means exactly the same thing
    as a long, descriptive document passage. Similarity scores are lower
    than they should be → you need a LOWER threshold → more noise.

  ASYMMETRIC EMBEDDING (e.g., Voyage AI):
    Documents and queries use DIFFERENT encoding functions.
    Doc:   embed(text, input_type="document") → vector that says
           "I ANSWER queries about ACI topology"
    Query: embed(text, input_type="query")   → vector that says
           "I am LOOKING FOR information about ACI topology"

    The model learns: what does a query that needs THIS document look like?
    This closes the semantic gap between short queries and long documents.

  EMPIRICAL IMPACT:
    Using "document" for queries (wrong):  ~0.71 average similarity on matching pairs
    Using "query" for queries (correct):   ~0.89 average similarity on matching pairs
    That 18-point gap means 10-20% worse retrieval Recall@5 in production.

  COMMON MISTAKE:
    Most tutorials embed both documents AND queries with the same call.
    This silently degrades retrieval quality.
    Always check: am I using input_type="document" for corpus and "query" for search?
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    show_model_comparison()
    show_model_recommendations()
    cost_comparison_at_scale()
    asymmetric_vs_symmetric_explained()
