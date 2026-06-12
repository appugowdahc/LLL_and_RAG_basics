"""
FILE: 02_generate_embeddings.py
LESSON: Phase 1 - Lesson 8 - Embeddings
TOPIC: Generating embeddings with Voyage AI — the API Anthropic recommends for RAG

WHAT THIS FILE TEACHES:
  - Setting up the Voyage AI client
  - The CRITICAL difference between input_type="document" vs "query"
  - Batch embedding (embedding multiple texts in one API call)
  - Rate limits and retry patterns
  - Caching embeddings to avoid re-embedding unchanged documents
  - Cost estimation for embedding a corpus at scale

WHY VOYAGE AI:
  Anthropic explicitly recommends Voyage AI as the embedding provider for RAG.
  Voyage models are trained specifically for retrieval (MTEB retrieval benchmark).
  They outperform OpenAI ada-002 and Cohere on most RAG benchmarks.
  The "document" vs "query" encoding is a deliberate asymmetric design
  that improves retrieval accuracy vs symmetric models.

VOYAGE API KEYS:
  Get a free key at https://www.voyageai.com/
  VOYAGE_API_KEY environment variable (set in .env)

INSTALL:
  pip install voyageai anthropic python-dotenv
"""

import os
import time
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# ─── Client Setup ─────────────────────────────────────────────────────────────

try:
    import voyageai
    # WHY api_key from environment:
    #   Never hardcode API keys. Load from .env or system environment.
    #   voyageai.Client() auto-reads VOYAGE_API_KEY from environment if not passed.
    vo = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY", ""))
    HAS_VOYAGE = bool(os.environ.get("VOYAGE_API_KEY"))
except ImportError:
    HAS_VOYAGE = False
    vo = None

import numpy as np


def mock_embedding(text: str, dims: int = 1024) -> list[float]:
    """
    Deterministic mock embedding for use when VOYAGE_API_KEY is not set.
    Uses a hash of the text to produce a consistent (but meaningless) vector.

    WHY deterministic hash (not random):
      np.random would produce different vectors on each run → similarity scores
      would be meaningless and misleading. The hash gives the SAME vector for
      the SAME text, so caching behavior is realistic even without the API.
    """
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.RandomState(seed)
    vec  = rng.randn(dims).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()   # unit normalize


# ─── Core Embedding Functions ─────────────────────────────────────────────────

def embed_documents(
    texts:     list[str],
    model:     str = "voyage-3",
    batch_size: int = 128,
) -> list[list[float]]:
    """
    Embed a list of document texts for indexing in the vector database.

    WHY input_type="document" NOT "query":
      Voyage AI uses ASYMMETRIC encoding:
      - "document" encodes for maximum retrieval coverage.
        The model asks: "what queries might this document answer?"
      - "query"    encodes for maximum match precision.
        The model asks: "what documents answer this question?"
      Using "document" for both → symmetric encoding → ~10-20% worse retrieval.
      This is one of the most common RAG mistakes.

    WHY batching:
      The Voyage API has a per-request limit (batch_size docs per call).
      Sending 10,000 documents one at a time = 10,000 API calls → slow + rate-limited.
      Batching to 128 per call = 79 API calls → 100× faster.

    Args:
        texts:      List of document strings to embed.
        model:      Voyage model name.
        batch_size: Max texts per API call.

    Returns:
        List of embedding vectors (same order as input texts).
    """

    if not HAS_VOYAGE:
        print(f"  (mock) Embedding {len(texts)} documents with deterministic mock vectors")
        return [mock_embedding(t) for t in texts]

    all_embeddings = []

    # WHY range(0, N, batch_size):
    #   Slides a window of batch_size across the texts list.
    #   Ensures no batch exceeds the API's per-call limit.
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]

        # WHY retry on rate limit:
        #   Voyage API has rate limits (tokens per minute).
        #   A simple retry with exponential backoff handles transient limits.
        for attempt in range(3):
            try:
                result = vo.embed(
                    texts      = batch,
                    model      = model,
                    input_type = "document",   # WHY "document": see docstring above
                )
                all_embeddings.extend(result.embeddings)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt   # WHY exponential backoff: 1s, 2s, 4s
                print(f"  API error, retrying in {wait}s: {e}")
                time.sleep(wait)

    return all_embeddings


def embed_query(
    query: str,
    model: str = "voyage-3",
) -> list[float]:
    """
    Embed a user query for semantic search.

    WHY input_type="query" NOT "document":
      See embed_documents() for the asymmetric encoding explanation.
      Query encoding is optimized for MATCHING documents, not for describing content.

    Args:
        query: The user's search question.
        model: Voyage model name.

    Returns:
        Single embedding vector for the query.
    """

    if not HAS_VOYAGE:
        print(f"  (mock) Embedding query: '{query[:50]}...' " if len(query) > 50 else f"  (mock) Embedding query: '{query}'")
        return mock_embedding(query)

    result = vo.embed(
        texts      = [query],
        model      = model,
        input_type = "query",   # WHY "query": optimized for retrieval matching
    )
    return result.embeddings[0]


# ─── Caching Layer ────────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    File-based embedding cache.
    Avoids re-embedding documents that haven't changed.

    WHY caching is CRITICAL for production RAG:
      Embedding 100,000 documents costs real money and time.
      When documents are updated (only 1% change daily), re-embedding ALL
      documents wastes 99% of the embedding cost.
      A cache keyed by content hash ensures we only embed NEW or CHANGED content.

    Cache key: SHA-256 hash of (text + model_name)
      WHY include model_name: same text + different model = different embedding
    """

    def __init__(self, cache_dir: str = "/tmp/embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # WHY exist_ok: safe to call multiple times

    def _cache_key(self, text: str, model: str) -> str:
        """SHA-256 hash of text+model → unique cache key."""
        payload = f"{model}:{text}"
        # WHY SHA-256 not MD5: SHA-256 has no known collisions. MD5 is sufficient
        # for cache keys but SHA-256 is standard practice for content hashing.
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, text: str, model: str) -> list[float] | None:
        """Return cached embedding or None if not cached."""
        key       = self._cache_key(text, model)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            # WHY json not pickle: json is human-readable and model-portable.
            # pickle ties to Python version and can be a security risk.
            return json.loads(cache_file.read_text())
        return None

    def set(self, text: str, model: str, embedding: list[float]):
        """Store embedding in cache."""
        key       = self._cache_key(text, model)
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(json.dumps(embedding))

    def embed_with_cache(
        self,
        texts: list[str],
        model: str = "voyage-3",
        input_type: str = "document",
    ) -> list[list[float]]:
        """
        Embed texts, using cache where available.
        Only calls API for uncached texts.

        Args:
            texts:      List of texts to embed.
            model:      Voyage model name.
            input_type: "document" or "query".

        Returns:
            List of embeddings in same order as input.
        """

        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts   = []

        # ── 1. Check cache for each text ──────────────────────────────────────
        for i, text in enumerate(texts):
            cached = self.get(text, model)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # ── 2. Embed only uncached texts ──────────────────────────────────────
        if uncached_texts:
            print(f"  Cache: {len(texts)-len(uncached_texts)} hits, {len(uncached_texts)} misses → embedding")

            if input_type == "query":
                new_embeddings = [embed_query(t, model) for t in uncached_texts]
            else:
                new_embeddings = embed_documents(uncached_texts, model)

            # Store in cache and fill results
            for i, (text, embedding) in zip(uncached_indices, zip(uncached_texts, new_embeddings)):
                self.set(text, model, embedding)
                results[i] = embedding
        else:
            print(f"  Cache: 100% hit rate — no API calls needed")

        return results


# ─── Cost Estimator ───────────────────────────────────────────────────────────

VOYAGE_PRICING = {
    "voyage-3":      0.06,    # $ per 1M tokens
    "voyage-3-lite": 0.02,    # $ per 1M tokens (cheaper, slightly lower quality)
    "voyage-code-3": 0.06,    # $ per 1M tokens (optimized for code)
}

def estimate_embedding_cost(
    texts:             list[str],
    model:             str = "voyage-3",
    chars_per_token:   float = 4.0,   # English prose approximation
) -> dict:
    """
    Estimate cost to embed a list of texts.

    WHY chars_per_token=4.0:
      English prose averages 4 characters per token.
      For code-heavy content use 3.3. For JSON/YAML use 2.5.
      This gives a rough token count from character count.
    """

    total_chars  = sum(len(t) for t in texts)
    total_tokens = int(total_chars / chars_per_token)
    price_per_m  = VOYAGE_PRICING.get(model, 0.06)
    cost         = total_tokens / 1_000_000 * price_per_m

    return {
        "model":         model,
        "num_texts":     len(texts),
        "total_chars":   total_chars,
        "est_tokens":    total_tokens,
        "price_per_m":   price_per_m,
        "estimated_cost": cost,
    }


# ─── Demo ─────────────────────────────────────────────────────────────────────

def run_embedding_demo():
    """
    Full embedding pipeline demo: documents + query + similarity search.
    """

    print("=" * 65)
    print("VOYAGE AI EMBEDDINGS: Document and Query Encoding")
    print("=" * 65)

    if not HAS_VOYAGE:
        print("\n  ⚠ VOYAGE_API_KEY not set — using deterministic mock embeddings.")
        print("  Results show correct behavior; add key to .env for real vectors.\n")

    # ── Documents to embed ────────────────────────────────────────────────────
    documents = [
        "Cisco ACI uses a Leaf-Spine topology. The APIC controller manages all fabric policy centrally.",
        "ReadyOps performs continuous validation across Live Operations and Production-Representative environments.",
        "Cisco Hypershield uses eBPF to enforce microsegmentation at the kernel level without dedicated appliances.",
        "ISE TrustSec assigns Security Group Tags (SGTs) at authentication and propagates them through the fabric.",
        "The Nexus 9000 series supports VXLAN EVPN for multi-tenant data center fabric deployments.",
        "Cisco SD-WAN (Viptela) provides software-defined WAN with centralized policy and zero-touch provisioning.",
        "My cat sat on the mat and ate a rat.",   # ← intentionally off-topic
        "The quick brown fox jumps over the lazy dog.",  # ← off-topic
    ]

    # ── 1. Embed documents ────────────────────────────────────────────────────
    print(f"\n  Step 1: Embed {len(documents)} documents (input_type='document')")
    start = time.perf_counter()
    doc_embeddings = embed_documents(documents, model="voyage-3")
    elapsed = time.perf_counter() - start

    print(f"  Embedding shape: {len(doc_embeddings)} vectors × {len(doc_embeddings[0])} dims")
    print(f"  Elapsed: {elapsed*1000:.0f}ms")
    print(f"  First vector (first 5 dims): {[round(x, 4) for x in doc_embeddings[0][:5]]}")

    # ── 2. Embed a query ──────────────────────────────────────────────────────
    query = "How does ACI manage network policy?"
    print(f"\n  Step 2: Embed query (input_type='query')")
    print(f"  Query: '{query}'")

    query_embedding = embed_query(query, model="voyage-3")
    print(f"  Query vector (first 5 dims): {[round(x, 4) for x in query_embedding[:5]]}")

    # ── 3. Semantic search ─────────────────────────────────────────────────────
    print(f"\n  Step 3: Find top-3 most similar documents")

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    scores = [
        (cosine_sim(query_embedding, doc_emb), doc)
        for doc_emb, doc in zip(doc_embeddings, documents)
    ]
    scores.sort(key=lambda x: -x[0])

    print(f"\n  {'Score':<10} {'Document'}")
    print(f"  {'─'*10} {'─'*55}")
    for score, doc in scores:
        marker = " ← TOP RESULT" if score == scores[0][0] else ""
        print(f"  {score:.4f}    {doc[:60]}{marker}")

    # ── 4. Cost estimation ─────────────────────────────────────────────────────
    print(f"\n  Step 4: Cost estimation for a larger corpus")
    large_corpus = documents * 1000   # simulate 8,000 documents
    cost_info    = estimate_embedding_cost(large_corpus, model="voyage-3")

    print(f"  Documents: {cost_info['num_texts']:,}")
    print(f"  Est. tokens: {cost_info['est_tokens']:,}")
    print(f"  Est. cost: ${cost_info['estimated_cost']:.4f}")

    # ── 5. Cache demo ─────────────────────────────────────────────────────────
    print(f"\n  Step 5: Caching demo")
    cache = EmbeddingCache()

    print("  First pass (cache cold — all misses):")
    _ = cache.embed_with_cache(documents[:3], model="voyage-3")

    print("  Second pass (cache warm — all hits):")
    _ = cache.embed_with_cache(documents[:3], model="voyage-3")

    print("  Third pass (mixed — 2 cached, 1 new):")
    mixed = documents[:2] + ["New document about Cisco Intersight management."]
    _ = cache.embed_with_cache(mixed, model="voyage-3")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_embedding_demo()
