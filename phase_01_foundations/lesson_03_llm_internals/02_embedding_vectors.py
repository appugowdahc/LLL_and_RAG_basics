"""
FILE: 02_embedding_vectors.py
LESSON: Phase 1 - Lesson 3 - How LLMs Work Internally
TOPIC: Embedding Vectors — How tokens become meaning

WHAT THIS FILE TEACHES:
  - What an embedding vector is mathematically
  - Why embeddings capture semantic meaning (not just position)
  - How to generate embeddings using the Anthropic / Claude API
  - How to measure similarity between embeddings (cosine similarity)
  - Why embedding quality determines RAG retrieval quality

CONCEPT: What is an Embedding?
───────────────────────────────
At the input layer of a transformer, each token ID is looked up in a
learned matrix called the Embedding Matrix:

  Embedding Matrix shape: [vocab_size × d_model]
  Example: [100,277 tokens × 4,096 dimensions]

  token_id 1234 → row 1234 of this matrix → a 4096-dimensional vector

This vector IS the token's "meaning" — as learned from training data.
Similar words end up close together in this high-dimensional space.

SEMANTIC GEOMETRY:
  king - man + woman ≈ queen        (famous analogy)
  Paris - France + Italy ≈ Rome     (capital relationship)
  happy - good + bad ≈ sad          (sentiment axis)

These relationships emerge from training — they are NOT programmed.

WHY THIS POWERS RAG:
  In RAG, we embed BOTH the user query AND all document chunks.
  Then we find chunks whose embedding is closest to the query embedding.
  "Closest in embedding space" = "most semantically relevant".
  This is the core of vector search (Phase 3).

INSTALL:
  pip install anthropic numpy python-dotenv
"""

import os
import math
from dotenv import load_dotenv
import anthropic

# WHY numpy:
#   Numpy provides efficient array operations (dot products, norms).
#   We use it to compute cosine similarity between embedding vectors.
#   In production RAG, vector databases handle this — but numpy helps us
#   understand the math before we abstract it away.
import numpy as np

load_dotenv()

client = anthropic.Anthropic()


# ─── Embedding Generation ────────────────────────────────────────────────────

def get_embedding(text: str, model: str = "voyage-3") -> list[float]:
    """
    Generate an embedding vector for a piece of text.

    Anthropic recommends Voyage AI embeddings for RAG with Claude.
    Voyage-3 produces 1024-dimensional vectors.

    NOTE: This requires a VOYAGE_API_KEY environment variable.
    If you only have ANTHROPIC_API_KEY, see the fallback function below.

    Args:
        text:  The text to embed (a sentence, paragraph, or chunk).
        model: The embedding model to use.

    Returns:
        List of floats (the embedding vector).
    """

    # WHY import voyageai only here (lazy import):
    #   If the user doesn't have voyage installed, this file still runs
    #   up to this point without crashing. Fail only when the function is called.
    try:
        import voyageai  # pip install voyageai
        vo = voyageai.Client()

        # WHY input_type="document" vs "query":
        #   Voyage AI has two modes:
        #     "document" → embedding a chunk to be stored in the vector DB
        #     "query"    → embedding a user query to search with
        #   They use slightly different representations optimized for each role.
        #   ALWAYS use "document" when indexing, "query" when searching.
        #   Mixing them degrades retrieval quality significantly.
        result = vo.embed([text], model=model, input_type="document")
        return result.embeddings[0]

    except ImportError:
        # Fallback: use a mock embedding for demonstration purposes
        # In a real system you'd install voyageai or use OpenAI's API
        print("  [Note: voyageai not installed — using mock embeddings for demo]")
        return _mock_embedding(text)


def _mock_embedding(text: str, dim: int = 128) -> list[float]:
    """
    Generate a deterministic mock embedding for demonstration purposes.

    WHY mock embeddings:
      Allows us to demonstrate the MATH of cosine similarity without
      requiring API keys. The math is identical — only the quality differs.

      In production: NEVER use mock embeddings — quality determines recall.

    HOW it works:
      Uses a character-level hash to seed numpy random.
      Same text always gives same vector (deterministic).
      Similar texts give moderately similar vectors (not as good as real embeddings).
    """

    # WHY sum(ord(c) for c in text) % (2**32):
    #   Creates a deterministic integer seed from the text.
    #   % (2**32) keeps it in valid numpy seed range [0, 2^32-1].
    seed = sum(ord(c) for c in text) % (2 ** 32)

    # WHY np.random.seed():
    #   Seeds the random number generator so the SAME text always gives
    #   the SAME vector. Reproducibility is critical for debugging RAG.
    rng = np.random.default_rng(seed)

    # WHY standard normal distribution (mean=0, std=1):
    #   Real embedding models produce approximately unit-normal distributions.
    #   This makes cosine similarity meaningful (not dominated by one dimension).
    vector = rng.standard_normal(dim)

    # WHY L2 normalize (divide by magnitude):
    #   Normalizing to unit length makes cosine similarity = dot product.
    #   Most embedding models already output unit-length vectors.
    magnitude = np.linalg.norm(vector)
    if magnitude > 0:
        vector = vector / magnitude

    return vector.tolist()


# ─── Similarity Metrics ───────────────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    MATH:
      cos(θ) = (A · B) / (|A| × |B|)

      Where:
        A · B  = dot product (sum of element-wise products)
        |A|    = L2 norm of A (square root of sum of squares)
        |B|    = L2 norm of B

    RANGE: -1.0 to 1.0
      1.0  = identical direction (most similar)
      0.0  = perpendicular (unrelated)
     -1.0  = opposite direction (antonyms / opposites)

    WHY cosine over euclidean distance for text:
      Two documents about the same topic but different lengths will have
      vectors pointing in the same DIRECTION but different MAGNITUDES.
      Cosine similarity is magnitude-invariant — it measures angle only.
      This makes it better for semantic similarity of text.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Float in [-1.0, 1.0]. Higher = more similar.
    """

    # WHY np.array():
    #   Converts Python lists to numpy arrays for vectorized math.
    #   Element-wise operations on plain Python lists are 100x slower.
    a = np.array(vec_a)
    b = np.array(vec_b)

    # WHY np.dot(a, b):
    #   Computes the dot product: sum of (a[i] * b[i]) for all i.
    #   When vectors are unit-normalized, this equals cosine similarity directly.
    dot_product = np.dot(a, b)

    # WHY np.linalg.norm():
    #   Computes the L2 (Euclidean) norm: sqrt(sum of squares).
    #   Used to normalize the dot product to [-1, 1] range.
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # WHY check for zero magnitude:
    #   A zero vector has undefined cosine similarity.
    #   Guarding against division by zero prevents NaN results.
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # WHY float(np.clip(...)):
    #   Floating-point arithmetic can produce values like 1.0000000002 due to
    #   precision errors. np.clip constrains the result to [-1, 1].
    #   float() converts numpy float64 to Python float for clean printing.
    return float(np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0))


def euclidean_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute Euclidean (L2) distance between two embedding vectors.

    MATH:
      dist = sqrt(sum((a_i - b_i)^2))

    RANGE: 0 to infinity
      0.0 = identical vectors
      Higher = more different

    WHEN TO USE euclidean vs cosine:
      Cosine:     Best for semantic similarity of text (magnitude-invariant)
      Euclidean:  Better for low-dimensional spaces or when magnitude matters
                  (e.g., comparing image features where intensity matters)

    For RAG text retrieval: almost always use cosine similarity.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Non-negative float. Lower = more similar.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    # WHY np.linalg.norm(a - b):
    #   a - b computes element-wise difference.
    #   linalg.norm of the difference vector = Euclidean distance.
    return float(np.linalg.norm(a - b))


# ─── RAG Retrieval Demo ───────────────────────────────────────────────────────

def demonstrate_semantic_search():
    """
    Show how embedding similarity powers RAG retrieval.

    This is the CORE operation of RAG:
      1. Embed all document chunks (at index time)
      2. Embed user query (at query time)
      3. Find chunks with highest cosine similarity to query
      4. Inject those chunks into the LLM prompt

    In production: a vector database does steps 1-3 at scale.
    Here we compute it manually to understand the math.
    """

    print("\n" + "="*65)
    print("SEMANTIC SEARCH WITH EMBEDDINGS (RAG Core Operation)")
    print("="*65)

    # Document corpus — these would be chunks from your knowledge base
    # WHY diverse topics:
    #   Shows that semantically similar content scores high even when
    #   it doesn't share exact keywords with the query.
    documents = [
        "Vector databases store embedding vectors and support fast similarity search.",
        "Cosine similarity measures the angle between two vectors in high-dimensional space.",
        "Python is a popular programming language for data science and machine learning.",
        "Retrieval-Augmented Generation combines LLMs with external knowledge retrieval.",
        "A neural network learns by adjusting weights through backpropagation.",
        "The Eiffel Tower is located in Paris, France.",
        "RAG systems retrieve relevant document chunks before generating answers.",
        "Transformers use attention mechanisms to process sequences in parallel.",
    ]

    # User query — what we want to find relevant documents for
    query = "How does RAG retrieval work with vectors?"

    print(f"\n  Query: \"{query}\"")
    print(f"\n  Embedding {len(documents)} documents + query...")

    # Step 1: Embed the query
    # WHY embed the query separately:
    #   Query embedding uses "query" input_type (optimization for search).
    #   Document embeddings use "document" input_type (optimization for retrieval).
    query_embedding = _mock_embedding(query)  # Use real get_embedding() in production

    # Step 2: Embed all documents
    doc_embeddings = [_mock_embedding(doc) for doc in documents]

    # Step 3: Compute similarity between query and each document
    scores = []
    for doc, doc_emb in zip(documents, doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        dist = euclidean_distance(query_embedding, doc_emb)
        scores.append((sim, dist, doc))

    # Step 4: Sort by similarity (descending) — most similar first
    # WHY sorted() with reverse=True:
    #   Higher cosine similarity = more relevant.
    #   We want top-K most relevant chunks at the top.
    scores.sort(key=lambda x: x[0], reverse=True)

    print(f"\n  Results (sorted by cosine similarity to query):")
    print(f"  {'Rank':<5} {'Cosine':<8} {'L2 Dist':<10} Document")
    print(f"  {'─'*5} {'─'*8} {'─'*10} {'─'*45}")

    for rank, (sim, dist, doc) in enumerate(scores, 1):
        # WHY highlight top 3:
        #   In RAG, you typically retrieve top-K (K=3 to 10) chunks.
        #   These are the chunks injected into the LLM prompt.
        marker = "→" if rank <= 3 else " "
        print(f"  {marker}{rank:<4} {sim:>+.4f}  {dist:>8.4f}   {doc[:60]}...")

    print(f"\n  → Top 3 chunks would be injected into the RAG prompt.")
    print(f"  → The LLM answers ONLY based on these retrieved chunks.")
    print(f"\n  NOTE: With mock embeddings, similarity is hash-based, not semantic.")
    print(f"  With real Voyage/OpenAI embeddings, semantic relevance is accurate.")


def demonstrate_embedding_dimensions():
    """
    Show the relationship between embedding dimensionality and information capacity.

    WHY dimensions matter for RAG:
      More dimensions = richer semantic representation = better retrieval.
      But also = larger storage and slower search.

      Common embedding dimensions:
        text-embedding-ada-002 (OpenAI):  1,536 dimensions
        voyage-3 (Anthropic recommended): 1,024 dimensions
        BGE-large (open source):          1,024 dimensions
        all-MiniLM-L6-v2 (lightweight):    384 dimensions

      Rule of thumb for RAG:
        ≥ 768 dimensions for production
        ≥ 1024 for high-quality semantic search
    """

    print("\n" + "="*65)
    print("EMBEDDING DIMENSIONS: Capacity vs Cost Trade-off")
    print("="*65)

    sentence = "How does retrieval-augmented generation work?"

    dim_configs = [
        (64,   "Too small: loses semantic nuance"),
        (128,  "Minimal: useful for demos and learning"),
        (384,  "Lightweight: all-MiniLM-L6-v2, fast inference"),
        (768,  "Medium: BERT-base, acceptable quality"),
        (1024, "Production: Voyage-3, BGE-large (recommended)"),
        (1536, "High: text-embedding-ada-002 (OpenAI standard)"),
    ]

    print(f"\n  Text: \"{sentence}\"")
    print(f"\n  {'Dims':<6} {'Bytes':<8} {'Description'}")
    print(f"  {'─'*6} {'─'*8} {'─'*45}")

    for dims, desc in dim_configs:
        # WHY dims × 4 bytes:
        #   Each dimension is a float32 value = 4 bytes.
        #   1024 dims = 4,096 bytes = ~4KB per vector.
        #   At 1M chunks: 1,024 × 4 × 1,000,000 = ~4GB of vectors.
        bytes_per_vec = dims * 4
        print(f"  {dims:<6} {bytes_per_vec:<8} {desc}")

    # Storage calculation for a real RAG system
    print(f"\n  STORAGE ESTIMATE FOR PRODUCTION RAG SYSTEM:")
    print(f"  Assumptions: 1M document chunks, voyage-3 (1024 dims)")

    chunks        = 1_000_000
    dims          = 1024
    bytes_per_vec = dims * 4
    total_bytes   = chunks * bytes_per_vec
    total_gb      = total_bytes / (1024 ** 3)

    print(f"  Vectors only:     {total_gb:.1f} GB")
    print(f"  With metadata:    {total_gb * 1.5:.1f} GB (estimate)")
    print(f"  With index (HNSW): {total_gb * 3:.1f} GB (estimate)")
    print(f"  → Phase 3 (Vector Databases) covers storage and indexing strategies.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demonstrate_semantic_search()
    demonstrate_embedding_dimensions()
