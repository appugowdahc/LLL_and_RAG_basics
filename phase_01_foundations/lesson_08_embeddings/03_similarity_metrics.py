"""
FILE: 03_similarity_metrics.py
LESSON: Phase 1 - Lesson 8 - Embeddings
TOPIC: Similarity metrics — cosine, dot product, euclidean from scratch

WHAT THIS FILE TEACHES:
  - Implementing all three metrics from scratch with numpy
  - WHY each metric behaves differently on embedding vectors
  - When to use cosine vs dot product vs euclidean
  - How magnitude affects each metric (and why normalization fixes it)
  - Performance comparison: dot product vs cosine on unit vectors
  - The relationship between distance and similarity for retrieval ranking

INSTALL:
  pip install numpy
"""

import math
import time
import numpy as np


# ─── 1. Cosine Similarity ─────────────────────────────────────────────────────

def cosine_similarity_manual(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity: measures the angle between two vectors.

    Formula:
      cos(θ) = (A · B) / (||A|| × ||B||)

    Step by step:
      1. Compute the dot product: Σ(a_i × b_i)
         This is related to the cosine of the angle between the vectors.
      2. Compute each vector's L2 norm (magnitude): √(Σ a_i²)
      3. Divide the dot product by the product of the norms.
         This "cancels out" the magnitude, leaving only the angle.

    Return range: [-1.0, +1.0]
      +1.0: identical direction (semantically identical)
       0.0: perpendicular (semantically unrelated)
      -1.0: opposite direction (semantically opposite)
    """

    # WHY dot product:
    #   From the geometric definition: A·B = |A||B|cos(θ)
    #   Dividing by |A||B| isolates cos(θ).
    dot_product = np.dot(a, b)           # Σ(a_i × b_i)

    # WHY np.linalg.norm:
    #   Computes L2 norm = √(Σ x_i²). The "length" of the vector in N-D space.
    norm_a      = np.linalg.norm(a)
    norm_b      = np.linalg.norm(b)

    # WHY + 1e-10:
    #   Guard against division by zero if either vector is the zero vector.
    #   In practice embedding vectors are never zero, but defensive coding prevents
    #   silent NaN results.
    denominator = norm_a * norm_b + 1e-10

    # WHY np.clip to [-1, 1]:
    #   Floating-point arithmetic can produce values like 1.0000000000000002.
    #   math.acos(1.0000000000000002) raises ValueError — clip prevents this.
    return float(np.clip(dot_product / denominator, -1.0, 1.0))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance = 1 - cosine_similarity.
    Converts similarity to a distance metric (higher = more different).

    WHY cosine distance (not similarity) for some vector DBs:
      Some vector databases (Qdrant, Weaviate) use DISTANCE for indexing.
      Lower distance = more similar (opposite of similarity).
      Range: [0, 2] where 0 = identical, 2 = opposite.
    """
    return 1.0 - cosine_similarity_manual(a, b)


# ─── 2. Dot Product ───────────────────────────────────────────────────────────

def dot_product_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dot product similarity: the raw dot product of two vectors.

    Formula: A · B = Σ(a_i × b_i)

    For UNIT VECTORS (||v|| = 1):
      A · B = |A||B|cos(θ) = 1 × 1 × cos(θ) = cos(θ)
      Dot product IS cosine similarity when vectors are normalized.

    For NON-UNIT VECTORS:
      Dot product conflates angle AND magnitude.
      A long vector has high dot product with everything — misleading.
      Use cosine similarity instead to factor out magnitude.

    WHY vector DBs prefer dot product over cosine:
      When vectors ARE normalized, dot product = cosine.
      Dot product avoids the two norm computations → 2× faster.
      Major vector DBs (Pinecone, FAISS IP index) use dot product and
      REQUIRE pre-normalized vectors.

    Return range: unbounded when vectors are not normalized.
      For unit vectors: same as cosine [-1, 1].
    """
    return float(np.dot(a, b))   # Σ(a_i × b_i)


# ─── 3. Euclidean Distance ────────────────────────────────────────────────────

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance: straight-line distance between two points in N-D space.

    Formula: d(A, B) = √(Σ(a_i - b_i)²)

    For UNIT VECTORS:
      There is a direct relationship to cosine similarity:
      d(A, B)² = 2 - 2 × cosine_similarity(A, B)
      So euclidean distance and cosine similarity give the SAME ranking
      for normalized vectors.

    WHY euclidean is RARELY used for semantic embeddings:
      It measures GEOMETRIC distance, not just angular direction.
      If a document model scales vectors by document length, long documents
      would have larger magnitudes → larger euclidean distance from everything.
      Cosine similarity normalizes this away.

    USE CASES for euclidean in embeddings:
      - Image embeddings where magnitude encodes feature strength
      - Embeddings where magnitude conveys information (e.g., popularity-weighted)
      - When magnitude-based reranking is desired

    Return range: [0, ∞) where 0 = identical vectors.
    """
    diff = a - b                         # element-wise difference
    return float(np.sqrt(np.sum(diff**2)))   # √(Σ diff²)


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Convert euclidean distance to a similarity score in [0, 1].
    Used when APIs expect similarity (higher = better) but you need euclidean.

    WHY 1/(1+d):
      Maps distance [0, ∞) to similarity (1, 0].
      d=0 → similarity=1.0 (identical)
      d→∞ → similarity→0.0 (completely different)
      Never negative — unlike cosine which can be [-1, 1].
    """
    d = euclidean_distance(a, b)
    return 1.0 / (1.0 + d)   # WHY +1: prevents division by zero when d=0


# ─── 4. Magnitude Effect Demonstration ────────────────────────────────────────

def magnitude_effect_demo():
    """
    Demonstrate how magnitude affects each metric.
    Shows WHY normalization is essential for semantic similarity.
    """

    print("=" * 65)
    print("MAGNITUDE EFFECT: Why normalization matters")
    print("=" * 65)

    # Base vectors (semantic content)
    base_a = np.array([1.0, 0.5, -0.3, 0.8])   # "networking" direction
    base_b = np.array([0.9, 0.4, -0.2, 0.7])   # similar to base_a

    # Scaled versions (same DIRECTION, different MAGNITUDE)
    # WHY scale: simulates what happens if embedding magnitudes vary by doc length
    scale_factor = 5.0
    scaled_a = base_a * scale_factor   # same direction, 5× longer

    print(f"\n  Vectors:")
    print(f"    base_a:   {base_a}  |norm|={np.linalg.norm(base_a):.3f}")
    print(f"    base_b:   {base_b}  |norm|={np.linalg.norm(base_b):.3f}")
    print(f"    scaled_a: {scaled_a}  |norm|={np.linalg.norm(scaled_a):.3f}")

    print(f"\n  base_a vs base_b (same scale):")
    print(f"    Cosine similarity:    {cosine_similarity_manual(base_a, base_b):.4f}")
    print(f"    Dot product:          {dot_product_similarity(base_a, base_b):.4f}")
    print(f"    Euclidean distance:   {euclidean_distance(base_a, base_b):.4f}")

    print(f"\n  scaled_a vs base_b (scaled_a is 5× longer, SAME direction):")
    print(f"    Cosine similarity:    {cosine_similarity_manual(scaled_a, base_b):.4f}  ← UNCHANGED ✓")
    print(f"    Dot product:          {dot_product_similarity(scaled_a, base_b):.4f}  ← 5× LARGER ✗ (misleading)")
    print(f"    Euclidean distance:   {euclidean_distance(scaled_a, base_b):.4f}  ← LARGER ✗ (misleading)")

    print(f"""
  KEY TAKEAWAY:
    Cosine similarity is INVARIANT to magnitude scaling.
    Dot product and euclidean are AFFECTED by magnitude.
    For semantic similarity, use cosine or normalize first then use dot product.
""")


# ─── 5. Speed Comparison ─────────────────────────────────────────────────────

def speed_comparison(dims: int = 1024, n_docs: int = 10_000, n_trials: int = 5):
    """
    Compare speed of each similarity metric for retrieval at scale.
    Shows why dot product dominates production vector DBs.
    """

    print("\n" + "=" * 65)
    print(f"SPEED COMPARISON: {n_docs:,} documents × {dims} dimensions")
    print("=" * 65)

    rng = np.random.RandomState(42)

    # Unit-normalized document matrix (n_docs × dims)
    # WHY pre-normalize: all production vector DBs pre-normalize at index time,
    # not at query time. Query time normalization would double the cost.
    raw_docs  = rng.randn(n_docs, dims)
    norms     = np.linalg.norm(raw_docs, axis=1, keepdims=True)
    docs      = raw_docs / norms   # unit-normalized

    # Unit-normalized query
    query_raw = rng.randn(dims)
    query     = query_raw / np.linalg.norm(query_raw)

    print()
    for name, fn in [
        ("Dot product (numpy matmul)", lambda: np.dot(docs, query)),
        ("Cosine (full formula)",      lambda: np.array([cosine_similarity_manual(docs[i], query) for i in range(n_docs)])),
        ("Euclidean",                  lambda: np.array([euclidean_distance(docs[i], query) for i in range(n_docs)])),
    ]:
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            result = fn()
            times.append(time.perf_counter() - t0)

        median_ms = sorted(times)[n_trials // 2] * 1000
        print(f"  {name:<40} {median_ms:>8.2f} ms  ({n_docs/median_ms*1000:,.0f} docs/sec)")

    print(f"""
  WHY DOT PRODUCT IS FASTEST:
    np.dot(docs, query) is a single BLAS matrix-vector multiply — highly optimized.
    Cosine computed per-element in Python is 100-1000× slower.
    In production: pre-normalize ALL vectors at index time, then use dot product.
    FAISS and other vector DBs do this automatically with "Inner Product" (IP) index.
""")


# ─── 6. Metric Selection Guide ────────────────────────────────────────────────

def metric_selection_guide():
    """
    Decision table for choosing the right similarity metric.
    """

    print("\n" + "=" * 65)
    print("METRIC SELECTION GUIDE")
    print("=" * 65)

    scenarios = [
        {
            "scenario": "Semantic search with normalized embedding vectors",
            "metric":   "Dot product (= cosine for unit vectors)",
            "why":      "Fastest, most widely supported by vector DBs",
        },
        {
            "scenario": "Semantic search, vectors NOT pre-normalized",
            "metric":   "Cosine similarity",
            "why":      "Magnitude-invariant — essential without normalization",
        },
        {
            "scenario": "Image embeddings where magnitude = feature strength",
            "metric":   "Euclidean distance",
            "why":      "Magnitude differences carry information, don't discard them",
        },
        {
            "scenario": "Sparse keyword search (TF-IDF vectors)",
            "metric":   "Dot product",
            "why":      "TF-IDF vectors ARE normalized per-document in BM25",
        },
        {
            "scenario": "Hybrid dense+sparse search",
            "metric":   "Cosine for dense, BM25 for sparse, then fuse",
            "why":      "Different metrics per component, combine scores afterward",
        },
        {
            "scenario": "Clustering embeddings (k-means)",
            "metric":   "Euclidean",
            "why":      "k-means centroid update requires euclidean space arithmetic",
        },
    ]

    for s in scenarios:
        print(f"\n  Scenario: {s['scenario']}")
        print(f"  Metric:   {s['metric']}")
        print(f"  Why:      {s['why']}")


# ─── 7. Relationship Between Distance and Similarity ─────────────────────────

def distance_similarity_relationship():
    """
    Show the mathematical relationship between cosine similarity and euclidean
    distance for unit-normalized vectors.
    """

    print("\n" + "=" * 65)
    print("DISTANCE ↔ SIMILARITY: The relationship for unit vectors")
    print("=" * 65)

    print(f"""
  For unit-normalized vectors (|v| = 1):

  IDENTITY:
    euclidean²(A, B) = 2 - 2 × cosine_sim(A, B)
    euclidean(A, B)  = √(2 - 2 × cosine_sim)

  This means for unit vectors, cosine similarity and euclidean distance
  give IDENTICAL RANKINGS — they are mathematically equivalent.

  LOOKUP TABLE:
    cosine_sim   euclidean_dist   interpretation
    ──────────────────────────────────────────────
    1.000        0.000            identical vectors
    0.999        0.045            near-identical
    0.950        0.316            very similar
    0.800        0.632            similar
    0.500        1.000            somewhat related
    0.000        1.414            unrelated
   -0.500        1.732            somewhat opposite
   -1.000        2.000            exact opposites (e.g. "good" vs "bad")
""")

    # Verify the identity empirically
    print("  Empirical verification:")
    rng = np.random.RandomState(7)
    for _ in range(5):
        a = rng.randn(512); a /= np.linalg.norm(a)
        b = rng.randn(512); b /= np.linalg.norm(b)

        cos_sim   = cosine_similarity_manual(a, b)
        euc_dist  = euclidean_distance(a, b)
        predicted = math.sqrt(max(0, 2 - 2 * cos_sim))  # WHY max(0,...): float precision guard

        print(f"    cos_sim={cos_sim:.4f}  euclidean={euc_dist:.4f}  predicted={predicted:.4f}  "
              f"match={'✓' if abs(euc_dist-predicted) < 1e-5 else '✗'}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    magnitude_effect_demo()
    speed_comparison(dims=1024, n_docs=10_000, n_trials=3)
    metric_selection_guide()
    distance_similarity_relationship()
