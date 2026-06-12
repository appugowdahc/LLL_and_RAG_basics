"""
FILE: 01_what_are_embeddings.py
LESSON: Phase 1 - Lesson 8 - Embeddings
TOPIC: What are embeddings? — vectors, semantic space, intuition from scratch

WHAT THIS FILE TEACHES:
  - What an embedding vector actually IS (a list of floats)
  - How meaning becomes geometry in high-dimensional space
  - Why similar concepts cluster together
  - Demonstrating semantic properties: analogies, clustering, outliers
  - Building intuition WITHOUT needing an API — pure numpy

WHY START WITH NUMPY (not an API):
  Understanding embeddings as MATH OBJECTS first prevents the "magic box" trap.
  Once you know what the vector IS, the API call is just a function that produces it.
  The math here is what the API produces — simplified to 4 dimensions for visualization.

INSTALL:
  pip install numpy
"""

import math
import numpy as np


# ─── What a Vector IS ─────────────────────────────────────────────────────────

def explain_vector_as_data():
    """
    Show what an embedding vector looks like as raw data.
    Strip away the mysticism — it's just a list of floats.
    """

    print("=" * 65)
    print("WHAT IS AN EMBEDDING VECTOR?")
    print("=" * 65)

    # A 1024-dimensional embedding (as from Voyage-3)
    # In reality these values are learned — here we show the shape
    # WHY 1024: that's the output dimension of voyage-3
    example_embedding = np.array([
        0.0234, -0.1823,  0.3412, -0.0891,  0.2234,  0.4512,
       -0.0123,  0.1892, -0.3241,  0.0512,  0.1923, -0.2341,
    ])  # Truncated for display

    print(f"\n  An embedding is just a numpy array (list of floats):")
    print(f"    Type:  {type(example_embedding)}")
    print(f"    Shape: {example_embedding.shape}  (shown: 12 of 1,024 typical dims)")
    print(f"    Dtype: {example_embedding.dtype}")
    print(f"    Values (first 12): {example_embedding.round(4)}")
    print(f"    L2 norm: {np.linalg.norm(example_embedding):.4f}  "
          f"(embedding models normalize to ~1.0)")

    print(f"""
  KEY FACTS:
    - 1,024 floats × 4 bytes (float32) = 4,096 bytes = 4 KB per embedding
    - You cannot read individual dimensions — they have no human-interpretable meaning
    - The entire 1,024-number list together encodes the semantic content
    - Two vectors with similar lists of numbers have similar meaning
""")


# ─── 4-Dimensional Semantic Space (Visualizable) ──────────────────────────────

# WHY 4 dimensions (not 1024):
#   You cannot visualize 1024 dimensions. We use 4 hand-crafted dimensions to
#   build intuition. Real embeddings have the SAME mathematical properties
#   but in 1024 dimensions instead of 4.
#
# Dimensions:
#   0: technical/non-technical  (-1 = casual, +1 = technical)
#   1: networking/non-networking (-1 = unrelated, +1 = networking)
#   2: security/non-security    (-1 = unrelated, +1 = security)
#   3: positive/negative sentiment (-1 = negative, +1 = positive)

SEMANTIC_VECTORS = {
    # Network & infrastructure texts
    "ACI Leaf-Spine topology":          np.array([ 0.9,  0.9,  0.1,  0.3]),
    "APIC controller manages fabric":   np.array([ 0.9,  0.9,  0.1,  0.3]),
    "VXLAN encapsulation protocol":     np.array([ 0.9,  0.9,  0.0,  0.2]),
    "network connectivity issue":       np.array([ 0.7,  0.9, -0.1, -0.5]),
    "link is down":                     np.array([ 0.5,  0.9, -0.1, -0.6]),
    "packet loss detected":             np.array([ 0.6,  0.9, -0.1, -0.4]),

    # Security texts
    "Hypershield eBPF enforcement":     np.array([ 0.9,  0.5,  0.9,  0.2]),
    "ISE TrustSec SGT policy":          np.array([ 0.9,  0.6,  0.9,  0.2]),
    "microsegmentation access control": np.array([ 0.8,  0.5,  0.9,  0.2]),
    "firewall rule deny":               np.array([ 0.6,  0.3,  0.9, -0.3]),

    # Off-topic texts
    "My cat is hungry":                 np.array([-0.8, -0.9, -0.9,  0.3]),
    "weather is sunny today":           np.array([-0.9, -0.9, -0.9,  0.8]),
    "great Italian restaurant":         np.array([-0.7, -0.8, -0.9,  0.9]),
}


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Unit-normalize a vector (scale to L2 norm = 1).

    WHY normalize:
      Embedding models output unit-normalized vectors.
      Normalizing makes cosine_sim = dot_product (simpler + faster).
      Without normalization, long texts would have larger magnitudes
      simply because they have more content — confounding similarity.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm   # WHY divide by norm: scales |v| to 1.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.

    Formula: cos(θ) = (a · b) / (|a| × |b|)

    WHY this formula:
      dot(a, b) = |a| |b| cos(θ)  — from the geometric definition of dot product
      Dividing by magnitudes cancels them → pure angle measurement.
      cos(0°) = 1.0  → identical direction → same meaning
      cos(90°) = 0.0 → perpendicular → unrelated
      cos(180°) = -1.0 → opposite directions → opposite meaning
    """
    a_norm = normalize(a)
    b_norm = normalize(b)
    # WHY np.clip: floating-point arithmetic can produce values like 1.0000001
    # which cause math.acos to error. Clip to [-1, 1] for safety.
    return float(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))


# ─── Demonstrate Semantic Properties ─────────────────────────────────────────

def demonstrate_semantic_clustering():
    """
    Show that semantically related texts cluster together in vector space.
    This is the CORE property that makes RAG work.
    """

    print("\n" + "=" * 65)
    print("SEMANTIC CLUSTERING: Similar texts → similar vectors")
    print("=" * 65)

    query = "network connectivity issue"
    query_vec = SEMANTIC_VECTORS[query]

    print(f"\n  Query: '{query}'")
    print(f"\n  {'Text':<40} {'Similarity':>12}  {'Relationship'}")
    print(f"  {'─'*40} {'─'*12}  {'─'*25}")

    results = []
    for text, vec in SEMANTIC_VECTORS.items():
        if text == query:
            continue
        sim = cosine_similarity(query_vec, vec)
        results.append((text, sim))

    # Sort by similarity descending
    results.sort(key=lambda x: -x[1])

    for text, sim in results:
        # Classify relationship based on similarity
        if sim > 0.95:
            rel = "Near-identical meaning"
        elif sim > 0.80:
            rel = "Same topic"
        elif sim > 0.50:
            rel = "Related domain"
        elif sim > 0.10:
            rel = "Loosely related"
        else:
            rel = "Unrelated"

        bar_len = max(0, int((sim + 1) / 2 * 20))   # map [-1,1] to [0,20]
        bar     = "█" * bar_len

        print(f"  {text:<40} {sim:>12.4f}  {rel}")

    print(f"""
  KEY INSIGHT:
    "link is down" and "packet loss detected" have HIGH similarity to
    "network connectivity issue" — even with ZERO shared words.
    The embedding model learned these all describe network problems.

    "My cat is hungry" has NEGATIVE or near-zero similarity —
    completely different semantic space.

    THIS IS WHY RAG WORKS:
      When a user asks "why is my network down?",
      the embedding of that query is close to embeddings of chunks about
      "connectivity issues", "link failures", "packet loss" — all retrieved,
      even if none use the exact words from the query.
""")


def demonstrate_analogies():
    """
    Show the famous semantic analogy property in vector space:
      King - Man + Woman ≈ Queen
    Our version: Technical + Security - Networking ≈ Security Tool
    """

    print("\n" + "=" * 65)
    print("VECTOR ANALOGIES: Arithmetic on meaning")
    print("=" * 65)

    # Vector arithmetic: "networking security" - "networking" + "non-networking" = "general security"
    v_network_security  = normalize(SEMANTIC_VECTORS["ISE TrustSec SGT policy"])
    v_networking        = normalize(SEMANTIC_VECTORS["VXLAN encapsulation protocol"])
    v_non_networking    = normalize(np.array([0.9, -0.9, 0.9, 0.2]))   # technical, non-network, security

    # WHY this analogy:
    #   "ISE TrustSec" (networking security) - networking component + non-networking
    #   should point toward something that is security but not network-specific.
    result_vec = v_network_security - v_networking + v_non_networking

    print(f"\n  Operation: 'ISE TrustSec SGT policy' - 'VXLAN protocol' + 'general security'")
    print(f"  Result vector: {result_vec.round(3)}")

    print(f"\n  Most similar to the result vector:")
    similarities = []
    for text, vec in SEMANTIC_VECTORS.items():
        sim = cosine_similarity(result_vec, vec)
        similarities.append((text, sim))
    similarities.sort(key=lambda x: -x[1])

    for text, sim in similarities[:5]:
        print(f"    {sim:.4f} — {text}")

    print(f"\n  In real 1024-dimensional space:")
    print(f"  king - man + woman ≈ queen")
    print(f"  Paris - France + Italy ≈ Rome")
    print(f"  These analogies emerge from the training data — the model")
    print(f"  absorbed relationships between concepts from billions of text examples.")


def demonstrate_outlier_detection():
    """
    Show how embeddings can detect the 'odd one out' in a list.
    Practical for data quality checks in RAG pipelines.
    """

    print("\n" + "=" * 65)
    print("OUTLIER DETECTION: Finding the odd one out")
    print("=" * 65)

    groups = [
        {
            "label": "Network troubleshooting terms",
            "items": [
                ("network connectivity issue", SEMANTIC_VECTORS["network connectivity issue"]),
                ("link is down",              SEMANTIC_VECTORS["link is down"]),
                ("packet loss detected",      SEMANTIC_VECTORS["packet loss detected"]),
                ("My cat is hungry",          SEMANTIC_VECTORS["My cat is hungry"]),  # ← OUTLIER
            ],
        },
        {
            "label": "Cisco security technologies",
            "items": [
                ("Hypershield eBPF enforcement",     SEMANTIC_VECTORS["Hypershield eBPF enforcement"]),
                ("ISE TrustSec SGT policy",          SEMANTIC_VECTORS["ISE TrustSec SGT policy"]),
                ("microsegmentation access control", SEMANTIC_VECTORS["microsegmentation access control"]),
                ("VXLAN encapsulation protocol",     SEMANTIC_VECTORS["VXLAN encapsulation protocol"]),  # ← less security-specific
            ],
        },
    ]

    for group in groups:
        print(f"\n  Group: {group['label']}")
        texts  = [item[0] for item in group["items"]]
        vecs   = [item[1] for item in group["items"]]

        # Compute avg similarity of each item to all others
        avg_sims = []
        for i, (text, vec) in enumerate(zip(texts, vecs)):
            other_vecs = [v for j, v in enumerate(vecs) if j != i]
            sim_to_others = np.mean([cosine_similarity(vec, other) for other in other_vecs])
            avg_sims.append((text, sim_to_others))

        avg_sims.sort(key=lambda x: x[1])  # Lowest avg similarity = outlier

        for text, avg_sim in avg_sims:
            marker = " ← OUTLIER (lowest cohesion)" if avg_sim == avg_sims[0][1] else ""
            print(f"    {avg_sim:.4f} avg similarity — {text}{marker}")

    print(f"\n  RAG USE CASE:")
    print(f"  Before indexing chunks, compute avg pairwise similarity within each")
    print(f"  document section. Very low cohesion may indicate misformatted or")
    print(f"  garbled text that should be filtered before embedding.")


# ─── Embedding Storage and Scale ─────────────────────────────────────────────

def embedding_storage_analysis():
    """
    Calculate storage requirements for different embedding configurations.
    """

    print("\n" + "=" * 65)
    print("EMBEDDING STORAGE AT SCALE")
    print("=" * 65)

    configs = [
        ("voyage-3",        1024, "float32", 4),
        ("voyage-3-lite",    512, "float32", 4),
        ("text-ada-002",    1536, "float32", 4),
        ("voyage-3 (int8)", 1024, "int8",    1),
        ("voyage-3 (256d)", 256,  "float32", 4),  # Matryoshka truncated
    ]

    doc_counts = [10_000, 100_000, 1_000_000, 10_000_000]

    print(f"\n  {'Model':<25} {'Dims':>5} {'Dtype':>8} {'Bytes/Vec':>10} "
          + "  ".join(f"{n//1000}K docs" if n < 1_000_000 else f"{n//1_000_000}M docs"
                      for n in doc_counts))
    print(f"  {'─'*25} {'─'*5} {'─'*8} {'─'*10}  " + "  ".join(["─"*10]*len(doc_counts)))

    for model, dims, dtype, bytes_per_val in configs:
        bytes_per_vec = dims * bytes_per_val
        sizes = [
            n * bytes_per_vec / (1024**3)   # GB
            for n in doc_counts
        ]
        sizes_str = "  ".join(f"{s:.2f} GB" if s < 1000 else f"{s/1024:.2f} TB"
                               for s in sizes)
        print(f"  {model:<25} {dims:>5} {dtype:>8} {bytes_per_vec:>10,}  {sizes_str}")

    print(f"""
  PRACTICAL TAKEAWAYS:
    For < 100K documents: any configuration works, even in-memory with numpy.
    For 1M documents: voyage-3 at float32 = ~4 GB → needs a vector database (FAISS, Qdrant).
    For 10M+ documents: use int8 quantization or smaller dimensions.
    HNSW index overhead: approximately 1.5× the raw embedding size.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    explain_vector_as_data()
    demonstrate_semantic_clustering()
    demonstrate_analogies()
    demonstrate_outlier_detection()
    embedding_storage_analysis()
