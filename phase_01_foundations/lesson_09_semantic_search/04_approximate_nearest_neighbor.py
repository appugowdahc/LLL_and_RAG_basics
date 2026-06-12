"""
FILE: 04_approximate_nearest_neighbor.py
LESSON: Phase 1 - Lesson 9 - Semantic Search
TOPIC: Approximate Nearest Neighbor (ANN) — the algorithms behind fast vector search

WHAT THIS FILE TEACHES:
  - WHY exact search fails at scale (O(N) per query)
  - HNSW: the layered graph that powers Qdrant, Pinecone, and Weaviate
  - IVF (Inverted File Index): cluster-based coarse-to-fine search
  - PQ (Product Quantization): compressing vectors to reduce memory
  - The recall-latency-memory triangle: can only optimize 2 of 3
  - Practical guidance: which algorithm to use at what scale

NO LIBRARIES NEEDED: Conceptual implementations with numpy.

INSTALL: pip install numpy
"""

import math
import time
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─── Helper ───────────────────────────────────────────────────────────────────

def mock_embed(text: str, dims: int = 64) -> np.ndarray:
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.RandomState(seed)
    v    = rng.randn(dims).astype(np.float32)
    return v / np.linalg.norm(v)

def gen_random_vectors(n: int, dims: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    raw = rng.randn(n, dims).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


# ─── Algorithm 1: Brute Force (Baseline) ─────────────────────────────────────

def brute_force_search(
    corpus:  np.ndarray,   # (N, D)
    query:   np.ndarray,   # (D,)
    top_k:   int,
) -> np.ndarray:
    """
    Exact nearest neighbor by exhaustive dot product.
    O(N × D) per query.

    Used as ground truth to measure ANN recall.
    """
    scores = np.dot(corpus, query)   # (N,) dot products
    return np.argsort(-scores)[:top_k]


# ─── Algorithm 2: IVF (Inverted File Index) ───────────────────────────────────

class IVFIndex:
    """
    IVF: Cluster the corpus into C centroids (using k-means),
    then at query time search only the nearest n_probe clusters.

    ANALOGY:
      Librarian analogy: books are organized into C rooms by topic.
      To find a book, check the nearest n_probe rooms instead of all rooms.

    FLOW:
      BUILD:
        1. Run k-means on corpus → C centroid vectors
        2. Assign each document to its nearest centroid
        3. Store documents in "posting lists" per centroid

      QUERY:
        1. Find the n_probe centroids nearest to the query
        2. Retrieve all documents from those n_probe posting lists
        3. Score documents by exact dot product
        4. Return top-K

    RECALL vs SPEED:
      n_probe=1:   fast but low recall (~50-70%)
      n_probe=8:   good balance
      n_probe=C:   exact (but no speedup vs brute force)

    KEY PARAMETER:
      n_lists (C): number of clusters.
        Typical: C = sqrt(N). For 1M docs: C ≈ 1000 clusters.
        Too few: posting lists too large → slow.
        Too many: clusters too small → poor coverage per probe.
    """

    def __init__(self, n_lists: int = 16):
        """
        Args:
            n_lists: Number of IVF clusters (centroids).
        """
        self.n_lists     = n_lists
        self.centroids:  Optional[np.ndarray] = None   # (C, D)
        self.lists:      list[list[int]] = []           # posting lists: cluster → [doc_idx, ...]
        self._corpus:    Optional[np.ndarray] = None

    def _kmeans(self, vectors: np.ndarray, n_clusters: int, n_iter: int = 20) -> np.ndarray:
        """
        Simple k-means++ initialization + Lloyd's algorithm.

        WHY k-means++ (not random init):
          Random init can place multiple centroids in the same dense region.
          k-means++ spreads initial centroids far apart → better coverage.
        """
        N, D = vectors.shape
        rng  = np.random.RandomState(7)

        # k-means++ init: first centroid random, subsequent chosen with probability
        # proportional to squared distance from nearest existing centroid.
        centroids = [vectors[rng.randint(N)]]

        for _ in range(n_clusters - 1):
            # WHY min axis=0: squared distance to nearest existing centroid
            dists = np.min(
                np.sum((vectors[:, None] - np.array(centroids)[None]) ** 2, axis=-1),
                axis=1
            )
            probs = dists / dists.sum()
            centroids.append(vectors[rng.choice(N, p=probs)])

        centroids = np.array(centroids)

        for _ in range(n_iter):
            # Assign each vector to nearest centroid
            # WHY neg dot product (not euclidean): unit vectors → cosine = dot product
            assignments = np.argmax(np.dot(vectors, centroids.T), axis=1)

            new_centroids = np.zeros_like(centroids)
            for c in range(n_clusters):
                mask = assignments == c
                if mask.any():
                    new_centroids[c] = vectors[mask].mean(axis=0)
                    norm = np.linalg.norm(new_centroids[c])
                    if norm > 0:
                        new_centroids[c] /= norm   # WHY normalize: maintain unit sphere

            if np.allclose(centroids, new_centroids, atol=1e-4):
                break   # WHY early stop: converged → no more assignment changes
            centroids = new_centroids

        return centroids

    def build(self, vectors: np.ndarray):
        """Build the IVF index."""
        self._corpus   = vectors
        self.centroids = self._kmeans(vectors, self.n_lists)

        # Assign each document to its nearest centroid
        # WHY argmax(dot): unit vectors → highest dot product = nearest centroid
        assignments = np.argmax(np.dot(vectors, self.centroids.T), axis=1)

        self.lists = [[] for _ in range(self.n_lists)]
        for doc_idx, centroid_idx in enumerate(assignments):
            self.lists[centroid_idx].append(doc_idx)

    def search(
        self,
        query:    np.ndarray,
        top_k:    int = 10,
        n_probe:  int = 4,
    ) -> np.ndarray:
        """
        IVF search: probe n_probe clusters, rank candidates, return top-K.

        Args:
            query:   Unit-normalized query vector.
            top_k:   Results to return.
            n_probe: Number of clusters to search.
                     Higher → better recall, slower.
        """
        # Find nearest n_probe centroids
        centroid_scores   = np.dot(self.centroids, query)            # (C,)
        nearest_centroids = np.argsort(-centroid_scores)[:n_probe]  # WHY -: descending

        # Collect candidate document indices from those clusters
        candidate_indices = []
        for c_idx in nearest_centroids:
            candidate_indices.extend(self.lists[c_idx])

        if not candidate_indices:
            return np.array([], dtype=int)

        candidate_indices = np.array(candidate_indices)
        candidate_vectors = self._corpus[candidate_indices]      # (n_candidates, D)

        # Score candidates by exact dot product
        scores = np.dot(candidate_vectors, query)

        # Return top-K from candidates
        local_top = np.argsort(-scores)[:top_k]
        return candidate_indices[local_top]


# ─── Algorithm 3: Product Quantization (PQ) Memory Compression ────────────────

class ProductQuantizer:
    """
    PQ compresses vectors from float32 (4 bytes/dim) to uint8 (1 byte per subvector).
    Typical compression: 32-64× reduction in memory.

    HOW IT WORKS:
      1. Split the D-dimensional vector into M sub-vectors of D/M dimensions.
         e.g., D=1024, M=8 → 8 sub-vectors of 128 dims each.

      2. For each sub-space, train a codebook of K=256 centroids.
         (K=256 allows 1 uint8 to encode the cluster assignment.)

      3. To compress a vector: for each sub-space, find the nearest centroid.
         Store the centroid INDEX (0-255) not the actual sub-vector.
         1024 float32 = 4096 bytes → 8 uint8 = 8 bytes (512× compression!)

      4. To compute approximate dot product:
         Use precomputed lookup tables: for each sub-space, compute the
         dot product between the query sub-vector and all 256 centroids.
         Approximate dot product = sum of K lookups (one per sub-space).

    MEMORY SAVINGS:
      Without PQ: 1M × 1024 dims × 4 bytes = 4 GB
      With PQ M=8: 1M × 8 bytes = 8 MB (!!)
      With HNSW+PQ (FAISS IndexHNSWPQ): best of both worlds.

    ACCURACY LOSS:
      PQ introduces approximation error in similarity scores.
      Higher M (more sub-spaces) = better approximation but more memory.
      PQ is used with a re-ranking step: get top-100 via PQ, re-rank exactly.
    """

    def __init__(self, n_subspaces: int = 8, n_codes: int = 256):
        """
        Args:
            n_subspaces: M — number of sub-spaces to split each vector into.
            n_codes:     K — codebook size per sub-space (max 256 for uint8).
        """
        self.M          = n_subspaces
        self.K          = n_codes
        self.codebooks: Optional[np.ndarray] = None   # (M, K, D/M)
        self.codes_:    Optional[np.ndarray] = None   # (N, M) uint8

    def fit(self, vectors: np.ndarray):
        """
        Train codebooks using k-means on each sub-space.

        Args:
            vectors: (N, D) float32 corpus vectors.
        """
        N, D = vectors.shape
        assert D % self.M == 0, f"D={D} must be divisible by M={self.M}"
        sub_dim = D // self.M

        self.codebooks = np.zeros((self.M, self.K, sub_dim), dtype=np.float32)

        for m in range(self.M):
            # Extract sub-vectors for this sub-space
            sub_vectors = vectors[:, m * sub_dim : (m + 1) * sub_dim]

            # Simple k-means for this sub-space
            rng = np.random.RandomState(m)
            indices    = rng.choice(N, size=self.K, replace=False)
            centroids  = sub_vectors[indices].copy()

            for _ in range(10):   # WHY 10 iterations: quick convergence for sub-spaces
                dists = np.sum((sub_vectors[:, None] - centroids[None]) ** 2, axis=-1)
                assignments = np.argmin(dists, axis=1)

                new_centroids = np.array([
                    sub_vectors[assignments == k].mean(axis=0) if (assignments == k).any()
                    else centroids[k]
                    for k in range(self.K)
                ])
                centroids = new_centroids

            self.codebooks[m] = centroids

    def compress(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compress vectors to PQ codes.

        Returns:
            (N, M) uint8 array of codebook indices.
        """
        N, D  = vectors.shape
        sub_d = D // self.M
        codes = np.zeros((N, self.M), dtype=np.uint8)

        for m in range(self.M):
            sub_vecs = vectors[:, m * sub_d : (m + 1) * sub_d]
            # Find nearest centroid for each sub-vector
            dists       = np.sum((sub_vecs[:, None] - self.codebooks[m][None]) ** 2, axis=-1)
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

        return codes

    def approx_dot_product(
        self,
        codes:   np.ndarray,   # (N, M) uint8
        query:   np.ndarray,   # (D,)
    ) -> np.ndarray:
        """
        Approximate dot products using lookup tables.

        WHY lookup tables (not recompute):
          For each sub-space m, precompute dot product between query
          sub-vector and ALL K centroids. Store in a (M, K) table.
          Then for each document: sum the K table lookups → O(M) per doc.
          Without PQ: O(D) per doc.
          Speedup: D/M times faster. For D=1024, M=8: 128× faster score computation.
        """
        D     = query.shape[0]
        sub_d = D // self.M

        # Precompute lookup table: (M, K) — dot product of query sub-vector with each centroid
        lookup = np.zeros((self.M, self.K), dtype=np.float32)
        for m in range(self.M):
            q_sub           = query[m * sub_d : (m + 1) * sub_d]
            lookup[m]       = np.dot(self.codebooks[m], q_sub)   # (K,)

        # For each document: sum lookup values for its M codes
        # WHY vectorized: avoids Python loop over N documents
        approx_scores = np.zeros(len(codes), dtype=np.float32)
        for m in range(self.M):
            approx_scores += lookup[m][codes[:, m]]   # index into m-th lookup using codes

        return approx_scores

    def memory_bytes(self, N: int, D: int) -> dict:
        """Calculate memory savings from PQ compression."""
        raw_bytes = N * D * 4   # float32
        pq_bytes  = N * self.M  # uint8 codes
        cb_bytes  = self.M * self.K * (D // self.M) * 4  # codebooks
        total_pq  = pq_bytes + cb_bytes

        return {
            "raw_float32_mb":  raw_bytes  / (1024**2),
            "pq_codes_mb":     pq_bytes   / (1024**2),
            "codebooks_mb":    cb_bytes   / (1024**2),
            "total_pq_mb":     total_pq   / (1024**2),
            "compression_ratio": raw_bytes / total_pq,
        }


# ─── Algorithm Comparison ─────────────────────────────────────────────────────

def compare_ann_algorithms():
    """
    Compare brute force, IVF, and PQ on recall and speed.
    """

    print("=" * 65)
    print("ANN ALGORITHM COMPARISON")
    print("=" * 65)

    N    = 2_000
    D    = 128
    K    = 5
    rng  = np.random.RandomState(42)

    corpus = gen_random_vectors(N, D)

    # ── Build indexes ─────────────────────────────────────────────────────────
    ivf = IVFIndex(n_lists=20)
    t0  = time.perf_counter()
    ivf.build(corpus)
    ivf_build = time.perf_counter() - t0

    pq = ProductQuantizer(n_subspaces=8, n_codes=64)
    t0  = time.perf_counter()
    pq.fit(corpus)
    codes = pq.compress(corpus)
    pq_build = time.perf_counter() - t0

    print(f"\n  Corpus: {N:,} docs × {D} dims")
    print(f"  IVF build:   {ivf_build*1000:.0f}ms  (n_lists=20)")
    print(f"  PQ build:    {pq_build*1000:.0f}ms  (M=8, K=64)")

    # ── Benchmark queries ─────────────────────────────────────────────────────
    n_queries = 50
    queries   = gen_random_vectors(n_queries, D, seed=99)

    bf_times   = []
    ivf_times  = []
    pq_times   = []
    ivf_recalls = []
    pq_recalls  = []

    for q in queries:
        # Brute force (ground truth)
        t0 = time.perf_counter()
        bf_top = brute_force_search(corpus, q, K)
        bf_times.append(time.perf_counter() - t0)
        bf_set = set(bf_top.tolist())

        # IVF
        t0 = time.perf_counter()
        ivf_top = ivf.search(q, top_k=K, n_probe=4)
        ivf_times.append(time.perf_counter() - t0)
        ivf_recalls.append(len(set(ivf_top.tolist()) & bf_set) / K)

        # PQ
        t0 = time.perf_counter()
        pq_scores = pq.approx_dot_product(codes, q)
        pq_top    = np.argsort(-pq_scores)[:K]
        pq_times.append(time.perf_counter() - t0)
        pq_recalls.append(len(set(pq_top.tolist()) & bf_set) / K)

    avg = lambda lst: sum(lst)/len(lst)*1000

    print(f"\n  {'Algorithm':<25} {'Avg query ms':>14} {'Recall@'+str(K):>10}  Notes")
    print(f"  {'─'*25} {'─'*14} {'─'*10}  {'─'*25}")
    print(f"  {'Brute Force':<25} {avg(bf_times):>12.3f}ms {'1.000':>10}  exact, O(N×D)")
    print(f"  {'IVF (n_probe=4)':<25} {avg(ivf_times):>12.3f}ms {avg(ivf_recalls)/1000:>10.3f}  approx, O(N/C × D)")
    print(f"  {'PQ (M=8, K=64)':<25} {avg(pq_times):>12.3f}ms {avg(pq_recalls)/1000:>10.3f}  approx, O(M × K + N × M)")

    # ── Memory analysis ───────────────────────────────────────────────────────
    mem = pq.memory_bytes(N, D)
    raw_mb = N * D * 4 / (1024**2)

    print(f"\n  Memory comparison ({N:,} docs × {D} dims):")
    print(f"    Float32 (no compression): {raw_mb:.2f} MB")
    print(f"    PQ codes:                {mem['pq_codes_mb']:.2f} MB")
    print(f"    PQ codebooks:            {mem['codebooks_mb']:.2f} MB")
    print(f"    PQ total:                {mem['total_pq_mb']:.2f} MB")
    print(f"    Compression ratio:       {mem['compression_ratio']:.1f}×")


def ann_selection_guide():
    """
    Practical guide for choosing the right ANN algorithm.
    """

    print("\n" + "=" * 65)
    print("ANN ALGORITHM SELECTION GUIDE")
    print("=" * 65)

    print(f"""
  CORPUS SIZE         ALGORITHM           NOTES
  ─────────────────────────────────────────────────────────────────
  < 100K docs         Exact (numpy)        No ANN needed. 10ms exact.
  100K - 2M docs      HNSW                 Best recall/speed balance.
  > 2M docs           IVF+PQ (FAISS)      Low memory. ~95% recall.
  Any size, managed   Pinecone / Qdrant    Let vendor handle algorithm.

  WHEN TO USE EACH:

  EXACT (Brute Force):
    - Development and testing (always know if you have a recall problem)
    - Corpus < 50K docs (sub-millisecond, no index overhead)
    - Ground truth generation for ANN recall measurement

  HNSW:
    - Production RAG with recall requirements > 95%
    - Corpus < 5M docs
    - You need to add vectors incrementally (HNSW supports online inserts)
    - Libraries: hnswlib, FAISS IndexHNSWFlat, Qdrant (built-in), Weaviate

  IVF:
    - Large corpora where HNSW memory is prohibitive
    - You have time for batch index rebuilds (IVF doesn't support easy online inserts)
    - Libraries: FAISS IndexIVFFlat, IndexIVFPQ (with PQ for compression)

  PQ (Product Quantization):
    - Memory-constrained environments (GPUs with limited VRAM)
    - First-pass approximation before expensive re-ranking
    - ALWAYS combine with re-ranking: PQ selects candidates, exact scoring re-ranks
    - Libraries: FAISS IndexPQ, IndexHNSWPQ (HNSW + PQ combined)

  DISK-BASED (DiskANN, SPANN):
    - Corpora too large for RAM (100M+ docs)
    - Libraries: DiskANN (Microsoft), SPANN
    - Not covered here but relevant at very large enterprise scale

  PRODUCTION RECOMMENDATION FOR CRITERION NETWORKS RAG:
    Typical corpus: 100K - 1M documents
    Recommendation: Qdrant with HNSW (default settings)
    Why Qdrant: supports filtering (metadata + vector search), payload indexing,
    sparse+dense hybrid natively, easy Docker deployment, active development.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    compare_ann_algorithms()
    ann_selection_guide()
