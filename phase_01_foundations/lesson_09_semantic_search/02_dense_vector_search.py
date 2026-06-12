"""
FILE: 02_dense_vector_search.py
LESSON: Phase 1 - Lesson 9 - Semantic Search
TOPIC: Dense vector search — brute force vs approximate, and HNSW intuition

WHAT THIS FILE TEACHES:
  - Brute-force (exact) nearest neighbor search with numpy
  - WHY brute force is slow at scale and needs ANN indexing
  - HNSW (Hierarchical Navigable Small World) — how the layered graph works
  - Recall vs latency tradeoff in ANN search
  - How production vector DBs (Qdrant, FAISS, Pinecone) implement this
  - Practical parameters: ef_construction, ef_search, M

NO REAL FAISS/QDRANT NEEDED:
  This file builds an in-memory brute-force index and simulates
  HNSW routing behavior. Real libraries do the same math + GPU acceleration.

INSTALL: pip install numpy
"""

import time
import heapq
import hashlib
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─── Mock Embedding Generator ─────────────────────────────────────────────────

def mock_embed(text: str, dims: int = 128) -> np.ndarray:
    """Deterministic mock embedding (unit-normalized)."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.RandomState(seed)
    vec  = rng.randn(dims).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ─── Exact (Brute-Force) Index ────────────────────────────────────────────────

class ExactIndex:
    """
    Exhaustive nearest-neighbor search.
    Computes cosine similarity against EVERY vector in the index.

    WHY this exists:
      - Correct by definition: always returns the true top-K.
      - Fast enough for < 50,000 vectors with numpy BLAS.
      - Useful as a GROUND TRUTH to measure ANN recall.

    Complexity:
      Build: O(N) — just store the vectors
      Query: O(N × D) — dot product with every vector
        N = corpus size, D = dimensions

    When to use:
      - Development and testing (exact results)
      - Small corpora (< 50K docs)
      - As a benchmark to measure ANN accuracy
    """

    def __init__(self, dims: int):
        self.dims        = dims
        self._vectors:   Optional[np.ndarray] = None   # shape: (N, D)
        self._ids:       list[str]            = []

    def add(self, ids: list[str], vectors: np.ndarray):
        """
        Add vectors to the index.

        Args:
            ids:     Document IDs.
            vectors: Unit-normalized vectors, shape (N, D).
        """
        assert vectors.shape[1] == self.dims, f"Expected {self.dims} dims, got {vectors.shape[1]}"

        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])   # WHY vstack: concatenate rows

        self._ids.extend(ids)

    def search(
        self,
        query:  np.ndarray,
        top_k:  int = 10,
    ) -> list[tuple[str, float]]:
        """
        Find top-K most similar vectors using exact dot product search.

        Complexity: O(N × D) — scores every vector in the corpus.

        WHY np.dot(matrix, query):
          Single BLAS matrix-vector multiply → computes ALL N dot products
          in one optimized operation. Much faster than a Python loop.
        """
        if self._vectors is None:
            return []

        # Unit-normalize query (defensive: should already be normalized)
        qnorm = query / (np.linalg.norm(query) + 1e-10)

        # Compute cosine similarity to all vectors at once
        # WHY this works: self._vectors rows are unit-normalized,
        # so dot product = cosine similarity
        scores = np.dot(self._vectors, qnorm)   # shape: (N,)

        # Get top-K indices (argpartition is faster than full argsort for large N)
        # WHY argpartition: O(N) vs O(N log N) for argsort; we only need top-K positions
        if len(scores) <= top_k:
            top_indices = np.argsort(-scores)
        else:
            # Get indices of top-K (unordered), then sort just those K
            top_k_unordered = np.argpartition(-scores, top_k)[:top_k]
            top_indices      = top_k_unordered[np.argsort(-scores[top_k_unordered])]

        return [
            (self._ids[i], float(scores[i]))
            for i in top_indices
        ]

    @property
    def size(self) -> int:
        return len(self._ids)


# ─── Simplified HNSW (Conceptual Implementation) ─────────────────────────────

class SimpleHNSW:
    """
    A SIMPLIFIED HNSW implementation for teaching purposes.
    NOT production-quality — real HNSW (FAISS, hnswlib) uses C++ and SIMD.

    WHAT THIS TEACHES:
      - The multi-layer graph structure (why "hierarchical")
      - The greedy search algorithm
      - How ef_construction controls quality vs build speed
      - Why HNSW achieves O(log N) query time

    REAL HNSW LIBRARIES:
      hnswlib: pip install hnswlib  (C++ bindings, fast)
      faiss:   pip install faiss-cpu  (Meta's production library)
      Qdrant, Weaviate, Pinecone all use HNSW internally.
    """

    def __init__(
        self,
        dims:             int,
        M:                int   = 16,
        ef_construction:  int   = 200,
        max_layers:       int   = 4,
    ):
        """
        Args:
            dims:            Vector dimension.
            M:               Max connections per node per layer.
                             Higher → better recall, more memory.
                             Default 16. Use 32-64 for high-recall needs.
            ef_construction: Size of dynamic candidate list during build.
                             Higher → better quality index, slower build.
                             Default 200.
            max_layers:      Number of graph layers.
                             Real HNSW computes this from M: ~log(1/p_level) × M.
        """
        self.dims             = dims
        self.M                = M
        self.ef_construction  = ef_construction
        self.max_layers       = max_layers

        self._vectors: list[np.ndarray] = []
        self._ids:     list[str]        = []

        # Graph: layers[layer][node_idx] = list of neighbor node indices
        # WHY nested structure: upper layers are sparse, lower layers dense.
        self._graph:   list[dict[int, list[int]]] = [{}]

        # Entry point: the single node that starts every search
        # WHY single entry: HNSW builds a funnel — searches converge from top layer
        self._entry_point: Optional[int] = None

    def _assign_layer(self, idx: int) -> int:
        """
        Randomly assign a node to a max layer.
        The distribution is exponential: most nodes in layer 0,
        few nodes in layer 1, very few in layer 2, etc.

        WHY exponential: creates the "small world" long-range shortcut structure.
        Upper layers are sparse → fast long-range navigation.
        Lower layers are dense → precise local search.
        """
        # WHY ln(self.M): matches the original HNSW paper's formula
        # for ml (the level multiplier)
        level_mult = 1.0 / math.log(max(self.M, 2))

        # Sample from exponential distribution
        rng   = np.random.RandomState(idx)
        level = int(-math.log(rng.random()) * level_mult)

        return min(level, self.max_layers - 1)

    def _greedy_search_layer(
        self,
        query:          np.ndarray,
        entry_node:     int,
        layer:          int,
        ef:             int,
    ) -> list[tuple[float, int]]:
        """
        Greedy beam search within one layer of the HNSW graph.

        Algorithm:
          Start at entry_node.
          Maintain a CANDIDATE set (to explore) and a VISITED set.
          At each step: expand the closest unvisited candidate.
          Stop when the closest candidate is farther than the furthest result found.

        Args:
            query:       Unit-normalized query vector.
            entry_node:  Starting node index.
            layer:       Which layer of the graph to search.
            ef:          Beam width (number of candidates to maintain).

        Returns:
            List of (distance, node_idx) — the ef nearest neighbors found.
        """

        # WHY min-heap for candidates: always pop the closest node next (greedy)
        # WHY max-heap for results: efficiently drop the farthest result when full
        entry_dist = float(1 - np.dot(query, self._vectors[entry_node]))

        # Candidate heap: (distance, node_idx) — min-heap (closest = smallest dist)
        candidates = [(entry_dist, entry_node)]

        # Results heap: (-distance, node_idx) — max-heap (furthest = most negative)
        results    = [(-entry_dist, entry_node)]

        visited    = {entry_node}

        while candidates:
            dist_c, c = heapq.heappop(candidates)

            # WHY this stopping condition:
            #   If the closest candidate is farther than the furthest result,
            #   all remaining candidates are also farther → we can't improve results.
            farthest_result_dist = -results[0][0]
            if dist_c > farthest_result_dist and len(results) >= ef:
                break

            # Explore c's neighbors in this layer
            neighbors = self._graph[layer].get(c, [])

            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                dist_n = float(1 - np.dot(query, self._vectors[neighbor]))

                if len(results) < ef or dist_n < -results[0][0]:
                    heapq.heappush(candidates, (dist_n, neighbor))
                    heapq.heappush(results,   (-dist_n, neighbor))

                    # WHY trim results to ef: maintains beam width
                    if len(results) > ef:
                        heapq.heappop(results)  # removes farthest result

        return [(- neg_d, idx) for neg_d, idx in results]

    def add(self, doc_id: str, vector: np.ndarray):
        """Add a single vector to the HNSW graph."""
        idx        = len(self._vectors)
        self._vectors.append(vector / (np.linalg.norm(vector) + 1e-10))
        self._ids.append(doc_id)

        # Ensure graph layers exist up to assigned_layer
        assigned_layer = self._assign_layer(idx)
        while len(self._graph) <= assigned_layer:
            self._graph.append({})

        if self._entry_point is None:
            self._entry_point = idx
            return

        # Navigate from top layer to assigned_layer using greedy search
        # This finds the best neighbors at each layer to connect to
        entry = self._entry_point
        top_layer = len(self._graph) - 1

        for layer in range(top_layer, assigned_layer, -1):
            if layer < len(self._graph):
                neighbors_found = self._greedy_search_layer(
                    vector, entry, layer, ef=1
                )
                if neighbors_found:
                    entry = neighbors_found[0][1]

        # Connect the new node to its neighbors at each layer up to assigned_layer
        for layer in range(min(assigned_layer, len(self._graph) - 1), -1, -1):
            candidates = self._greedy_search_layer(
                vector, entry, layer, ef=self.ef_construction
            )

            # Select M nearest neighbors for this layer
            neighbors = sorted(candidates, key=lambda x: x[0])[:self.M]
            neighbor_ids = [n[1] for n in neighbors]

            self._graph[layer][idx] = neighbor_ids

            # Add reverse connections (bidirectional graph)
            for neighbor_id in neighbor_ids:
                if neighbor_id not in self._graph[layer]:
                    self._graph[layer][neighbor_id] = []
                self._graph[layer][neighbor_id].append(idx)

                # Trim neighbor's connections if over M limit
                if len(self._graph[layer][neighbor_id]) > self.M * 2:
                    # Keep M*2 closest connections only
                    nbr_vec    = self._vectors[neighbor_id]
                    conn_dists = [
                        (float(1 - np.dot(nbr_vec, self._vectors[c])), c)
                        for c in self._graph[layer][neighbor_id]
                    ]
                    conn_dists.sort()
                    self._graph[layer][neighbor_id] = [c for _, c in conn_dists[:self.M]]

        # Update entry point to highest-layer node
        if assigned_layer == len(self._graph) - 1:
            self._entry_point = idx

    def search(
        self,
        query:    np.ndarray,
        top_k:    int = 10,
        ef_search: int = 100,
    ) -> list[tuple[str, float]]:
        """
        Search the HNSW graph for top-K nearest neighbors.

        Algorithm:
          1. Start at entry point (top layer).
          2. Greedily navigate toward query at each layer.
          3. At bottom layer, run beam search with width ef_search.
          4. Return top-K from the ef_search candidates found.

        Args:
            query:     Query vector.
            top_k:     Number of results to return.
            ef_search: Beam width. Higher → better recall, slower.

        WHY ef_search > top_k:
          Greedy search can "miss" the true nearest neighbors by going down
          a suboptimal path. A larger beam width increases the probability
          of finding the true top-K.
          ef_search=100 for top_k=10 → recall ~97%.
          ef_search=10  for top_k=10 → recall ~70%.
        """
        if self._entry_point is None:
            return []

        qnorm = query / (np.linalg.norm(query) + 1e-10)
        entry = self._entry_point

        # Navigate from top layer to layer 1, tracking closest node
        for layer in range(len(self._graph) - 1, 0, -1):
            if layer < len(self._graph):
                found = self._greedy_search_layer(qnorm, entry, layer, ef=1)
                if found:
                    entry = found[0][1]

        # Full beam search at layer 0
        results = self._greedy_search_layer(qnorm, entry, 0, ef=max(ef_search, top_k))

        results.sort(key=lambda x: x[0])  # sort by distance ascending
        top_results = results[:top_k]

        return [
            (self._ids[idx], 1.0 - dist)   # WHY 1-dist: convert distance back to similarity
            for dist, idx in top_results
        ]

    @property
    def size(self) -> int:
        return len(self._ids)


# ─── Performance Comparison ───────────────────────────────────────────────────

def compare_index_performance(
    n_docs: int = 1_000,
    dims:   int = 128,
    top_k:  int = 5,
):
    """
    Compare exact vs HNSW search for speed and recall.
    """

    print("=" * 65)
    print(f"INDEX COMPARISON: {n_docs:,} docs × {dims} dims × top-{top_k}")
    print("=" * 65)

    # Generate random unit-normalized vectors
    rng     = np.random.RandomState(42)
    raw     = rng.randn(n_docs, dims).astype(np.float32)
    norms   = np.linalg.norm(raw, axis=1, keepdims=True)
    vectors = raw / norms

    ids     = [f"doc_{i:05d}" for i in range(n_docs)]

    # ── Build exact index ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    exact = ExactIndex(dims=dims)
    exact.add(ids, vectors)
    exact_build = time.perf_counter() - t0

    # ── Build HNSW index ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    hnsw = SimpleHNSW(dims=dims, M=16, ef_construction=50)
    for i, (doc_id, vec) in enumerate(zip(ids, vectors)):
        hnsw.add(doc_id, vec)
    hnsw_build = time.perf_counter() - t0

    print(f"\n  Build time:  Exact={exact_build*1000:.0f}ms  HNSW={hnsw_build*1000:.0f}ms")

    # ── Query benchmark ───────────────────────────────────────────────────────
    n_queries    = 20
    query_vecs   = rng.randn(n_queries, dims).astype(np.float32)
    query_vecs  /= np.linalg.norm(query_vecs, axis=1, keepdims=True)

    exact_times  = []
    hnsw_times   = []
    recalls      = []

    for qvec in query_vecs:
        t0 = time.perf_counter()
        exact_results = exact.search(qvec, top_k=top_k)
        exact_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        hnsw_results = hnsw.search(qvec, top_k=top_k, ef_search=50)
        hnsw_times.append(time.perf_counter() - t0)

        # Recall: fraction of exact top-K found by HNSW
        exact_ids = {r[0] for r in exact_results}
        hnsw_ids  = {r[0] for r in hnsw_results}
        recalls.append(len(exact_ids & hnsw_ids) / max(len(exact_ids), 1))

    avg_exact_ms = sum(exact_times) / n_queries * 1000
    avg_hnsw_ms  = sum(hnsw_times)  / n_queries * 1000
    avg_recall   = sum(recalls) / n_queries * 100

    print(f"  Query latency: Exact={avg_exact_ms:.2f}ms  HNSW={avg_hnsw_ms:.2f}ms")
    print(f"  HNSW Recall@{top_k}: {avg_recall:.1f}%")
    print(f"  HNSW speedup:  {avg_exact_ms/max(avg_hnsw_ms,0.01):.1f}×")

    print(f"""
  NOTE: This simplified HNSW is for teaching only.
  Real HNSW (hnswlib, FAISS) uses C++ SIMD → 10-100× faster.
  At 1M docs: hnswlib queries in ~2ms vs ~100ms brute force.
""")


def hnsw_parameter_guide():
    """
    Practical guide to HNSW parameters for production RAG.
    """

    print("\n" + "=" * 65)
    print("HNSW PARAMETER GUIDE (for production vector DBs)")
    print("=" * 65)

    print(f"""
  M (connections per node per layer):
    M=8:   Low memory, lower recall (~85-90%)
    M=16:  Good balance (DEFAULT) — recall ~95-97%
    M=32:  High recall (~99%), 2× memory vs M=16
    M=64:  Near-exact, use only if recall >99% required

  ef_construction (beam width during BUILD):
    ef_construction=100:  Fast build, recall ~90%
    ef_construction=200:  DEFAULT — good balance
    ef_construction=500:  Slow build, recall ~99%
    Rule: ef_construction ≥ M × 2

  ef_search (beam width during QUERY):
    ef_search=10:   Fastest, recall ~70%  (for high-QPS low-accuracy)
    ef_search=50:   Good balance, recall ~92%
    ef_search=100:  DEFAULT — recall ~97%
    ef_search=500:  Near-exact, recall ~99.5%
    Rule: ef_search ≥ top_k × 10

  PRODUCTION RECOMMENDATIONS (for RAG):
    Corpus < 100K docs:  Use EXACT search (numpy). No ANN needed.
    Corpus 100K-1M docs: HNSW with M=16, ef_construction=200, ef_search=100
    Corpus > 1M docs:    Consider IVF+PQ or disk-based index (see Lesson 9 file 4)

  VECTOR DATABASE DEFAULTS:
    Qdrant:    M=16, ef_construction=100, ef_search=128  ← good for RAG
    Weaviate:  M=64, ef_construction=128, ef_search=64
    Pinecone:  Managed — no manual tuning needed
    FAISS:     Manual configuration (IndexHNSWFlat)
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    compare_index_performance(n_docs=500, dims=64, top_k=5)
    hnsw_parameter_guide()
