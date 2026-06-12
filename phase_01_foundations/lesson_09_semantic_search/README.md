# Phase 1 — Lesson 9: Semantic Search

## Definition

**Semantic search** finds documents based on *meaning*, not exact keyword matches.
It uses embedding vectors to measure conceptual similarity between a query and stored content.

In RAG, semantic search is the retrieval step that decides which chunks the LLM sees.
Bad retrieval → bad answers, regardless of how good the LLM is.

---

## The Retrieval Spectrum

```
KEYWORD SEARCH          SEMANTIC SEARCH         HYBRID SEARCH
(BM25 / TF-IDF)        (Dense vectors)         (BM25 + Dense)
      │                       │                       │
      ▼                       ▼                       ▼
Exact word match       Meaning match            Both signals
"ACI contracts"    →   "EPG communication"  →   Best of both
 finds "ACI"       →   finds "EPG comm"      →   finds either
 misses "EPG"          even with 0 shared words

PRECISION:  HIGH              MEDIUM-HIGH           HIGHEST
RECALL:     LOW               HIGH                  HIGHEST
SPEED:      VERY FAST         FAST                  MODERATE
COST:       ZERO              EMBEDDING API $       EMBEDDING API $
```

---

## BM25: The Keyword Search Baseline

```
BM25 (Best Match 25) is the gold standard keyword search algorithm.
It improves on TF-IDF with two key features:
  1. Term frequency saturation: more occurrences of a word gives diminishing returns
  2. Document length normalization: shorter documents get a relevance boost

FORMULA:
  BM25(d, q) = Σ_t [ IDF(t) × TF_sat(t,d) ]

  IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)
    N  = total documents in corpus
    df = documents containing term t
    (rare terms get higher IDF weight)

  TF_sat(t,d) = tf × (k1 + 1) / (tf + k1 × (1 - b + b × |d|/avgdl))
    tf    = term frequency in document d
    |d|   = document length (tokens)
    avgdl = average document length in corpus
    k1    = 1.5  (saturation parameter — 1.2-2.0 typical)
    b     = 0.75 (length normalization — 0=no norm, 1=full norm)

WHEN BM25 WINS OVER DENSE:
  - Exact product names: "Nexus 9336C-FX2" → only BM25 finds this reliably
  - Error codes: "CSCvh12345" → typo/rare term BM25 handles better
  - Named entities: specific person names, IP addresses, serial numbers
  - Out-of-distribution queries: technical jargon not in embedding training data
```

---

## HNSW: How Vector Databases Index Embeddings

```
PROBLEM:
  1M documents × 1024 dims → brute-force cosine: 1M dot products per query.
  At 0.1μs each → 100ms per query → too slow for interactive RAG.

SOLUTION: HNSW (Hierarchical Navigable Small World)
  Approximate Nearest Neighbor (ANN) index that finds the ~K closest vectors
  in O(log N) time instead of O(N).

STRUCTURE:
  Layer 2 (sparse): ● ─────────────────────── ●    ← long-range connections
                              │
  Layer 1 (medium): ● ── ● ── ● ── ●           ─ ● ─ ●
                                   │
  Layer 0 (dense):  ●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●

  Search: start at top layer, greedily move toward query,
          descend to denser layers, find exact neighborhood.

PARAMETERS:
  ef_construction = 200  (quality during index build: higher → better but slower)
  M               = 16   (connections per node: higher → better but more memory)
  ef_search       = 100  (beam width during search: higher → better but slower)

ACCURACY vs SPEED:
  ef_search=10:  ~70% recall, 0.5ms query
  ef_search=100: ~97% recall, 2ms query
  ef_search=500: ~99% recall, 8ms query

For RAG: ef_search=100 gives 97% recall at 2ms — more than sufficient.
```

---

## Hybrid Search: Combining BM25 + Dense

```
WHY HYBRID:
  Dense search misses exact technical terms → false negatives
  BM25 misses semantic matches → also false negatives
  Combined → captures both → highest recall

FUSION STRATEGIES:

1. RECIPROCAL RANK FUSION (RRF) — most common:
   RRF_score(d) = Σ_list 1 / (k + rank_in_list)
   k = 60 (constant that smooths rank differences)
   WHY k=60: empirically found to be optimal across many benchmarks

   Example:
     Doc A: BM25 rank=1, Dense rank=3 → RRF = 1/61 + 1/63 = 0.0321
     Doc B: BM25 rank=5, Dense rank=1 → RRF = 1/65 + 1/61 = 0.0319
     Doc C: BM25 rank=2, Dense rank=2 → RRF = 1/62 + 1/62 = 0.0323 ← wins

2. WEIGHTED SUM:
   score = α × dense_score + (1-α) × bm25_score
   α is a tunable hyperparameter (default α=0.5)
   Requires score normalization (both systems use different scales!)

3. CASCADE (re-rank):
   Dense retrieves top-100, BM25 re-ranks the 100 → return top-5
   OR BM25 retrieves top-1000, dense re-ranks → return top-5

RRF is preferred because:
  - No score normalization needed (only ranks matter)
  - Robust to outliers (rank 1 vs rank 2 matters; score 0.99 vs 0.95 doesn't)
  - Simple to implement
  - No hyperparameter to tune (k=60 works well everywhere)
```

---

## Metadata Filtering

```
Vector DBs support filtering on metadata fields alongside similarity search.
This dramatically narrows the search space before computing similarity.

EXAMPLE METADATA:
  { "source": "aci_guide.pdf",
    "date": "2025-01-15",
    "tier": "core",
    "language": "en",
    "section": "chapter_3" }

FILTER MODES:
  Pre-filter:  Apply metadata filter BEFORE ANN search
               → only search vectors matching the filter
               → fast but may miss results if filter is too narrow
               → risk: if filter returns <ef_search candidates, recall drops

  Post-filter: Run ANN search first, then filter the top-K results
               → full recall on the ANN side, filter after
               → risk: top-K after filter may be much less than K

  In-filter (HNSW aware):
               Many modern vector DBs (Qdrant, Weaviate) interleave filtering
               with HNSW traversal → best of both worlds
               → Qdrant calls this "payload indexing"
```

---

## Files in This Lesson

| File | What It Teaches |
|------|-----------------|
| 01_bm25_keyword_search.py | BM25 algorithm from scratch, TF-IDF intuition, IDF weighting |
| 02_dense_vector_search.py | Exhaustive search vs ANN, HNSW concepts, brute-force numpy index |
| 03_hybrid_search.py | RRF fusion, weighted sum, cascade re-ranking |
| 04_approximate_nearest_neighbor.py | HNSW, IVF, PQ — concepts and tradeoffs |
| 05_metadata_filtering.py | Pre/post/in-filter, filter index design |
| 06_mini_project_search_engine.py | Full hybrid search engine: BM25 + dense + RRF + metadata filter |

---

## Interview Questions

Q1: What is BM25 and how does it improve on TF-IDF?
A: BM25 adds two improvements to TF-IDF: (1) Term frequency saturation via the k1
   parameter — additional occurrences of a word give diminishing extra score,
   preventing extremely common terms from dominating. (2) Document length normalization
   via the b parameter — shorter documents get a relevance boost so short docs with a
   single matching term compete fairly against long docs.

Q2: Why does hybrid search outperform either BM25 or dense search alone?
A: BM25 excels at exact term matching (product names, error codes, named entities)
   but misses semantic matches. Dense search captures meaning but can miss exact
   technical terms not well-represented in the embedding space. Hybrid search using
   RRF fusion captures signals from both channels, typically improving Recall@10
   by 5-15% over the best single-method baseline.

Q3: What is HNSW and why does every production vector DB use it?
A: HNSW (Hierarchical Navigable Small World) is an ANN index that organizes vectors
   in a layered graph structure. Search starts at sparse upper layers (fast long-range
   navigation) and descends to dense lower layers (fine-grained neighborhood). This
   achieves O(log N) query time vs O(N) for brute force. At 1M vectors, HNSW queries
   in ~2ms vs ~100ms brute force, with ~97% recall — making it the standard for
   production vector databases (FAISS, Qdrant, Weaviate, Pinecone all use it).

Q4: What is Reciprocal Rank Fusion and why is k=60?
A: RRF(d) = Σ 1/(k + rank_i) across all result lists. The k=60 constant smooths
   differences between highly-ranked results, preventing rank 1 from overwhelming
   rank 2. k=60 was empirically determined to perform well across diverse benchmarks
   (Cormack et al. 2009). RRF's key advantage: it requires no score normalization
   since only ranks matter, making it robust when combining systems with different
   score scales.

Q5: What is the difference between pre-filter and post-filter in vector search?
A: Pre-filter applies metadata conditions before ANN search, reducing the candidate
   pool. This is fast but can hurt recall if the filter leaves too few candidates
   for the HNSW graph to traverse effectively (a filter returning only 50 docs in a
   1M corpus means HNSW can't use its layered structure). Post-filter runs full ANN
   search then applies the filter — preserves recall but wastes compute on filtered
   results. Modern vector DBs (Qdrant) use in-filter: interleave metadata checks with
   HNSW traversal, providing pre-filter speed with post-filter recall.

---

## Quiz

1. Write the BM25 term saturation formula and explain what k1 and b control.
2. What does "approximate" mean in Approximate Nearest Neighbor? What is the recall tradeoff?
3. You have 3 result lists from BM25, dense search, and a re-ranker.
   Doc A has ranks 1, 4, 2. Doc B has ranks 2, 1, 4. Using RRF with k=60,
   which document has the higher score?
4. Why might a pure dense vector search fail to find a document containing the
   string "CSCvh23456" (a Cisco bug ID)?
5. You have a 2M document corpus filtered to 500 documents by metadata.
   Should you use pre-filter or post-filter? Why?
