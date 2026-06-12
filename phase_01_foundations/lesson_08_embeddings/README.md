# Phase 1 — Lesson 8: Embeddings

## Definition

An **embedding** is a fixed-length vector of floating-point numbers that represents
the *meaning* of a piece of text in a high-dimensional geometric space.

```
"The network is down"   → [0.12, -0.34, 0.87, ..., 0.22]  (1024 numbers)
"Connectivity failure"  → [0.11, -0.31, 0.89, ..., 0.19]  ← nearly identical vector
"My cat is hungry"      → [-0.45, 0.78, -0.12, ..., 0.63]  ← far from both above
```

Texts with **similar meaning** produce vectors that are **close together** in this space.
This is the entire foundation of semantic search in RAG.

---

## Why Embeddings Exist (The Problem They Solve)

```
KEYWORD SEARCH (old approach):
  Query: "network connectivity issue"
  Document contains: "link is down" → NOT FOUND
  Keyword search MISSES this because "connectivity" ≠ "link"

EMBEDDING SEARCH (semantic approach):
  Query: "network connectivity issue" → vector A
  Document: "link is down"           → vector B
  distance(A, B) = 0.08              → VERY CLOSE → FOUND

  The embedding model learned that "connectivity issue" and "link is down"
  describe the same real-world situation, even with zero shared words.
```

---

## How Embeddings Are Created

```
ARCHITECTURE: Encoder-only transformer (BERT-style)

Input text → Tokenizer → Token embeddings → Transformer layers → [CLS] token
                                                                       ↓
                                                              Single pooled vector
                                                              (e.g. 1024 dimensions)

KEY DIFFERENCE from LLMs:
  LLMs (GPT, Claude) are DECODER-ONLY: generate tokens left-to-right.
  Embedding models are ENCODER-ONLY: read the full text BIDIRECTIONALLY.
  Bidirectional attention → richer semantic representation.
  No text generation capability.

POOLING:
  After the encoder, you have one vector per token.
  Pooling collapses them to ONE vector for the whole text:
    CLS pooling:   use the [CLS] token's vector (BERT approach)
    Mean pooling:  average all token vectors (most common for modern models)
    Max pooling:   take the max value per dimension (rare)
```

---

## Embedding Dimensions and What They Mean

```
Dimension = number of floating-point values in the vector.

  Model                   Dimensions   Storage/vector   Notes
  ──────────────────────────────────────────────────────────────
  OpenAI text-ada-002        1,536      6 KB (float32)   Common baseline
  Voyage-3                   1,024      4 KB             Best for RAG (Anthropic rec.)
  Voyage-3-lite               512       2 KB             Fast, efficient
  Voyage-code-3             1,024      4 KB             Optimized for code
  Cohere embed-v3           1,024      4 KB             Multilingual strong
  BGE-M3 (open source)      1,024      4 KB             Free, strong multilingual

WHAT EACH DIMENSION REPRESENTS:
  Dimensions are NOT interpretable individually.
  Dimension 47 doesn't mean "sentence_length" or "topic_networking."
  The transformer learned abstract features across ALL dimensions jointly.
  The only thing that matters is the RELATIVE position of vectors.

DIMENSIONALITY REDUCTION (for production at scale):
  1,024 float32 = 4 KB per vector
  1,000,000 documents × 4 KB = 4 GB just for embeddings
  Reduce to 256 dims → 1 GB (acceptable for most deployments)
  Matryoshka Representation Learning (MRL) enables truncation without quality loss.
```

---

## Similarity Metrics

```
COSINE SIMILARITY (most common for embeddings):
  Measures the ANGLE between two vectors, regardless of magnitude.
  cosine_sim(A, B) = (A · B) / (|A| × |B|)
  Range: -1 (opposite) to +1 (identical)

  WHY cosine not euclidean for embeddings:
    Embedding models normalize vectors to unit length (|v| = 1).
    For unit vectors: cosine_sim = dot_product (same thing).
    Euclidean distance conflates direction AND magnitude.
    Cosine focuses on direction = semantic content.

DOT PRODUCT (equivalent to cosine for normalized vectors):
  A · B = Σ(a_i × b_i)
  Fastest to compute. Vector DBs optimize for this.
  Requires unit-normalized vectors to behave like cosine similarity.

EUCLIDEAN DISTANCE:
  distance = √(Σ(a_i - b_i)²)
  Use when magnitude matters (e.g., popularity weighting).
  Rarely used for pure semantic similarity.
```

---

## Voyage AI (Anthropic's Recommended Embedding Provider)

```
WHY VOYAGE AI:
  Anthropic explicitly recommends Voyage AI as the embedding partner for RAG.
  Voyage-3 is trained specifically for retrieval tasks (not just sentence similarity).
  Strong performance on MTEB (Massive Text Embedding Benchmark).
  Input token limit: 32,000 tokens per document.

VOYAGE API:
  import voyageai
  vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

  result = vo.embed(
      texts       = ["doc 1 text", "doc 2 text"],
      model       = "voyage-3",
      input_type  = "document",  # "document" for indexing, "query" for search
  )
  embeddings = result.embeddings   # list of float lists

INPUT TYPE (IMPORTANT):
  "document": used when embedding chunks for the vector index
  "query":    used when embedding the user's search query
  Using the SAME input_type for both is a common mistake —
  document vs query encodings are trained differently for retrieval.
```

---

## Files in This Lesson

| File | What It Teaches |
|------|-----------------|
| 01_what_are_embeddings.py | Vector space, dimensions, semantic clustering, numpy demo |
| 02_generate_embeddings.py | Voyage AI API, batch embedding, document vs query types |
| 03_similarity_metrics.py | Cosine, dot product, euclidean from scratch + comparison |
| 04_embedding_models_comparison.py | Model landscape, MTEB scores, trade-offs |
| 05_embedding_quality_eval.py | Precision@K, Recall@K, MRR, NDCG evaluation metrics |
| 06_mini_project_semantic_search.py | Full semantic search: embed → index → query → rank |

---

## Interview Questions

Q1: What is an embedding and how is it created?
A: An embedding is a fixed-length vector of floats representing text meaning.
   Created by encoder-only transformers (BERT-style) that process text bidirectionally
   through multiple attention layers. The final hidden states are pooled (usually mean
   pooling) into a single vector. The model is trained so similar texts produce similar
   vectors — typically via contrastive learning (similar pairs pulled together, different
   pairs pushed apart).

Q2: Why do you use different input_type for documents vs queries in Voyage AI?
A: Retrieval-optimized embedding models are trained with ASYMMETRIC objectives.
   A query is short and often incomplete ("network down"). A document is long and
   specific. The model learns a document representation that "answers" potential
   queries, and a query representation that "matches" document answers.
   Using "document" for both leads to 10-20% worse retrieval performance.

Q3: What is cosine similarity and why is it preferred over Euclidean distance?
A: Cosine similarity measures the angle between vectors: (A·B)/(|A||B|).
   Range is -1 to +1. For unit-normalized embedding vectors, it equals the dot product.
   Euclidean distance measures geometric distance including magnitude.
   Since embedding models typically output unit-normalized vectors, cosine similarity
   (or equivalently, dot product) captures the semantic direction without being
   influenced by vector magnitude artifacts.

Q4: A document collection has 1M chunks. What are the storage implications of
    using 1024-dimension float32 embeddings?
A: 1M × 1024 × 4 bytes = ~4 GB for embeddings alone, plus metadata and index overhead.
   A typical HNSW index adds ~1.5× overhead → ~6 GB total.
   Options: use smaller dimensions (512 → 2 GB), quantize to int8 (1 GB),
   or use Matryoshka embeddings truncated to 256 dims (1 GB, ~3% quality loss).

Q5: What is the difference between an embedding model and a language model?
A: Embedding models are ENCODER-ONLY: they read text bidirectionally and output a
   single fixed-size vector — no text generation. LLMs are typically DECODER-ONLY:
   they generate text autoregressively left-to-right. Embedding models have much
   fewer parameters (100M-500M vs 7B-405B for LLMs) and are 100× cheaper to run.
   You use an embedding model to INDEX and SEARCH; you use an LLM to GENERATE answers.

---

## Quiz

1. What type of transformer architecture do embedding models use, and how does it
   differ from GPT-style models?
2. Write the cosine similarity formula and explain each term.
3. Why must you use input_type="query" for search queries and input_type="document"
   for indexed chunks in Voyage AI?
4. 500,000 documents × 768 float32 dimensions. How many GB for embeddings?
5. What is the MTEB benchmark and what does it measure?
