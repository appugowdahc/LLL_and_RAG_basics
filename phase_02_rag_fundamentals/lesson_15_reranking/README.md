# Phase 2 — Lesson 15: Reranking

## Why Retrieval Ranking Is Never Good Enough

Dense retrieval uses **bi-encoders**: the query and every document are embedded *independently*, then compared by cosine similarity. This is fast — O(1) per document after indexing — but it is a shallow comparison. The query and document never "see" each other during scoring.

**Reranking** uses a more powerful model to re-score the top-K retrieved candidates by considering the query and document *together*. The reranker sees both at once and can understand nuanced relevance relationships that a pure cosine similarity score misses.

```
┌─────────────────────────────────────────────────────────────────────┐
│  TWO-STAGE RETRIEVAL PIPELINE                                       │
│                                                                     │
│  Stage 1: Retrieval (Fast, Approximate)                             │
│  ─────────────────────────────────────                              │
│  Query ──▶ Bi-encoder ──▶ Cosine similarity ──▶ Top-K candidates   │
│  (query and docs encoded independently — no cross-attention)        │
│                                                                     │
│  Stage 2: Reranking (Slow, Precise)                                 │
│  ─────────────────────────────────                                  │
│  [Query + Doc_1] ──▶ Cross-encoder ──▶ relevance score 0.92        │
│  [Query + Doc_2] ──▶ Cross-encoder ──▶ relevance score 0.71        │
│  [Query + Doc_3] ──▶ Cross-encoder ──▶ relevance score 0.88        │
│                         ↓                                           │
│              Re-sort by cross-encoder score                         │
│              Top-M (M << K) sent to LLM                             │
└─────────────────────────────────────────────────────────────────────┘
```

**Why two stages?** You cannot run a cross-encoder over the entire corpus — it takes O(N) forward passes, each with full attention between query and document. That's 10–100× slower than bi-encoder retrieval. So you:
1. Use the fast bi-encoder to narrow from N documents to K candidates (K = 50–200).
2. Use the precise cross-encoder to re-rank K candidates and select the top M (M = 3–10) for the LLM context.

---

## Three Reranking Approaches

### 1. Cross-Encoder Reranker
A transformer model (usually BERT or a fine-tuned variant) receives `[CLS] query [SEP] document [SEP]` as a single input. The output is a single relevance score.

```
Input:  [CLS] "APIC cluster HA requirement" [SEP] "The APIC requires 3 nodes for quorum." [SEP]
Output: 0.94   ← strong relevance

Input:  [CLS] "APIC cluster HA requirement" [SEP] "BGP route reflection concepts." [SEP]
Output: 0.11   ← weak relevance
```

**Models**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast), `cross-encoder/ms-marco-electra-base` (accurate).
**Library**: `sentence-transformers` — `CrossEncoder` class.

---

### 2. Cohere Rerank API
Cohere provides a managed reranking endpoint. Send query + list of documents → get ranked list with relevance scores. No model hosting required.

```python
co.rerank(
    query     = "APIC HA requirements",
    documents = [doc1, doc2, doc3, ...],
    model     = "rerank-v3.5",
    top_n     = 5,
)
```

**Best for**: Production where you don't want to host a cross-encoder.
**Cost**: ~$0.001 per 1K results.

---

### 3. LLM-as-Judge Reranker
Use the generation LLM to score each candidate's relevance on a 1–10 scale. Slow and expensive — but the most semantically rich signal.

```
Prompt: "On a scale 1–10, how relevant is this passage to the query?
         Query: 'APIC HA requirements'
         Passage: 'The APIC cluster requires 3 nodes for quorum...'
         Reply with just the number."
Response: "9"
```

**Best for**: High-stakes retrieval where precision matters more than cost (compliance, legal).

---

## Reranking Metrics

| Metric | What it measures |
|---|---|
| **MRR@K** (Mean Reciprocal Rank) | How high the first relevant doc appears |
| **NDCG@K** (Normalized DCG) | Full ranking quality with graded relevance |
| **Precision@K** | Fraction of top-K that are relevant |
| **Recall@K** | Fraction of all relevant docs in top-K |

Reranking maximizes **Precision@M** (fewer, better chunks to LLM) while relying on Stage 1 for **Recall@K**.

---

## When to Rerank

| Use reranking | Skip reranking |
|---|---|
| Multiple documents look similar in score | Clear top-1 with large score gap |
| Query is short or ambiguous | Long precise query with exact-term match |
| Context window is expensive (budget M < 5) | Latency-critical path (< 100ms SLA) |
| High-stakes answers (compliance, support) | Simple FAQ / lookup |

---

## Interview Questions

**Q: What is the difference between a bi-encoder and a cross-encoder?**
A: A bi-encoder encodes query and document *independently* into separate embeddings, then computes similarity. Fast but shallow — it can't model query-document interactions. A cross-encoder encodes the query and document *together* in a single forward pass with full cross-attention between them. Much more accurate but requires one inference call per (query, document) pair — scales as O(K) at reranking time, not O(N) at indexing time.

**Q: Why not just use a cross-encoder for retrieval directly?**
A: Cross-encoders require a fresh inference for every (query, document) pair. With 1M documents, that's 1M inference calls per query — unacceptably slow. Bi-encoders pre-compute document embeddings; at query time only one embedding is needed, followed by a fast vector similarity search. Reranking uses cross-encoders only on the short candidate list (50–200 docs) returned by stage 1.

**Q: What is the key tradeoff when choosing how many candidates to rerank (K)?**
A: K is a precision-recall-latency triangle. Larger K → higher recall (less chance of missing a relevant doc before reranking) but higher reranking latency and cost. Smaller K → faster but risks cutting relevant docs before the reranker sees them. Typical values: K=50 for fast retrieval, K=100-200 for high-recall scenarios. The reranker then selects top-M (M=3–10) for the LLM.

**Q: When would you use LLM-as-judge reranking over a cross-encoder?**
A: LLM-as-judge is best when: (1) the relevance criterion is complex and domain-specific — a fine-tuned cross-encoder on MS-MARCO may not understand your specific scoring rubric; (2) you need the reranker to explain *why* it ranked a document higher (auditability); (3) you're in a regulated industry where the reranking rationale must be human-readable. The tradeoff: 5–20× higher cost and latency vs a cross-encoder.

---

## Quiz

1. A cross-encoder improves on bi-encoder retrieval because:
   a) It is faster at indexing time
   **b) It scores query and document jointly, enabling cross-attention between them**
   c) It uses BM25 under the hood
   d) It eliminates the need for embeddings

2. The two-stage pipeline structure (retrieve K, rerank to M) exists because:
   a) Cross-encoders can't process long documents
   **b) Cross-encoders are too slow to score all N corpus documents at query time**
   c) Bi-encoders are more accurate
   d) It reduces storage requirements

3. MRR@K measures:
   a) The fraction of top-K documents that are relevant
   **b) The reciprocal of the rank of the first relevant document**
   c) The total recall across the corpus
   d) The hallucination rate of retrieved documents

4. You should skip reranking when:
   a) The query is ambiguous
   b) Context budget is tight
   **c) Latency is the primary constraint and retrieval already has a clear top result**
   d) Multiple documents have similar scores

5. Cohere Rerank is preferable to a hosted cross-encoder when:
   a) You want the lowest possible latency
   b) You need explainability
   **c) You want accurate reranking without model hosting infrastructure**
   d) The corpus has fewer than 100 documents
