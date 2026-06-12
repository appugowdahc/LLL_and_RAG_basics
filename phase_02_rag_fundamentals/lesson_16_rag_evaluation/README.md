# Phase 2 — Lesson 16: RAG Evaluation

## Why Evaluation Is Non-Negotiable

A RAG pipeline has many moving parts — chunking strategy, embedding model, retrieval K, reranker, prompt template, generation model. Without evaluation you cannot answer:

- Is the pipeline actually improving over no-RAG?
- Which change made things better or worse?
- What breaks under real queries?

**Without measurement, you are guessing.**

```
┌──────────────────────────────────────────────────────────────────────┐
│  WHAT CAN GO WRONG SILENTLY IN RAG                                   │
│                                                                      │
│  Retrieval fails:                                                    │
│    - Top chunk is topically related but doesn't contain the answer   │
│    - Relevant chunk exists but ranks #12, never reaches the LLM      │
│                                                                      │
│  Generation fails:                                                   │
│    - LLM uses context but also injects parametric knowledge          │
│    - Answer is fluent but fabricated ("hallucinated but confident")  │
│    - Answer is technically grounded but doesn't address the question │
│                                                                      │
│  Both fail:                                                          │
│    - Query is about a topic not in the knowledge base                │
│    - LLM refuses to answer ("I don't see that in the context")       │
│      even though the answer IS present in chunk 4                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## The Four Core RAGAS Metrics

RAGAS (Retrieval-Augmented Generation Assessment, Es et al. 2023) defines four metrics that cover both retrieval quality and generation quality.

### 1. Faithfulness
*Is the answer grounded in the retrieved context?*

```
Score = (# claims in answer that are supported by context) / (# total claims)

Example:
  Answer: "APIC requires 3 nodes for HA. It was released in 2012."
  Context: "APIC requires 3 nodes for HA."
  Claims:  ["APIC requires 3 nodes for HA"] → supported ✓
           ["APIC released in 2012"]         → NOT in context ✗
  Faithfulness = 1/2 = 0.50
```

**Target**: ≥ 0.90 in production. Below 0.80 indicates significant hallucination.

---

### 2. Answer Relevance
*Does the answer actually address the question?*

```
Method: Generate N "reverse questions" from the answer, embed each,
        measure cosine similarity with the original question.
        High similarity → answer is on-topic.

Example:
  Question: "How many APIC nodes are required for HA?"
  Answer:   "The APIC cluster requires a minimum of 3 nodes."
  Reverse Q: "What is the minimum APIC cluster size?"
  Similarity to original Q: 0.92 → high relevance
```

**Target**: ≥ 0.85. Low scores indicate the LLM answered the wrong question or avoided the question.

---

### 3. Context Precision
*Are the retrieved chunks actually useful?*

```
Score = (# relevant chunks in top-K) / K

Example with K=5:
  Retrieved: [relevant, relevant, irrelevant, relevant, irrelevant]
  Precision  = 3/5 = 0.60
```

**Measures**: Retrieval quality — are you retrieving noise along with signal?

---

### 4. Context Recall
*Did retrieval find all the chunks needed to answer the question?*

```
Score = (# answer claims supported by retrieved context)
      / (# answer claims in the ideal/reference answer)

Example:
  Ideal answer has 3 claims. Retrieved context supports 2 of them.
  Context Recall = 2/3 = 0.67
```

**Measures**: Whether the retrieval stage missed critical information.

---

## Evaluation Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  GOLDEN DATASET                                                      │
│  (query, ideal_answer, relevant_chunk_ids)                           │
│          ↓                                                           │
│  FOR EACH QUERY:                                                     │
│    1. Retrieve K chunks   → compute Context Precision + Recall       │
│    2. Generate answer     → compute Faithfulness + Answer Relevance  │
│          ↓                                                           │
│  AGGREGATE across all queries → report mean metrics                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Building a Golden Dataset

| Field | Description |
|---|---|
| `query` | Real user question (from production logs or curated) |
| `ideal_answer` | Ground-truth answer written by domain expert |
| `relevant_chunk_ids` | IDs of all chunks that should be retrieved |
| `answer_claims` | Atomic claims that the ideal answer makes |

**Sources for golden datasets**:
1. Production query logs + human annotation
2. Domain expert interviews (what questions do ops engineers ask?)
3. LLM-generated synthetic data + human review (cheapest to bootstrap)
4. BEIR benchmark for generic retrieval evaluation

---

## Interview Questions

**Q: What is faithfulness in RAG and why does it matter?**
A: Faithfulness measures whether each claim in the generated answer is supported by the retrieved context — not by the LLM's parametric memory. A faithfulness score of 1.0 means every statement in the answer can be traced to a retrieved chunk. It matters because low faithfulness means the LLM is hallucinating beyond the provided context, making the RAG pipeline unreliable. The context was retrieved specifically to constrain the LLM — if the LLM ignores it, you've paid retrieval cost for nothing.

**Q: What is the difference between context precision and context recall?**
A: Context precision measures what fraction of the retrieved chunks are actually relevant — it penalizes noise retrieval. Context recall measures whether retrieval found all the chunks needed to answer the question — it penalizes missed relevant content. Both matter: high precision alone (only 1 chunk, which is relevant) may miss information; high recall alone (retrieve everything) buries the answer in noise. RAGAS targets ≥ 0.8 for both.

**Q: How does RAGAS measure answer relevance without human labels?**
A: RAGAS generates N "reverse questions" from the answer using the LLM (e.g., "What question does this answer?") and then computes the cosine similarity between those reverse questions and the original query. If the answer is on-topic, the reverse questions should be semantically similar to the original question. This is reference-free — no human labels needed — but requires an LLM call per answer.

**Q: How do you build a golden dataset for RAG evaluation?**
A: Three approaches: (1) Mine production query logs and have domain experts annotate which answers are correct and which chunks are relevant — high quality but slow and expensive. (2) Use an LLM to generate (query, answer) pairs from your knowledge base, then have humans review — fast to bootstrap, needs careful quality control. (3) Repurpose existing domain QA datasets if your domain overlaps. For the retrieval side specifically, create QA pairs where each answer can only come from specific chunks — this lets you measure context recall precisely.

---

## Quiz

1. A RAG system with faithfulness = 0.60 means:
   a) 60% of queries were answered correctly
   **b) 40% of claims in generated answers are not supported by retrieved context**
   c) 60% of retrieved chunks are relevant
   d) The LLM uses only 60% of the context

2. Context recall measures:
   a) How many chunks were retrieved
   **b) Whether all needed information was found in retrieval**
   c) How relevant the top retrieved chunk is
   d) Whether the answer cites its sources

3. Answer relevance uses "reverse question generation" because:
   a) It's cheaper than faithfulness scoring
   **b) It is reference-free — no human-labeled answers needed**
   c) It measures hallucination directly
   d) It requires fewer LLM calls

4. A production RAG system should target faithfulness ≥:
   a) 0.50
   b) 0.70
   **c) 0.90**
   d) 1.00 always

5. The best source of a golden evaluation dataset is:
   a) Synthetic LLM-generated data only
   **b) Real production queries + domain expert annotation**
   c) BEIR benchmark (always applicable)
   d) Random sampling of the knowledge base
