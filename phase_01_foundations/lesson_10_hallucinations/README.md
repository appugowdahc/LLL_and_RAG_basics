# Phase 1 — Lesson 10: Hallucinations

## What Are Hallucinations?

A **hallucination** is when an LLM generates text that is **confidently wrong** — factually incorrect, fabricated, or unsupported by any real evidence — while sounding completely plausible.

The term comes from psychiatry (perception without stimulus). In LLMs it means: **output without grounding in real-world fact or provided context**.

---

## Taxonomy: Three Types of Hallucination

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HALLUCINATION TAXONOMY                           │
├───────────────────┬─────────────────────────────────────────────────┤
│  TYPE             │  DESCRIPTION + EXAMPLE                          │
├───────────────────┼─────────────────────────────────────────────────┤
│  INTRINSIC        │  Contradicts provided context.                  │
│  (worst)          │  Context: "ACI v6.0 supports 200 leafs"         │
│                   │  Output:  "ACI supports up to 100 leaf switches" │
│                   │  WHY dangerous: user trusts context is the       │
│                   │  ground truth, but LLM ignores it.               │
├───────────────────┼─────────────────────────────────────────────────┤
│  EXTRINSIC        │  Adds info not in context, possibly wrong.       │
│                   │  Context: "ACI uses APIC for management"         │
│                   │  Output:  "ACI uses APIC, which requires 5       │
│                   │            nodes for HA" (APIC needs 3, not 5)  │
│                   │  WHY happens: LLM fills gaps from parametric     │
│                   │  memory, which may be outdated or wrong.         │
├───────────────────┼─────────────────────────────────────────────────┤
│  CONFABULATION    │  Fabricates entities that don't exist.           │
│                   │  "The Cisco ACI paper by Johnson et al. (2019)   │
│                   │   describes the algorithm in detail."            │
│                   │  WHY happens: LLM learned citation patterns but  │
│                   │  has no specific memory → interpolates a fake.   │
└───────────────────┴─────────────────────────────────────────────────┘
```

---

## Root Causes

### 1. Parametric vs Non-Parametric Memory

```
┌────────────────────────────────────────────────────────────────────┐
│  PARAMETRIC MEMORY (in weights)                                    │
│    Baked into model at training time. Can't be updated.            │
│    Stale after training cutoff.                                    │
│    Compressed → details lost → model guesses (hallucinates).       │
│                                                                    │
│  NON-PARAMETRIC MEMORY (retrieved at inference)                    │
│    Fetched fresh from a knowledge store at query time.             │
│    Always up to date. Exact, not compressed.                       │
│    THIS is what RAG provides.                                      │
└────────────────────────────────────────────────────────────────────┘
```

### 2. Decoding Under Uncertainty

When the model is "uncertain" (the next-token probability distribution is flat), it still **must pick a token**. It picks a plausible-sounding one, not necessarily a correct one.

Temperature > 0 amplifies this. Even at temperature=0 (greedy decoding), hallucinations occur — they are a **training and grounding problem**, not just a sampling problem.

### 3. Training Data Artifacts

- **Sycophancy**: Model learned to agree with users → affirms false premises.
- **Frequency bias**: Rare facts are poorly learned; common names/places are over-generated.
- **Pattern completion**: "The capital of Australia is..." → model completes "Sydney" (popular, wrong) not "Canberra" (correct, less frequent in web text).

### 4. Context Blindness

See "Lost in the Middle" (Lesson 6). Even when the correct answer IS in the context, the model may fail to attend to it and fall back on parametric memory — an intrinsic hallucination.

---

## Why RAG Is the Primary Mitigation

```
┌───────────────────────────────────────────────────────────────────┐
│  WITHOUT RAG                    WITH RAG                          │
│                                                                   │
│  Query ──► LLM ──► Answer       Query ──► Retriever ──► Context  │
│                                              │                    │
│  Source: parametric memory                  ▼                    │
│  (stale, compressed, uncertain)    LLM + Context ──► Answer      │
│                                                                   │
│                                   Source: retrieved documents     │
│                                   (current, exact, citeable)      │
└───────────────────────────────────────────────────────────────────┘
```

RAG mitigates hallucinations by:
1. **Grounding**: answer must come from retrieved context, not memory.
2. **Verifiability**: citations allow post-hoc fact-checking.
3. **Freshness**: retrieval fetches current docs; parametric memory is frozen.
4. **Scope control**: "Answer only from the provided context" in the system prompt.

**Limitations**: RAG does NOT eliminate hallucinations. The model can still:
- Ignore the context and answer from memory (context blindness).
- Mix context with memory (extrinsic hallucination).
- Misinterpret correct context and produce wrong answer.

---

## Faithfulness vs Relevance

| Metric | Meaning | How to Measure |
|---|---|---|
| **Faithfulness** | Is the answer supported by the retrieved context? | NLI model checks if context entails answer |
| **Answer Relevance** | Does the answer address the question? | Embedding sim between query and answer |
| **Context Relevance** | Is the retrieved context actually relevant? | Embedding sim between query and chunks |
| **Grounding rate** | % of claims in the answer that are in the context | Claim extraction + entailment check |

RAGAS (Retrieval Augmented Generation Assessment) formalizes these four metrics.

---

## Detection Techniques

| Technique | How It Works | Cost |
|---|---|---|
| **Self-consistency** | Sample N answers at temp>0; if they diverge, uncertainty is high | N × inference cost |
| **NLI entailment** | Check: does context entail the answer? DeBERTa/NLI model | Fast (small NLI model) |
| **Source attribution** | Ask LLM to cite which chunk each sentence comes from | Near-zero (prompt engineering) |
| **Logit-based confidence** | Average token log-probability of the answer | Requires logprob access |
| **LLM-as-judge** | Ask a second LLM to critique the answer vs the context | 1 extra LLM call |

---

## Interview Questions

**Q: What is the difference between intrinsic and extrinsic hallucination?**
A: Intrinsic = contradicts provided context. Extrinsic = adds info not in context (may or may not be factually wrong, but is ungrounded). Intrinsic is more dangerous because the ground truth was available and ignored.

**Q: Why does RAG reduce but not eliminate hallucinations?**
A: RAG reduces hallucinations by replacing parametric memory with retrieved context. But the model still generates from that context probabilistically — it can misread, extrapolate, or ignore parts of the context. The "lost in the middle" problem means information in the middle of a long context is less likely to be used. Additionally, if retrieval fails (wrong chunks retrieved), the model may fill gaps with parametric memory.

**Q: What is the faithfulness metric in RAGAS?**
A: Faithfulness measures whether every claim in the generated answer is supported by the retrieved context. A score of 1.0 means every statement is grounded; 0.0 means none are. Computed by: extracting statements from the answer, then using an NLI model to check if the context entails each statement.

**Q: How does temperature affect hallucinations?**
A: Higher temperature increases randomness → more hallucination risk. BUT hallucinations occur even at temperature=0 (greedy decoding), because the model may not have the correct information in parametric memory — no amount of sampling determinism fixes a knowledge gap. Temperature controls variance, not accuracy.

**Q: What is sycophancy and how does it cause hallucinations?**
A: Sycophancy is when an LLM agrees with or validates a user's (possibly false) premise to appear helpful. Example: user says "I read that ACI needs 10 APIC nodes" → sycophantic LLM says "Yes, 10 APIC nodes provide the best HA" (wrong). RLHF training can inadvertently reward agreement because human raters prefer being validated.

---

## Quiz

1. A user asks: "According to your documents, how many APIC nodes does ACI require?" The LLM correctly retrieved the right chunk but still says "5 nodes" (the real answer is 3). What type of hallucination is this?
   a) Confabulation
   b) Extrinsic
   **c) Intrinsic** ← correct: context was present, model ignored/misread it

2. Which technique detects hallucinations WITHOUT needing a second model call?
   a) LLM-as-judge
   b) NLI entailment
   **c) Source attribution via prompt engineering** ← ask the LLM to cite; free

3. At temperature=0, hallucinations:
   a) Are impossible
   b) Increase significantly
   **c) Still occur due to knowledge gaps in parametric memory**

4. RAGAS faithfulness score of 0.4 means:
   **a) 40% of answer claims are supported by retrieved context**
   b) 40% of relevant docs were retrieved
   c) The answer is 40% semantically similar to the query

5. The primary advantage of RAG over a standalone LLM for factual accuracy:
   a) Larger context window
   **b) Non-parametric memory — answers grounded in retrieved, citable documents**
   c) Faster inference
