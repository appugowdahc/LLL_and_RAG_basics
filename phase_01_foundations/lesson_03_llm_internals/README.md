# Phase 1 — Lesson 3: How LLMs Work Internally

## Overview

When you call `client.messages.create(...)`, a precise sequence of mathematical
operations happens inside the model. Understanding this pipeline is essential for:
  - Debugging why a model gives wrong answers
  - Understanding why context window size matters
  - Designing better prompts
  - Understanding embeddings (Lesson 8) which power all of RAG

---

## The Complete Internal Pipeline

```
YOUR TEXT (string)
      │
      ▼
┌─────────────────────────────────────────┐
│  STEP 1: TOKENIZATION                   │
│                                         │
│  "Hello RAG" → [15496, 432, 38, 38]     │
│                                         │
│  - Byte-Pair Encoding (BPE) algorithm   │
│  - Maps subwords to integer IDs         │
│  - Vocabulary size: ~50,000-100,000     │
└────────────────┬────────────────────────┘
                 │  List[int]  (token IDs)
                 ▼
┌─────────────────────────────────────────┐
│  STEP 2: TOKEN EMBEDDING LOOKUP         │
│                                         │
│  token_id 15496 → [0.23, -0.11, 0.87,  │
│                    ..., 0.44]           │
│                   (4096-dimensional)    │
│                                         │
│  - Embedding matrix: vocab_size × d_model │
│  - Each token maps to a dense vector    │
│  - This vector = "initial meaning"      │
└────────────────┬────────────────────────┘
                 │  Tensor[seq_len, d_model]
                 ▼
┌─────────────────────────────────────────┐
│  STEP 3: POSITIONAL ENCODING            │
│                                         │
│  Adds position info to each token vec:  │
│  token[0] += pos_enc(0)                 │
│  token[1] += pos_enc(1)  ... etc.       │
│                                         │
│  WHY: Transformers have no inherent     │
│  sense of order — position must be      │
│  explicitly injected.                   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐  ─┐
│  STEP 4: TRANSFORMER BLOCK (×N layers) │   │
│                                         │   │
│  ┌─────────────────────────────────┐   │   │
│  │  Multi-Head Self-Attention       │   │   │
│  │  Each token "looks at" all others│   │   │  Repeated
│  │  and updates its representation  │   │   │  32–96 times
│  └────────────┬────────────────────┘   │   │
│               │                         │   │
│  ┌────────────▼────────────────────┐   │   │
│  │  Feed-Forward Network (FFN)     │   │   │
│  │  2 linear layers + activation   │   │   │
│  │  Transforms each token vector   │   │   │
│  └────────────┬────────────────────┘   │   │
│               │                         │   │
│  (LayerNorm + Residual Connections)     │   │
└───────────────┬─────────────────────────┘  ─┘
                │
                ▼
┌─────────────────────────────────────────┐
│  STEP 5: LM HEAD (Language Model Head)  │
│                                         │
│  Final vector (d_model) → Logits        │
│  Linear projection: d_model → vocab_size│
│                                         │
│  Output: score for EVERY token in vocab │
│  [Paris: 12.4, Lyon: 3.2, Nice: 2.1...] │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  STEP 6: SOFTMAX + SAMPLING             │
│                                         │
│  Logits → Probabilities via softmax     │
│  Sample next token based on temperature │
│                                         │
│  → Token ID → Detokenize → Character(s) │
└─────────────────────────────────────────┘
                 │
                 ▼  (repeat steps 3–6 for each new token)
           GENERATED TEXT
```

---

## Key Model Dimensions (Real Numbers)

| Model        | d_model | Layers | Heads | Params  | Context |
|--------------|---------|--------|-------|---------|---------|
| GPT-2 Small  | 768     | 12     | 12    | 117M    | 1,024   |
| GPT-3        | 12,288  | 96     | 96    | 175B    | 4,096   |
| Llama-3-8B   | 4,096   | 32     | 32    | 8B      | 128K    |
| Llama-3-70B  | 8,192   | 80     | 64    | 70B     | 128K    |
| Claude Sonnet| ~8,192  | ~60    | ~64   | ~70B*   | 200K    |

*Anthropic doesn't publish exact architecture details

---

## Why This Matters for RAG

```
CONCEPT               RAG CONNECTION
─────────────────────────────────────────────────────────────────
Token embedding       Embeddings are extracted from this layer for vector search
Attention mechanism   Explains why retrieved context in the prompt gets "attended to"
Context window        Hard limit on how many retrieved chunks fit in one call
Autoregressive decoding  Why streaming works / why long outputs are slow
Softmax temperature   Controls hallucination risk vs. answer confidence
```

---

## Files in This Lesson

| File                              | What It Teaches                                |
|-----------------------------------|------------------------------------------------|
| 01_tokenization_pipeline.py       | BPE, token IDs, special tokens, edge cases     |
| 02_embedding_vectors.py           | Embed tokens/sentences, compare distances      |
| 03_attention_intuition.py         | Attention scores, why position matters         |
| 04_autoregressive_decoding.py     | Token-by-token generation, streaming           |
| 05_mini_project_llm_visualizer.py | Full pipeline visualizer: text → tokens → gen  |

---

## Interview Questions

Q1: What is tokenization and why don't LLMs use words as their basic unit?
A: Tokenization splits text into subword units using Byte-Pair Encoding (BPE).
   Words have infinite forms (plurals, conjugations, typos). Subwords handle all
   cases with a fixed vocabulary of ~50k tokens. Also handles any language.

Q2: What is d_model and what does it represent?
A: d_model is the dimensionality of the token representation vector throughout
   the transformer. Every token is represented as a d_model-dimensional vector
   that evolves through each layer. Larger d_model = more expressive representations.

Q3: Why do transformers need positional encoding?
A: The attention mechanism is permutation-invariant — it treats all tokens equally
   regardless of position. Positional encoding explicitly injects order information
   so the model knows "token 3 comes before token 4".

Q4: What is the LM Head?
A: A linear projection that maps the final hidden state vector (d_model dimensions)
   to a score (logit) for every token in the vocabulary. Softmax converts logits
   to probabilities. The model samples the next token from this distribution.

Q5: What is the difference between the embedding layer and the LM Head in most LLMs?
A: They are often the SAME weight matrix (weight tying). The embedding matrix maps
   token IDs to vectors (input side). The LM Head maps vectors back to token IDs
   (output side). Sharing weights reduces parameters and improves training efficiency.

---

## Quiz

1. What are the 6 steps in the LLM forward pass?
2. What algorithm does tokenization use? Why subwords instead of words?
3. Why does the transformer need positional encoding?
4. What does the LM Head produce? What are logits?
5. What is autoregressive generation?
