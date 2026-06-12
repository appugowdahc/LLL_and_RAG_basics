# Phase 1 — Lesson 4: Transformer Architecture

## Origin

Introduced in "Attention Is All You Need" (Vaswani et al., Google, 2017).
Replaced RNNs and LSTMs entirely for sequence modeling tasks.
The core insight: **attention alone** (no recurrence, no convolution) is enough.

---

## Three Transformer Variants

```
ENCODER-ONLY                DECODER-ONLY               ENCODER-DECODER
(BERT, RoBERTa)             (GPT, Claude, Llama)       (T5, BART)
─────────────────           ──────────────────         ────────────────────
Input → Encoder → CLS       Input → Decoder → next     Input → Encoder
                 token                        token               │
TASK:                        TASK:                               ▼
  Classification              Generation                  Encoder states
  NER, Similarity             Completion                        │
  Embedding                   Q&A, Chat                         ▼
                              RAG answers              Input → Decoder → output
                                     ↑
                              THIS IS WHAT                TASK:
                              CLAUDE/GPT USE              Translation
                                                          Summarization
                                                          Seq2Seq
```

**For RAG: we almost exclusively use Decoder-only models (Claude, GPT-4, Llama).**

---

## Full Decoder-Only Transformer Architecture

```
INPUT TEXT: "The cat sat on"
       │
       ▼
┌──────────────────────────────────────────┐
│  TOKENIZER                               │
│  "The cat sat on" → [464, 3797, 3332, 319]│
└──────────────┬───────────────────────────┘
               │ token_ids: [seq_len]
               ▼
┌──────────────────────────────────────────┐
│  TOKEN EMBEDDING LAYER                   │
│  Lookup: token_id → d_model vector       │
│  Matrix: [vocab_size × d_model]          │
│  Output: [seq_len × d_model]             │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  POSITIONAL ENCODING                     │
│  Add position info to each token vector  │
│  (Sinusoidal or RoPE)                    │
│  Output: [seq_len × d_model]             │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐  ─┐
│  TRANSFORMER BLOCK × N_LAYERS            │   │
│  ┌────────────────────────────────────┐  │   │
│  │  1. PRE-LAYER NORM (RMSNorm/LN)    │  │   │
│  │     Normalize input before attn    │  │   │
│  └────────────────┬───────────────────┘  │   │
│                   │                      │   │
│  ┌────────────────▼───────────────────┐  │   │
│  │  2. MULTI-HEAD CAUSAL ATTENTION    │  │   │
│  │     (each token attends to past)   │  │   │  Repeated
│  │     Q,K,V projections              │  │   │  N times
│  │     Causal mask (no future peek)   │  │   │  (32–96)
│  └────────────────┬───────────────────┘  │   │
│                   │                      │   │
│  ┌────────────────▼───────────────────┐  │   │
│  │  3. RESIDUAL CONNECTION            │  │   │
│  │     output = x + attn(norm(x))     │  │   │
│  └────────────────┬───────────────────┘  │   │
│                   │                      │   │
│  ┌────────────────▼───────────────────┐  │   │
│  │  4. PRE-LAYER NORM (again)         │  │   │
│  └────────────────┬───────────────────┘  │   │
│                   │                      │   │
│  ┌────────────────▼───────────────────┐  │   │
│  │  5. FEED-FORWARD NETWORK (FFN)     │  │   │
│  │     Linear(d_model → 4×d_model)    │  │   │
│  │     Activation (SwiGLU / GELU)     │  │   │
│  │     Linear(4×d_model → d_model)    │  │   │
│  └────────────────┬───────────────────┘  │   │
│                   │                      │   │
│  ┌────────────────▼───────────────────┐  │   │
│  │  6. RESIDUAL CONNECTION            │  │   │
│  │     output = x + ffn(norm(x))      │  │   │
│  └────────────────┬───────────────────┘  │   │
└────────────────────┼─────────────────────┘  ─┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  FINAL LAYER NORM                        │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  LM HEAD                                 │
│  Linear: [d_model → vocab_size]          │
│  Output: logits [seq_len × vocab_size]   │
└──────────────┬───────────────────────────┘
               │
               ▼
         NEXT TOKEN PROBS
         Sample → "the" (next token)
```

---

## Key Components Explained

### 1. Residual Connections (Skip Connections)
```
input ──────────────────────────┐
  │                             │
  ▼                             │ (skip)
[Attention or FFN block]        │
  │                             │
  ▼                             │
output_of_block                 │
  │                             │
  └──── + (add) ◄───────────────┘
             │
         final_output = input + block(input)
```
WHY: Gradients flow directly through the skip connection during backprop.
     Prevents vanishing gradients in very deep networks.
     Network can learn identity function (ignore a block) if needed.

### 2. Layer Normalization
```
input vector x of dimension d_model
  │
  ▼
mean μ = (1/d) × Σ x_i
var  σ² = (1/d) × Σ (x_i - μ)²
  │
  ▼
x_norm = (x - μ) / sqrt(σ² + ε)
  │
  ▼
output = γ × x_norm + β        (γ, β are learned scale/shift parameters)
```
WHY: Stabilizes training by keeping activations in a consistent range.
     Without it: deep networks develop exploding or vanishing activations.

### 3. Feed-Forward Network (FFN / MLP)
```
input: [seq_len × d_model]
  │
  ▼
Linear: [d_model → d_ff]     (d_ff = 4 × d_model typically)
  │
  ▼
Activation: GELU or SwiGLU
  │
  ▼
Linear: [d_ff → d_model]
  │
  ▼
output: [seq_len × d_model]
```
WHY: Attention captures WHICH information to use (routing).
     FFN transforms WHAT the information means (computation).
     Together: a complete reasoning step.

### 4. KV Cache (Critical for RAG Performance)
```
WITHOUT KV Cache:
  Token 1: process tokens [1]
  Token 2: process tokens [1,2]       ← recomputes token 1!
  Token 3: process tokens [1,2,3]     ← recomputes tokens 1,2!
  Token N: process tokens [1,...,N]   ← O(N²) total work!

WITH KV Cache:
  Token 1: compute K1,V1 → cache
  Token 2: compute K2,V2 → cache (reuse K1,V1 from cache)
  Token 3: compute K3,V3 → cache (reuse K1,V1,K2,V2 from cache)
  Token N: compute KN,VN only → O(N) total work!
```
WHY it matters for RAG: Long prompts with retrieved documents fill the KV cache.
KV cache size determines maximum context window in production deployments.

---

## Parameter Count Formula

```
For a transformer with:
  L = number of layers
  d = d_model (hidden size)
  d_ff = feed-forward size (usually 4×d)
  V = vocabulary size

Parameters per layer:
  Attention:    4 × d²          (Q, K, V, Output projections)
  FFN:          2 × d × d_ff   (up-projection + down-projection)
  LayerNorms:   4 × d            (2 norms × 2 params each)
  Total/layer:  4d² + 2d×d_ff + 4d

Total model:
  Embedding:   V × d
  Layers:      L × (4d² + 2d×d_ff + 4d)
  LM Head:     V × d (often shared with embedding)

Example: Llama-3-8B
  d=4096, L=32, d_ff=14336, V=128,256
  Per layer: 4×4096² + 2×4096×14336 = 67M + 117M = 184M params
  All layers: 32 × 184M = 5.9B
  Embeddings: 128,256 × 4096 = 0.5B
  Total ≈ 6.4B (rest from biases, norms = ~8B labeled)
```

---

## Files in This Lesson

| File                              | What It Teaches                                  |
|-----------------------------------|--------------------------------------------------|
| 01_architecture_overview.py       | Component inventory, shape flow, model configs   |
| 02_encoder_vs_decoder.py          | BERT vs GPT vs T5 behavioral comparison          |
| 03_layer_components.py            | LayerNorm, Residual, FFN from scratch (numpy)    |
| 04_parameter_counter.py           | Count params for any transformer config          |
| 05_kv_cache_demo.py               | KV cache mechanics + performance impact          |
| 06_mini_project_transformer_inspector.py | Full config inspector + RAG fit analysis |

---

## Interview Questions

Q1: What are the two sub-layers in every transformer block?
A: (1) Multi-Head Self-Attention and (2) Position-wise Feed-Forward Network.
   Both are wrapped with residual connections and layer normalization.

Q2: What is a residual connection and why is it essential?
A: output = input + f(input). The skip connection lets gradients flow directly
   through during backpropagation, solving the vanishing gradient problem
   and allowing networks of 96+ layers to train stably.

Q3: What is the difference between Pre-LN and Post-LN transformers?
A: Post-LN (original paper): norm after attention+residual. Harder to train.
   Pre-LN (modern LLMs): norm before the attention block. More stable training.
   All modern LLMs (GPT-4, Claude, Llama) use Pre-LN.

Q4: Why does the FFN use 4×d_model hidden size?
A: Empirically found to be a good balance. The FFN is where most factual
   knowledge is stored (research shows facts are memorized in FFN weights).
   Larger FFN = more capacity to store knowledge.

Q5: What is a KV cache and why does it matter for RAG?
A: KV cache stores computed Key and Value matrices for previous tokens,
   avoiding recomputation on each new token. In RAG with long contexts
   (many retrieved chunks), KV cache prevents O(N²) recomputation,
   making inference linear in the number of new tokens generated.

---

## Quiz

1. Name the 6 sub-components in a single transformer decoder block.
2. What does a residual connection add mathematically?
3. Why is Pre-LN (modern) better than Post-LN (original)?
4. What are the two linear layers in the FFN and what are their dimensions?
5. How does the KV cache save computation? What is the memory trade-off?
