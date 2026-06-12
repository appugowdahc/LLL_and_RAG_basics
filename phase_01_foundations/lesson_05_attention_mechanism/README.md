# Phase 1 — Lesson 5: Attention Mechanism

## The Central Idea

Attention answers one question: **"When computing the new meaning of token X,
how much should I borrow from token Y?"**

Every word's meaning depends on context. "Bank" in "river bank" vs "bank account"
needs different representations — attention learns to look at surrounding words
to build context-dependent representations.

---

## Scaled Dot-Product Attention — The Full Math

```
Given:
  Query  Q: [seq_len × d_k]   "What am I looking for?"
  Key    K: [seq_len × d_k]   "What information do I hold?"
  Value  V: [seq_len × d_v]   "What is my actual content?"

Step 1 — Compute raw scores (how relevant is each key to this query?):
  scores = Q × Kᵀ                      shape: [seq_len × seq_len]
  scores[i][j] = dot(Q[i], K[j])
                = how much token i's query matches token j's key

Step 2 — Scale (prevent softmax saturation):
  scores = scores / √d_k

  WHY √d_k: For random vectors, E[dot(q,k)] = 0, Var[dot(q,k)] = d_k
  Dividing by √d_k makes Var = 1 → softmax gradients stay healthy.

Step 3 — Mask (decoder only — prevent future token access):
  scores[i][j] = -∞   if j > i   (future token → zero attention)

Step 4 — Softmax (convert to probabilities):
  weights = softmax(scores)            shape: [seq_len × seq_len]
  weights[i] sums to 1.0
  weights[i][j] = how much token i attends to token j

Step 5 — Weighted sum of values:
  output = weights × V                 shape: [seq_len × d_v]
  output[i] = Σ_j weights[i][j] × V[j]
            = weighted blend of all token values, weighted by relevance

Final:
  Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```

---

## Multi-Head Attention — Why Multiple Heads

```
SINGLE HEAD:
  One Q, K, V projection → one attention pattern
  Can only capture ONE relationship type at a time

MULTI-HEAD (H heads):
  H independent Q, K, V projections → H parallel attention patterns
  Each head learns a DIFFERENT relationship:
    Head 1 → syntactic (subject-verb agreement)
    Head 2 → semantic  (word meaning similarity)
    Head 3 → positional (nearby tokens)
    Head 4 → coreference (pronoun → noun)
    ...

FORMULA:
  For each head h:
    head_h = Attention(Q × Wq_h, K × Wk_h, V × Wv_h)
    where Wq_h, Wk_h, Wv_h ∈ [d_model × d_k]  (d_k = d_model / H)

  MultiHead(Q,K,V) = Concat(head_1, ..., head_H) × W_O
  where W_O ∈ [H×d_v × d_model]

SHAPE FLOW:
  Input:           [seq_len × d_model]
  Per head:        [seq_len × d_k]     (d_k = d_model / H)
  Attention output: [seq_len × d_v]   per head
  After concat:    [seq_len × H×d_v]  = [seq_len × d_model]
  After W_O:       [seq_len × d_model]
```

---

## Three Types of Attention

```
1. SELF-ATTENTION (decoder, causal)
   Q, K, V all from the SAME sequence.
   Each token looks at itself + all PAST tokens.
   Causal mask prevents looking at future tokens.
   Used in: GPT, Claude, Llama (all decoder-only LLMs)

2. BIDIRECTIONAL SELF-ATTENTION (encoder)
   Q, K, V all from the SAME sequence.
   Each token can look at ALL tokens (past AND future).
   No causal mask.
   Used in: BERT, RoBERTa, embedding models

3. CROSS-ATTENTION (encoder-decoder)
   Q comes from the DECODER sequence.
   K, V come from the ENCODER output.
   The decoder attends to the full input encoding.
   Used in: T5, BART, original transformer (translation)
   RAG connection: cross-attention is conceptually what happens
   when the LLM reads retrieved context — it "attends to" those tokens.
```

---

## Positional Encodings

```
PROBLEM: Attention is permutation-invariant.
  Attention("cat sat mat") = Attention("mat cat sat")  ← same result!
  The model has no built-in sense of order.

SOLUTION: Positional Encoding (PE) — explicitly inject position info.

SINUSOIDAL (original paper):
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  Added to token embeddings. Deterministic (not learned).
  Generalizes to longer sequences than seen in training.

LEARNED POSITIONAL EMBEDDING:
  A learned table: [max_seq_len × d_model]
  Looked up like token embeddings. Simple, effective.
  Cannot generalize beyond max_seq_len seen in training.
  Used in: original GPT-1, GPT-2

ROTARY POSITION EMBEDDING (RoPE):
  Modern standard (Llama, Mistral, Qwen, Falcon).
  Rotates Q and K vectors by position-dependent angles.
  Encodes RELATIVE position — token i attends to token j
  based on their DISTANCE (i-j), not absolute positions.
  Better generalization to longer contexts.

  RoPE(q, pos) = q × [cos(pos×θ) | sin(pos×θ)]
  RoPE(k, pos) = k × [cos(pos×θ) | sin(pos×θ)]
  dot(RoPE(q,m), RoPE(k,n)) depends only on (m-n)

ALiBi (Attention with Linear Biases):
  Subtracts a linear penalty proportional to distance.
  score(i,j) = dot(Q_i, K_j) - slope_h × |i - j|
  Each head has a different slope.
  No learned position params. Strong length generalization.
  Used in: MPT, BLOOM.
```

---

## Flash Attention

```
PROBLEM: Standard attention is O(N²) in memory.
  For seq_len=200,000: 200k × 200k × 4 bytes = 160 GB — impossible!

STANDARD ATTENTION MEMORY:
  1. Store full attention matrix S = Q×Kᵀ: O(N²)
  2. Store softmax(S): O(N²)
  3. Compute output S×V: O(N²)

FLASH ATTENTION (Dao et al. 2022):
  Key insight: compute attention in TILES that fit in L2 cache (SRAM).
  Process one block of queries against blocks of keys/values.
  Accumulate the softmax incrementally (online softmax algorithm).
  Never materialize the full N×N attention matrix.

  Memory: O(N) — only store output, no full attention matrix
  Speed:  2-4× faster than standard attention (SRAM vs HBM bandwidth)
  Result: enables 100K+ token context windows in practice

Flash Attention 2 (2023): further optimized for modern GPUs
Flash Attention 3 (2024): optimized for H100 with async execution
```

---

## Why Attention Matters for RAG

```
RETRIEVED CONTEXT IN PROMPT:
  "Context: [retrieved chunk 1] [retrieved chunk 2] [retrieved chunk 3]
   Question: What is X?"

  When the model generates each answer token, the attention mechanism
  lets it "look back" at any retrieved chunk.

  Attention heads specialized for:
    - Semantic similarity → attend to chunks about the same topic
    - Coreference → track "it", "this", "the system" across chunks
    - Factual recall → find the specific sentence answering the question

IMPLICATION FOR RETRIEVAL QUALITY:
  If a retrieved chunk is NOT semantically relevant to the question,
  the model's attention weights to that chunk will be low.
  The model "ignores" irrelevant retrieved content.
  → Retrieval quality (what you inject) matters, but the model self-filters.
  → However, irrelevant chunks STILL consume context window tokens.
  → Still worth retrieving only relevant content (efficiency + focus).
```

---

## Files in This Lesson

| File                              | What It Teaches                                     |
|-----------------------------------|-----------------------------------------------------|
| 01_single_head_attention.py       | Full math: Q,K,V, scores, softmax, output           |
| 02_multi_head_attention.py        | Multi-head from scratch, concat, projection         |
| 03_positional_encodings.py        | Sinusoidal, RoPE, ALiBi — code + visualization      |
| 04_flash_attention_concept.py     | Memory complexity, tiling concept, benchmarks       |
| 05_attention_patterns.py          | Visualize what heads actually learn                 |
| 06_mini_project_attention_visualizer.py | RAG prompt attention map                      |

---

## Interview Questions

Q1: Write the attention formula and explain each term.
A: Attention(Q,K,V) = softmax(QKᵀ/√d_k) × V
   Q = what each token seeks; K = what each token has; V = actual content.
   QKᵀ = similarity scores; /√d_k = scaling; softmax = probabilities; ×V = weighted blend.

Q2: Why scale by √d_k?
A: For random unit vectors in d_k dimensions, dot products have variance d_k.
   Without scaling, large d_k → large dot products → softmax saturates near 0/1
   → near-zero gradients → training fails. Dividing by √d_k normalizes variance to 1.

Q3: What is the difference between RoPE and sinusoidal positional encoding?
A: Sinusoidal adds absolute position vectors to embeddings at the input.
   RoPE rotates Q and K vectors inside each attention head by position-dependent angles,
   encoding RELATIVE positions. dot(RoPE(q,m), RoPE(k,n)) = f(m-n only).
   RoPE generalizes better to longer contexts.

Q4: What is Flash Attention and why is it important for RAG?
A: Flash Attention computes attention in SRAM tiles, never materializing the O(N²)
   attention matrix. Reduces memory from O(N²) to O(N), enabling long contexts (200K+)
   needed to fit many retrieved chunks in a single RAG call.

Q5: What is cross-attention and how does it relate to RAG?
A: Cross-attention has Q from one sequence and K,V from another. In encoder-decoder
   models, the decoder attends to encoded input. In RAG terms, the LLM effectively
   cross-attends to retrieved context tokens when generating answers.

---

## Quiz

1. Write the 5-step attention computation with shapes for each step.
2. Why divide attention scores by √d_k?
3. What is the difference between causal (masked) and bidirectional attention?
4. What is RoPE and why is it better than learned positional embeddings?
5. What memory problem does Flash Attention solve, and how does tiling fix it?
