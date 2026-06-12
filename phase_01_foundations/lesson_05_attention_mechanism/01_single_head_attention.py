"""
FILE: 01_single_head_attention.py
LESSON: Phase 1 - Lesson 5 - Attention Mechanism
TOPIC: Single-Head Scaled Dot-Product Attention — complete math from scratch

WHAT THIS FILE TEACHES:
  - Build Q, K, V projections from token embeddings
  - Compute attention scores step-by-step with shapes at every line
  - Apply causal mask (why and how)
  - Numerically stable softmax (why this matters)
  - Compute weighted output
  - Visualize the attention weight matrix

THE CORE FORMULA (memorize this):
  Attention(Q, K, V) = softmax( Q × Kᵀ / √d_k ) × V

INSTALL:
  pip install numpy
"""

import math
import numpy as np


# ─── Weight Initialization ────────────────────────────────────────────────────

def init_projection(d_in: int, d_out: int, seed: int) -> np.ndarray:
    """
    Initialize a linear projection matrix with Xavier/Glorot initialization.

    WHY Xavier init (√(2/(d_in + d_out))):
      Keeps the variance of activations consistent across layers.
      Too large → exploding activations early in training.
      Too small → vanishing activations — gradients can't flow.
      Xavier is the theoretical optimum for linear + tanh layers.
      For attention in transformers, it keeps initial dot products
      in a reasonable range before the √d_k scaling takes effect.

    Args:
        d_in:  Input dimension.
        d_out: Output dimension.
        seed:  Random seed for reproducibility.

    Returns:
        Weight matrix [d_in × d_out].
    """

    rng   = np.random.default_rng(seed)
    scale = math.sqrt(2.0 / (d_in + d_out))

    # WHY standard_normal × scale (not uniform):
    #   Normal distribution is better for deep nets (central limit theorem).
    #   Multiplying by scale achieves Xavier variance without changing the
    #   distribution shape.
    return rng.standard_normal((d_in, d_out)) * scale


# ─── The Attention Components ─────────────────────────────────────────────────

class SingleHeadAttention:
    """
    Single-head scaled dot-product attention — the foundational building block.

    Every multi-head attention module is just H independent instances of this,
    run in parallel and then concatenated.

    DIMENSIONS:
      d_model: Token embedding dimension (e.g., 512, 768, 4096)
      d_k:     Query/Key projection dimension
      d_v:     Value projection dimension
      seq_len: Number of tokens in the sequence

    In practice d_k = d_v = d_model / num_heads, but we keep them
    separate here to make the shape math explicit.
    """

    def __init__(self, d_model: int, d_k: int, d_v: int):
        self.d_model = d_model
        self.d_k     = d_k
        self.d_v     = d_v

        # Three learned projection matrices — the ONLY parameters in attention.
        # WHY three separate projections:
        #   Q projection: learns "what to search for"
        #   K projection: learns "what to advertise as having"
        #   V projection: learns "what content to contribute"
        #   Separating them allows each role to specialise independently.
        self.W_q = init_projection(d_model, d_k, seed=1)   # [d_model × d_k]
        self.W_k = init_projection(d_model, d_k, seed=2)   # [d_model × d_k]
        self.W_v = init_projection(d_model, d_v, seed=3)   # [d_model × d_v]

    def project_qkv(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project input embeddings to Q, K, V spaces.

        Args:
            x: Token embeddings [seq_len × d_model]

        Returns:
            Q [seq_len × d_k]
            K [seq_len × d_k]
            V [seq_len × d_v]
        """

        # WHY @ (matmul, not element-wise):
        #   Each token embedding (row of x) is linearly projected to a lower-
        #   dimensional space. @ broadcasts over the seq_len dimension automatically:
        #   [seq_len × d_model] @ [d_model × d_k] = [seq_len × d_k]
        Q = x @ self.W_q   # [seq_len × d_k]
        K = x @ self.W_k   # [seq_len × d_k]
        V = x @ self.W_v   # [seq_len × d_v]

        return Q, K, V

    def compute_attention_scores(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Compute raw (pre-softmax) attention scores.

        MATH:
          scores = Q × Kᵀ / √d_k

          scores[i][j] = dot(Q[i], K[j]) / √d_k
                       = (how much token i "wants" what token j "has") / scale

        Args:
            Q: [seq_len × d_k]
            K: [seq_len × d_k]

        Returns:
            scores: [seq_len × seq_len]
        """

        # WHY K.T (transpose):
        #   We need the dot product of every Q-row with every K-row.
        #   Q @ K.T achieves this efficiently:
        #   [seq_len × d_k] @ [d_k × seq_len] = [seq_len × seq_len]
        #   Element [i][j] = dot(Q[i], K[j])
        raw_scores = Q @ K.T                  # [seq_len × seq_len]

        # WHY divide by √d_k:
        #   Q and K are d_k-dimensional vectors. For random unit vectors,
        #   E[dot(q,k)] = 0  but  Var[dot(q,k)] = d_k
        #   So std dev ≈ √d_k. Dividing re-normalizes std to ~1.
        #
        #   WITHOUT scaling: at d_k=64, dot products are in range [-8, 8]
        #   → softmax pushes extreme values to near 0 or 1
        #   → near-zero gradients (vanishing gradient problem in training)
        #
        #   WITH scaling: dot products stay in [-1, 1] range
        #   → softmax stays in a well-behaved gradient region
        scaled_scores = raw_scores / math.sqrt(self.d_k)

        return scaled_scores                  # [seq_len × seq_len]

    def apply_causal_mask(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply causal (autoregressive) mask — prevent attending to future tokens.

        WHY causal masking:
          During generation, token[i] must NOT see token[i+1], [i+2], ...
          If it could: the model would "cheat" by reading future tokens
          during training, and at inference time those tokens don't exist yet.

          The mask sets future positions to -∞ so softmax gives them prob ≈ 0.

        SHAPE:
          scores: [seq_len × seq_len]
          mask:   [seq_len × seq_len] — lower triangular (1=attend, 0=masked)

        Args:
            scores: Raw attention scores [seq_len × seq_len].

        Returns:
            Masked scores [seq_len × seq_len].
        """

        seq_len = scores.shape[0]

        # WHY np.tril (lower triangular):
        #   tril creates a matrix where:
        #     mask[i][j] = 1  if j <= i   (past/present — allowed)
        #     mask[i][j] = 0  if j > i    (future — blocked)
        #   This enforces: token i can attend to tokens 0..i only.
        mask = np.tril(np.ones((seq_len, seq_len)))

        # WHY -1e9 (not -inf):
        #   True -inf causes NaN in exp(-inf) = 0 → 0/0 = NaN when ALL are -inf.
        #   -1e9 is large enough that exp(-1e9) ≈ 0 (effectively zero probability)
        #   while remaining numerically safe.
        masked_scores = np.where(mask == 1, scores, -1e9)

        return masked_scores                  # [seq_len × seq_len]

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Numerically stable softmax along a specified axis.

        FORMULA:
          softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))

        WHY subtract max (numerical stability):
          exp(1000) = overflow (inf).
          exp(1000 - 1000) = exp(0) = 1   (safe).
          Subtracting max doesn't change the ratio (numerator and denominator
          both multiply by exp(-max), which cancels).

        WHY keepdims=True:
          Preserves axis dimensions so subtraction broadcasts correctly.
          Without it: shapes won't align in the division step.

        Args:
            x:    Input array.
            axis: Axis to compute softmax over (default: last axis).

        Returns:
            Probability distribution along axis (sums to 1).
        """

        # Stability: subtract row-wise max before exponentiating
        x_max = np.max(x, axis=axis, keepdims=True)  # [seq_len × 1]
        exp_x = np.exp(x - x_max)                    # [seq_len × seq_len]

        # Normalize: divide by sum along the same axis
        # WHY keepdims again: ensures shapes align for element-wise division
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def forward(
        self,
        x:     np.ndarray,
        causal: bool = True,
        return_weights: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Full single-head attention forward pass.

        Args:
            x:              Token embeddings [seq_len × d_model].
            causal:         If True, apply causal mask (for generation).
            return_weights: If True, also return attention weight matrix.

        Returns:
            output:          [seq_len × d_v]  — updated token representations.
            attn_weights:    [seq_len × seq_len] or None.
        """

        # Step 1 → Project to Q, K, V
        Q, K, V = self.project_qkv(x)              # each [seq_len × d_k or d_v]

        # Step 2 → Compute scores
        scores = self.compute_attention_scores(Q, K) # [seq_len × seq_len]

        # Step 3 → Causal mask (optional, for decoder models)
        if causal:
            scores = self.apply_causal_mask(scores)  # [seq_len × seq_len]

        # Step 4 → Softmax → attention weights
        attn_weights = self.softmax(scores)          # [seq_len × seq_len]

        # Step 5 → Weighted sum of values
        # WHY attn_weights @ V:
        #   attn_weights[i] is a probability distribution over all tokens.
        #   Multiplying by V gives a weighted blend: token i's output is a
        #   combination of all tokens' values, weighted by relevance.
        #   [seq_len × seq_len] @ [seq_len × d_v] = [seq_len × d_v]
        output = attn_weights @ V                    # [seq_len × d_v]

        return output, (attn_weights if return_weights else None)


# ─── Visualizations ───────────────────────────────────────────────────────────

def visualize_attention_matrix(tokens: list[str], weights: np.ndarray, title: str):
    """
    Print an ASCII heatmap of the attention weight matrix.

    Rows = source token (attending FROM)
    Cols = target token (attending TO)
    Darker cell = higher attention weight.

    WHY visualize:
      Attention weights reveal what the model "thinks is relevant."
      In RAG, visualizing attention helps debug why retrieved context
      is or isn't being used in the generated answer.
    """

    shades = [" ", "░", "▒", "▓", "█"]

    def weight_to_shade(w: float) -> str:
        # Map [0, 1] → shade index
        idx = min(int(w * len(shades)), len(shades) - 1)
        return shades[idx]

    n = len(tokens)
    col_w = 8

    print(f"\n  {title}")
    print(f"  (row = FROM token, col = TO token | darker = higher attention)")

    # Header
    header = " " * 12 + "".join(f"{t[:col_w-1]:<{col_w}}" for t in tokens)
    print(f"\n  {header}")

    for i, from_tok in enumerate(tokens):
        row = f"  {from_tok[:10]:<12}"
        for j in range(n):
            w = weights[i][j]
            shade = weight_to_shade(w)
            row += f" {shade}{w:.3f}"
        print(row)


def step_by_step_demo():
    """
    Walk through the COMPLETE attention forward pass with print at each step.
    Uses a small example to make every shape and value inspectable.
    """

    print("=" * 65)
    print("SINGLE-HEAD ATTENTION: Step-by-Step Forward Pass")
    print("=" * 65)

    # Small example: 5 tokens, d_model=8, d_k=4, d_v=4
    # WHY small dimensions:
    #   Real d_model=4096, d_k=128 → matrices are too large to print.
    #   d_model=8, d_k=4 shows IDENTICAL computation at human-readable scale.
    D_MODEL, D_K, D_V = 8, 4, 4
    tokens  = ["[SYS]", "The", "RAG", "retrieves", "docs"]
    SEQ_LEN = len(tokens)

    # Simulate token embeddings (in real model: from embedding lookup table)
    np.random.seed(42)
    x = np.random.randn(SEQ_LEN, D_MODEL) * 0.5  # [5 × 8]

    print(f"\n  Config: d_model={D_MODEL}, d_k={D_K}, d_v={D_V}")
    print(f"  Tokens: {tokens}")
    print(f"  Input x shape: {x.shape}")

    # Build attention module
    attn = SingleHeadAttention(D_MODEL, D_K, D_V)

    # ── Step 1: Project ──────────────────────────────────────────────────────
    Q, K, V = attn.project_qkv(x)
    print(f"\n  STEP 1 — Q,K,V Projections:")
    print(f"    Q shape: {Q.shape}  (x @ W_q:  [{SEQ_LEN}×{D_MODEL}] @ [{D_MODEL}×{D_K}])")
    print(f"    K shape: {K.shape}  (x @ W_k)")
    print(f"    V shape: {V.shape}  (x @ W_v)")
    print(f"    Q[0] ('{tokens[0]}'): {Q[0].round(3)}")

    # ── Step 2: Raw scores ───────────────────────────────────────────────────
    raw_scores = Q @ K.T
    print(f"\n  STEP 2 — Raw Scores (Q @ Kᵀ):")
    print(f"    Shape: {raw_scores.shape}")
    print(f"    scores[0] ('{tokens[0]}' attends to all): {raw_scores[0].round(3)}")
    print(f"    Raw score variance: {raw_scores.var():.3f}  (expected ≈ d_k={D_K})")

    # ── Step 3: Scaled scores ────────────────────────────────────────────────
    scaled = raw_scores / math.sqrt(D_K)
    print(f"\n  STEP 3 — Scaled Scores (÷ √{D_K} = ÷{math.sqrt(D_K):.2f}):")
    print(f"    Scaled variance: {scaled.var():.3f}  (expected ≈ 1.0)")
    print(f"    scores[0] after scaling: {scaled[0].round(3)}")

    # ── Step 4: Causal mask ──────────────────────────────────────────────────
    masked = attn.apply_causal_mask(scaled)
    print(f"\n  STEP 4 — Causal Mask Applied:")
    print(f"    Upper triangle → -1e9 (future tokens blocked)")
    print(f"    Row 2 ('{tokens[2]}') masked scores: {masked[2].round(1)}")
    print(f"      Positions 0,1,2 visible; positions 3,4 → -1e9")

    # ── Step 5: Softmax ──────────────────────────────────────────────────────
    weights = attn.softmax(masked)
    print(f"\n  STEP 5 — Softmax → Attention Weights:")
    print(f"    Each row sums to 1.0")
    for i, tok in enumerate(tokens):
        row = weights[i]
        print(f"    {tok:<12}: {' '.join(f'{w:.3f}' for w in row)}")

    # ── Step 6: Weighted output ──────────────────────────────────────────────
    output = weights @ V
    print(f"\n  STEP 6 — Output (weights @ V):")
    print(f"    Shape: {output.shape}")
    print(f"    output[0] ('{tokens[0]}'): {output[0].round(3)}")
    print(f"    Each output token is a weighted blend of all V vectors.")

    # Visualize
    visualize_attention_matrix(tokens, weights, "Causal Attention Weights (5-token example)")


def bidirectional_vs_causal_demo():
    """
    Compare the attention weight matrices for causal vs bidirectional attention.

    CAUSAL (decoder):
      Token[i] sees only past tokens. Upper triangle is zero.
      Matrix is lower-triangular.

    BIDIRECTIONAL (encoder):
      Token[i] sees all tokens. All cells are non-zero.
      Matrix is symmetric-ish.
    """

    D_MODEL, D_K, D_V = 8, 4, 4
    tokens = ["Paris", "is", "the", "capital", "of", "France"]
    SEQ_LEN = len(tokens)

    np.random.seed(10)
    x    = np.random.randn(SEQ_LEN, D_MODEL) * 0.5
    attn = SingleHeadAttention(D_MODEL, D_K, D_V)

    _, causal_w  = attn.forward(x, causal=True)
    _, bidir_w   = attn.forward(x, causal=False)

    print("\n" + "=" * 65)
    print("CAUSAL vs BIDIRECTIONAL ATTENTION PATTERNS")
    print("=" * 65)

    visualize_attention_matrix(tokens, causal_w,  "CAUSAL (Decoder — future tokens masked)")
    visualize_attention_matrix(tokens, bidir_w,   "BIDIRECTIONAL (Encoder — all tokens visible)")

    # Show the key difference
    print(f"\n  KEY DIFFERENCE:")
    print(f"  Causal:       'France' (last) only attends to ['Paris','is','the','capital','of','France']")
    print(f"  Bidirectional:'is' attends to ALL including ['the','capital','of','France'] (future)")
    print(f"\n  WHY it matters for RAG:")
    print(f"  Decoder (causal) models generate answers token-by-token.")
    print(f"  But they CAN read all retrieved context (it's in the 'past' of the prompt).")
    print(f"  Retrieved context placed BEFORE the question gets full causal attention.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    step_by_step_demo()
    bidirectional_vs_causal_demo()
