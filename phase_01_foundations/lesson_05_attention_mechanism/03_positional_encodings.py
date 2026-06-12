"""
FILE: 03_positional_encodings.py
LESSON: Phase 1 - Lesson 5 - Attention Mechanism
TOPIC: Positional Encodings — Sinusoidal, RoPE, ALiBi

WHAT THIS FILE TEACHES:
  - WHY position encoding is needed (attention is permutation-invariant)
  - Sinusoidal PE (original transformer, deterministic)
  - Learned PE (GPT-2 style, trainable table)
  - RoPE — Rotary Position Embedding (Llama-3, Mistral, modern standard)
  - ALiBi — Attention with Linear Biases (MPT, BLOOM)
  - How each affects context window extrapolation (longer seqs than trained on)

WHY THIS MATTERS FOR RAG:
  RAG prompts are long: system prompt + multiple retrieved chunks + query.
  Positional encoding determines whether the model can RELIABLY handle
  those long sequences:
  - Sinusoidal/Learned PE: quality degrades past training max length
  - RoPE: designed for extrapolation — better at longer contexts
  - ALiBi: penalizes distant tokens — implicit chunking behavior

INSTALL:
  pip install numpy
"""

import math
import numpy as np


# ─── 1. Sinusoidal Positional Encoding ────────────────────────────────────────

def sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Classic sinusoidal positional encoding from "Attention Is All You Need".

    FORMULA:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

      Where:
        pos   = token position (0, 1, 2, ...)
        i     = dimension index (0 to d_model/2)
        10000 = base frequency (determines wavelength spread)

    INTUITION:
      Different dimensions encode position at different frequencies:
        dim 0,1   → high freq (changes every token, like a fast clock)
        dim d-2,d → low freq  (changes slowly, like a slow clock)
      Together they form a unique "fingerprint" for every position.
      Like a binary counter: fast bits flip often, slow bits rarely.

    PROPERTIES:
      - Deterministic (no learned params)
      - Position-encoding is ABSOLUTE (not relative)
      - Can generalize to longer sequences (not seen in training)
        because the pattern is a fixed mathematical function

    Args:
        seq_len: Number of positions to encode.
        d_model: Embedding dimension.

    Returns:
        PE matrix [seq_len × d_model].
    """

    # Build output matrix
    pe = np.zeros((seq_len, d_model))

    # Position vector [0, 1, 2, ..., seq_len-1]
    # WHY reshape to (seq_len, 1):
    #   Allows broadcasting across d_model dimensions below.
    positions = np.arange(seq_len).reshape(-1, 1)      # [seq_len × 1]

    # Dimension frequency denominators: 10000^(2i/d_model)
    # WHY np.arange(0, d_model, 2):
    #   We compute one sin+cos pair per pair of dimensions.
    #   Step 2 generates [0, 2, 4, ..., d_model-2] — the "2i" indices.
    i         = np.arange(0, d_model, 2)               # [d_model/2]
    div_term  = np.power(10000.0, i / d_model)         # [d_model/2]
    angles    = positions / div_term                    # [seq_len × d_model/2]

    # Fill even dimensions with sin, odd dimensions with cos
    pe[:, 0::2] = np.sin(angles)   # positions 0,2,4,...
    pe[:, 1::2] = np.cos(angles)   # positions 1,3,5,...

    return pe


# ─── 2. Rotary Position Embedding (RoPE) ──────────────────────────────────────

class RoPE:
    """
    Rotary Position Embedding (Su et al. 2021).
    Used in: Llama, Mistral, Qwen, Falcon, GPT-NeoX.

    CORE IDEA:
      Instead of ADDING a position vector to the embedding (additive PE),
      ROTATE the Q and K vectors by a position-dependent angle.

      For each pair of dimensions (2i, 2i+1) in Q or K:
        [q_2i,   q_2i+1 ] × [cos(m×θ_i)  -sin(m×θ_i)]  → rotated pair
                              [sin(m×θ_i)   cos(m×θ_i)]

      Where m = position, θ_i = 1 / 10000^(2i/d)

    KEY PROPERTY (why RoPE is superior):
      dot(RoPE(q, m), RoPE(k, n)) = f(q, k, m-n)
      The dot product depends ONLY on the RELATIVE position (m-n),
      not the absolute positions m and n.

      This means the model learns "how far apart are these tokens?"
      rather than "what absolute position are these tokens at?"
      → Better generalization to longer sequences.

    RAG BENEFIT:
      Long RAG prompts: system (0-200) + docs (200-5000) + query (5000-5100).
      RoPE encodes relative distances within each section correctly.
      With absolute PE, positions 4000-5000 are "out of distribution."
    """

    def __init__(self, d_model: int, max_seq_len: int = 131072, base: float = 10000.0):
        """
        Args:
            d_model:    Dimension of Q and K vectors.
            max_seq_len: Maximum sequence length to precompute.
            base:       Base frequency (10000 original, 500000 Llama-3).
        """

        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # Precompute frequency bands: θ_i = 1 / base^(2i/d)
        # WHY precompute:
        #   θ values are fixed (depend only on d_model and base).
        #   Precomputing avoids recomputing them on every forward pass.
        i       = np.arange(0, d_model, 2, dtype=np.float64)  # [d_model/2]
        thetas  = 1.0 / (base ** (i / d_model))               # [d_model/2]

        # Precompute cos and sin for all positions
        positions = np.arange(max_seq_len, dtype=np.float64).reshape(-1, 1)  # [seq × 1]
        angles    = positions * thetas                         # [seq × d/2]

        # WHY store cos and sin separately:
        #   The rotation formula needs both: [cos, -sin; sin, cos].
        self.cos_table = np.cos(angles)  # [max_seq_len × d_model/2]
        self.sin_table = np.sin(angles)  # [max_seq_len × d_model/2]

    def rotate(self, x: np.ndarray, position: int) -> np.ndarray:
        """
        Apply RoPE rotation to a single token's Q or K vector at a given position.

        ROTATION FORMULA (per dimension pair 2i, 2i+1):
          x_rotated[2i]   = x[2i]   × cos(pos×θ_i) - x[2i+1] × sin(pos×θ_i)
          x_rotated[2i+1] = x[2i+1] × cos(pos×θ_i) + x[2i]   × sin(pos×θ_i)

        This is a 2D rotation matrix applied to each dimension pair.

        Args:
            x:        Single token vector [d_model].
            position: Integer position index.

        Returns:
            Rotated vector [d_model].
        """

        cos = self.cos_table[position]   # [d_model/2]
        sin = self.sin_table[position]   # [d_model/2]

        # Split into pairs
        x_even = x[0::2]   # dimensions 0, 2, 4, ... (x_2i)
        x_odd  = x[1::2]   # dimensions 1, 3, 5, ... (x_2i+1)

        # Apply rotation to each pair
        # WHY this formula:
        #   A 2D rotation matrix multiplied by [x_even, x_odd] gives:
        #   [x_even × cos - x_odd × sin,
        #    x_even × sin + x_odd × cos]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd  = x_even * sin + x_odd * cos

        # Interleave back
        result = np.empty_like(x)
        result[0::2] = rotated_even
        result[1::2] = rotated_odd

        return result

    def apply_to_sequence(self, x: np.ndarray) -> np.ndarray:
        """
        Apply RoPE to all tokens in a sequence.

        Args:
            x: [seq_len × d_model]  (Q or K matrix)

        Returns:
            Rotated [seq_len × d_model]
        """

        seq_len, d = x.shape
        result = np.empty_like(x)
        for pos in range(seq_len):
            result[pos] = self.rotate(x[pos], pos)
        return result


def demonstrate_rope_relative_position():
    """
    Show that RoPE encodes RELATIVE position (dot product depends on m-n only).

    This is the key property that makes RoPE better for long-context RAG.
    """

    print("=" * 65)
    print("RoPE: RELATIVE POSITION PROPERTY")
    print("=" * 65)

    D = 8   # Small d_model for clarity
    rope = RoPE(d_model=D, max_seq_len=100)

    np.random.seed(5)
    q = np.random.randn(D) * 0.5   # A single query vector
    k = np.random.randn(D) * 0.5   # A single key vector

    # Test: dot(RoPE(q, m), RoPE(k, n)) should depend only on (m-n)
    # Pairs with same relative distance should give similar dot products
    test_pairs = [
        (0, 1),   # distance 1
        (5, 6),   # distance 1 (different absolute positions)
        (50, 51), # distance 1 (very different absolute positions)
        (0, 5),   # distance 5
        (10, 15), # distance 5 (different absolute positions)
        (0, 10),  # distance 10
    ]

    print(f"\n  Query q: {q[:4].round(3)}...  Key k: {k[:4].round(3)}...")
    print(f"\n  {'Position pair':<18} {'Distance':<12} {'dot(RoPE(q,m), RoPE(k,n))'}")
    print(f"  {'─'*18} {'─'*12} {'─'*28}")

    for m, n in test_pairs:
        q_rot = rope.rotate(q, m)
        k_rot = rope.rotate(k, n)
        dot   = np.dot(q_rot, k_rot)
        dist  = abs(m - n)
        print(f"  ({m:>3}, {n:>3}){'':<8}  dist={dist:<8}  {dot:>+.6f}")

    print(f"\n  OBSERVATION:")
    print(f"  Pairs (0,1), (5,6), (50,51) all have distance=1 → similar dot products.")
    print(f"  Pairs (0,5), (10,15) both have distance=5 → similar dot products.")
    print(f"  Absolute positions don't matter — only relative distance.")


# ─── 3. ALiBi (Attention with Linear Biases) ─────────────────────────────────

def alibi_bias(n_heads: int, seq_len: int) -> np.ndarray:
    """
    ALiBi: Attention with Linear Biases (Press et al. 2022).
    Used in: MPT, BLOOM.

    FORMULA:
      score(i, j) = dot(Q_i, K_j) / √d_k - slope_h × |i - j|

      Where slope_h = 2^(-8/H × h)  for head h in 0..H-1

    INTUITION:
      Instead of modifying Q and K (like RoPE), ALiBi SUBTRACTS a penalty
      proportional to the distance between tokens, from the attention score.
      Far-apart tokens are penalized more → model prefers nearby context.
      Each head has a different slope → different "locality" preferences.

    PROPERTIES:
      - No learned positional parameters
      - Strong extrapolation (tested on 3x longer seqs than trained)
      - Implicit locality bias (penalizes distant context)
      - Simpler than RoPE (just an additive bias)

    RAG IMPACT:
      The distance penalty means tokens in later retrieved chunks
      are "further" from the query and get penalized.
      For long RAG contexts, chunk ORDER matters with ALiBi.
      → Place most important chunks CLOSEST to the question.

    Returns:
        Bias matrix [n_heads × seq_len × seq_len]
    """

    # Compute head-specific slopes
    # WHY 2^(-8/H × h):
    #   Geometric sequence: head 0 has largest slope (most local)
    #   Head H-1 has smallest slope (most global view).
    slopes = np.array([
        2 ** (-8 / n_heads * (h + 1))
        for h in range(n_heads)
    ])

    # Build distance matrix
    # distance[i][j] = |i - j|
    positions = np.arange(seq_len)
    # WHY outer subtraction:
    #   creates matrix where [i][j] = i - j
    #   We take absolute value for symmetric distance.
    dist = np.abs(positions.reshape(-1, 1) - positions.reshape(1, -1))  # [seq_len × seq_len]

    # Per-head biases: slope × distance
    # WHY [:, None, None]:
    #   Broadcasts slopes [n_heads] over [seq_len × seq_len]
    biases = -slopes[:, None, None] * dist[None, :, :]  # [n_heads × seq_len × seq_len]

    return biases


def compare_all_encodings():
    """
    Side-by-side comparison of all positional encoding methods.
    """

    print("\n" + "=" * 65)
    print("POSITIONAL ENCODING COMPARISON")
    print("=" * 65)

    comparisons = [
        {
            "name":       "Sinusoidal (original transformer)",
            "type":       "Absolute, deterministic",
            "formula":    "PE(pos, 2i) = sin(pos / 10000^(2i/d))",
            "params":     "0 (no learned parameters)",
            "extrapolate":"Moderate — works beyond training length but degrades",
            "used_in":    "Original Transformer, T5 (relative variant)",
            "rag_note":   "Adequate for short contexts, avoid for 50K+ tokens",
        },
        {
            "name":       "Learned PE (GPT-2 style)",
            "type":       "Absolute, learned",
            "formula":    "PE = lookup table[position_id]",
            "params":     "max_seq_len × d_model",
            "extrapolate":"Poor — no extrapolation beyond max_seq_len seen in training",
            "used_in":    "GPT-1, GPT-2, early BERT variants",
            "rag_note":   "Don't use for RAG — hard context window cap",
        },
        {
            "name":       "RoPE (Rotary PE)",
            "type":       "Relative, deterministic",
            "formula":    "Rotate(Q, pos) × Rotate(K, pos) → f(m-n)",
            "params":     "0 (computed from position, not learned)",
            "extrapolate":"Good — encodes relative positions → length generalization",
            "used_in":    "Llama-3, Mistral, Qwen, Falcon, GPT-NeoX",
            "rag_note":   "Best choice for long-context RAG (128K-1M tokens)",
        },
        {
            "name":       "ALiBi (Linear Biases)",
            "type":       "Relative, deterministic bias",
            "formula":    "score(i,j) -= slope_h × |i - j|",
            "params":     "0 (slopes are fixed per head index)",
            "extrapolate":"Excellent — explicitly tested at 3x training length",
            "used_in":    "MPT, BLOOM, some Falcon variants",
            "rag_note":   "Good for RAG but penalizes distant chunks — order matters",
        },
    ]

    for enc in comparisons:
        print(f"\n  ┌─ {enc['name']}")
        print(f"  │  Type:        {enc['type']}")
        print(f"  │  Formula:     {enc['formula']}")
        print(f"  │  Parameters:  {enc['params']}")
        print(f"  │  Extrapolate: {enc['extrapolate']}")
        print(f"  │  Used in:     {enc['used_in']}")
        print(f"  └  RAG Note:    {enc['rag_note']}")

    print(f"\n  RECOMMENDATION FOR RAG:")
    print(f"  Use models with RoPE (Llama-3, Mistral) for long-context RAG.")
    print(f"  Avoid models with Learned PE for contexts > 2048 tokens.")
    print(f"  ALiBi models work well but be aware of chunk ordering impact.")


def visualize_pe_patterns():
    """
    Show how sinusoidal PE creates unique position fingerprints.
    Each position has a unique vector — the model uses this to distinguish positions.
    """

    print("\n" + "=" * 65)
    print("SINUSOIDAL PE: Position Fingerprints")
    print("=" * 65)

    D_MODEL = 16
    SEQ_LEN = 8

    pe = sinusoidal_encoding(SEQ_LEN, D_MODEL)

    print(f"\n  PE matrix shape: {pe.shape}  [{SEQ_LEN} positions × {D_MODEL} dimensions]")
    print(f"\n  First 4 dimensions per position (shows frequency pattern):")
    print(f"\n  {'Pos':<6}  " + "  ".join(f"dim{i:<3}" for i in range(4)))
    print(f"  {'─'*6}  " + "  ".join("─"*6 for _ in range(4)))

    for pos in range(SEQ_LEN):
        vals = "  ".join(f"{pe[pos, i]:>+.3f}" for i in range(4))
        print(f"  {pos:<6}  {vals}")

    # Show distance property: nearby positions → similar PE → higher dot product
    print(f"\n  Dot product between PE vectors (similar = nearby positions):")
    print(f"  {'─'*50}")
    print(f"  {'Pair':<12} {'Distance':<12} {'PE dot product'}")
    print(f"  {'─'*12} {'─'*12} {'─'*15}")

    for pos_a, pos_b in [(0,1), (0,2), (0,4), (0,7), (3,4), (3,7)]:
        dist = abs(pos_a - pos_b)
        dot  = np.dot(pe[pos_a], pe[pos_b])
        bar  = "█" * int(max(0, dot) * 5)
        print(f"  ({pos_a},{pos_b}){'':<7} dist={dist:<9} {dot:>+.4f}  {bar}")

    print(f"\n  OBSERVATION: Nearby positions (small distance) have higher PE similarity.")
    print(f"  This lets attention naturally prefer nearby context — a useful inductive bias.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demonstrate_rope_relative_position()
    compare_all_encodings()
    visualize_pe_patterns()
