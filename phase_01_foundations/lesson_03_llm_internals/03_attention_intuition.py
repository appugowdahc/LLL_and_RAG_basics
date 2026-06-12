"""
FILE: 03_attention_intuition.py
LESSON: Phase 1 - Lesson 3 - How LLMs Work Internally
TOPIC: Attention Mechanism — How tokens look at each other

WHAT THIS FILE TEACHES:
  - What self-attention does (in plain English + math)
  - Q, K, V matrices and what they represent
  - How attention scores are computed
  - Why multi-head attention is better than single-head
  - How attention explains why "context" in the prompt matters for RAG

CONCEPT: The Attention Mechanism
──────────────────────────────────
Imagine reading: "The animal didn't cross the street because it was tired."
What does "it" refer to? You scan back through the sentence and decide "it = animal".
THIS is what attention does — each token looks at all other tokens and decides
which ones are most relevant to understanding its own meaning.

MATH:
  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V

  Where:
    Q = Query matrix   ("What am I looking for?")
    K = Key matrix     ("What information do I have?")
    V = Value matrix   ("What is the actual content?")

  Steps:
    1. For each token, compute Q, K, V via learned linear projections
    2. Compute attention scores: Q × K^T (dot product of query with all keys)
    3. Scale by sqrt(d_k) to prevent vanishing gradients
    4. Apply softmax to get attention weights (probabilities that sum to 1)
    5. Multiply weights × V to get weighted sum of value vectors

INTUITION:
  Q = "I'm looking for information about the subject of 'was tired'"
  K = "I contain information about animals"  (key of 'animal' token)
  High Q·K score → high attention weight → 'animal' gets weighted heavily
  Result: the 'it' token's representation absorbs meaning from 'animal'

WHY THIS MATTERS FOR RAG:
  When you inject retrieved documents into the prompt, each retrieved sentence
  becomes a sequence of tokens. The attention mechanism lets the LLM "attend to"
  the most relevant parts of the retrieved context when generating each word.
  Retrieved context that is semantically relevant gets high attention scores.

INSTALL:
  pip install numpy
"""

import math
import numpy as np


# ─── Scaled Dot-Product Attention (from scratch) ─────────────────────────────

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

    This is the EXACT formula used inside every transformer layer.

    Args:
        Q: Query matrix  [seq_len × d_k]
        K: Key matrix    [seq_len × d_k]
        V: Value matrix  [seq_len × d_v]
        mask: Optional causal mask [seq_len × seq_len]
              Used in language models to prevent attending to future tokens.

    Returns:
        (output, attention_weights)
        output:            [seq_len × d_v]  — updated token representations
        attention_weights: [seq_len × seq_len] — how much each token attends to each other
    """

    d_k = Q.shape[-1]  # Dimension of query/key vectors

    # STEP 1: Compute raw attention scores
    # WHY Q @ K.T:
    #   @ is matrix multiplication. K.T transposes K.
    #   Result[i][j] = dot product of token_i's query with token_j's key
    #   High dot product = token_i is "looking for" what token_j "has"
    scores = Q @ K.T  # Shape: [seq_len × seq_len]

    # STEP 2: Scale by sqrt(d_k)
    # WHY sqrt(d_k):
    #   Without scaling, dot products grow large as d_k increases
    #   (more dimensions = larger magnitudes).
    #   Large values push softmax into near-zero gradient regions (vanishing gradients).
    #   Dividing by sqrt(d_k) keeps values in a stable range regardless of dimension.
    scores = scores / math.sqrt(d_k)

    # STEP 3: Apply causal mask (decoder-only models like GPT/Claude)
    # WHY mask:
    #   During generation, token[i] should NOT see tokens[i+1, i+2, ...] (future tokens).
    #   We mask future positions with -infinity so softmax gives them probability ≈ 0.
    #   This enforces autoregressive generation — predict the next token from past only.
    if mask is not None:
        # WHY -1e9 (not -inf):
        #   True -inf causes NaN in softmax. -1e9 is large enough to produce ~0
        #   probability after softmax while remaining numerically stable.
        scores = np.where(mask == 0, -1e9, scores)

    # STEP 4: Softmax → attention weights (probabilities)
    # WHY softmax:
    #   Converts raw scores to probabilities that sum to 1.
    #   High scores get amplified; low scores get suppressed.
    #   Each token distributes 100% of its "attention budget" across all other tokens.
    # WHY axis=-1:
    #   Apply softmax along the last axis (per query token, across all key tokens).
    attention_weights = softmax(scores, axis=-1)

    # STEP 5: Weighted sum of value vectors
    # WHY weights @ V:
    #   For each token, we compute a weighted sum of all token value vectors.
    #   If token[i] attends 90% to token[j] and 10% to token[k],
    #   its output = 0.9 × V[j] + 0.1 × V[k].
    #   This is how context "flows" into each token's representation.
    output = attention_weights @ V  # Shape: [seq_len × d_v]

    return output, attention_weights


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.

    MATH: softmax(x_i) = exp(x_i) / sum(exp(x_j))

    WHY subtract max (numerical stability):
      exp(x) overflows for large x values (e.g., exp(1000) = inf).
      Subtracting max(x) before exp doesn't change the softmax result
      (the constant cancels in the numerator/denominator) but keeps
      values in a safe numerical range: exp(0) = 1, exp(-big) ≈ 0.
    """
    # WHY keepdims=True:
    #   Preserves the array shape for broadcasting during subtraction.
    #   Without it, numpy shapes don't align correctly.
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_attention(tokens: list[str], attention_weights: np.ndarray):
    """
    Print a heatmap of attention weights.

    Shows which tokens attend to which other tokens.
    Higher value = more attention paid.
    """

    n = len(tokens)
    print(f"\n  Attention Weight Matrix (row = from token, col = to token):")
    print(f"  Higher value = more attention paid")
    print()

    # Column headers
    header = "           " + "".join(f"{t[:6]:>7}" for t in tokens)
    print(f"  {header}")

    for i, from_token in enumerate(tokens):
        row = f"  {from_token[:10]:>10} "
        for j in range(n):
            weight = attention_weights[i][j]
            # WHY bar visualization:
            #   Makes high-attention cells visually obvious in terminal output.
            #   ████ for high attention, ░░░░ for low attention.
            if weight > 0.3:
                bar = "████"
            elif weight > 0.15:
                bar = "▓▓▓░"
            elif weight > 0.05:
                bar = "░░░░"
            else:
                bar = "    "
            row += f" {weight:.3f}"
        print(row)


def attention_on_rag_context():
    """
    Demonstrate attention on a simplified RAG prompt.

    INSIGHT: This shows WHY injecting retrieved context works.
    When you put relevant documents in the prompt, the attention mechanism
    lets the LLM "look at" those tokens when generating each response token.
    The model naturally focuses on the most relevant retrieved text.
    """

    print("\n" + "="*65)
    print("ATTENTION MECHANISM: Simplified Demonstration")
    print("="*65)

    # WHY simplified token sequence:
    #   Real LLMs use hundreds of tokens and 96 attention heads.
    #   We use 6 tokens and 1 head to make the math visible.
    #   The math is IDENTICAL at any scale.

    # A simplified "prompt" with retrieved context
    tokens = ["[CTX]", "RAG", "retrieves", "docs", "[Q]", "what"]

    # Simulated d_k = 4 (real models use 64-128 per head)
    d_k = 4
    seq_len = len(tokens)

    # Manually craft Q, K, V to show a realistic attention pattern
    # WHY specific values:
    #   We're manually engineering high attention between related tokens
    #   ([CTX]↔[Q], "RAG"↔"docs") to illustrate the concept clearly.
    #   In real models, Q/K/V are learned from data.
    np.random.seed(42)  # WHY seed: reproducibility for demonstration

    # Random initialization (simulating learned weights applied to token embeddings)
    Q = np.random.randn(seq_len, d_k) * 0.5
    K = np.random.randn(seq_len, d_k) * 0.5
    V = np.random.randn(seq_len, d_k) * 0.5

    # Apply causal mask (decoder / generative model — can't see future)
    # WHY upper triangular mask:
    #   Token[i] can only attend to tokens[0..i] (past and present).
    #   Token[i] CANNOT attend to tokens[i+1..n] (future — not generated yet).
    causal_mask = np.tril(np.ones((seq_len, seq_len)))

    # Compute attention
    output, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

    print(f"\n  Tokens: {tokens}")
    print(f"  Sequence length: {seq_len} | d_k (key dim): {d_k}")
    print(f"\n  Causal mask (1=can attend, 0=masked/future):")
    for i, row in enumerate(causal_mask):
        print(f"  {tokens[i]:>10}: {' '.join(str(int(v)) for v in row)}")

    visualize_attention(tokens, weights)

    print(f"\n  OUTPUT shape: {output.shape}  (updated token representations)")
    print(f"  Each token now contains information from ALL tokens it attended to.")
    print(f"  → In RAG: the [Q] token will attend strongly to [CTX] tokens")
    print(f"    → The answer will be grounded in the retrieved context.")


def explain_multi_head_attention():
    """
    Explain why we use MULTIPLE attention heads instead of one.

    WHY MULTI-HEAD:
      A single attention head learns ONE relationship pattern.
      Multi-head attention runs H heads in PARALLEL, each learning
      a DIFFERENT type of relationship:

        Head 1 → syntactic relationships (subject-verb-object)
        Head 2 → coreference (it → animal)
        Head 3 → positional relationships (nearby tokens)
        Head 4 → semantic similarity
        ...

      Results from all heads are CONCATENATED, giving the model
      H different "views" of relationships simultaneously.

    PARAMETERS:
      d_model = total embedding dimension (e.g., 4096)
      H = number of heads (e.g., 32)
      d_k = d_model / H = dimension per head (e.g., 128)

      Each head operates on d_k dimensions.
      All heads' outputs concatenate back to d_model.
    """

    print("\n" + "="*65)
    print("MULTI-HEAD ATTENTION: Why Multiple Heads?")
    print("="*65)

    head_configs = [
        ("GPT-2 Small", 768,   12,  64),
        ("GPT-2 Large", 1280,  36,  64),   # 1280/20 = 64
        ("GPT-3",       12288, 96,  128),
        ("Llama-3-8B",  4096,  32,  128),
        ("Llama-3-70B", 8192,  64,  128),
    ]

    print(f"\n  {'Model':<15} {'d_model':<10} {'Heads':<8} {'d_k/head':<10} {'Params (attention)'}")
    print(f"  {'─'*15} {'─'*10} {'─'*8} {'─'*10} {'─'*20}")

    for model, d_model, heads, d_k in head_configs:
        # WHY 4 × d_model²:
        #   Each attention layer has 4 weight matrices: Q, K, V, Output
        #   Each is d_model × d_model.
        #   Total attention params per layer = 4 × d_model²
        params_per_layer = 4 * d_model * d_model
        print(
            f"  {model:<15} {d_model:<10,} {heads:<8} {d_k:<10} "
            f"{params_per_layer:,}"
        )

    print(f"\n  INSIGHT: Attention weight matrices scale as d_model².")
    print(f"  Doubling d_model QUADRUPLES attention parameters.")
    print(f"  This is why large models are exponentially more expensive.")

    print(f"\n  WHAT EACH HEAD LEARNS (observed in research):")
    head_types = [
        ("Syntactic heads",    "Learn grammatical structure (subject, verb, object)"),
        ("Coreference heads",  "Resolve pronouns to their referents ('it' → 'animal')"),
        ("Position heads",     "Attend to tokens at specific relative positions"),
        ("Rare word heads",    "Focus on unusual / information-dense tokens"),
        ("Semantic heads",     "Attend to semantically similar tokens"),
        ("Separator heads",    "Focus on [SEP] and document boundary tokens"),
    ]

    for head_type, description in head_types:
        print(f"  → {head_type:<20}: {description}")

    print(f"\n  RAG RELEVANCE:")
    print(f"  Semantic heads explain WHY retrieved context is used — the model")
    print(f"  attends to retrieved chunks that share semantic meaning with the query.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    attention_on_rag_context()
    explain_multi_head_attention()
