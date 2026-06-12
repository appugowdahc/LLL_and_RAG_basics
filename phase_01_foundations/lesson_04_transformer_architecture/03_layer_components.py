"""
FILE: 03_layer_components.py
LESSON: Phase 1 - Lesson 4 - Transformer Architecture
TOPIC: Layer components built from scratch — LayerNorm, Residual, FFN

WHAT THIS FILE TEACHES:
  - Implement LayerNorm (RMSNorm) from scratch with full math
  - Implement Residual Connection and show WHY gradients need it
  - Implement Feed-Forward Network (FFN) with GELU and SwiGLU activation
  - Understand Pre-LN vs Post-LN and why modern LLMs use Pre-LN
  - See how these components interact in a single transformer block

WHY BUILD FROM SCRATCH:
  Every RAG engineer eventually needs to customize a transformer:
    - Add a retrieval adapter layer
    - Build a custom embedding layer for domain-specific tokens
    - Modify attention for longer context windows
  Understanding the components lets you modify them correctly.

INSTALL:
  pip install numpy
"""

import math
import numpy as np


# ─── 1. Layer Normalization ────────────────────────────────────────────────────

class LayerNorm:
    """
    Standard Layer Normalization (from "Layer Normalization", Ba et al. 2016).

    FORMULA:
      y = γ × (x - μ) / sqrt(σ² + ε) + β

      Where:
        x  = input vector of shape [d_model]
        μ  = mean of x across d_model dimensions
        σ² = variance of x across d_model dimensions
        ε  = small constant for numerical stability (typically 1e-5)
        γ  = learned scale parameter (initialized to 1.0)
        β  = learned bias parameter  (initialized to 0.0)

    WHY normalize ACROSS features (not batch):
      Batch Normalization normalizes across the batch dimension.
      For sequences of variable length, batch stats are unstable.
      LayerNorm normalizes each token's vector independently —
      works perfectly for sequences of any length.

    WHY learnable γ and β:
      Pure normalization would lose all amplitude information.
      γ and β let the model re-scale and re-shift after normalization.
      The network can learn to "undo" normalization if needed.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps     = eps

        # WHY initialize γ=1 and β=0:
        #   At initialization: LayerNorm is identity (normalized output).
        #   During training, γ and β are learned to add back useful scale/shift.
        self.gamma = np.ones(d_model)   # scale: [d_model]
        self.beta  = np.zeros(d_model)  # shift: [d_model]

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Args:
            x: Input tensor [seq_len × d_model] or [d_model]

        Returns:
            (normalized output, debug_info dict)
        """

        # WHY axis=-1:
        #   For input [seq_len × d_model], we normalize EACH TOKEN (row) independently.
        #   axis=-1 means "across the last dimension" (d_model), not across tokens.
        #   Each token gets its own μ and σ — tokens are normalized independently.
        mu  = x.mean(axis=-1, keepdims=True)         # [seq_len × 1]
        var = x.var( axis=-1, keepdims=True)          # [seq_len × 1]

        # WHY + eps:
        #   If var=0 (all identical values), sqrt(0) = 0 → division by zero.
        #   Adding eps=1e-5 prevents this without meaningfully changing the result.
        x_norm = (x - mu) / np.sqrt(var + self.eps)  # [seq_len × d_model]

        # Apply learned scale (gamma) and shift (beta)
        # WHY multiply by gamma AFTER normalizing:
        #   Normalization makes the output have unit variance.
        #   gamma lets the model control the scale of each dimension independently.
        out = self.gamma * x_norm + self.beta

        return out, {
            "input_mean":  float(x.mean()),
            "input_std":   float(x.std()),
            "output_mean": float(out.mean()),
            "output_std":  float(out.std()),
        }


class RMSNorm:
    """
    Root Mean Square Layer Normalization (used in Llama, Mistral, modern LLMs).

    FORMULA:
      y = γ × x / RMS(x)

      Where:
        RMS(x) = sqrt((1/d) × Σ x_i²)

    DIFFERENCE FROM LAYERNORM:
      RMSNorm skips the mean subtraction (centering step).
      Empirically works as well as LayerNorm but is ~10% faster.
      No beta (bias) parameter — fewer parameters.

    WHY modern LLMs prefer RMSNorm:
      Faster to compute (no mean subtraction).
      Theoretically: only the scale (RMS) matters, not the offset.
      Llama-3, Mistral, Qwen all use RMSNorm.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps     = eps
        self.gamma   = np.ones(d_model)   # Only scale — no bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: [seq_len × d_model]
        Returns:
            Normalized tensor of same shape
        """

        # WHY mean of squares (not variance):
        #   RMS = sqrt(mean(x²)) — skips the mean subtraction of standard LN.
        #   This is the "R" in RMS (Root Mean Square).
        rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + self.eps)

        return self.gamma * (x / rms)


# ─── 2. Residual Connection ───────────────────────────────────────────────────

class ResidualConnection:
    """
    Residual (skip) connection as used in transformers.

    FORMULA:
      output = x + sublayer(norm(x))    [Pre-LN, modern]
      output = norm(x + sublayer(x))    [Post-LN, original paper]

    WHY residual connections exist:
      Without them, training deep networks (32+ layers) fails because:
        - Gradients vanish: ∂L/∂x gets multiplied by small numbers through layers
        - Or explode: gradients grow exponentially
        - Network can't learn identity (skipping a useless layer)

      With residuals:
        - Gradient flows directly through the skip: ∂(x + f(x))/∂x = 1 + ∂f/∂x
        - Even if ∂f/∂x → 0 (vanishing), gradient = 1 (flows through skip)
        - Network can learn f(x) = 0 to effectively "disable" a layer

    Pre-LN vs Post-LN:
      Post-LN (original): norm(x + sublayer(x))
        - Gradient of x through LayerNorm can be small → less stable
        - Requires careful learning rate warmup

      Pre-LN (modern — GPT-4, Claude, Llama):
        x + sublayer(norm(x))
        - x passes through residual UNNORMALIZED (gradient = 1)
        - More stable training, especially for very deep models
    """

    def __init__(self, d_model: int, use_pre_ln: bool = True):
        # WHY use_pre_ln=True default:
        #   All modern production LLMs use Pre-LN.
        #   Post-LN is only kept for historical reference.
        self.norm     = RMSNorm(d_model)
        self.use_pre_ln = use_pre_ln

    def forward_pre_ln(self, x: np.ndarray, sublayer_fn) -> np.ndarray:
        """
        Pre-LN: output = x + sublayer(norm(x))
        Modern architecture — more stable training.
        """
        normed = self.norm.forward(x)   # normalize first
        out    = sublayer_fn(normed)    # apply sublayer to normalized input
        return x + out                  # add unnormalized x (residual skip)

    def forward_post_ln(self, x: np.ndarray, sublayer_fn) -> np.ndarray:
        """
        Post-LN: output = norm(x + sublayer(x))
        Original paper architecture — less stable for deep models.
        """
        out = sublayer_fn(x)            # apply sublayer to raw input
        return self.norm.forward(x + out)  # add residual THEN normalize


# ─── 3. Feed-Forward Network ──────────────────────────────────────────────────

class FeedForwardNetwork:
    """
    Position-wise Feed-Forward Network as used in transformer blocks.

    ARCHITECTURE:
      FFN(x) = W₂ × activation(W₁ × x + b₁) + b₂

      Where:
        W₁: [d_model → d_ff]   (expansion: d_ff = 4 × d_model typically)
        W₂: [d_ff → d_model]   (compression back)
        activation: GELU or SwiGLU

    WHY "position-wise":
      The same FFN is applied INDEPENDENTLY to each token position.
      No information flows between positions in the FFN (only in attention).
      The FFN learns a transformation per token, not across tokens.

    WHY 4× expansion:
      Empirically found to be optimal by the original transformer paper.
      Recent research shows knowledge/facts are STORED in FFN weights.
      Larger FFN = more factual capacity = better retrieval of memorized facts.

    ACTIVATION FUNCTIONS:
      GELU (Gaussian Error Linear Unit):
        Used in BERT, GPT-2/3.
        Smooth, differentiable approximation of ReLU.
        GELU(x) ≈ x × 0.5 × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))

      SwiGLU (Swish-Gated Linear Unit):
        Used in Llama, PaLM, modern LLMs.
        Empirically better than GELU.
        SwiGLU(x, gate) = x × sigmoid(gate)  (gating mechanism)
        Uses 3 weight matrices instead of 2.
    """

    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu"):
        self.d_model    = d_model
        self.d_ff       = d_ff
        self.activation = activation

        # Initialize weight matrices (in real models, learned via backprop)
        # WHY scale by sqrt(2/(d_model + d_ff)):
        #   Xavier/Glorot initialization — keeps activation variance stable.
        #   Too large: exploding activations. Too small: vanishing activations.
        scale_1 = math.sqrt(2.0 / (d_model + d_ff))
        scale_2 = math.sqrt(2.0 / (d_ff   + d_model))

        rng          = np.random.default_rng(42)
        self.W1      = rng.standard_normal((d_model, d_ff))  * scale_1
        self.b1      = np.zeros(d_ff)
        self.W2      = rng.standard_normal((d_ff, d_model))  * scale_2
        self.b2      = np.zeros(d_model)

        # SwiGLU needs an extra gate matrix
        if activation == "swiglu":
            self.W_gate = rng.standard_normal((d_model, d_ff)) * scale_1
            self.b_gate = np.zeros(d_ff)

    def gelu(self, x: np.ndarray) -> np.ndarray:
        """
        GELU activation: Gaussian Error Linear Unit.

        GELU(x) = x × Φ(x)
        Where Φ(x) = CDF of standard normal distribution.

        Approximation (tanh form):
          GELU(x) ≈ 0.5x × (1 + tanh(√(2/π) × (x + 0.044715x³)))

        WHY GELU over ReLU:
          ReLU(x) = max(0, x) — hard cutoff at 0.
          GELU is smooth: small negative inputs get small (not zero) outputs.
          Smooth gradients → better training of deep networks.
        """
        # WHY 0.044715: empirical constant from the GELU paper approximation
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

    def swiglu(self, x: np.ndarray) -> np.ndarray:
        """
        SwiGLU: Swish-Gated Linear Unit (Shazeer, 2020).

        SwiGLU(x) = (W₁x) × sigmoid(W_gate × x)

        WHY gating:
          The sigmoid gate acts as a soft selector — it learns WHEN to let
          information through. The gated design gives more expressive power
          than a pure activation function.

        WHY 3 weight matrices (W1, W_gate, W2) instead of 2:
          The gate requires its own projection.
          To keep params equal to a standard FFN, d_ff is reduced by 2/3
          (d_ff = 2/3 × 4 × d_model ≈ 2.67 × d_model for Llama).
        """
        # Linear projection (content path)
        linear = x @ self.W1 + self.b1          # [seq_len × d_ff]
        # Gate path
        gate   = x @ self.W_gate + self.b_gate  # [seq_len × d_ff]
        # Swish activation on gate = x × sigmoid(x)
        swish_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
        # Element-wise multiplication (gating)
        return linear * swish_gate              # [seq_len × d_ff]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: [seq_len × d_model]
        Returns:
            [seq_len × d_model]  (same shape in, same shape out)
        """

        # Step 1: Expand (d_model → d_ff)
        # WHY @ (matrix multiply):
        #   Broadcasting over the seq_len dimension.
        #   Each token's d_model vector → d_ff vector independently.
        if self.activation == "swiglu":
            hidden = self.swiglu(x)             # [seq_len × d_ff]
        else:
            hidden = x @ self.W1 + self.b1      # [seq_len × d_ff]
            hidden = self.gelu(hidden)           # apply GELU activation

        # Step 2: Compress (d_ff → d_model)
        out = hidden @ self.W2 + self.b2        # [seq_len × d_model]

        return out


# ─── 4. Complete Transformer Block ────────────────────────────────────────────

def demonstrate_transformer_block():
    """
    Assemble a single transformer block from our components and run a forward pass.
    Shows how all three components interact.
    """

    print("=" * 65)
    print("COMPLETE TRANSFORMER BLOCK: Forward Pass")
    print("=" * 65)

    # Config for a mini GPT-2-like model
    D_MODEL = 64    # Small for fast demo (real: 768-8192)
    D_FF    = 256   # 4 × d_model
    SEQ_LEN = 8     # 8 tokens

    # Simulate input: 8 tokens after positional encoding
    np.random.seed(7)
    x = np.random.randn(SEQ_LEN, D_MODEL) * 0.5  # [8 × 64]

    print(f"\n  Config: d_model={D_MODEL}, d_ff={D_FF}, seq_len={SEQ_LEN}")
    print(f"\n  Input x: shape={x.shape}")
    print(f"  Input stats: mean={x.mean():.4f}, std={x.std():.4f}")

    # Initialize components
    ln1 = RMSNorm(D_MODEL)            # Pre-LN before attention
    ln2 = RMSNorm(D_MODEL)            # Pre-LN before FFN
    ffn = FeedForwardNetwork(D_MODEL, D_FF, activation="gelu")

    # ── Simulated Multi-Head Attention ────────────────────────────────────────
    # WHY we simulate with random output:
    #   Full attention is in Lesson 3 (03_attention_intuition.py).
    #   Here we focus on HOW attention output feeds into residual + FFN.
    #   The shape contract is what matters: [seq_len × d_model] in and out.
    def mock_attention(x_norm):
        """Mock attention: small random perturbation to simulate context mixing."""
        rng = np.random.default_rng(42)
        return x_norm + rng.standard_normal(x_norm.shape) * 0.1

    # ── Full Block Forward Pass (Pre-LN style) ────────────────────────────────

    # STEP A: Pre-LayerNorm → Attention → Residual
    print(f"\n  STEP A: Attention sub-layer (Pre-LN)")
    x_norm_1 = ln1.forward(x)           # normalize input
    attn_out  = mock_attention(x_norm_1) # attention (mocked)
    x         = x + attn_out            # RESIDUAL: x = x + attention(norm(x))

    print(f"  After norm:     mean={x_norm_1.mean():.4f}, std={x_norm_1.std():.4f}")
    print(f"  After residual: mean={x.mean():.4f}, std={x.std():.4f}")

    # STEP B: Pre-LayerNorm → FFN → Residual
    print(f"\n  STEP B: FFN sub-layer (Pre-LN)")
    x_norm_2 = ln2.forward(x)            # normalize before FFN
    ffn_out   = ffn.forward(x_norm_2)    # FFN: expand → activate → compress
    x         = x + ffn_out              # RESIDUAL: x = x + ffn(norm(x))

    print(f"  After FFN norm:     mean={x_norm_2.mean():.4f}, std={x_norm_2.std():.4f}")
    print(f"  After FFN residual: mean={x.mean():.4f}, std={x.std():.4f}")

    print(f"\n  Output x: shape={x.shape}")
    print(f"  ✓ Shape unchanged: [{SEQ_LEN} × {D_MODEL}]")
    print(f"  ✓ Each token's representation has been enriched by:")
    print(f"    1. Attention: mixing information from other tokens")
    print(f"    2. FFN: per-token non-linear transformation")


def compare_ln_vs_rms_norm():
    """
    Show the difference between LayerNorm and RMSNorm computationally.
    """

    print("\n" + "=" * 65)
    print("LAYERNORM vs RMSNORM COMPARISON")
    print("=" * 65)

    D_MODEL = 8
    x = np.array([[3.0, -1.0, 2.0, 0.5, -2.0, 1.5, 0.0, -0.5]])  # [1 × 8]

    print(f"\n  Input: {x[0].tolist()}")

    # LayerNorm
    ln = LayerNorm(D_MODEL)
    ln_out, ln_info = ln.forward(x)
    print(f"\n  LayerNorm:")
    print(f"    Input  stats: mean={ln_info['input_mean']:.4f}, std={ln_info['input_std']:.4f}")
    print(f"    Output stats: mean={ln_info['output_mean']:.4f}, std={ln_info['output_std']:.4f}")
    print(f"    Output: {ln_out[0].round(4).tolist()}")

    # RMSNorm
    rms = RMSNorm(D_MODEL)
    rms_out = rms.forward(x)
    print(f"\n  RMSNorm (no mean subtraction):")
    print(f"    Output: {rms_out[0].round(4).tolist()}")
    print(f"    RMS of input: {math.sqrt((x**2).mean()):.4f}")

    print(f"\n  Difference (LN - RMS): {(ln_out - rms_out)[0].round(4).tolist()}")
    print(f"  WHY different: LayerNorm centers (subtracts mean), RMSNorm only scales.")
    print(f"  Modern LLMs use RMSNorm — same quality, ~10% faster computation.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demonstrate_transformer_block()
    compare_ln_vs_rms_norm()
