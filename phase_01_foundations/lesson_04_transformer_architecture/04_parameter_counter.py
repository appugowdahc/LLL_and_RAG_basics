"""
FILE: 04_parameter_counter.py
LESSON: Phase 1 - Lesson 4 - Transformer Architecture
TOPIC: Count every parameter in a transformer — know your model's cost

WHAT THIS FILE TEACHES:
  - Exact formula for parameter count at each component
  - How to verify published param counts (Llama-3-8B, GPT-2, etc.)
  - How params map to GPU memory requirement
  - Why knowing param count matters for RAG infrastructure decisions

WHY PARAMETER COUNT MATTERS FOR RAG:
  Self-hosting:
    1B params × 2 bytes (fp16) = 2GB GPU memory minimum
    Plus KV cache, activations, optimizer states
    Llama-3-8B  ≈  16 GB  → fits on 1× RTX 4090 (24GB)
    Llama-3-70B ≈ 140 GB  → needs 2× A100 80GB or 4× RTX 4090

  API pricing is proportional to model size:
    Haiku (small)  →  cheapest
    Sonnet (mid)   →  balanced
    Opus (large)   →  most expensive

  Understanding scale helps you:
    - Choose the right model tier for your RAG use case
    - Estimate infrastructure costs
    - Plan batching and concurrency
"""

import math
from dataclasses import dataclass, field


@dataclass
class TransformerConfig:
    """
    Complete configuration for a decoder-only transformer.

    WHY @dataclass:
      Auto-generates __repr__, __eq__, and allows clean initialization.
      Fields are clearly documented — self-describing config object.
    """
    name:            str
    d_model:         int      # Hidden dimension (embedding size)
    n_layers:        int      # Number of transformer blocks
    n_heads:         int      # Number of attention heads
    d_ff:            int      # FFN hidden size (usually 4 × d_model)
    vocab_size:      int      # Vocabulary size
    context_window:  int      # Maximum sequence length
    tie_embeddings:  bool = True   # Share embedding and LM head weights
    use_gqa:         bool = False  # Grouped Query Attention (Llama-3, Mistral)
    n_kv_heads:      int  = 0      # Number of KV heads (for GQA, < n_heads)
    use_rmsnorm:     bool = True   # RMSNorm (modern) vs LayerNorm (original)

    def __post_init__(self):
        # WHY __post_init__:
        #   Runs after dataclass __init__. Validates and computes derived fields.
        if self.use_gqa and self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads // 4  # Default: 1/4 of query heads
        elif not self.use_gqa:
            self.n_kv_heads = self.n_heads  # Standard MHA: KV heads = Q heads

    @property
    def d_k(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads


def count_parameters(cfg: TransformerConfig) -> dict:
    """
    Count every parameter in a transformer model with exact formulas.

    PARAMETER INVENTORY:
      1. Token Embedding Matrix:     vocab_size × d_model
      2. Per transformer layer:
         a. Attention Q projection:  d_model × d_model
         b. Attention K projection:  d_model × (n_kv_heads × d_k)
         c. Attention V projection:  d_model × (n_kv_heads × d_k)
         d. Attention O projection:  d_model × d_model
         e. FFN W1 (up-proj):       d_model × d_ff
         f. FFN W2 (down-proj):     d_ff    × d_model
         g. LayerNorm 1 params:     d_model × 2  (γ and β, or just γ for RMSNorm)
         h. LayerNorm 2 params:     d_model × 2
      3. Final LayerNorm:            d_model × 2
      4. LM Head:                    vocab_size × d_model  (may be tied with embedding)

    Args:
        cfg: TransformerConfig instance.

    Returns:
        dict with parameter counts broken down by component.
    """

    d   = cfg.d_model
    L   = cfg.n_layers
    dff = cfg.d_ff
    V   = cfg.vocab_size
    H   = cfg.n_heads
    Hkv = cfg.n_kv_heads  # Number of KV heads
    dk  = cfg.d_k          # Dimension per head

    # ── 1. Embedding Layer ────────────────────────────────────────────────────
    # WHY vocab_size × d_model:
    #   Each of the V tokens in the vocabulary has a d_model-dimensional vector.
    embedding_params = V * d

    # ── 2. Per-Layer Attention Parameters ─────────────────────────────────────
    # Q projection: d_model → d_model (all heads combined)
    # WHY d × d: Each of the d_model input features projected to d_model output.
    q_proj_params = d * d

    # K, V projections: for Grouped Query Attention (GQA), fewer KV heads
    # WHY GQA reduces params:
    #   Standard MHA: n_heads KV heads → K_params = d × d
    #   GQA (Llama-3): n_kv_heads < n_heads → K_params = d × (n_kv_heads × dk)
    #   Reduces memory for KV cache at inference time.
    kv_dim       = Hkv * dk  # Total KV projection dimension
    k_proj_params = d * kv_dim
    v_proj_params = d * kv_dim

    # Output projection: concatenated heads → d_model
    o_proj_params = d * d

    # Total attention params per layer
    attn_params_per_layer = q_proj_params + k_proj_params + v_proj_params + o_proj_params

    # ── 3. Per-Layer FFN Parameters ────────────────────────────────────────────
    # W1: d_model → d_ff (expand)
    # W2: d_ff → d_model (compress)
    # WHY no bias for modern LLMs:
    #   Many modern LLMs (Llama, Mistral) omit bias terms to reduce params
    #   and because RMSNorm + careful init makes biases unnecessary.
    w1_params = d * dff
    w2_params = dff * d
    ffn_params_per_layer = w1_params + w2_params

    # ── 4. LayerNorm Parameters Per Layer ─────────────────────────────────────
    # RMSNorm: only gamma (1 param per dimension)
    # LayerNorm: gamma + beta (2 params per dimension)
    norm_params_per = d if cfg.use_rmsnorm else 2 * d
    norm_params_per_layer = 2 * norm_params_per   # 2 norms per block

    # ── 5. Total Per-Layer Parameters ─────────────────────────────────────────
    per_layer_params = attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer

    # ── 6. All Layers ─────────────────────────────────────────────────────────
    all_layers_params = L * per_layer_params

    # ── 7. Final LayerNorm ────────────────────────────────────────────────────
    final_norm_params = norm_params_per

    # ── 8. LM Head ────────────────────────────────────────────────────────────
    # WHY tie_embeddings:
    #   The LM head maps d_model → vocab_size — same shape as the embedding matrix.
    #   Weight tying shares these weights: embedding and LM head use the same matrix.
    #   Saves V × d_model parameters and improves training (Inan et al. 2016).
    lm_head_params = 0 if cfg.tie_embeddings else V * d

    # ── Total ─────────────────────────────────────────────────────────────────
    total_params = (
        embedding_params
        + all_layers_params
        + final_norm_params
        + lm_head_params
    )

    return {
        "model":                  cfg.name,
        "embedding_params":       embedding_params,
        "per_layer_attn":         attn_params_per_layer,
        "per_layer_ffn":          ffn_params_per_layer,
        "per_layer_norm":         norm_params_per_layer,
        "per_layer_total":        per_layer_params,
        "all_layers":             all_layers_params,
        "final_norm":             final_norm_params,
        "lm_head":                lm_head_params,
        "total_params":           total_params,
        "total_params_B":         total_params / 1e9,
        # Memory estimates
        "fp32_bytes":             total_params * 4,     # 4 bytes per float32
        "fp16_bytes":             total_params * 2,     # 2 bytes per float16
        "int8_bytes":             total_params * 1,     # 1 byte per int8
        "fp16_gb":                total_params * 2 / (1024**3),
    }


def print_parameter_breakdown(cfg: TransformerConfig):
    """Print a detailed parameter breakdown table for a model config."""

    params = count_parameters(cfg)

    print(f"\n{'='*65}")
    print(f"PARAMETER COUNT: {params['model']}")
    print(f"Config: d={cfg.d_model}, L={cfg.n_layers}, H={cfg.n_heads}, "
          f"Hkv={cfg.n_kv_heads}, d_ff={cfg.d_ff}, V={cfg.vocab_size:,}")
    print(f"{'='*65}")

    components = [
        ("Token Embedding",     params["embedding_params"]),
        ("All Attention Layers",params["all_layers"] * params["per_layer_attn"] // params["all_layers"] * cfg.n_layers
          if params["all_layers"] > 0 else params["per_layer_attn"] * cfg.n_layers),
        ("All FFN Layers",      params["per_layer_ffn"] * cfg.n_layers),
        ("All LayerNorm",       params["per_layer_norm"] * cfg.n_layers + params["final_norm"]),
        ("LM Head (untied)",    params["lm_head"] if params["lm_head"] > 0 else 0),
    ]

    # Recompute correctly
    components = [
        ("Token Embedding",             params["embedding_params"]),
        (f"Attention ×{cfg.n_layers}",  params["per_layer_attn"] * cfg.n_layers),
        (f"FFN ×{cfg.n_layers}",        params["per_layer_ffn"]  * cfg.n_layers),
        (f"LayerNorm ×{cfg.n_layers}+1",params["per_layer_norm"] * cfg.n_layers + params["final_norm"]),
        ("LM Head (if untied)",         params["lm_head"]),
    ]

    for name, count in components:
        pct = count / params["total_params"] * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:<28}: {count:>14,}  ({pct:5.1f}%) {bar}")

    print(f"  {'─'*65}")
    print(f"  {'TOTAL':<28}: {params['total_params']:>14,}  (100.0%)")
    print(f"  {'Total (Billions)':<28}: {params['total_params_B']:>14.2f}B")

    print(f"\n  Per-layer breakdown:")
    print(f"    Q/K/V/O attention:  {params['per_layer_attn']:>12,}")
    print(f"    FFN (W1+W2):        {params['per_layer_ffn']:>12,}")
    print(f"    LayerNorms:         {params['per_layer_norm']:>12,}")
    print(f"    Total per layer:    {params['per_layer_total']:>12,}")

    print(f"\n  Memory requirements (weights only):")
    print(f"    fp32 (training):   {params['fp32_bytes'] / (1024**3):>6.1f} GB")
    print(f"    fp16 (inference):  {params['fp16_bytes'] / (1024**3):>6.1f} GB")
    print(f"    int8 (quantized):  {params['int8_bytes'] / (1024**3):>6.1f} GB")

    # GPU recommendation
    fp16_gb = params["fp16_bytes"] / (1024**3)
    overhead = fp16_gb * 1.3  # KV cache + activations ≈ 30% overhead

    print(f"\n  GPU recommendation (fp16 + 30% overhead = {overhead:.0f} GB total):")
    gpu_options = [
        ("RTX 4090",   24),
        ("A100 40GB",  40),
        ("A100 80GB",  80),
        ("H100 80GB",  80),
        ("H100 NVL",   94),
        ("A100×2",    160),
        ("H100×2",    160),
        ("H100×4",    320),
        ("H100×8",    640),
    ]
    recommended = [(name, mem) for name, mem in gpu_options if mem >= overhead]
    if recommended:
        name, mem = recommended[0]
        print(f"    Minimum: {name} ({mem}GB)")
    else:
        print(f"    Requires multi-node cluster ({overhead:.0f}GB needed)")


def verify_published_counts():
    """
    Verify our formula against published parameter counts.
    Shows how close our estimates are to the real numbers.
    """

    configs_to_verify = [
        TransformerConfig(
            name="GPT-2 Small",
            d_model=768, n_layers=12, n_heads=12,
            d_ff=3072, vocab_size=50257, context_window=1024,
            tie_embeddings=True, use_gqa=False, use_rmsnorm=False
        ),
        TransformerConfig(
            name="Llama-3-8B",
            d_model=4096, n_layers=32, n_heads=32,
            d_ff=14336, vocab_size=128256, context_window=131072,
            tie_embeddings=False, use_gqa=True, n_kv_heads=8, use_rmsnorm=True
        ),
        TransformerConfig(
            name="Llama-3-70B",
            d_model=8192, n_layers=80, n_heads=64,
            d_ff=28672, vocab_size=128256, context_window=131072,
            tie_embeddings=False, use_gqa=True, n_kv_heads=8, use_rmsnorm=True
        ),
    ]

    published_counts_B = {
        "GPT-2 Small": 0.117,
        "Llama-3-8B":  8.0,
        "Llama-3-70B": 70.0,
    }

    print(f"\n{'='*60}")
    print("FORMULA VERIFICATION vs PUBLISHED COUNTS")
    print(f"{'='*60}")
    print(f"\n  {'Model':<18} {'Our estimate':>14} {'Published':>12} {'Error':>8}")
    print(f"  {'─'*18} {'─'*14} {'─'*12} {'─'*8}")

    for cfg in configs_to_verify:
        params  = count_parameters(cfg)
        our_B   = params["total_params_B"]
        pub_B   = published_counts_B[cfg.name]
        err_pct = abs(our_B - pub_B) / pub_B * 100
        print(f"  {cfg.name:<18}   {our_B:>10.2f}B   {pub_B:>8.2f}B   {err_pct:>6.1f}%")

    print(f"\n  Note: Small errors come from architectural details not captured")
    print(f"  in this simplified model (bias terms, rotary embedding params, etc.)")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Analyze Llama-3-8B in detail
    llama_8b = TransformerConfig(
        name="Llama-3-8B",
        d_model=4096, n_layers=32, n_heads=32,
        d_ff=14336,   vocab_size=128256, context_window=131072,
        tie_embeddings=False, use_gqa=True, n_kv_heads=8, use_rmsnorm=True
    )
    print_parameter_breakdown(llama_8b)

    # Analyze Llama-3-70B
    llama_70b = TransformerConfig(
        name="Llama-3-70B",
        d_model=8192, n_layers=80, n_heads=64,
        d_ff=28672,   vocab_size=128256, context_window=131072,
        tie_embeddings=False, use_gqa=True, n_kv_heads=8, use_rmsnorm=True
    )
    print_parameter_breakdown(llama_70b)

    # Verify against published
    verify_published_counts()
