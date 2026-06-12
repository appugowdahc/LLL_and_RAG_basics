"""
FILE: 01_architecture_overview.py
LESSON: Phase 1 - Lesson 4 - Transformer Architecture
TOPIC: Component inventory, data shape flow, real model configurations

WHAT THIS FILE TEACHES:
  - Every named component in a decoder-only transformer
  - Exact tensor shapes at each stage (with real numbers)
  - Real model configurations (GPT-2 → Llama-3-70B)
  - How increasing scale (d_model, layers, heads) grows parameters
  - How to read a model config and understand what it means for RAG

WHY THIS MATTERS FOR RAG:
  - d_model determines embedding quality (bigger = more expressive)
  - n_layers determines reasoning depth (more = better multi-hop)
  - context_window determines max retrieved context (Phase 5 chunking)
  - Parameters determine GPU memory needed for self-hosting open models
"""

import math


# ─── Real Model Configurations ────────────────────────────────────────────────

# WHY a list of dicts (not classes):
#   Model configs are data, not behavior. Dicts are readable, serializable,
#   and easy to iterate over for comparison tables.
#   In production: these come from model cards or HuggingFace config.json.
MODEL_CONFIGS = [
    {
        "name":           "GPT-2 Small",
        "d_model":        768,
        "n_layers":       12,
        "n_heads":        12,
        "d_ff":           3072,    # 4 × d_model
        "vocab_size":     50257,
        "context_window": 1024,
        "params_B":       0.117,   # billion
        "type":           "decoder-only",
        "use_case":       "Learning / historical reference",
    },
    {
        "name":           "GPT-2 XL",
        "d_model":        1600,
        "n_layers":       48,
        "n_heads":        25,
        "d_ff":           6400,
        "vocab_size":     50257,
        "context_window": 1024,
        "params_B":       1.5,
        "type":           "decoder-only",
        "use_case":       "Learning / local experiments",
    },
    {
        "name":           "Llama-3-8B",
        "d_model":        4096,
        "n_layers":       32,
        "n_heads":        32,
        "d_ff":           14336,   # ~3.5× d_model (SwiGLU uses different ratio)
        "vocab_size":     128256,
        "context_window": 131072,  # 128K tokens
        "params_B":       8.0,
        "type":           "decoder-only",
        "use_case":       "Local RAG, edge deployment, fine-tuning",
    },
    {
        "name":           "Llama-3-70B",
        "d_model":        8192,
        "n_layers":       80,
        "n_heads":        64,
        "d_ff":           28672,
        "vocab_size":     128256,
        "context_window": 131072,
        "params_B":       70.0,
        "type":           "decoder-only",
        "use_case":       "High-quality local RAG, enterprise self-hosting",
    },
    {
        "name":           "Claude Sonnet 4.6 (est.)",
        "d_model":        8192,    # Anthropic doesn't publish — estimated
        "n_layers":       60,      # estimated
        "n_heads":        64,      # estimated
        "d_ff":           32768,
        "vocab_size":     100000,  # estimated
        "context_window": 200000,
        "params_B":       70.0,    # estimated ~70B
        "type":           "decoder-only",
        "use_case":       "Production RAG — best cost/quality balance",
    },
    {
        "name":           "BERT-base",
        "d_model":        768,
        "n_layers":       12,
        "n_heads":        12,
        "d_ff":           3072,
        "vocab_size":     30522,
        "context_window": 512,
        "params_B":       0.11,
        "type":           "encoder-only",
        "use_case":       "Embedding generation, classification (NOT generation)",
    },
]


def print_model_comparison_table():
    """
    Print a side-by-side comparison of all model configurations.

    WHY compare:
      When choosing a model for RAG, these configs tell you:
        - context_window: how many retrieved chunks fit in one call
        - params_B: how much GPU memory is needed to self-host
        - d_model: quality of internal representations
    """

    print("=" * 90)
    print("TRANSFORMER MODEL CONFIGURATIONS")
    print("=" * 90)

    # Header
    headers = ["Model", "d_model", "Layers", "Heads", "d_ff", "Vocab", "Ctx(K)", "Params(B)"]
    widths  = [22,        8,         7,        6,       7,      7,       7,        10]

    header_row = "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(f"\n  {header_row}")
    print("  " + "─" * 80)

    for cfg in MODEL_CONFIGS:
        row = "  ".join([
            f"{cfg['name']:<22}",
            f"{cfg['d_model']:>8,}",
            f"{cfg['n_layers']:>7}",
            f"{cfg['n_heads']:>6}",
            f"{cfg['d_ff']:>7,}",
            f"{cfg['vocab_size']:>7,}",
            f"{cfg['context_window']//1000:>7}",
            f"{cfg['params_B']:>10.1f}",
        ])
        print(f"  {row}  [{cfg['type']}]")

    print(f"\n  Note: Claude configs are community estimates — Anthropic doesn't publish architecture details.")


def trace_tensor_shapes(config: dict):
    """
    Show exactly how tensor shapes change through a transformer forward pass.

    WHY trace shapes:
      Shape mismatches are the #1 cause of bugs when building custom
      transformer components. Knowing the expected shape at each stage
      lets you debug and verify implementations quickly.

    Args:
        config: A model config dict from MODEL_CONFIGS.
    """

    name    = config["name"]
    d       = config["d_model"]
    L       = config["n_layers"]
    H       = config["n_heads"]
    d_ff    = config["d_ff"]
    V       = config["vocab_size"]
    d_k     = d // H  # dimension per attention head

    # Example sequence: a RAG prompt with retrieved context
    # seq_len represents: system_prompt + retrieved_chunks + user_query
    seq_len = 512  # tokens in this example prompt

    print(f"\n{'='*65}")
    print(f"TENSOR SHAPE TRACE: {name}")
    print(f"Config: d_model={d}, layers={L}, heads={H}, d_ff={d_ff}")
    print(f"Input sequence length: {seq_len} tokens")
    print(f"{'='*65}")

    # Stage by stage
    stages = [
        ("Input token IDs",          f"[{seq_len}]",
         "List of integer token IDs (0 to vocab_size-1)"),

        ("Token Embedding Lookup",   f"[{seq_len} × {d}]",
         f"Each token ID → d_model={d} dimensional vector"),

        ("+ Positional Encoding",    f"[{seq_len} × {d}]",
         "Same shape — position info ADDED in-place"),

        ("── Transformer Block ×1 ──", "─"*20, ""),

        ("  Pre-LayerNorm",           f"[{seq_len} × {d}]",
         "Normalize before attention (stable training)"),

        ("  Q projection",            f"[{seq_len} × {d}]",
         f"Linear: d={d} → d={d}  (then split into {H} heads of {d_k} each)"),

        ("  K projection",            f"[{seq_len} × {d}]",
         f"Same as Q — used as 'keys' for attention scoring"),

        ("  V projection",            f"[{seq_len} × {d}]",
         f"Same — used as 'values' that get aggregated"),

        ("  Split to heads",          f"[{seq_len} × {H} × {d_k}]",
         f"Reshape d_model into {H} heads of d_k={d_k} each"),

        ("  Attention scores",        f"[{H} × {seq_len} × {seq_len}]",
         f"Q×K^T per head: [{seq_len}×{d_k}] × [{d_k}×{seq_len}] = [{seq_len}×{seq_len}]"),

        ("  After softmax",           f"[{H} × {seq_len} × {seq_len}]",
         "Attention weights (probabilities summing to 1 per row)"),

        ("  Attended values",         f"[{H} × {seq_len} × {d_k}]",
         f"weights × V: [{seq_len}×{seq_len}] × [{seq_len}×{d_k}]"),

        ("  Concatenate heads",       f"[{seq_len} × {d}]",
         f"Merge {H} heads × {d_k} back to {d}"),

        ("  Output projection",       f"[{seq_len} × {d}]",
         f"Linear: d={d} → d={d}  (mixes head outputs)"),

        ("  Residual add",            f"[{seq_len} × {d}]",
         "x = x + attention_output (skip connection)"),

        ("  Pre-LayerNorm (FFN)",     f"[{seq_len} × {d}]",
         "Normalize before FFN"),

        ("  FFN up-projection",       f"[{seq_len} × {d_ff}]",
         f"Linear: {d} → {d_ff}  (expand by {d_ff//d}×)"),

        ("  FFN activation",          f"[{seq_len} × {d_ff}]",
         "GELU or SwiGLU — introduces non-linearity"),

        ("  FFN down-projection",     f"[{seq_len} × {d}]",
         f"Linear: {d_ff} → {d}  (compress back)"),

        ("  Residual add",            f"[{seq_len} × {d}]",
         "x = x + ffn_output (skip connection)"),

        ("── × L layers total ──",    f"L={L}", ""),

        ("Final LayerNorm",           f"[{seq_len} × {d}]",
         "Normalize final hidden states"),

        ("LM Head projection",        f"[{seq_len} × {V}]",
         f"Linear: {d} → {V}  (one score per vocab token)"),

        ("Softmax (last position)",   f"[{V}]",
         "Probabilities for the NEXT token (only last position used)"),

        ("Sample next token",         "token_id (scalar)",
         "Sample from probability distribution → one integer"),
    ]

    for stage_name, shape, note in stages:
        if "──" in stage_name:
            # Separator line
            print(f"\n  {'─'*60}")
            print(f"  {stage_name}")
            print(f"  {'─'*60}")
        else:
            # WHY f-string alignment:
            #   Consistent column widths make the shape column easy to scan.
            print(f"  {stage_name:<35} {shape:<25} {note[:30] if note else ''}")

    # Memory for the attention score matrix (most memory-intensive operation)
    attn_bytes = H * seq_len * seq_len * 4  # float32 per element
    attn_mb    = attn_bytes / (1024**2)
    print(f"\n  Peak memory (attention scores): {attn_mb:.1f} MB")
    print(f"  WHY attention is memory-intensive: [{H}×{seq_len}×{seq_len}] matrix")
    print(f"  At seq_len=200,000 (Claude's max): {H * 200000 * 200000 * 4 / (1024**3):.0f} GB!")
    print(f"  → Flash Attention (used in production) reduces this to O(seq_len) memory.")


def calculate_flops_per_token(config: dict) -> dict:
    """
    Estimate FLOPs (floating-point operations) for generating ONE token.

    WHY FLOPs matter for RAG:
      FLOPs per token × tokens_per_second = GPU utilization.
      Larger models need more FLOPs → slower generation → higher latency.
      For RAG: latency is user-facing — optimizing FLOPs matters.

    A100 GPU: ~312 TFLOPS (fp16)
    H100 GPU: ~989 TFLOPS (fp16)
    """

    d    = config["d_model"]
    L    = config["n_layers"]
    d_ff = config["d_ff"]

    # Approximate FLOPs per transformer layer per token
    # WHY 2× in matrix multiplications:
    #   Each multiply-add = 2 FLOPs (one multiply + one add).
    attn_flops  = 2 * 4 * d * d   # Q,K,V,Output projections: 4 × [d×d] matmuls
    ffn_flops   = 2 * 2 * d * d_ff  # Two linear layers: 2 × [d×d_ff] matmuls
    flops_layer = attn_flops + ffn_flops

    total_flops = L * flops_layer

    # Estimate time on an A100 (312 TFLOPS fp16)
    a100_tflops     = 312e12   # 312 × 10^12 FLOPS/second
    h100_tflops     = 989e12
    time_a100_ms    = (total_flops / a100_tflops) * 1000
    time_h100_ms    = (total_flops / h100_tflops) * 1000

    return {
        "model":             config["name"],
        "flops_per_token":   total_flops,
        "flops_billions":    total_flops / 1e9,
        "time_a100_ms":      round(time_a100_ms, 4),
        "time_h100_ms":      round(time_h100_ms, 4),
        "tokens_per_sec_a100": round(1000 / time_a100_ms) if time_a100_ms > 0 else 0,
    }


def print_flops_comparison():
    """Compare FLOPs and theoretical throughput across models."""

    print(f"\n{'='*75}")
    print("FLOPs PER TOKEN (Theoretical Throughput)")
    print(f"{'='*75}")
    print(f"\n  {'Model':<25} {'GFLOPs/tok':<14} {'A100 ms/tok':<14} {'A100 tok/s':<12}")
    print(f"  {'─'*25} {'─'*14} {'─'*14} {'─'*12}")

    for cfg in MODEL_CONFIGS:
        if cfg["type"] == "encoder-only":
            continue  # Skip BERT (not used for generation)
        info = calculate_flops_per_token(cfg)
        print(
            f"  {info['model']:<25}"
            f"  {info['flops_billions']:<12.1f}"
            f"  {info['time_a100_ms']:<12.4f}"
            f"  {info['tokens_per_sec_a100']:<12,}"
        )

    print(f"\n  Note: Actual throughput is lower due to memory bandwidth limits,")
    print(f"  batching overhead, and KV cache I/O. These are theoretical ceilings.")
    print(f"\n  RAG IMPLICATION:")
    print(f"  Faster tok/s = lower latency per RAG response.")
    print(f"  Smaller models (Haiku/Llama-8B) → higher throughput → better UX.")
    print(f"  Use the smallest model that meets your quality bar.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Model comparison table
    print_model_comparison_table()

    # 2. Shape trace for Llama-3-8B (good learning example — real open-source config)
    llama_8b = next(c for c in MODEL_CONFIGS if "8B" in c["name"])
    trace_tensor_shapes(llama_8b)

    # 3. FLOPs comparison
    print_flops_comparison()
