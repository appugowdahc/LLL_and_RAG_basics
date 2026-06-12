"""
FILE: 04_flash_attention_concept.py
LESSON: Phase 1 - Lesson 5 - Attention Mechanism
TOPIC: Flash Attention — memory-efficient attention for long-context RAG

WHAT THIS FILE TEACHES:
  - WHY standard attention is memory-bottlenecked (O(N²) problem)
  - HOW Flash Attention solves it with tiled SRAM computation
  - The online softmax algorithm (softmax without materializing the matrix)
  - Memory complexity: standard O(N²) → Flash O(N)
  - Why this is the enabling technology for 200K+ context RAG
  - Practical implications: what to look for when choosing infrastructure

NOTE:
  Flash Attention is a CUDA kernel — it's not implementable in pure Python
  without CUDA access. This file teaches the ALGORITHM and CONCEPT with
  a Python simulation of the tiling logic and memory analysis.

INSTALL:
  pip install numpy
"""

import math
import numpy as np
import time


# ─── Memory Analysis ──────────────────────────────────────────────────────────

def analyze_attention_memory(seq_len: int, d_model: int, n_heads: int, dtype_bytes: int = 2):
    """
    Compute peak GPU memory needed for standard vs Flash Attention.

    STANDARD ATTENTION peak memory (per layer):
      1. Score matrix S = QKᵀ:  [n_heads × seq_len × seq_len] × dtype_bytes
      2. Softmax(S):             same size
      3. Output = softmax(S) × V: [seq_len × d_model] × dtype_bytes

    FLASH ATTENTION peak memory (per layer):
      Only the final output: [seq_len × d_model] × dtype_bytes
      The N×N score matrix is NEVER materialized.
      Intermediate tiles fit in L2 cache (SRAM), not HBM.

    Args:
        seq_len:     Number of tokens in the sequence.
        d_model:     Model hidden dimension.
        n_heads:     Number of attention heads.
        dtype_bytes: Bytes per value (2=fp16, 4=fp32).

    Returns:
        dict with memory breakdowns for both methods.
    """

    d_k = d_model // n_heads

    # Standard attention: must store the full N×N attention matrix
    # WHY ×2: both the raw scores AND the softmax probabilities are stored
    std_score_matrix_bytes = 2 * n_heads * seq_len * seq_len * dtype_bytes
    std_qkv_bytes          = 3 * seq_len * d_model * dtype_bytes   # Q, K, V
    std_output_bytes       = seq_len * d_model * dtype_bytes
    std_total_bytes        = std_score_matrix_bytes + std_qkv_bytes + std_output_bytes

    # Flash attention: score matrix is NEVER materialized
    # Only stores Q, K, V, output, and a small per-token statistics (m, l)
    flash_qkv_bytes    = 3 * seq_len * d_model * dtype_bytes   # Q, K, V (same)
    flash_output_bytes = seq_len * d_model * dtype_bytes
    flash_stats_bytes  = 2 * seq_len * n_heads * dtype_bytes   # m and l per head per token
    flash_total_bytes  = flash_qkv_bytes + flash_output_bytes + flash_stats_bytes

    return {
        "seq_len":            seq_len,
        "d_model":            d_model,
        "n_heads":            n_heads,
        # Standard attention
        "std_score_matrix_gb": std_score_matrix_bytes / (1024**3),
        "std_total_gb":        std_total_bytes / (1024**3),
        # Flash attention
        "flash_total_gb":     flash_total_bytes / (1024**3),
        # Reduction
        "memory_reduction_x": std_total_bytes / max(flash_total_bytes, 1),
    }


def show_memory_scaling():
    """
    Show how attention memory scales with sequence length.
    Reveals why long-context RAG REQUIRES Flash Attention.
    """

    print("=" * 70)
    print("ATTENTION MEMORY: Standard O(N²) vs Flash Attention O(N)")
    print("Config: d_model=4096, n_heads=32, fp16 (Claude Sonnet estimate)")
    print("=" * 70)

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 200000]

    print(f"\n  {'Seq Len':>10}  {'Standard (GB)':>16}  {'Flash (GB)':>12}  "
          f"{'Reduction':>12}  {'Feasible?':>10}")
    print(f"  {'─'*10}  {'─'*16}  {'─'*12}  {'─'*12}  {'─'*10}")

    A100_HBM_GB = 80  # A100 80GB HBM capacity

    for seq in seq_lengths:
        info = analyze_attention_memory(seq, d_model=4096, n_heads=32)

        std_feasible   = "✓" if info["std_total_gb"] < A100_HBM_GB else "✗ OOM"
        flash_feasible = "✓" if info["flash_total_gb"] < A100_HBM_GB else "✗ OOM"

        feasibility = f"Std:{std_feasible:<6} Flash:{flash_feasible}"

        print(
            f"  {seq:>10,}"
            f"  {info['std_total_gb']:>14.2f} GB"
            f"  {info['flash_total_gb']:>10.2f} GB"
            f"  {info['memory_reduction_x']:>10.1f}×"
            f"  {feasibility}"
        )

    print(f"\n  RAG IMPLICATION:")
    print(f"  Claude Sonnet's 200K context window REQUIRES Flash Attention.")
    print(f"  Without it: standard attention at 200K = {analyze_attention_memory(200000, 4096, 32)['std_score_matrix_gb']:.0f} GB just for scores!")
    print(f"  With Flash Attention: fits in A100 80GB alongside model weights.")


# ─── Online Softmax (Core Algorithm Concept) ──────────────────────────────────

def online_softmax_demo():
    """
    Demonstrate the ONLINE SOFTMAX algorithm — the mathematical trick
    that makes Flash Attention possible.

    PROBLEM:
      Standard softmax needs to see ALL values to compute the denominator:
        softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
      You MUST compute ALL exp(x_j) before normalizing any x_i.

    ONLINE SOFTMAX SOLUTION:
      Process values in TILES. Maintain running statistics:
        m = running max  (for numerical stability)
        l = running denominator (sum of exp)
      When you see a new tile, UPDATE m and l without revisiting old tiles.
      Final result = same as standard softmax, computed in tiles.

    WHY THIS ENABLES FLASH ATTENTION:
      The tile (block) fits in fast SRAM (L2 cache, ~40MB).
      HBM (GPU memory, ~80GB) is only read/written once per tile.
      Total HBM reads: O(N) instead of O(N²) for the full matrix.
    """

    print("\n" + "=" * 65)
    print("ONLINE SOFTMAX: The Algorithm Behind Flash Attention")
    print("=" * 65)

    # Simulate attention scores for one row (one query attending to N keys)
    N = 16  # sequence length (scores for one query)
    np.random.seed(7)
    scores = np.random.randn(N) * 2  # raw attention scores

    # ── Standard softmax (all at once) ───────────────────────────────────────
    def standard_softmax(x):
        # Step 1: compute max for stability
        m = x.max()
        # Step 2: compute exp(x - max) for all at once
        exp_x = np.exp(x - m)
        # Step 3: normalize
        return exp_x / exp_x.sum()

    # ── Online softmax (tile by tile) ─────────────────────────────────────────
    def online_softmax(x, tile_size: int = 4):
        """
        Process scores in tiles of size tile_size.
        Maintain running max m and running sum l.
        Produce IDENTICAL result to standard softmax.
        """

        N = len(x)

        # Running statistics
        m_running = -np.inf  # running maximum (for numerical stability)
        l_running = 0.0      # running denominator (sum of exp)

        # First pass: compute running m and l (tile by tile)
        for start in range(0, N, tile_size):
            tile = x[start : start + tile_size]

            # Update running max
            m_tile  = tile.max()
            m_new   = max(m_running, m_tile)

            # WHY re-scale l_running:
            #   When max increases from m_old to m_new, previous exp() values
            #   were computed with a different base.
            #   Correction factor: exp(m_old - m_new) rescales them correctly.
            l_running = l_running * np.exp(m_running - m_new) + np.sum(np.exp(tile - m_new))
            m_running = m_new

        # Second pass: compute final probabilities using accumulated m and l
        result = np.exp(x - m_running) / l_running

        return result

    # Compare
    std = standard_softmax(scores)
    onl = online_softmax(scores, tile_size=4)

    print(f"\n  Input scores (first 8): {scores[:8].round(3)}")
    print(f"\n  Standard softmax (all at once):")
    print(f"  {std[:8].round(4)}...")
    print(f"  Sum = {std.sum():.6f}")

    print(f"\n  Online softmax (tile_size=4, tile-by-tile):")
    print(f"  {onl[:8].round(4)}...")
    print(f"  Sum = {onl.sum():.6f}")

    max_diff = np.abs(std - onl).max()
    print(f"\n  Max difference: {max_diff:.2e}  ({'✓ identical' if max_diff < 1e-10 else '✗ different'})")
    print(f"\n  KEY INSIGHT:")
    print(f"  Online softmax produces the EXACT same result but processes")
    print(f"  only {4} scores at a time — never loading all {N} at once.")
    print(f"  Scale this to seq_len=200,000 and tile_size=1024:")
    print(f"  200 tiles × 1024 scores × 4 bytes = 800KB  (fits in SRAM!)")
    print(f"  vs 200,000 × 200,000 × 4 bytes = 160GB  (impossible!)")


# ─── Flash Attention Tiling Simulation ───────────────────────────────────────

def simulate_flash_attention_tiling(seq_len: int = 16, tile_size: int = 4):
    """
    Simulate the TILING PATTERN of Flash Attention.
    Shows which blocks are loaded from HBM on each step.

    In real Flash Attention:
      - Q tiles stay in SRAM (inner loop)
      - K,V tiles are streamed from HBM (outer loop)
      - Score tiles computed, softmax applied incrementally, output accumulated
      - HBM writes: only the output matrix O (once per Q tile)
    """

    print(f"\n  Flash Attention Tiling (seq_len={seq_len}, tile_size={tile_size})")
    print(f"  Q = outer loop (stays in SRAM), K/V = inner loop (streamed from HBM)")
    print()

    q_tiles = math.ceil(seq_len / tile_size)
    kv_tiles = math.ceil(seq_len / tile_size)

    hbm_reads = 0

    for qi in range(q_tiles):
        q_start = qi * tile_size
        q_end   = min(q_start + tile_size, seq_len)

        print(f"  Load Q[{q_start}:{q_end}] into SRAM", end="")
        hbm_reads += 1

        for kvi in range(kv_tiles):
            kv_start = kvi * tile_size
            kv_end   = min(kv_start + tile_size, seq_len)

            # Causal: skip future K,V tiles
            if kv_start > q_end:
                continue

            print(f" → Process K/V[{kv_start}:{kv_end}]", end="")
            hbm_reads += 1

        print(f" → Write O[{q_start}:{q_end}] to HBM")
        hbm_reads += 1   # Write output

    std_hbm_ops = seq_len * seq_len  # Full N×N matrix read+write
    print(f"\n  Flash HBM operations: ~{hbm_reads}")
    print(f"  Standard HBM ops (N²):  {std_hbm_ops}")
    print(f"  HBM reduction: {std_hbm_ops / hbm_reads:.0f}×")
    print(f"  WHY faster: HBM bandwidth is the bottleneck on modern GPUs.")
    print(f"  A100: 2TB/s HBM vs 19TB/s SRAM. Reading O(N) not O(N²) from HBM = big win.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    show_memory_scaling()
    online_softmax_demo()
    simulate_flash_attention_tiling(seq_len=16, tile_size=4)
