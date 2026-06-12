"""
FILE: 02_multi_head_attention.py
LESSON: Phase 1 - Lesson 5 - Attention Mechanism
TOPIC: Multi-Head Attention — parallel heads, concat, output projection

WHAT THIS FILE TEACHES:
  - Why single-head attention is limited
  - How multiple heads run in parallel
  - The concatenation and output projection step
  - Grouped Query Attention (GQA) — Llama-3, Mistral optimization
  - How different heads specialize for different relationships
  - Multi-Query Attention (MQA) — extreme GQA

CORE INSIGHT:
  Multi-head attention = run H independent attention modules in parallel,
  each with its own Q,K,V weights (but smaller d_k = d_model/H),
  then concatenate and project back to d_model.

  This gives the model H "perspectives" on each token's relationships —
  H different learned relationship types simultaneously.

INSTALL:
  pip install numpy
"""

import math
import numpy as np
from lesson_05_attention_mechanism.01_single_head_attention import (
    SingleHeadAttention,
    init_projection,
)


# ─── Multi-Head Attention ─────────────────────────────────────────────────────

class MultiHeadAttention:
    """
    Multi-Head Self-Attention as used in decoder-only transformers (GPT, Claude).

    FORMULA:
      MultiHead(Q,K,V) = Concat(head_1, ..., head_H) × W_O

      Where:
        head_h = Attention(x×Wq_h, x×Wk_h, x×Wv_h)
        Wq_h, Wk_h, Wv_h ∈ [d_model × d_k]   (d_k = d_model / H)
        W_O               ∈ [H×d_v × d_model]

    PARAMETER COUNT:
      Per head: 3 × d_model × d_k  (Q, K, V projections)
      All H heads: 3 × H × d_model × d_k = 3 × d_model²  (since H × d_k = d_model)
      Output projection W_O: d_model × d_model
      Total: 4 × d_model²  (same as 4 linear layers of size d_model × d_model)

    This is important: MHA total params = 4 × d_model², regardless of H.
    H affects parallelism and expressiveness, not total parameter count.
    """

    def __init__(self, d_model: int, n_heads: int):
        """
        Args:
            d_model: Hidden dimension (must be divisible by n_heads).
            n_heads: Number of attention heads.
        """

        # WHY assert divisibility:
        #   d_k = d_model / n_heads must be an integer.
        #   If not, head dimensions don't split evenly.
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.d_model = d_model
        self.n_heads = n_heads

        # WHY d_k = d_model // n_heads:
        #   Each head operates in a LOWER-dimensional space.
        #   All heads combined still cover the full d_model dimensions.
        #   This keeps total compute the same as single-head at d_model.
        self.d_k = d_model // n_heads   # per-head key/query dimension
        self.d_v = d_model // n_heads   # per-head value dimension

        # WHY H separate attention modules (not one big one):
        #   Each head has its OWN Q,K,V weights → learns different patterns.
        #   A single head with d_k=d_model would be "full attention" but
        #   wouldn't learn multiple independent relationship types.
        self.heads = [
            SingleHeadAttention(d_model, self.d_k, self.d_v)
            for _ in range(n_heads)
        ]

        # Output projection: [H × d_v × d_model] = [d_model × d_model]
        # WHY project after concat:
        #   Each head outputs d_v dimensions. After concat: H × d_v = d_model.
        #   W_O mixes information across heads — allows each head's output
        #   to influence all d_model dimensions of the final representation.
        self.W_O = init_projection(d_model, d_model, seed=99)  # [d_model × d_model]

    def forward(
        self,
        x: np.ndarray,
        causal: bool = True,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Multi-head attention forward pass.

        Args:
            x:      Token embeddings [seq_len × d_model].
            causal: If True, apply causal mask.

        Returns:
            output:       [seq_len × d_model]
            all_weights:  list of H attention weight matrices, each [seq_len × seq_len]
        """

        head_outputs = []
        all_weights  = []

        # Run all H heads in PARALLEL (here: sequential loop for clarity)
        # In real GPUs: all heads run simultaneously via batched matrix ops.
        for h, head in enumerate(self.heads):
            head_out, weights = head.forward(x, causal=causal, return_weights=True)
            head_outputs.append(head_out)  # each [seq_len × d_v]
            all_weights.append(weights)    # each [seq_len × seq_len]

        # Concatenate head outputs along the last (feature) axis
        # WHY np.concatenate(axis=-1):
        #   Each head gives [seq_len × d_v].
        #   After concat: [seq_len × (H × d_v)] = [seq_len × d_model].
        #   This merges H independent views of each token into one vector.
        concat = np.concatenate(head_outputs, axis=-1)  # [seq_len × d_model]

        # Output projection: mix information across heads
        # WHY multiply by W_O:
        #   Raw concatenation = each head's output in separate, independent channels.
        #   W_O allows information from head 1 to interact with head 2's output.
        #   This cross-head mixing is essential for the heads to cooperate.
        output = concat @ self.W_O  # [seq_len × d_model]

        return output, all_weights


# ─── Grouped Query Attention (GQA) ────────────────────────────────────────────

class GroupedQueryAttention:
    """
    Grouped Query Attention (GQA) — used in Llama-3, Mistral, Falcon.

    MOTIVATION:
      Standard MHA: n_heads K heads, n_heads V heads.
      KV cache per token: 2 × n_heads × d_k × n_layers bytes.
      For Llama-3-70B (n_heads=64, d_k=128): 2 × 64 × 128 × 80 = ~1.3 MB per token.
      At 128K context: 1.3 MB × 128,000 = 166 GB KV cache → doesn't fit!

    GQA SOLUTION:
      Use n_heads QUERY heads but only n_kv_heads KEY/VALUE heads.
      Multiple query heads SHARE the same K and V.
      n_kv_heads = n_heads / group_size  (e.g., 64 Q heads → 8 KV heads)

      KV cache reduction: 64× → 8× fewer KV heads = 8× smaller KV cache.
      Quality: minimal degradation because KV heads still capture full context.

    SPECTRUM:
      MHA:  n_kv_heads = n_heads       (full KV per head)
      GQA:  n_kv_heads = n_heads / G   (G query heads share 1 KV head)
      MQA:  n_kv_heads = 1             (ALL query heads share 1 KV head — extreme)

    Llama-3-8B:  n_heads=32, n_kv_heads=8  (4 Q heads per KV head)
    Llama-3-70B: n_heads=64, n_kv_heads=8  (8 Q heads per KV head)
    Mistral-7B:  n_heads=32, n_kv_heads=8
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        """
        Args:
            d_model:    Hidden dimension.
            n_heads:    Number of QUERY heads.
            n_kv_heads: Number of KEY/VALUE heads (must divide n_heads evenly).
        """

        assert n_heads % n_kv_heads == 0, (
            f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
        )

        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_heads // n_kv_heads  # query heads per KV head

        self.d_k = d_model // n_heads    # per query-head dimension
        self.d_v = d_model // n_heads    # per value-head dimension

        # WHY n_heads Q projections but only n_kv_heads K,V projections:
        #   Each query head learns to look for different things.
        #   But the keys and values that are searched are SHARED.
        #   Think of it as: many different search terms, one document corpus.
        self.W_q_all = [
            init_projection(d_model, self.d_k, seed=h)
            for h in range(n_heads)
        ]

        # WHY n_kv_heads projections (not n_heads):
        #   This is the core GQA saving. Fewer K,V projections → smaller KV cache.
        self.W_k_all = [
            init_projection(d_model, self.d_k, seed=n_heads + h)
            for h in range(n_kv_heads)
        ]
        self.W_v_all = [
            init_projection(d_model, self.d_v, seed=2 * n_heads + h)
            for h in range(n_kv_heads)
        ]

        self.W_O = init_projection(d_model, d_model, seed=999)

    def forward(self, x: np.ndarray, causal: bool = True) -> np.ndarray:
        """
        GQA forward pass.
        Each query head uses one K,V pair, shared within its group.

        Returns:
            output: [seq_len × d_model]
        """

        seq_len = x.shape[0]
        head_outputs = []

        for h in range(self.n_heads):
            # Which KV head does this query head belong to?
            # WHY h // group_size:
            #   Heads 0,1,2,3 share KV head 0.
            #   Heads 4,5,6,7 share KV head 1. Etc.
            kv_idx = h // self.group_size

            # Project this query head
            Q = x @ self.W_q_all[h]          # [seq_len × d_k]

            # Use the SHARED K and V for this group
            K = x @ self.W_k_all[kv_idx]     # [seq_len × d_k]
            V = x @ self.W_v_all[kv_idx]     # [seq_len × d_v]

            # Standard scaled dot-product attention
            scores = Q @ K.T / math.sqrt(self.d_k)  # [seq_len × seq_len]

            if causal:
                mask   = np.tril(np.ones((seq_len, seq_len)))
                scores = np.where(mask == 1, scores, -1e9)

            weights   = self._softmax(scores)
            head_out  = weights @ V            # [seq_len × d_v]
            head_outputs.append(head_out)

        concat = np.concatenate(head_outputs, axis=-1)   # [seq_len × d_model]
        return concat @ self.W_O

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ─── Comparison Demo ──────────────────────────────────────────────────────────

def compare_mha_vs_gqa():
    """
    Compare parameter counts and KV cache sizes for MHA vs GQA.
    Show why GQA is essential for models with long context windows.
    """

    print("=" * 65)
    print("MHA vs GQA: Parameter Count and KV Cache Comparison")
    print("=" * 65)

    configs = [
        # (name, d_model, n_layers, n_heads, n_kv_heads, max_seq_len)
        ("GPT-2 (MHA)",        768,  12, 12, 12,    1024),
        ("Llama-3-8B (GQA)",  4096,  32, 32,  8, 131072),
        ("Llama-3-70B (GQA)", 8192,  80, 64,  8, 131072),
        ("Mistral-7B (GQA)",  4096,  32, 32,  8,  32768),
    ]

    print(f"\n  {'Model':<22} {'n_heads':<10} {'n_kv':<8} {'KV params':<14} {'KV cache @max ctx (GB)'}")
    print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*14} {'─'*22}")

    for name, d, n_layers, n_heads, n_kv, max_seq in configs:
        d_k = d // n_heads

        # KV projection parameters (W_k + W_v per layer)
        kv_params = 2 * n_layers * d * (n_kv * d_k)

        # KV cache at max sequence length (fp16 = 2 bytes)
        # Formula: 2 (K+V) × n_layers × n_kv_heads × max_seq × d_k × 2 bytes
        d_head    = 128  # standard
        kv_cache  = 2 * n_layers * n_kv * max_seq * d_head * 2
        kv_gb     = kv_cache / (1024**3)

        print(
            f"  {name:<22}  {n_heads:<10}  {n_kv:<8}"
            f"  {kv_params:>12,}  {kv_gb:>10.1f} GB"
        )

    print(f"\n  KEY INSIGHT:")
    print(f"  Llama-3-70B at 128K context: KV cache would be 100+ GB with MHA (n_kv=64).")
    print(f"  GQA (n_kv=8) reduces it 8× → fits alongside model weights on multi-GPU systems.")


def visualize_head_specialization():
    """
    Run MHA and show how different heads produce different attention patterns.
    Each head SHOULD (with real trained weights) specialize differently.
    With random weights this is approximate — real specialization emerges from training.
    """

    print("\n" + "=" * 65)
    print("MULTI-HEAD ATTENTION: Different Heads, Different Patterns")
    print("=" * 65)

    D_MODEL = 16
    N_HEADS =  4
    tokens  = ["[SYS]", "RAG", "retrieves", "relevant", "docs", "for", "LLM"]
    SEQ_LEN = len(tokens)

    np.random.seed(7)
    x   = np.random.randn(SEQ_LEN, D_MODEL) * 0.5
    mha = MultiHeadAttention(D_MODEL, N_HEADS)
    _, all_weights = mha.forward(x, causal=True)

    # Show first 3 heads — each has a different pattern
    head_roles = [
        "Head 0 (syntactic — tends to focus on nearby tokens)",
        "Head 1 (semantic — tends to focus on related content tokens)",
        "Head 2 (positional — tends to focus on specific positions)",
    ]

    for h, (weights, role) in enumerate(zip(all_weights[:3], head_roles)):
        # Simplify display: show only the last token's attention
        last_attn = weights[-1]  # How the last token attends to all others
        print(f"\n  {role}")
        print(f"  '{tokens[-1]}' attends to:")
        for tok, w in zip(tokens, last_attn):
            bar = "█" * int(w * 30)
            print(f"    {tok:<12}: {w:.4f} {bar}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Demo 1: Full MHA forward pass
    print("=" * 65)
    print("MULTI-HEAD ATTENTION: Forward Pass")
    print("=" * 65)

    D_MODEL = 16
    N_HEADS =  4
    SEQ_LEN =  5
    tokens  = ["[CTX]", "Paris", "is", "the", "capital"]

    np.random.seed(0)
    x   = np.random.randn(SEQ_LEN, D_MODEL) * 0.5
    mha = MultiHeadAttention(D_MODEL, N_HEADS)

    output, all_weights = mha.forward(x, causal=True)

    print(f"\n  d_model={D_MODEL}, n_heads={N_HEADS}, d_k={D_MODEL//N_HEADS}")
    print(f"  Input  shape: {x.shape}")
    print(f"  Output shape: {output.shape}  ← same as input! Shape preserved.")
    print(f"  Number of attention weight matrices: {len(all_weights)}")
    print(f"  Each matrix shape: {all_weights[0].shape}")

    # Show total parameters
    q_params   = N_HEADS * D_MODEL * (D_MODEL // N_HEADS)  # all Q projections
    k_params   = N_HEADS * D_MODEL * (D_MODEL // N_HEADS)  # all K projections
    v_params   = N_HEADS * D_MODEL * (D_MODEL // N_HEADS)  # all V projections
    wo_params  = D_MODEL * D_MODEL                          # output projection
    total_attn = q_params + k_params + v_params + wo_params
    print(f"\n  Total attention params: {total_attn:,}  = 4 × d_model² = 4 × {D_MODEL}² = {4*D_MODEL**2}")

    # Demo 2: MHA vs GQA comparison
    compare_mha_vs_gqa()

    # Demo 3: Head specialization
    visualize_head_specialization()
