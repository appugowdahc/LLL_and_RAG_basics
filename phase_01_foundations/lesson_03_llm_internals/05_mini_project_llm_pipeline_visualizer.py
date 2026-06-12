"""
FILE: 05_mini_project_llm_pipeline_visualizer.py
LESSON: Phase 1 - Lesson 3 - How LLMs Work Internally
TOPIC: Mini-Project — Full LLM Internal Pipeline Visualizer

WHAT THIS FILE TEACHES:
  - Ties together ALL concepts from Lesson 3 in one pipeline
  - Visualizes: tokenization → embedding → attention → generation → output
  - Shows the EXACT data shape at every stage of the pipeline
  - Builds intuition for what happens "inside the box" on each API call
  - Produces a human-readable report of the full internal pipeline

THIS IS THE FOUNDATION FOR UNDERSTANDING RAG:
  Every RAG API call goes through this same pipeline.
  Phase 2-6 will build the RETRIEVAL layer that feeds into this pipeline.
  Understanding the pipeline helps you debug retrieval quality issues.
"""

import os
import math
import time
import numpy as np
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

# WHY import tiktoken here:
#   We use tiktoken to show REAL tokenization of the prompt.
#   This makes the visualization accurate and educational.
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


# ─── Pipeline Stages ──────────────────────────────────────────────────────────

def stage_1_tokenize(text: str) -> dict:
    """
    STAGE 1: Tokenization
    Text string → List of integer token IDs
    """

    if HAS_TIKTOKEN:
        token_ids = TOKENIZER.encode(text)
        token_strings = [TOKENIZER.decode([tid]) for tid in token_ids]
    else:
        # Fallback: simple word tokenization for visualization
        # WHY fallback: keeps the demo runnable without tiktoken
        words = text.split()
        token_ids     = [hash(w) % 50000 for w in words]
        token_strings = words

    return {
        "stage":          "Tokenization",
        "input":          text[:80] + ("..." if len(text) > 80 else ""),
        "token_ids":      token_ids,
        "token_strings":  token_strings,
        "token_count":    len(token_ids),
        "vocab_size":     TOKENIZER.n_vocab if HAS_TIKTOKEN else 50000,
    }


def stage_2_embed(token_ids: list[int], d_model: int = 4096) -> dict:
    """
    STAGE 2: Token Embedding Lookup
    List of token IDs → [seq_len × d_model] tensor

    In real transformers, this is a lookup into the embedding matrix:
      embedding_matrix: [vocab_size × d_model]
      token_embedding[i] = embedding_matrix[token_id[i]]

    We simulate with deterministic random vectors (same concept, fake values).
    """

    seq_len = len(token_ids)

    # WHY list of arrays:
    #   Each token ID maps to a d_model-dimensional vector.
    #   The full embedding is [seq_len × d_model].
    embeddings = []
    for tid in token_ids:
        # Deterministic embedding: same token always gets same vector
        rng = np.random.default_rng(tid % (2**32))
        vec = rng.standard_normal(d_model)
        # WHY L2 normalize:
        #   Token embeddings in real LLMs are approximately unit-normalized.
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        embeddings.append(vec)

    embedding_matrix = np.array(embeddings)  # [seq_len × d_model]

    return {
        "stage":            "Token Embedding",
        "input_shape":      f"[{seq_len}] token IDs",
        "output_shape":     f"[{seq_len} × {d_model}]",
        "embedding_matrix": embedding_matrix,
        "d_model":          d_model,
        "seq_len":          seq_len,
        "total_floats":     seq_len * d_model,
        "memory_bytes":     seq_len * d_model * 4,  # float32 = 4 bytes
    }


def stage_3_positional_encoding(embedding_matrix: np.ndarray) -> dict:
    """
    STAGE 3: Positional Encoding
    Add position information to each token embedding.

    Uses the classic sinusoidal encoding from "Attention Is All You Need":
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    WHY sinusoidal:
      - Produces unique encoding for each position
      - Relative positions have predictable relationships
      - Generalizes to longer sequences than seen in training
      - Modern LLMs use RoPE (Rotary Position Embedding) instead,
        but sinusoidal is easiest to understand

    WHY ADD (not concatenate):
      Adding position to token embedding means the model learns to
      DISENTANGLE positional and semantic information during training.
      Concatenating would double d_model and double parameter count.
    """

    seq_len, d_model = embedding_matrix.shape

    # Build sinusoidal positional encoding matrix
    pe = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # WHY 10000^(2i/d_model):
            #   Creates different frequencies for different dimensions.
            #   Low dimensions use high frequency (changes rapidly with position).
            #   High dimensions use low frequency (changes slowly).
            #   This creates a unique "fingerprint" for each position.
            denom = 10000 ** (2 * i / d_model)
            pe[pos, i]     = math.sin(pos / denom)  # even dimensions: sin
            if i + 1 < d_model:
                pe[pos, i+1] = math.cos(pos / denom)  # odd dimensions: cos

    # Add positional encoding to token embeddings
    positioned = embedding_matrix + pe

    return {
        "stage":            "Positional Encoding",
        "method":           "Sinusoidal (classic) — modern LLMs use RoPE",
        "input_shape":      f"[{seq_len} × {d_model}]",
        "output_shape":     f"[{seq_len} × {d_model}] (same shape, position info added)",
        "output_matrix":    positioned,
        "pe_sample_pos0":   pe[0, :4].tolist(),   # First 4 dims of position 0
        "pe_sample_pos1":   pe[1, :4].tolist(),   # First 4 dims of position 1
    }


def stage_4_transformer_layers(matrix: np.ndarray, n_layers: int = 3) -> dict:
    """
    STAGE 4: Transformer Layers (Simplified)
    Shows data shape through N transformer layers.

    In reality: each layer has full multi-head attention + FFN.
    Here: we show the SHAPE transformations and add small random perturbations
    to simulate the actual information mixing that happens.

    Each layer:
      1. Multi-Head Self-Attention: [seq_len × d_model] → [seq_len × d_model]
      2. Add & LayerNorm
      3. Feed-Forward Network: [seq_len × d_model] → [seq_len × d_model]
      4. Add & LayerNorm

    KEY INSIGHT:
      Input shape = Output shape (for each layer AND for the whole stack).
      The transformer enriches each token's representation by mixing information
      from other tokens — but the shape stays constant throughout.
    """

    seq_len, d_model = matrix.shape
    current = matrix.copy()
    layer_info = []

    for layer_num in range(n_layers):
        # Simulate attention: slight mixing between tokens
        # WHY random mixing: represents information flowing between tokens
        noise = np.random.default_rng(layer_num).standard_normal(current.shape) * 0.01
        current = current + noise

        # Simulate LayerNorm: normalize each token's vector
        # WHY LayerNorm: stabilizes training by normalizing activations
        mean = current.mean(axis=-1, keepdims=True)
        std  = current.std(axis=-1,  keepdims=True) + 1e-8
        current = (current - mean) / std

        layer_info.append({
            "layer":           layer_num + 1,
            "shape":           current.shape,
            "mean_activation": float(current.mean()),
            "std_activation":  float(current.std()),
        })

    return {
        "stage":        "Transformer Layers",
        "n_layers":     n_layers,
        "note":         f"Real models: 32-96 layers. We show {n_layers} for speed.",
        "input_shape":  f"[{seq_len} × {d_model}]",
        "output_shape": f"[{seq_len} × {d_model}] (shape unchanged, meaning enriched)",
        "layer_info":   layer_info,
        "output":       current,
    }


def stage_5_lm_head(final_hidden: np.ndarray, vocab_size: int = 1000) -> dict:
    """
    STAGE 5: LM Head — Project to Vocabulary
    [seq_len × d_model] → [seq_len × vocab_size] (logits for next token prediction)

    The LAST token's logits determine what the NEXT token will be.
    The other tokens' logits are used during training (teacher forcing)
    but not during inference.

    WHY a linear projection:
      The LM head is a single linear layer: logits = W × hidden_state + b
      Where W is [d_model × vocab_size].
      This projects the rich d_model representation to a score for every possible next token.
    """

    seq_len, d_model = final_hidden.shape

    # Simulate the LM Head projection matrix (normally d_model × vocab_size)
    # WHY seed=0: reproducibility
    rng = np.random.default_rng(0)
    lm_head_weights = rng.standard_normal((d_model, vocab_size)) * 0.02

    # Project all positions (but only the LAST position's logits matter at inference)
    logits = final_hidden @ lm_head_weights  # [seq_len × vocab_size]

    # Get the LAST position's logits (this predicts the NEXT token)
    last_logits = logits[-1]  # [vocab_size]

    # Apply softmax to get probabilities
    exp_logits = np.exp(last_logits - last_logits.max())  # stable softmax
    probs = exp_logits / exp_logits.sum()

    # Top-5 most likely next tokens (by index — real model uses real vocab)
    top5_indices = np.argsort(probs)[::-1][:5]
    top5_probs   = probs[top5_indices]

    return {
        "stage":             "LM Head (Language Model Head)",
        "input_shape":       f"[{seq_len} × {d_model}]",
        "weight_shape":      f"[{d_model} × {vocab_size}]",
        "output_shape":      f"[{seq_len} × {vocab_size}] logits",
        "active_position":   "LAST token position (predicts next token)",
        "top5_token_ids":    top5_indices.tolist(),
        "top5_probs":        [round(float(p), 4) for p in top5_probs],
        "entropy":           float(-np.sum(probs * np.log(probs + 1e-10))),
        "note": "In production: vocab_size=50,277-100,277. We use 1,000 for speed."
    }


# ─── Full Pipeline Visualizer ─────────────────────────────────────────────────

def visualize_llm_pipeline(prompt: str):
    """
    Run the complete simulated LLM pipeline on a prompt and print
    a stage-by-stage report showing data shapes and transformations.

    This is what happens INSIDE the model on every API call.
    """

    print("\n" + "="*65)
    print("LLM INTERNAL PIPELINE VISUALIZER")
    print(f"Input: \"{prompt[:60]}{'...' if len(prompt)>60 else ''}\"")
    print("="*65)

    # STAGE 1: Tokenization
    print("\n▶ STAGE 1: TOKENIZATION")
    s1 = stage_1_tokenize(prompt)
    print(f"  Input:       string ({len(prompt)} characters)")
    print(f"  Tokenizer:   cl100k_base (vocab={s1['vocab_size']:,})")
    print(f"  Token count: {s1['token_count']}")
    print(f"  Token IDs:   {s1['token_ids'][:10]}{'...' if len(s1['token_ids'])>10 else ''}")
    print(f"  Token text:  {s1['token_strings'][:8]}{'...' if len(s1['token_strings'])>8 else ''}")
    print(f"  Output:      List[int] of length {s1['token_count']}")

    # STAGE 2: Embedding
    print("\n▶ STAGE 2: TOKEN EMBEDDING LOOKUP")
    # WHY d_model=64 here (not 4096):
    #   Using 4096 would make numpy operations slow in a demo.
    #   d_model=64 shows identical structure at 64× faster speed.
    D_MODEL = 64
    s2 = stage_2_embed(s1["token_ids"], d_model=D_MODEL)
    print(f"  Embedding matrix: [{s1['vocab_size']:,} × {D_MODEL}] (total parameters: {s1['vocab_size']*D_MODEL:,})")
    print(f"  Input:  [{s1['token_count']} token IDs]")
    print(f"  Output: {s2['output_shape']}  tensor")
    print(f"  Memory: {s2['memory_bytes']:,} bytes = {s2['memory_bytes']/1024:.1f} KB")
    print(f"  Sample (token 0, first 6 dims): {s2['embedding_matrix'][0, :6].round(3).tolist()}")

    # STAGE 3: Positional Encoding
    print("\n▶ STAGE 3: POSITIONAL ENCODING")
    s3 = stage_3_positional_encoding(s2["embedding_matrix"])
    print(f"  Method:        {s3['method']}")
    print(f"  Input:         {s3['input_shape']}")
    print(f"  Output:        {s3['output_shape']}")
    print(f"  PE(pos=0) sample: {[round(v,3) for v in s3['pe_sample_pos0']]}")
    print(f"  PE(pos=1) sample: {[round(v,3) for v in s3['pe_sample_pos1']]}")
    print(f"  WHY different: Each position gets a unique sinusoidal fingerprint.")

    # STAGE 4: Transformer Layers
    print("\n▶ STAGE 4: TRANSFORMER LAYERS")
    N_LAYERS = 3  # Show 3 layers (real: 32-96)
    s4 = stage_4_transformer_layers(s3["output_matrix"], n_layers=N_LAYERS)
    print(f"  Simulated layers: {N_LAYERS}  (production: 32-96)")
    print(f"  Input:  {s4['input_shape']}")
    print(f"  Output: {s4['output_shape']}")
    print(f"  Shape is unchanged — each layer ENRICHES, not transforms shape.")
    for layer in s4["layer_info"]:
        print(f"    Layer {layer['layer']}: shape={layer['shape']} | "
              f"mean={layer['mean_activation']:+.4f} | std={layer['std_activation']:.4f}")

    # STAGE 5: LM Head
    print("\n▶ STAGE 5: LM HEAD (Vocabulary Projection)")
    s5 = stage_5_lm_head(s4["output"], vocab_size=1000)
    print(f"  Weight matrix:    {s5['weight_shape']}")
    print(f"  Input:            {s5['input_shape']}")
    print(f"  Output:           {s5['output_shape']}")
    print(f"  Active position:  {s5['active_position']}")
    print(f"  Top-5 next token probabilities (by simulated token ID):")
    for tid, prob in zip(s5["top5_token_ids"], s5["top5_probs"]):
        bar = "█" * int(prob * 100)
        print(f"    token_id={tid:<5}: {prob:.4f} {bar}")
    print(f"  Distribution entropy: {s5['entropy']:.3f} nats")
    print(f"  {s5['note']}")

    # Summary
    print("\n" + "─"*65)
    print("PIPELINE SUMMARY")
    print("─"*65)
    print(f"  Input:  \"{prompt[:50]}\"")
    print(f"  Stages: text → token_ids → embeddings → positioned → enriched → logits → next_token")
    print(f"  Shape flow:")
    print(f"    string({len(prompt)} chars)")
    print(f"    → [{s1['token_count']}] token IDs")
    print(f"    → [{s1['token_count']} × {D_MODEL}] embeddings (d_model={D_MODEL})")
    print(f"    → [{s1['token_count']} × {D_MODEL}] + positional encoding")
    print(f"    → [{s1['token_count']} × {D_MODEL}] × {N_LAYERS} transformer layers")
    print(f"    → [1000] logits for last position → sample next token")
    print(f"\n  This pipeline runs ONCE per token generated.")
    print(f"  A 200-token response = 200 forward passes through all {N_LAYERS} layers.")
    print(f"  Real models: 32-96 layers. That's why long outputs are slow.")


def run_real_pipeline_with_api(prompt: str):
    """
    Run the REAL pipeline via the API and report metrics.
    Shows that the simulated pipeline above matches real API behavior.
    """

    print("\n" + "="*65)
    print("REAL PIPELINE VIA CLAUDE API")
    print("="*65)

    # Count tokens BEFORE calling (from Lesson 1)
    token_count = client.messages.count_tokens(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": prompt}]
    ).input_tokens

    print(f"\n  Input: \"{prompt[:60]}\"")
    print(f"  Input tokens: {token_count}")
    print(f"  Running inference...", end="", flush=True)

    start = time.perf_counter()
    first_token_time = None
    output_tokens = 0
    full_response = ""

    # WHY stream for this demo:
    #   Streaming lets us measure TTFT accurately.
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for delta in stream.text_stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            full_response += delta
            output_tokens += 1

    total_time = time.perf_counter() - start
    ttft = first_token_time - start if first_token_time else 0

    print(f"\n\n  Response: \"{full_response.strip()[:80]}\"")
    print(f"\n  Pipeline Metrics:")
    print(f"    Input tokens:        {token_count}")
    print(f"    Output tokens:       {output_tokens} (approx)")
    print(f"    TTFT:                {ttft*1000:.0f}ms")
    print(f"    Total latency:       {total_time*1000:.0f}ms")
    print(f"    Throughput:          {output_tokens/total_time:.0f} tokens/sec")
    print(f"\n  INSIGHT: Throughput = how fast the transformer forward pass runs on Anthropic's GPU cluster.")
    print(f"  Higher throughput = faster generation = better UX in RAG applications.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_prompt = "What is Retrieval-Augmented Generation and why is it important?"

    # Simulated pipeline (no API call needed — shows the math)
    visualize_llm_pipeline(test_prompt)

    # Real pipeline via API
    run_real_pipeline_with_api(test_prompt)
