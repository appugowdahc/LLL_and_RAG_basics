"""
FILE: 02_encoder_vs_decoder.py
LESSON: Phase 1 - Lesson 4 - Transformer Architecture
TOPIC: Encoder-only vs Decoder-only vs Encoder-Decoder — What to use when

WHAT THIS FILE TEACHES:
  - Structural difference between encoder-only and decoder-only
  - Why decoder-only (causal) models are used for RAG generation
  - Why encoder-only models (BERT) are used for embeddings
  - How to recognize which architecture a model uses from its behavior
  - Which architecture to choose for each RAG component

KEY INSIGHT FOR RAG:
  A RAG system uses BOTH architectures:

  ┌─────────────────────────────────────────────────────────┐
  │                    RAG PIPELINE                         │
  │                                                         │
  │  Document chunks                                        │
  │       │                                                 │
  │       ▼                                                 │
  │  [ENCODER-ONLY model]  ← BGE, E5, all-MiniLM            │
  │  Embeds chunks → vectors stored in vector DB            │
  │                                                         │
  │  User query                                             │
  │       │                                                 │
  │       ▼                                                 │
  │  [ENCODER-ONLY model]  ← same model                     │
  │  Embeds query → search vector                           │
  │       │                                                 │
  │       ▼ (retrieved chunks)                              │
  │  [DECODER-ONLY model]  ← Claude, GPT-4, Llama           │
  │  Generates answer from retrieved context                │
  └─────────────────────────────────────────────────────────┘

INSTALL:
  pip install anthropic python-dotenv
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# ─── Architecture Comparison ──────────────────────────────────────────────────

def print_architecture_comparison():
    """
    Side-by-side comparison of the three transformer variants.
    """

    print("=" * 70)
    print("THREE TRANSFORMER ARCHITECTURES")
    print("=" * 70)

    architectures = {
        "ENCODER-ONLY": {
            "examples":       ["BERT", "RoBERTa", "all-MiniLM", "BGE", "E5"],
            "attention_type": "Bidirectional (each token sees ALL other tokens)",
            "training_task":  "Masked Language Modeling (predict masked tokens)",
            "output":         "Rich contextual embedding for EVERY input token",
            "best_for": [
                "Generating text embeddings for semantic search",
                "Sentence similarity / semantic matching",
                "Named entity recognition (NER)",
                "Text classification",
            ],
            "cannot_do": [
                "Text generation (no autoregressive decoding)",
                "Q&A without a separate prediction head",
                "Multi-turn conversation",
            ],
            "rag_role":  "RETRIEVER — embed chunks and queries for vector search",
            "why_bidirectional": (
                "Knowing all context before AND after a token produces better embeddings."
                " 'I bank at the river bank.' — seeing 'river' helps encode both 'bank's correctly."
            ),
        },
        "DECODER-ONLY": {
            "examples":       ["GPT-4", "Claude", "Llama-3", "Mistral", "Gemini"],
            "attention_type": "Causal / Unidirectional (each token sees only past tokens)",
            "training_task":  "Next-token prediction (autoregressive language modeling)",
            "output":         "Next token probabilities (generates text token by token)",
            "best_for": [
                "Text generation (answers, summaries, essays)",
                "Q&A and conversation (chat)",
                "Reasoning over retrieved context (RAG generation)",
                "Code generation",
                "Instruction following",
            ],
            "cannot_do": [
                "Direct sentence embedding (needs pooling tricks — degrades quality)",
                "Bidirectional understanding of full document",
            ],
            "rag_role":  "GENERATOR — produce answers grounded in retrieved context",
            "why_causal": (
                "During training, we predict next tokens from left to right."
                " Causal masking enforces this and enables autoregressive generation at inference."
            ),
        },
        "ENCODER-DECODER": {
            "examples":       ["T5", "BART", "mT5", "Flan-T5"],
            "attention_type": "Encoder: bidirectional | Decoder: causal + cross-attention",
            "training_task":  "Span corruption (T5) / Denoising (BART)",
            "output":         "Generated sequence conditioned on full input encoding",
            "best_for": [
                "Translation (input → output in different language)",
                "Summarization (long doc → short summary)",
                "Structured extraction with known schema",
            ],
            "cannot_do": [
                "Long-form generation at scale (less common in modern systems)",
            ],
            "rag_role": "Niche use in RAG — occasionally used for structured extraction",
            "why_cross_attention": (
                "Decoder attends to encoder's output (the input encoding) via cross-attention."
                " Allows generated output to 'see' the full input at every step."
            ),
        },
    }

    for arch_name, info in architectures.items():
        print(f"\n{'─'*65}")
        print(f"  {arch_name}")
        print(f"{'─'*65}")
        print(f"  Examples:       {', '.join(info['examples'])}")
        print(f"  Attention:      {info['attention_type']}")
        print(f"  Training task:  {info['training_task']}")
        print(f"  Output:         {info['output']}")
        print(f"\n  Best for:")
        for use in info["best_for"]:
            print(f"    ✓ {use}")
        print(f"\n  Cannot do:")
        for no in info["cannot_do"]:
            print(f"    ✗ {no}")
        print(f"\n  RAG Role:  {info['rag_role']}")


# ─── Bidirectional vs Causal Attention Demo ───────────────────────────────────

def demonstrate_bidirectional_vs_causal():
    """
    Show the practical consequence of bidirectional vs causal attention.

    BIDIRECTIONAL (BERT-style):
      Processes the ENTIRE input at once, in both directions.
      Great for understanding — each token sees full context.
      Cannot generate: once you have a token, you can't add the next one
      without re-processing the whole sequence.

    CAUSAL (GPT/Claude-style):
      Each token only sees past tokens.
      Enables generation: append one token and run forward pass → next token.
      Slightly worse at "understanding" tasks vs bidirectional of same size.

    We demonstrate this with Claude by showing how context (before AND after)
    affects understanding.
    """

    print("\n" + "=" * 65)
    print("BIDIRECTIONAL vs CAUSAL: Practical Impact")
    print("=" * 65)

    # Task: word sense disambiguation
    # The word "bank" has different meanings depending on surrounding context.
    # Bidirectional models handle this better because they see ALL context.
    # Causal models at token "bank" haven't seen "river" yet if it comes after.

    test_sentences = [
        {
            "sentence":  "I deposit money at the bank every Friday.",
            "ambiguous": "bank",
            "context":   "financial institution — 'deposit money' appears BEFORE 'bank'",
            "direction": "Context is BEFORE the ambiguous word (both arch handle this)",
        },
        {
            "sentence":  "The bank was muddy after the heavy rain near the river.",
            "ambiguous": "bank",
            "context":   "river bank — 'river' appears AFTER 'bank'",
            "direction": "Key context is AFTER ambiguous word (bidirectional advantage)",
        },
    ]

    for case in test_sentences:
        print(f"\n  Sentence: \"{case['sentence']}\"")
        print(f"  Ambiguous word: '{case['ambiguous']}'")
        print(f"  Key context: {case['context']}")
        print(f"  Architecture note: {case['direction']}")

    # Demo with Claude (decoder-only): show it can resolve ambiguity
    # Note: Claude can handle this because it's large and powerful — but
    # an encoder model would get the token embedding right from the start.
    print(f"\n  --- Testing disambiguation with Claude (decoder-only) ---")
    print(f"  (Claude sees tokens left-to-right but has learned strong priors)")

    for case in test_sentences:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=30,
            temperature=0,
            messages=[{
                "role": "user",
                "content": (
                    f"In this sentence: \"{case['sentence']}\"\n"
                    f"What does the word '{case['ambiguous']}' mean? "
                    f"Reply with 2-3 words only."
                )
            }]
        )
        print(f"\n  Sentence: {case['sentence'][:50]}...")
        print(f"  Claude's interpretation of '{case['ambiguous']}': "
              f"{response.content[0].text.strip()}")


# ─── Why Decoder-Only Dominates Modern RAG ────────────────────────────────────

def explain_decoder_dominance():
    """
    Explain why modern RAG systems use decoder-only models for generation.

    Historical context:
      Early RAG papers (2020) used encoder-decoder (T5-style) generators.
      Modern RAG (2022+) shifted entirely to decoder-only (GPT/Claude-style).

    WHY decoder-only won:
      1. Scale efficiency: decoder-only at same params outperforms encoder-decoder
      2. In-context learning: decoder-only is better at few-shot prompting
      3. Instruction following: RLHF training works better on decoder-only
      4. API availability: all major LLM APIs are decoder-only
      5. Context length: decoder-only models have longer context windows
    """

    print("\n" + "=" * 65)
    print("WHY DECODER-ONLY DOMINATES MODERN RAG")
    print("=" * 65)

    reasons = [
        {
            "reason": "Scale efficiency",
            "detail": (
                "At equal parameter count, decoder-only outperforms encoder-decoder "
                "on generation tasks. Encoder-decoder 'wastes' parameters on the "
                "encoder that could be decoder layers."
            ),
        },
        {
            "reason": "In-context learning (ICL)",
            "detail": (
                "Decoder-only models are much better at few-shot prompting. "
                "The causal attention pattern is ideal for learning patterns "
                "from examples shown earlier in the context."
            ),
        },
        {
            "reason": "RLHF / instruction tuning",
            "detail": (
                "Reinforcement Learning from Human Feedback works naturally "
                "on decoder-only because the generation process is differentiable. "
                "This gives us helpful, harmless models like Claude."
            ),
        },
        {
            "reason": "Long context windows",
            "detail": (
                "Decoder-only models have pushed context to 200K-1M tokens. "
                "For RAG: longer context = more retrieved chunks = better answers."
            ),
        },
        {
            "reason": "KV cache efficiency",
            "detail": (
                "Decoder-only KV cache grows linearly with sequence length. "
                "The large retrieved context only gets encoded once per call, "
                "then cached for each output token — making RAG inference fast."
            ),
        },
    ]

    for i, reason in enumerate(reasons, 1):
        print(f"\n  {i}. {reason['reason']}")
        print(f"     {reason['detail']}")

    # Show with Claude how instruction following enables reliable RAG
    print(f"\n  --- Practical Demo: Instruction Following Quality ---")

    strict_rag_prompt = """RULES (follow EXACTLY):
1. Answer ONLY using the context below.
2. If the answer is not in context, say "NOT IN CONTEXT".
3. Include the source document name in parentheses.
4. Maximum 2 sentences.

Context:
[rag_overview.pdf] RAG stands for Retrieval-Augmented Generation. It was introduced
in a 2020 paper by Lewis et al. at Facebook AI Research. RAG combines retrieval
systems with language model generation to produce grounded answers.

[vector_db.pdf] Pinecone is a managed vector database. It supports filtered ANN search
and scales to billions of vectors. Pricing is based on storage and query volume.

Questions:"""

    questions = [
        "Who invented RAG?",
        "What is Pinecone's pricing model?",
        "What is the GDP of France?",  # Not in context — should refuse
    ]

    for q in questions:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=80,
            temperature=0,
            messages=[{
                "role": "user",
                "content": strict_rag_prompt + f"\n{q}"
            }]
        )
        print(f"\n  Q: {q}")
        print(f"  A: {response.content[0].text.strip()}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_architecture_comparison()
    demonstrate_bidirectional_vs_causal()
    explain_decoder_dominance()
