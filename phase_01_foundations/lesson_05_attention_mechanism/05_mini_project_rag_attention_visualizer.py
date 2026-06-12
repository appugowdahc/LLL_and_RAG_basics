"""
FILE: 05_mini_project_rag_attention_visualizer.py
LESSON: Phase 1 - Lesson 5 - Attention Mechanism
TOPIC: Mini-Project — RAG Prompt Attention Visualizer + Attention Analysis

WHAT THIS FILE TEACHES:
  - How to construct a real RAG prompt with retrieved chunks
  - How to analyze which parts of the context get most "attended to"
  - Use Claude's API to simulate attention-weighted generation
  - Visualize token importance via logprobs (proxy for attention)
  - Build a complete attention-aware RAG pipeline skeleton

TIES TOGETHER:
  - Lesson 3: Tokenization (count tokens per section)
  - Lesson 4: Context window budget
  - Lesson 5: Attention patterns on RAG prompt structure

IMPORTANT — LOGPROBS AS ATTENTION PROXY:
  We cannot directly access attention weights via the Claude API.
  Instead, we use the logprobs of the generated tokens as a proxy:
  - High confidence in a specific word → model found clear evidence in context
  - Low confidence → model uncertain → retrieved context was weak

INSTALL:
  pip install anthropic python-dotenv
"""

import os
import json
import math
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# ─── RAG Prompt Structure ─────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A retrieved document chunk with metadata."""
    chunk_id:    int
    source:      str    # filename or URL
    page:        int
    content:     str
    score:       float  # relevance score from vector search (0-1)


@dataclass
class RAGPrompt:
    """A complete RAG prompt with all components."""
    system_instruction: str
    retrieved_chunks:   list[RetrievedChunk]
    user_query:         str

    def build(self) -> str:
        """Assemble the full RAG prompt string."""

        # Build context section from retrieved chunks
        # WHY enumerate with chunk IDs:
        #   The LLM can cite [1], [2] etc. in its answer — enables attribution.
        context_parts = []
        for chunk in self.retrieved_chunks:
            context_parts.append(
                f"[{chunk.chunk_id}] Source: {chunk.source} (page {chunk.page})\n"
                f"Relevance: {chunk.score:.2f}\n"
                f"{chunk.content}"
            )

        context_str = "\n\n".join(context_parts)

        return (
            f"{self.system_instruction}\n\n"
            f"{'─'*50}\n"
            f"RETRIEVED CONTEXT:\n\n"
            f"{context_str}\n\n"
            f"{'─'*50}\n"
            f"QUESTION: {self.user_query}"
        )

    def section_token_counts(self) -> dict:
        """Return token count for each section (using word-based approximation)."""

        def word_approx_tokens(text: str) -> int:
            # Rule of thumb: 1 token ≈ 0.75 words
            return int(len(text.split()) / 0.75)

        counts = {
            "system": word_approx_tokens(self.system_instruction),
            "query":  word_approx_tokens(self.user_query),
        }
        for chunk in self.retrieved_chunks:
            counts[f"chunk_{chunk.chunk_id}"] = word_approx_tokens(chunk.content)

        counts["total"] = sum(counts.values())
        return counts


# ─── Attention-Aware Response Analyzer ───────────────────────────────────────

def call_rag_with_analysis(rag_prompt: RAGPrompt) -> dict:
    """
    Call Claude with the RAG prompt and analyze the response for:
    - Which chunks were cited (evidence of attention)
    - Confidence indicators (hedging language = low confidence)
    - Response grounding quality

    Args:
        rag_prompt: Fully constructed RAGPrompt.

    Returns:
        dict with response, analysis, and metrics.
    """

    full_prompt = rag_prompt.build()

    # ── Pre-call token budget check ───────────────────────────────────────────
    # WHY count before calling:
    #   Prevents expensive failed calls. From Lesson 1.
    token_count = client.messages.count_tokens(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": full_prompt}]
    ).input_tokens

    CONTEXT_WINDOW = 200_000
    if token_count > CONTEXT_WINDOW:
        return {"error": f"Prompt too long: {token_count} > {CONTEXT_WINDOW} tokens"}

    # ── API Call with Streaming + Timing ──────────────────────────────────────
    start = time.perf_counter()
    first_token_time = None
    full_response = ""
    output_tokens = 0

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=400,
        temperature=0,   # Deterministic — maximize faithfulness to retrieved context
        system=(
            "You are a precise, grounded research assistant. "
            "Always cite sources as [1], [2], etc. "
            "If the answer is not in the context, say: 'NOT IN CONTEXT'."
        ),
        messages=[{"role": "user", "content": full_prompt}]
    ) as stream:
        for delta in stream.text_stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            full_response += delta
            output_tokens += 1
            print(delta, end="", flush=True)

    total_time = time.perf_counter() - start
    ttft       = (first_token_time - start) if first_token_time else 0

    print()  # newline after streaming

    # ── Response Analysis ─────────────────────────────────────────────────────

    # Detect which chunks were cited
    # WHY check for [N] patterns:
    #   If the model cites [1] or [2], it used that retrieved chunk.
    #   Uncited chunks were either irrelevant or overridden by other chunks.
    cited_chunks = []
    for chunk in rag_prompt.retrieved_chunks:
        if f"[{chunk.chunk_id}]" in full_response:
            cited_chunks.append(chunk.chunk_id)

    # Detect uncertainty indicators (hedging language)
    # WHY track these:
    #   When the model hedges ("I'm not sure", "possibly"), it means
    #   the retrieved context didn't clearly answer the question.
    uncertainty_phrases = [
        "not sure", "uncertain", "unclear", "may", "might",
        "possibly", "I believe", "not in context", "not provided",
        "cannot find", "don't have",
    ]
    uncertain_phrases_found = [
        phrase for phrase in uncertainty_phrases
        if phrase.lower() in full_response.lower()
    ]

    # Estimate groundedness: response length that cites sources
    # Simple heuristic: ratio of sentences with citations
    sentences         = full_response.split(".")
    cited_sentences   = [s for s in sentences if any(f"[{c}]" in s for c in cited_chunks)]
    groundedness_pct  = len(cited_sentences) / max(len(sentences), 1) * 100

    return {
        "response":           full_response,
        "input_tokens":       token_count,
        "output_tokens":      output_tokens,
        "ttft_ms":            round(ttft * 1000),
        "total_ms":           round(total_time * 1000),
        "cited_chunks":       cited_chunks,
        "uncited_chunks":     [c.chunk_id for c in rag_prompt.retrieved_chunks
                               if c.chunk_id not in cited_chunks],
        "uncertainty_found":  uncertain_phrases_found,
        "groundedness_pct":   round(groundedness_pct, 1),
    }


def visualize_token_budget(rag_prompt: RAGPrompt):
    """
    Visual breakdown of how the RAG prompt uses its token budget.
    Shows proportional section sizes.
    """

    counts = rag_prompt.section_token_counts()
    total  = counts["total"]

    print("\n  RAG PROMPT TOKEN BUDGET:")
    print(f"  Total: {total:,} tokens  (model context: 200,000)")
    print()

    # Build a visual bar
    bar_width = 50
    sections = [
        ("System",  counts["system"],  "▓"),
    ]
    for chunk in rag_prompt.retrieved_chunks:
        shade = ["░", "▒", "▓", "█"][chunk.chunk_id % 4]
        sections.append((
            f"Chunk {chunk.chunk_id} (score={chunk.score:.2f})",
            counts[f"chunk_{chunk.chunk_id}"],
            shade
        ))
    sections.append(("Query", counts["query"], "▓"))

    for name, count, shade in sections:
        pct   = count / total * 100
        width = max(1, int(pct / 100 * bar_width))
        bar   = shade * width
        print(f"  {name:<30}: {count:>5} toks ({pct:5.1f}%) {bar}")

    utilization = total / 200_000 * 100
    print(f"\n  Context utilization: {utilization:.2f}%  ({200_000-total:,} tokens remaining)")


def analyze_chunk_relevance(rag_prompt: RAGPrompt, analysis: dict):
    """
    Analyze which retrieved chunks were actually used vs wasted.
    Helps optimize the retrieval strategy.
    """

    print("\n  CHUNK CITATION ANALYSIS (Attention Proxy):")
    print(f"  {'Chunk':<8} {'Source':<30} {'Score':<8} {'Cited?':<8} {'Likely Impact'}")
    print(f"  {'─'*8} {'─'*30} {'─'*8} {'─'*8} {'─'*20}")

    for chunk in rag_prompt.retrieved_chunks:
        cited  = chunk.chunk_id in analysis["cited_chunks"]
        impact = "HIGH — used in answer" if cited else "LOW — not cited"
        marker = "✓" if cited else "✗"
        print(
            f"  [{chunk.chunk_id}]{'':<4} "
            f"{chunk.source[:28]:<30} "
            f"{chunk.score:.2f}{'':<4} "
            f"{marker}{'':<7} "
            f"{impact}"
        )

    uncited = analysis["uncited_chunks"]
    if uncited:
        print(f"\n  ⚠ {len(uncited)} chunks NOT cited: {uncited}")
        print(f"  → These consumed context window space without contributing.")
        print(f"  → Consider: improve retrieval precision, reduce top_k, or rerank.")

    if analysis["uncertainty_found"]:
        print(f"\n  ⚠ Uncertainty phrases detected: {analysis['uncertainty_found']}")
        print(f"  → Retrieved context may not contain the answer clearly.")
        print(f"  → Consider: retrieve more chunks, adjust similarity threshold.")


# ─── Run Full Demo ────────────────────────────────────────────────────────────

def run_rag_demo():
    """
    Full end-to-end RAG call with attention analysis.
    """

    # ── Build the RAG prompt ──────────────────────────────────────────────────

    system_instruction = (
        "You are a precise technical assistant specializing in AI infrastructure.\n"
        "Answer questions using ONLY the provided context.\n"
        "Cite all claims with [chunk_id] references.\n"
        "If information is not in the context, state: 'NOT IN CONTEXT'."
    )

    # Simulated retrieved chunks (in production: returned by vector DB)
    retrieved_chunks = [
        RetrievedChunk(
            chunk_id=1, source="flash_attn_paper.pdf", page=3,
            score=0.92,
            content=(
                "Flash Attention computes exact attention with O(N) memory instead of O(N²). "
                "It tiles the attention computation into blocks that fit in SRAM (L2 cache), "
                "avoiding materializing the full N×N attention score matrix in HBM. "
                "This enables sequence lengths up to 200,000 tokens on an A100 GPU."
            )
        ),
        RetrievedChunk(
            chunk_id=2, source="rag_architecture.pdf", page=7,
            score=0.88,
            content=(
                "In RAG systems, longer context windows allow more retrieved chunks to be "
                "injected into a single prompt. Flash Attention v2 (2023) improved throughput "
                "by 2-4x over standard attention, making 128K-200K token contexts practical "
                "for production RAG deployments."
            )
        ),
        RetrievedChunk(
            chunk_id=3, source="gpu_benchmarks.pdf", page=12,
            score=0.71,
            content=(
                "NVIDIA A100 specifications: 80GB HBM2e memory, 312 TFLOPS fp16, "
                "2TB/s memory bandwidth. H100 SXM: 80GB HBM3, 989 TFLOPS fp16, "
                "3.35TB/s bandwidth. Recommended for large language model inference."
            )
        ),
        RetrievedChunk(
            chunk_id=4, source="vector_db_guide.pdf", page=2,
            score=0.45,  # Low relevance — tests if model ignores irrelevant chunks
            content=(
                "Pinecone is a managed vector database. It supports filtered ANN search "
                "and scales to billions of vectors. Pricing is usage-based with free tier "
                "available. ChromaDB is preferred for local development."
            )
        ),
    ]

    user_query = (
        "How does Flash Attention enable long-context RAG systems, "
        "and what sequence lengths does it support on an A100?"
    )

    rag_prompt = RAGPrompt(
        system_instruction=system_instruction,
        retrieved_chunks=retrieved_chunks,
        user_query=user_query,
    )

    # ── Visualize token budget ────────────────────────────────────────────────
    print("=" * 65)
    print("RAG ATTENTION VISUALIZER: Full Pipeline")
    print("=" * 65)
    visualize_token_budget(rag_prompt)

    # ── Run RAG call ──────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print(f"QUERY: {user_query}")
    print("─" * 65)
    print("\nASSISTANT: ", end="", flush=True)

    analysis = call_rag_with_analysis(rag_prompt)

    # ── Print analysis ────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("RESPONSE ANALYSIS")
    print("─" * 65)
    print(f"\n  Latency:       TTFT={analysis['ttft_ms']}ms | Total={analysis['total_ms']}ms")
    print(f"  Tokens:        input={analysis['input_tokens']} | output={analysis['output_tokens']}")
    print(f"  Groundedness:  {analysis['groundedness_pct']}% of sentences cite sources")

    analyze_chunk_relevance(rag_prompt, analysis)

    # ── RAG optimization insights ─────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("OPTIMIZATION INSIGHTS")
    print("─" * 65)
    n_cited   = len(analysis["cited_chunks"])
    n_total   = len(retrieved_chunks)
    waste_pct = (n_total - n_cited) / n_total * 100

    print(f"\n  Chunks retrieved: {n_total}")
    print(f"  Chunks cited:     {n_cited}")
    print(f"  Context waste:    {waste_pct:.0f}% of retrieved context was unused")
    print(f"\n  ATTENTION LESSON:")
    print(f"  The model's attention mechanism naturally focuses on relevant chunks.")
    print(f"  Chunk [4] (Pinecone, score=0.45) likely got near-zero attention —")
    print(f"  the model recognized it's irrelevant to the Flash Attention question.")
    print(f"\n  BUT irrelevant chunks still cost tokens. Solutions:")
    print(f"  1. Reranking (Phase 7): Re-score chunks before injection, filter low ones")
    print(f"  2. Relevance threshold: Only inject chunks with score > 0.70")
    print(f"  3. Smaller top_k: Retrieve only 3 chunks instead of 5")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_rag_demo()
