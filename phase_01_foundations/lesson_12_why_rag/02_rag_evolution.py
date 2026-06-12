"""
FILE: 02_rag_evolution.py
LESSON: Phase 1 - Lesson 12 - Why RAG Was Invented
TOPIC: Three generations of RAG and how each improved on the last

WHAT THIS FILE TEACHES:
  - Naive RAG: the original pattern and its failure modes
  - Advanced RAG: what was added and why each addition matters
  - Agentic RAG: the current state of the art
  - Modular RAG: compositional design (retrievers, rerankers, generators as swappable modules)
  - Benchmarks: how each generation improves faithfulness and recall
  - WHY the progression happened: which production failures drove each upgrade

INSTALL: no external dependencies
"""

from dataclasses import dataclass, field
from typing import Optional


# ─── RAG Generations ──────────────────────────────────────────────────────────

@dataclass
class RAGGeneration:
    """One generation of the RAG architecture."""
    name:           str
    years:          str
    pipeline:       list[str]      # ordered steps
    improvements:   list[str]      # over previous generation
    failure_modes:  list[str]      # what still goes wrong
    example_stack:  str            # concrete tech stack
    faithfulness:   str            # approximate faithfulness improvement
    recall:         str            # approximate recall improvement


RAG_GENERATIONS = [

    RAGGeneration(
        name            = "Naive RAG",
        years           = "2020–2022",
        pipeline        = [
            "1. User query",
            "2. Dense retrieval (single vector search)",
            "3. Top-K chunks concatenated into context",
            "4. LLM generates answer",
            "5. Answer returned to user",
        ],
        improvements    = [
            "First working RAG — establishes the retrieve-then-generate pattern.",
            "Non-parametric knowledge access: retrieval provides docs not in training data.",
            "Can answer questions about private corpora.",
        ],
        failure_modes   = [
            "Low retrieval precision: top-K often contains irrelevant chunks.",
            "No query understanding: typos or ambiguous queries break retrieval.",
            "Fixed K: doesn't adapt to query complexity.",
            "No faithfulness check: hallucinations from context misreading go undetected.",
            "No metadata: can't filter by date, product, or version.",
            "Dense-only: misses exact keyword matches (bug IDs, model numbers).",
        ],
        example_stack   = "LangChain + OpenAI Embeddings + FAISS + GPT-3.5",
        faithfulness    = "~40–60% (no grounding enforcement)",
        recall          = "~50–70% (single modality, no reranking)",
    ),

    RAGGeneration(
        name            = "Advanced RAG",
        years           = "2022–2024",
        pipeline        = [
            "1. User query",
            "2. Query rewriting / expansion (HyDE, multi-query)",
            "3. Hybrid retrieval: BM25 + dense + metadata filter",
            "4. RRF fusion of multiple ranked lists",
            "5. Reranker (cross-encoder) re-scores top-N candidates",
            "6. Context compression (remove irrelevant sentences)",
            "7. LLM generates answer with citation requirement",
            "8. Post-generation faithfulness check",
            "9. Answer + citations returned",
        ],
        improvements    = [
            "Query rewriting: handles typos, ambiguity, and multi-intent queries.",
            "Hybrid search: BM25 catches exact terms; dense catches semantics.",
            "Metadata filtering: scope by date, product, version, source type.",
            "Reranking: precision@5 significantly better than retrieval rank alone.",
            "Context compression: reduces noise before LLM call.",
            "Citations: every sentence attributable to a source chunk.",
            "Faithfulness check: low-scoring answers flagged or re-queried.",
        ],
        failure_modes   = [
            "Still one-shot: can't ask follow-up retrieval for ambiguous answers.",
            "No tool use: arithmetic and filtering still fail.",
            "Static context: retrieved docs not updated mid-conversation.",
            "Query rewriting adds latency (extra LLM call).",
            "Cross-encoder reranking is expensive at scale.",
        ],
        example_stack   = "LlamaIndex + Voyage AI + Qdrant + BM25 + Cohere Reranker + Claude",
        faithfulness    = "~75–85% (citations + strict prompting)",
        recall          = "~80–90% (hybrid + reranking)",
    ),

    RAGGeneration(
        name            = "Agentic / Modular RAG",
        years           = "2024–present",
        pipeline        = [
            "1. User query → query router (limitation detection)",
            "2. Planner: decide retrieval strategy, tools needed, steps",
            "3. Multi-step retrieval loop:",
            "   a. Execute retrieval step (search, API call, tool)",
            "   b. Evaluate retrieved content: is the question answered?",
            "   c. If not: refine query and loop",
            "4. Tool use: code interpreter for arithmetic, SQL for filtering",
            "5. Context assembly with budget management",
            "6. LLM generates answer with chain-of-thought + citations",
            "7. Faithfulness check + self-critique",
            "8. If low confidence: ask clarifying question or escalate",
            "9. Answer + citations + confidence score returned",
        ],
        improvements    = [
            "Multi-hop: can retrieve → reason → retrieve again for complex queries.",
            "Tool use: delegates arithmetic, filtering, logic to code interpreter.",
            "Self-correction: model evaluates its own answer and re-retrieves if needed.",
            "Query routing: different pipeline for different query types.",
            "Real-time retrieval: can call live APIs (APIC, ServiceNow) not just vector DB.",
            "Memory integration: episodic memory (conversation history) managed explicitly.",
            "Confidence-aware: knows when to escalate rather than guess.",
        ],
        failure_modes   = [
            "Higher latency: multiple retrieval loops take more time.",
            "Higher cost: planner + tool calls + multiple LLM calls per query.",
            "Harder to debug: multi-step plans are opaque without good tracing.",
            "Planner errors: if the planner chooses wrong tools, answer degrades.",
        ],
        example_stack   = "Anthropic Claude + tool_use API + Qdrant + Elasticsearch + Python interpreter",
        faithfulness    = "~90–95% (self-correction + multi-hop verification)",
        recall          = "~90–95% (multi-step retrieval + query adaptation)",
    ),
]


# ─── Display ──────────────────────────────────────────────────────────────────

def display_rag_generations():
    """Walk through all three RAG generations."""

    print("=" * 72)
    print("THREE GENERATIONS OF RAG: How the Architecture Evolved")
    print("=" * 72)

    for gen in RAG_GENERATIONS:
        print(f"\n  ══ {gen.name.upper()} ({gen.years}) ══")
        print(f"  Faithfulness: {gen.faithfulness}")
        print(f"  Recall:       {gen.recall}")
        print(f"  Stack:        {gen.example_stack}")

        print(f"\n  Pipeline:")
        for step in gen.pipeline:
            print(f"    {step}")

        if RAG_GENERATIONS.index(gen) > 0:
            print(f"\n  Improvements over previous generation:")
            for imp in gen.improvements:
                print(f"    + {imp}")

        print(f"\n  Remaining failure modes:")
        for fm in gen.failure_modes:
            print(f"    ✗ {fm}")


# ─── The Paper Timeline ───────────────────────────────────────────────────────

def rag_paper_timeline():
    """
    Key papers that shaped RAG's evolution.
    Each paper introduced a specific component that became standard.
    """

    print("\n" + "=" * 72)
    print("KEY PAPERS: The Research That Built Modern RAG")
    print("=" * 72)

    papers = [
        {
            "year":        "2017",
            "title":       "Attention is All You Need",
            "authors":     "Vaswani et al. (Google)",
            "contribution": "Transformer architecture — the foundation of both the retriever and generator",
            "lesson_ref":  "Lesson 4: Transformer Architecture",
        },
        {
            "year":        "2019",
            "title":       "Dense Passage Retrieval for Open-Domain QA",
            "authors":     "Karpukhin et al. (Facebook AI)",
            "contribution": "Proved dense embeddings outperform BM25 for semantic retrieval",
            "lesson_ref":  "Lesson 8: Embeddings",
        },
        {
            "year":        "2020",
            "title":       "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "authors":     "Lewis et al. (Facebook AI Research)",
            "contribution": "Named and formalized RAG; introduced parametric + non-parametric memory framing",
            "lesson_ref":  "Lesson 12 (this lesson)",
        },
        {
            "year":        "2021",
            "title":       "Improving Language Models by Retrieving from Trillions of Tokens",
            "authors":     "Borgeaud et al. (DeepMind) — RETRO",
            "contribution": "Showed retrieval can scale to 2 trillion token databases; retrieval during generation",
            "lesson_ref":  "Lesson 12",
        },
        {
            "year":        "2022",
            "title":       "Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)",
            "authors":     "Gao et al.",
            "contribution": "Generate a hypothetical document, then embed it for retrieval — better than query embedding alone",
            "lesson_ref":  "Lesson 13 (query rewriting)",
        },
        {
            "year":        "2023",
            "title":       "Lost in the Middle: How LLMs Use Long Contexts",
            "authors":     "Liu et al.",
            "contribution": "Proved U-shaped attention degradation; motivated chunk ordering strategies",
            "lesson_ref":  "Lesson 6: Context Window",
        },
        {
            "year":        "2023",
            "title":       "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
            "authors":     "Es et al.",
            "contribution": "Defined faithfulness, answer relevance, context relevance, context recall metrics",
            "lesson_ref":  "Lesson 10: Hallucinations",
        },
        {
            "year":        "2024",
            "title":       "RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture",
            "authors":     "Ovadia et al. (Bloomberg)",
            "contribution": "Empirical comparison: RAG outperforms fine-tuning for factual accuracy in domain Q&A",
            "lesson_ref":  "Lesson 12",
        },
        {
            "year":        "2024",
            "title":       "Self-RAG: Learning to Retrieve, Generate, and Critique",
            "authors":     "Asai et al.",
            "contribution": "Model learns to issue retrieval calls only when needed; self-reflects on answer quality",
            "lesson_ref":  "Lesson 12 (Agentic RAG)",
        },
    ]

    print(f"\n  {'Year':<6} {'Key contribution'}")
    print(f"  {'─'*6} {'─'*62}")
    for p in papers:
        print(f"\n  {p['year']:<6} {p['title']}")
        print(f"         {p['authors']}")
        print(f"         → {p['contribution']}")
        print(f"         ↳ See: {p['lesson_ref']}")


# ─── Component Map ────────────────────────────────────────────────────────────

def component_to_lesson_map():
    """
    Map every RAG pipeline component to the lesson that covers it.
    Confirms that Lessons 1–11 collectively cover all of modern RAG.
    """

    print("\n" + "=" * 72)
    print("COMPONENT MAP: Every RAG Part You've Already Learned")
    print("=" * 72)

    components = [
        ("LLM / Generator",           "Lessons 1–5", "Generative AI, LLMs, internals, transformer, attention"),
        ("Context window management",  "Lesson 6",    "Budget allocation, Lost in the Middle, compression"),
        ("Tokenization",               "Lesson 7",    "BPE, token cost, chunk sizing by content type"),
        ("Embedding model",            "Lesson 8",    "Voyage AI, cosine similarity, model selection"),
        ("BM25 keyword index",         "Lesson 9",    "TF saturation, inverted index, exact term matching"),
        ("Dense vector index",         "Lesson 9",    "HNSW, IVF, Product Quantization"),
        ("Hybrid search + RRF",        "Lesson 9",    "Reciprocal Rank Fusion, weighted fusion"),
        ("Metadata filtering",         "Lesson 9",    "Pre/post filter, enterprise schema"),
        ("Hallucination detection",    "Lesson 10",   "Faithfulness scoring, attribution, self-consistency"),
        ("Knowledge cutoff handling",  "Lesson 11",   "Cutoff detector, routing by staleness risk"),
        ("Query router",               "Lesson 11",   "Limitation signals → route to KB/tool/API"),
        ("Private knowledge base",     "Lesson 11",   "Fine-tuning vs RAG, memory types"),
        ("Why RAG exists",             "Lesson 12",   "Five gaps framework, generation evolution"),
    ]

    print(f"\n  {'RAG Component':<35} {'Lesson(s)':<12} {'Key concepts'}")
    print(f"  {'─'*35} {'─'*12} {'─'*30}")
    for comp, lessons, concepts in components:
        print(f"  {comp:<35} {lessons:<12} {concepts}")

    print(f"""
  STATUS: You now have all the foundational knowledge to build a production RAG system.
  The remaining Phase 1 lesson covers the full end-to-end pipeline integration.
  Phase 2 onwards covers advanced retrieval, reranking, evaluation, and production deployment.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    display_rag_generations()
    rag_paper_timeline()
    component_to_lesson_map()
