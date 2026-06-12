"""
FILE: 04_rag_as_mitigation.py
LESSON: Phase 1 - Lesson 10 - Hallucinations
TOPIC: How RAG reduces hallucinations — grounding, citation, faithfulness scoring

WHAT THIS FILE TEACHES:
  - Closed-book vs open-book prompting and their hallucination rates
  - WHY system prompt framing controls context adherence
  - Citation attribution as a retrieval audit trail
  - RAGAS-style faithfulness metric (simplified, no external model)
  - Residual hallucination modes that RAG does NOT fix
  - Prompt patterns that maximize faithfulness

INSTALL: pip install anthropic python-dotenv  (runs without API key too)
"""

import os
import re
import random
from dataclasses import dataclass, field
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    HAS_ANTHROPIC = False


# ─── Prompting Patterns ───────────────────────────────────────────────────────

# WHY three levels: shows that framing strength affects grounding compliance.
# Weak framing: model can add from parametric memory.
# Strict framing: model is constrained to context only.
# Audit framing: model MUST cite; enables automatic faithfulness check.

CLOSED_BOOK_PROMPT = """You are a helpful technical assistant.
Answer the user's question using your knowledge."""

OPEN_BOOK_WEAK_PROMPT = """You are a helpful technical assistant.
Use the provided context to help answer the question.
You may also use your general knowledge when relevant."""

OPEN_BOOK_STRICT_PROMPT = """You are a precise technical assistant.
Answer ONLY from the provided context below.
Do NOT add information from your training knowledge.
If the context does not contain the answer, say:
"I could not find this information in the provided documents."
Never speculate. Never estimate."""

OPEN_BOOK_AUDIT_PROMPT = """You are a precise technical assistant with citation requirements.
Answer ONLY from the provided context.
Every sentence in your answer must end with [ChunkN] citing the context chunk used.
If a claim has no support in the context, append [unsupported].
If the answer is not in the context, say: "Not found in provided documents."

Format:
<answer>
Sentence one. [Chunk1]
Sentence two. [Chunk2]
</answer>"""


def build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block.

    WHY explicit chunk numbers: enables citation audit.
    Without numbers, the model cites "the document" — unusable for verification.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", f"doc_{i}")
        parts.append(f"[Chunk{i}] ({source})\n{chunk['content']}")
    return "\n\n".join(parts)


# ─── Faithfulness Scoring ─────────────────────────────────────────────────────

@dataclass
class FaithfulnessScore:
    """Result of faithfulness evaluation for one answer."""
    answer:               str
    context:              str
    total_claims:         int
    supported_claims:     int
    unsupported_claims:   int
    faithfulness:         float      # supported / total (0.0–1.0)
    verdict:              str        # "faithful" | "partial" | "unfaithful"
    unsupported_text:     list[str]  # sentences flagged as unsupported


def score_faithfulness(answer: str, context: str, threshold: float = 0.30) -> FaithfulnessScore:
    """
    Compute a faithfulness score for an LLM answer against its context.

    This is a simplified implementation of the RAGAS faithfulness metric.

    Full RAGAS:
      1. Use an LLM to extract atomic claims from the answer.
      2. For each claim, use an NLI model to check: does context entail claim?
      3. faithfulness = supported_claims / total_claims

    Our approach:
      1. Split answer into sentences.
      2. Check each sentence against context using lexical overlap.
      3. Score = grounded sentences / total sentences.

    WHY this approach works for teaching:
      - No external model dependencies.
      - Shows the concept clearly.
      - Production note: replace step 2 with DeBERTa-mnli for accuracy.
    """

    def tokenize_content(text: str) -> set[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        return {t for t in tokens if len(t) > 3}

    # Strip citation tags from answer before scoring
    clean_answer = re.sub(r"\[Chunk\d+\]|\[unsupported\]", "", answer)

    sentences    = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_answer.strip()) if len(s.strip()) > 20]
    context_toks = tokenize_content(context)

    supported, unsupported = [], []
    for sent in sentences:
        sent_toks = tokenize_content(sent)
        if not sent_toks:
            continue
        overlap = len(sent_toks & context_toks) / len(sent_toks)
        if overlap >= threshold:
            supported.append(sent)
        else:
            unsupported.append(sent)

    total         = len(supported) + len(unsupported)
    faith_score   = len(supported) / max(total, 1)

    if faith_score >= 0.80:
        verdict = "faithful"
    elif faith_score >= 0.50:
        verdict = "partially faithful"
    else:
        verdict = "unfaithful (hallucinated)"

    return FaithfulnessScore(
        answer             = answer,
        context            = context,
        total_claims       = total,
        supported_claims   = len(supported),
        unsupported_claims = len(unsupported),
        faithfulness       = faith_score,
        verdict            = verdict,
        unsupported_text   = unsupported,
    )


# ─── Prompt Pattern Comparison ────────────────────────────────────────────────

def compare_prompt_patterns():
    """
    Show how different system prompt patterns affect hallucination risk.
    We simulate four answer styles corresponding to the four prompts above.
    """

    context = (
        "ReadyOps is Criterion Networks' continuous validation platform. "
        "It runs AI agent classes across two isolated environments: "
        "Production-Representative and Live Operations. Changes must pass "
        "a 100% validation gate before promotion to Live Operations."
    )

    question = "How does ReadyOps ensure safe deployments?"

    # Simulated answers for each prompt type
    answers = {
        "Closed-book (no context)": (
            "ReadyOps ensures safe deployments through automated testing, "
            "canary deployments, and blue-green rollout strategies. "
            "It integrates with CI/CD pipelines and supports rollback via "
            "Git-based versioning. The platform uses Kubernetes health checks "
            "and Prometheus alerting to validate deployment health."
        ),
        "Open-book weak (context + memory)": (
            "ReadyOps uses a Production-Representative environment to validate changes. "
            "It runs AI agent classes that check configuration compliance. "
            "The platform also integrates with Jira for ticketing and uses "
            "GitOps workflows for change management."
        ),
        "Open-book strict (context only)": (
            "ReadyOps ensures safe deployments by running AI agent classes across "
            "two isolated environments: Production-Representative and Live Operations. "
            "Changes must pass a 100% validation gate before promotion to Live Operations."
        ),
        "Open-book audit (context + citations)": (
            "ReadyOps is a continuous validation platform by Criterion Networks. [Chunk1] "
            "It operates across Production-Representative and Live Operations environments. [Chunk1] "
            "Promotion to Live Operations requires passing a 100% validation gate. [Chunk1]"
        ),
    }

    print("=" * 70)
    print("PROMPT PATTERN COMPARISON: Faithfulness by System Prompt Type")
    print("=" * 70)

    for prompt_type, answer in answers.items():
        result = score_faithfulness(answer, context)
        print(f"\n  [{prompt_type}]")
        print(f"  Answer: '{answer[:100]}...'")
        print(f"  Faithfulness: {result.faithfulness:.0%}  ({result.verdict})")
        if result.unsupported_text:
            print(f"  Hallucinated sentences:")
            for s in result.unsupported_text:
                print(f"    ✗ '{s}'")

    print(f"""
  INSIGHT:
    Closed-book prompt: pure parametric memory, 0% faithfulness to context.
    Weak open-book:     mixes context and memory — partial faithfulness.
    Strict open-book:   context-only, high faithfulness.
    Audit (citations):  same faithfulness + audit trail for verification.

    In production RAG: always use strict or audit prompts.
    The audit prompt is preferred because citations enable:
      1. Automatic faithfulness validation at query time.
      2. UI display of "answer sourced from [document X]".
      3. Post-hoc audit when a user disputes an answer.
""")


# ─── Residual Hallucination Modes ────────────────────────────────────────────

def residual_hallucination_modes():
    """
    RAG does NOT eliminate all hallucinations. Show the residual failure modes.
    """

    print("=" * 70)
    print("RESIDUAL HALLUCINATIONS: What RAG Doesn't Fix")
    print("=" * 70)

    modes = [
        {
            "mode":    "Retrieval failure",
            "explain": "If the relevant document is NOT in the knowledge base, "
                       "the retriever returns irrelevant chunks. The LLM then "
                       "answers from those chunks or falls back to parametric memory.",
            "example": "Q: 'What is ACI 6.2's new feature?' "
                       "If 6.2 docs aren't indexed, retrieval returns 6.0 docs. "
                       "LLM may confabulate a 6.2 feature.",
            "fix":     "Keep knowledge base current. Monitor retrieval quality. "
                       "Add 'no relevant documents found' handling.",
        },
        {
            "mode":    "Context misreading",
            "explain": "Model is given the correct document but misreads a number, "
                       "confuses two similar items, or misinterprets a table.",
            "example": "Context: 'ACI 4.x: max 180 leafs. ACI 6.x: max 200 leafs.' "
                       "Q: 'Max leafs in ACI 6.0?' "
                       "LLM answers '180' — picked the wrong row.",
            "fix":     "Structure context with clear labels. Use shorter chunks. "
                       "Test with confusion-prone queries.",
        },
        {
            "mode":    "Lost in the middle",
            "explain": "Correct document is retrieved but placed in the middle "
                       "of a long context. Model pays less attention to it.",
            "example": "5 chunks retrieved. Answer is in chunk 3 of 5. "
                       "Model answers from chunk 1 (primacy effect).",
            "fix":     "Reorder chunks: highest-relevance first and last. "
                       "Reduce top-K to reduce context length.",
        },
        {
            "mode":    "Extrinsic addition",
            "explain": "Model correctly uses retrieved context but also adds "
                       "extra information from parametric memory.",
            "example": "Context says 'ISE uses RADIUS.' "
                       "Model answers: 'ISE uses RADIUS (port 1812) and TACACS+ for device admin.'",
            "fix":     "Use strict system prompt: 'Do not add information not in the context.'",
        },
        {
            "mode":    "Temporal confusion",
            "explain": "Knowledge base has both old and new documents. Model "
                       "retrieves an outdated chunk and presents old info as current.",
            "example": "ACI 4.0 doc (max 180 leafs) retrieved alongside ACI 6.0 doc. "
                       "Model answers with 4.0 spec.",
            "fix":     "Metadata date filter: 'date >= last N months'. "
                       "Version-pin queries in the system prompt.",
        },
    ]

    for m in modes:
        print(f"\n  FAILURE MODE: {m['mode'].upper()}")
        print(f"  Problem: {m['explain']}")
        print(f"  Example: {m['example']}")
        print(f"  Fix:     {m['fix']}")

    print(f"""
  SUMMARY TABLE:

  Hallucination Type         RAG Fixes?  Remaining Mitigation
  ─────────────────────────  ──────────  ──────────────────────────────
  No context (closed-book)   YES         Ensure retrieval runs
  Stale parametric facts     YES         Keep KB current
  Fabricated citations       YES         Require [ChunkN] attribution
  Retrieval failure          PARTIAL     Knowledge base maintenance
  Context misreading         NO          Shorter chunks, re-ranking
  Lost in the middle         NO          Chunk ordering, top-K reduction
  Extrinsic addition         NO          Strict system prompt
  Temporal confusion         NO          Date metadata filtering
""")


# ─── Live API Demo (optional) ────────────────────────────────────────────────

def live_rag_vs_closed_demo():
    """
    If ANTHROPIC_API_KEY is set, call Claude with and without context
    to show the faithfulness difference live.
    """

    if not HAS_ANTHROPIC:
        print("  [Skipping live demo — ANTHROPIC_API_KEY not set or anthropic not installed]")
        return

    client = anthropic.Anthropic()

    context_chunks = [
        {"source": "readyops_guide_v2.md",
         "content": "ReadyOps runs AI agent classes across two isolated environments. "
                    "The Production-Representative environment receives changes first. "
                    "A 100% validation pass rate is required before promotion to Live Operations."},
        {"source": "readyops_guide_v2.md",
         "content": "ReadyOps agent classes include: Health and Posture agents (baseline monitoring), "
                    "Validation agents (pre-change tests), Operational agents (runbook execution), "
                    "and Stress and Adversarial agents (resilience testing)."},
    ]
    context_block = build_context_block(context_chunks)
    question = "How does ReadyOps handle a failed validation gate?"

    print("=" * 70)
    print("LIVE DEMO: Closed-book vs Open-book (grounded) response")
    print("=" * 70)

    print("\n  [CLOSED-BOOK — no context]")
    resp_closed = client.messages.create(
        model       = "claude-haiku-4-5-20251001",  # WHY Haiku: fast and cheap for demo
        max_tokens  = 200,
        system      = CLOSED_BOOK_PROMPT,
        messages    = [{"role": "user", "content": question}],
    )
    closed_answer = resp_closed.content[0].text
    print(f"  Answer: {closed_answer[:200]}")
    faith_c = score_faithfulness(closed_answer, context_block)
    print(f"  Faithfulness to context: {faith_c.faithfulness:.0%} ({faith_c.verdict})")

    print("\n  [OPEN-BOOK STRICT — with context]")
    full_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"
    resp_grounded = client.messages.create(
        model       = "claude-haiku-4-5-20251001",
        max_tokens  = 200,
        system      = OPEN_BOOK_STRICT_PROMPT,
        messages    = [{"role": "user", "content": full_prompt}],
    )
    grounded_answer = resp_grounded.content[0].text
    print(f"  Answer: {grounded_answer[:200]}")
    faith_g = score_faithfulness(grounded_answer, context_block)
    print(f"  Faithfulness to context: {faith_g.faithfulness:.0%} ({faith_g.verdict})")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    compare_prompt_patterns()
    residual_hallucination_modes()
    live_rag_vs_closed_demo()
