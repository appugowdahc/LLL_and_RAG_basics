"""
FILE: 04_sub_question_decomposition.py
LESSON: Phase 2 - Lesson 14 - Query Understanding and Rewriting
TOPIC: Sub-question decomposition — answer complex queries by splitting them

WHAT THIS FILE TEACHES:
  - Why complex multi-hop queries break single-chunk retrieval
  - Sub-question decomposition algorithm
  - Answering each sub-question independently with its own retrieval
  - Answer synthesis: combining partial answers into a final response
  - Sequential decomposition (answer feeds into next sub-question)
  - WHY decomposition is the foundation of agentic RAG

INSTALL: pip install anthropic python-dotenv numpy
"""

import os
import re
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    HAS_ANTHROPIC = False


# ─── Utilities ────────────────────────────────────────────────────────────────

def mock_embed(text: str, dims: int = 64) -> np.ndarray:
    keywords  = sorted(set(re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())))[:8]
    topic_key = " ".join(keywords)
    seed      = int(hashlib.md5(topic_key.encode()).hexdigest(), 16) % (2**32)
    rng       = np.random.RandomState(seed)
    full_rng  = np.random.RandomState(int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31))
    v = rng.randn(dims).astype(np.float32) + full_rng.randn(dims).astype(np.float32) * 0.15
    return v / (np.linalg.norm(v) + 1e-10)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                        b / (np.linalg.norm(b) + 1e-10)))


# ─── Sub-Question Generator ───────────────────────────────────────────────────

DECOMPOSE_SYSTEM = """You are a search query analyst.
Given a complex question, break it into 2-4 atomic sub-questions, each answerable
by a single retrieved document passage.

Rules:
  - Each sub-question must be self-contained (no pronouns referring to other questions)
  - Order them logically (foundational questions first)
  - Keep each sub-question short (< 15 words)
  - Output ONLY a numbered list, one sub-question per line"""

DECOMPOSE_USER = """Complex question: {question}

Break this into atomic sub-questions for retrieval:"""

# Mock decompositions for demo without API
MOCK_DECOMPOSITIONS: dict[str, list[str]] = {
    "readyops aci": [
        "What is ReadyOps?",
        "How does ReadyOps integrate with Cisco ACI?",
        "What validation steps does ReadyOps perform on ACI changes?",
    ],
    "difference between": [
        "What is {subject_a}?",
        "What is {subject_b}?",
        "What are the key differences between {subject_a} and {subject_b}?",
    ],
    "hypershield ebpf aci": [
        "What is Cisco Hypershield?",
        "What is eBPF and how does it enable policy enforcement?",
        "How does Hypershield integrate with ACI EPG policy?",
    ],
    "apic readyops": [
        "What is the APIC controller in ACI?",
        "How does ReadyOps consume APIC configuration?",
        "What validation does ReadyOps perform on APIC policy?",
    ],
}


def decompose_query(query: str) -> list[str]:
    """
    Break a complex query into atomic sub-questions.

    WHY decompose instead of sending the full query:
      A query like "What is ReadyOps and how does it integrate with ACI?"
      has TWO distinct information needs. A single embedding averages both
      topics — the resulting vector may not be close to either the ReadyOps
      intro chunk OR the ACI integration chunk.

      Decomposed:
        Q1: "What is ReadyOps?" → retrieves ReadyOps overview chunks
        Q2: "How does ReadyOps integrate with ACI?" → retrieves integration chunks

      Both chunks are now in context for the LLM to synthesize.
    """
    if HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 200,
            system     = DECOMPOSE_SYSTEM,
            messages   = [{"role": "user", "content": DECOMPOSE_USER.format(question=query)}],
        )
        text  = resp.content[0].text.strip()
        lines = re.findall(r"^\d+[.)]\s*(.+)$", text, re.MULTILINE)
        return lines if lines else [query]

    else:
        q_low = query.lower()
        for key, subs in MOCK_DECOMPOSITIONS.items():
            if all(w in q_low for w in key.split()):
                return subs
        # Heuristic fallback: split on " and "
        if " and " in q_low:
            parts = [p.strip() for p in query.split(" and ", maxsplit=1)]
            return [parts[0] + "?", "And " + parts[1] + "?"]
        return [query]


# ─── Sub-Question Retrieval and Answering ─────────────────────────────────────

@dataclass
class SubAnswer:
    """Answer to one sub-question with supporting context."""
    sub_question:   str
    retrieved_docs: list[dict]
    answer:         str           # either from LLM or mock
    confidence:     float         # 0.0–1.0 based on retrieval score


@dataclass
class DecomposedResult:
    """Full result of sub-question decomposition and synthesis."""
    original_query:  str
    sub_questions:   list[str]
    sub_answers:     list[SubAnswer]
    final_answer:    str

    def display(self):
        print(f"\n  Original: '{self.original_query}'")
        print(f"  Decomposed into {len(self.sub_questions)} sub-questions:")
        for i, (sq, sa) in enumerate(zip(self.sub_questions, self.sub_answers), 1):
            print(f"\n    Sub-Q {i}: '{sq}'")
            if sa.retrieved_docs:
                print(f"    Retrieved: '{sa.retrieved_docs[0]['content'][:70]}'")
            print(f"    Sub-answer: '{sa.answer[:100]}'")
        print(f"\n  Final synthesized answer:")
        print(f"    '{self.final_answer[:200]}'")


ANSWER_SYSTEM = """Answer the question using ONLY the provided context.
Be concise (1-2 sentences). If the context doesn't answer the question, say "Not found in context."
Do NOT add information from prior knowledge."""

SYNTHESIS_SYSTEM = """Given a set of sub-questions and their answers, synthesize a single
comprehensive answer to the original question. Be clear and concise.
Format: Answer the original question directly, incorporating all relevant sub-answers."""


def retrieve_for_subquestion(
    sub_question: str,
    corpus:       list[dict],
    top_k:        int = 3,
) -> list[dict]:
    """Retrieve top-K docs for a sub-question."""
    q_vec   = mock_embed(sub_question)
    scored  = sorted(
        [(doc, cosine_sim(q_vec, mock_embed(doc["content"]))) for doc in corpus],
        key=lambda x: -x[1]
    )[:top_k]
    return [doc for doc, _ in scored]


def answer_subquestion(sub_q: str, context_docs: list[dict]) -> tuple[str, float]:
    """
    Answer one sub-question using retrieved context.
    Returns (answer_text, confidence).
    WHY confidence from retrieval score: low retrieval score = weak grounding.
    """
    context = "\n".join(doc["content"] for doc in context_docs[:2])

    if HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 100,
            system     = ANSWER_SYSTEM,
            messages   = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {sub_q}"}],
        )
        answer = resp.content[0].text.strip()
    else:
        # Mock: extract relevant sentence from context
        answer = context_docs[0]["content"] if context_docs else "Not found in context."

    confidence = cosine_sim(mock_embed(sub_q), mock_embed(context)) if context else 0.0
    return answer, max(0.0, min(1.0, confidence))


def synthesize_answers(original_query: str, sub_answers: list[SubAnswer]) -> str:
    """
    Synthesize sub-answers into a final answer to the original query.
    """
    if HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        subs_text = "\n".join(
            f"Sub-question {i+1}: {sa.sub_question}\nAnswer: {sa.answer}"
            for i, sa in enumerate(sub_answers)
        )
        resp = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 200,
            system     = SYNTHESIS_SYSTEM,
            messages   = [{
                "role": "user",
                "content": f"Original question: {original_query}\n\n{subs_text}\n\nSynthesize a final answer:",
            }],
        )
        return resp.content[0].text.strip()
    else:
        # Combine sub-answers into one text
        parts = [f"{sa.answer}" for sa in sub_answers if "Not found" not in sa.answer]
        return " ".join(parts) if parts else "Information not found in context."


def decompose_and_answer(
    query:  str,
    corpus: list[dict],
) -> DecomposedResult:
    """
    Full pipeline: decompose → retrieve → answer each → synthesize.
    """
    sub_questions = decompose_query(query)
    sub_answers   = []

    for sq in sub_questions:
        docs                   = retrieve_for_subquestion(sq, corpus, top_k=3)
        answer, confidence     = answer_subquestion(sq, docs)
        sub_answers.append(SubAnswer(
            sub_question   = sq,
            retrieved_docs = docs,
            answer         = answer,
            confidence     = confidence,
        ))

    final = synthesize_answers(query, sub_answers)

    return DecomposedResult(
        original_query = query,
        sub_questions  = sub_questions,
        sub_answers    = sub_answers,
        final_answer   = final,
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────

CORPUS = [
    {"id": "c001", "content": "The APIC cluster manages all ACI fabric policy and requires minimum 3 nodes for HA."},
    {"id": "c002", "content": "EPGs define policy groups in ACI. Contracts allow inter-EPG traffic."},
    {"id": "c003", "content": "ReadyOps is Criterion Networks' continuous validation platform for network infrastructure."},
    {"id": "c004", "content": "ReadyOps validates changes in Production-Representative environment. 100% pass rate required."},
    {"id": "c005", "content": "ReadyOps consumes APIC policy snapshots to build the digital twin fabric model."},
    {"id": "c006", "content": "Hypershield uses eBPF for kernel-level policy at the workload without appliances."},
    {"id": "c007", "content": "Hypershield integrates with ACI EPG membership propagated from APIC for policy."},
    {"id": "c008", "content": "ACI 6.0 supports 200 leaf switches per pod with VXLAN fabric overlay."},
    {"id": "c009", "content": "ReadyOps agent classes include Health Posture, Validation, Operational, and Stress."},
    {"id": "c010", "content": "The promotion gate opens only when all ReadyOps validation tests pass at 100%."},
]


def run_decomposition_demo():
    """Compare single-query vs decomposed retrieval on complex questions."""

    print("=" * 70)
    print("SUB-QUESTION DECOMPOSITION: Handling Complex Multi-Part Queries")
    print("=" * 70)

    complex_queries = [
        "What is ReadyOps and how does it integrate with Cisco ACI?",
        "How does Hypershield use eBPF and how does it connect to ACI policy?",
    ]

    for q in complex_queries:
        print(f"\n  {'─'*65}")
        print(f"  COMPLEX QUERY: '{q}'")

        # Show single-query limitation
        q_vec    = mock_embed(q)
        single   = sorted([(doc, cosine_sim(q_vec, mock_embed(doc["content"])))
                           for doc in CORPUS], key=lambda x: -x[1])[:3]
        print(f"\n  Single-query top-3 (embedding blurs across topics):")
        for doc, score in single:
            print(f"    {score:.3f}  '{doc['content'][:65]}'")

        # Decompose and answer
        result = decompose_and_answer(q, CORPUS)
        result.display()


def sequential_decomposition_demo():
    """
    Show sequential decomposition where later sub-questions use earlier answers.
    This is the foundation of multi-hop agentic RAG.
    """

    print("\n" + "=" * 70)
    print("SEQUENTIAL DECOMPOSITION: Multi-Hop Agentic Pattern")
    print("=" * 70)

    print(f"""
  STANDARD DECOMPOSITION (parallel):
    All sub-questions run independently and in parallel.
    Suitable when sub-questions are independent of each other.

    Example:
      Q: "What is ReadyOps and how does it integrate with ACI?"
      Sub-Q 1: "What is ReadyOps?" → retrieve → answer A1
      Sub-Q 2: "How does ReadyOps integrate with ACI?" → retrieve → answer A2
      Synthesize A1 + A2 → final answer

  SEQUENTIAL DECOMPOSITION (multi-hop):
    Later sub-questions USE earlier answers to form their query.
    Required for multi-hop queries where you need to find A first, then use A.

    Example:
      Q: "What product does Criterion Networks use to validate ACI changes,
          and what are its agent classes?"
      Step 1: "What Criterion Networks product validates ACI changes?"
              → retrieve → answer: "ReadyOps"
      Step 2: "What are the ReadyOps agent classes?"  ← uses 'ReadyOps' from step 1
              → retrieve → answer: "Health Posture, Validation, Operational, Stress"
      Final: "Criterion Networks uses ReadyOps to validate ACI changes.
              Its agent classes are: Health and Posture, Validation, Operational,
              and Stress and Adversarial."

  WHY SEQUENTIAL IS THE FOUNDATION OF AGENTIC RAG:
    Each retrieval step can use the result of the previous step.
    The system can plan multiple steps before executing any of them,
    or it can plan one step, execute, observe the result, and plan next.
    This is what enables complex multi-hop reasoning over a knowledge base.
    Covered in depth in Phase 4: Agentic RAG.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_decomposition_demo()
    sequential_decomposition_demo()
