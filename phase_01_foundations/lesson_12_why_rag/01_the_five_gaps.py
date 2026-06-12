"""
FILE: 01_the_five_gaps.py
LESSON: Phase 1 - Lesson 12 - Why RAG Was Invented
TOPIC: The five knowledge gaps that RAG was built to close

WHAT THIS FILE TEACHES:
  - Precise articulation of each gap (NOT just "LLMs hallucinate")
  - Evidence for each gap with infrastructure examples
  - Mapping: gap → which RAG component closes it
  - What RAG does NOT close (and what closes those instead)
  - WHY understanding the gaps precisely matters for architecture decisions

INSTALL: no external dependencies
"""

from dataclasses import dataclass
from typing import Optional


# ─── Gap Framework ────────────────────────────────────────────────────────────

@dataclass
class KnowledgeGap:
    """
    One specific gap between what an LLM can do and what production needs.
    Includes a precise problem statement, evidence, the RAG component that
    closes it, and what remains after RAG.
    """
    gap_id:          int
    name:            str
    problem_precise: str             # Not "LLMs hallucinate" — be precise
    evidence:        list[str]       # Concrete examples
    rag_component:   str             # Which part of the RAG pipeline fixes this
    rag_closes:      bool            # Does RAG fully close this gap?
    residual_risk:   Optional[str]   # What remains after RAG
    lesson_ref:      str             # Which prior lesson covered this


FIVE_GAPS = [

    KnowledgeGap(
        gap_id          = 1,
        name            = "Knowledge Cutoff",
        problem_precise = (
            "LLM training data ends at a specific date. Any fact published after "
            "that date is unknown to the model. The model cannot know what it was "
            "never trained on — it will either say 'I don't know' or hallucinate "
            "a plausible-sounding answer using nearby parametric knowledge."
        ),
        evidence        = [
            "ACI 6.0 released after many models' training cutoffs → model gives 5.x specs",
            "CVE published last week → model has no knowledge of it",
            "Your internal document written yesterday → model has never seen it",
            "Competitor announced EOL last month → model says product is still sold",
        ],
        rag_component   = "Knowledge base with continuous indexing. Fresh docs are indexed as they are created/updated. Retrieval bypasses parametric memory entirely for these queries.",
        rag_closes      = True,
        residual_risk   = "Retrieval failure: if the updated doc isn't in the KB yet, or if the retriever returns the wrong chunk, the gap remains.",
        lesson_ref      = "Lesson 11: Knowledge Cutoff",
    ),

    KnowledgeGap(
        gap_id          = 2,
        name            = "Hallucination",
        problem_precise = (
            "When a model lacks a specific fact in parametric memory, it generates "
            "the most statistically likely next token — which may be factually wrong. "
            "The generated answer is delivered with the same fluency and confidence "
            "as a correct answer. The user cannot distinguish them without verification."
        ),
        evidence        = [
            "Model says APIC needs '5 nodes for HA' (actually 3) — close enough to trick a non-expert",
            "Model fabricates a Cisco advisory reference that doesn't exist",
            "Model says ACI 6.0 max leafs is 180 (actually 200) — off by one version",
            "Sycophantic model agrees with user's false premise about port numbers",
        ],
        rag_component   = "Retrieved context as ground truth + strict system prompt ('answer only from context') + source attribution ([ChunkN] citations). Post-generation faithfulness check.",
        rag_closes      = True,
        residual_risk   = "Intrinsic hallucination: model may still misread or ignore the correct context (Lost in the Middle). Extrinsic addition: model may add from memory even with strict prompting.",
        lesson_ref      = "Lesson 10: Hallucinations",
    ),

    KnowledgeGap(
        gap_id          = 3,
        name            = "Private Data Access",
        problem_precise = (
            "An LLM trained on public internet data has never seen your organization's "
            "private knowledge: internal runbooks, customer configurations, SLAs, "
            "incident history, architecture docs, proprietary templates. This is not "
            "a hallucination problem — the model simply has no knowledge to recall. "
            "No amount of prompting can surface information the model never learned."
        ),
        evidence        = [
            "User asks for the change management SLA → model cannot answer (never in training data)",
            "User asks about customer ACME's EPG topology → model cannot answer",
            "User asks which team is on-call this week → model cannot answer",
            "User asks for the post-mortem from last Tuesday's fabric outage → model cannot answer",
        ],
        rag_component   = "Private knowledge base: index internal documents (runbooks, configs, incident reports, SLAs) into a private vector store. Never public. Only accessible via authenticated retrieval.",
        rag_closes      = True,
        residual_risk   = "Data not indexed: if a doc was never added to the KB, retrieval cannot find it. Data staleness: if the KB is not updated after doc changes, content is stale.",
        lesson_ref      = "Lesson 11: No Private Data",
    ),

    KnowledgeGap(
        gap_id          = 4,
        name            = "Context Window Overload",
        problem_precise = (
            "An LLM cannot have its entire knowledge base in the context window. "
            "Even with a 200K-token window: (1) cost scales linearly with input tokens, "
            "(2) model attention degrades for middle-position content, (3) filling the "
            "window with all documents reduces answer precision (noise overwhelms signal). "
            "The model needs exactly the relevant information — not everything."
        ),
        evidence        = [
            "100-doc knowledge base at 400 tokens/doc = 40K tokens input = $0.12/call at Sonnet pricing",
            "10K-doc knowledge base = impossible to fit even at 200K window",
            "At 50 chunks, the answer-bearing chunk at position 25 has ~60% recall (Lost in the Middle)",
            "Loading irrelevant context causes the model to hallucinate by mixing irrelevant content",
        ],
        rag_component   = "Retrieval: select only the top-3 to top-5 most relevant chunks per query. This fits within token budget while maximizing attention quality for the relevant content.",
        rag_closes      = True,
        residual_risk   = "Retrieval precision: if the wrong 5 chunks are retrieved, the correct answer is not in context. Reranking (Lesson 13) reduces this.",
        lesson_ref      = "Lesson 11: Context Window Limits + Lesson 6: Context Window",
    ),

    KnowledgeGap(
        gap_id          = 5,
        name            = "No Citations or Auditability",
        problem_precise = (
            "A standalone LLM generates assertions without sources. In enterprise and "
            "regulated contexts, every factual claim must be traceable to a source "
            "document. Without citations, there is no way to: audit compliance, "
            "dispute an answer, version-control a knowledge base, or build user trust "
            "in the system's outputs. 'Trust the AI' is not acceptable in production."
        ),
        evidence        = [
            "Security audit: 'Prove that the CVE guidance came from Cisco PSIRT'",
            "Compliance review: 'Which policy document justifies this configuration recommendation?'",
            "Incident post-mortem: 'What was the LLM's reasoning for recommending this change?'",
            "User dispute: 'You said X but our runbook says Y — where did X come from?'",
        ],
        rag_component   = "Chunk metadata: every retrieved chunk carries source, date, version, author. System prompt requires [ChunkN] citations. UI surfaces citations as clickable links to source documents.",
        rag_closes      = True,
        residual_risk   = "Attribution quality: only as good as the metadata attached to chunks at index time. Missing or stale metadata = unreliable citations.",
        lesson_ref      = "Lesson 10: Hallucination Detection + Lesson 9: Metadata Filtering",
    ),
]


# ─── Display ──────────────────────────────────────────────────────────────────

def display_five_gaps():
    """
    Walk through each gap with full evidence and resolution.
    """

    print("=" * 72)
    print("THE FIVE KNOWLEDGE GAPS THAT MOTIVATED RAG")
    print("=" * 72)
    print(f"""
  These are NOT vague complaints about LLMs. Each gap is a precise,
  measurable failure mode in a production enterprise AI system.
  RAG exists because each of these gaps was causing real production failures.
""")

    for gap in FIVE_GAPS:
        closed_str = "CLOSED" if gap.rag_closes else "PARTIAL"
        print(f"\n  ══ GAP {gap.gap_id}: {gap.name.upper()} [{closed_str} by RAG] ══")
        print(f"  Lesson reference: {gap.lesson_ref}")
        print(f"\n  Problem:")
        for line in _wrap(gap.problem_precise, 64):
            print(f"    {line}")
        print(f"\n  Evidence (real production scenarios):")
        for ev in gap.evidence:
            print(f"    • {ev}")
        print(f"\n  RAG solution:")
        for line in _wrap(gap.rag_component, 64):
            print(f"    {line}")
        if gap.residual_risk:
            print(f"\n  Residual risk after RAG:")
            for line in _wrap(gap.residual_risk, 64):
                print(f"    ⚠ {line}")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    return lines


# ─── Gap Coverage Table ───────────────────────────────────────────────────────

def gap_coverage_table():
    """
    Show the gap coverage matrix: which approaches close which gaps.
    """

    print("\n" + "=" * 72)
    print("GAP COVERAGE MATRIX: RAG vs Alternatives")
    print("=" * 72)

    approaches = [
        "Pure LLM",
        "Fine-tuning",
        "Large context window",
        "Traditional search",
        "Naive RAG (basic)",
        "Advanced RAG (hybrid+rerank)",
        "Agentic RAG (tools+loops)",
    ]

    gaps = ["Cutoff", "Hallucin.", "Private", "Context", "Citations"]

    coverage = {
        "Pure LLM":                       [False, False, False, False, False],
        "Fine-tuning":                     [True,  False, True,  False, False],
        "Large context window":            [False, False, False, "PART",False],
        "Traditional search":              [True,  "N/A", True,  True,  True ],
        "Naive RAG (basic)":               [True,  "PART",True,  True,  "PART"],
        "Advanced RAG (hybrid+rerank)":    [True,  True,  True,  True,  True ],
        "Agentic RAG (tools+loops)":       [True,  True,  True,  True,  True ],
    }

    def fmt(v) -> str:
        if v is True:    return "  YES  "
        if v is False:   return "  NO   "
        if v == "PART":  return " PART. "
        if v == "N/A":   return "  N/A  "
        return "  ?    "

    print(f"\n  {'Approach':<35} " + " ".join(f"{g:^8}" for g in gaps))
    print(f"  {'─'*35} " + " ".join("─"*8 for _ in gaps))

    for approach, vals in coverage.items():
        row = f"  {approach:<35} " + " ".join(fmt(v) for v in vals)
        print(row)

    print(f"""
  INSIGHT:
    Only Advanced RAG and Agentic RAG close ALL five gaps.
    Fine-tuning helps with cutoff and private data but fails on hallucination
    and citations — making it insufficient for production enterprise use alone.
    Traditional search (Elasticsearch) handles knowledge gaps but provides
    no synthesis — users must read and reason themselves.

    The winning architecture: Advanced/Agentic RAG — not one of the alternatives,
    but a synthesis that wraps the LLM with retrieval infrastructure.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    display_five_gaps()
    gap_coverage_table()
