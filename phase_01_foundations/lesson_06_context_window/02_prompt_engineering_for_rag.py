"""
FILE: 02_prompt_engineering_for_rag.py
LESSON: Phase 1 - Lesson 6 - Context Window
TOPIC: Prompt engineering specifically for RAG — how to structure your context
       so the LLM uses it faithfully, stays grounded, and cites sources.

WHAT THIS FILE TEACHES:
  - The anatomy of a high-quality RAG system prompt
  - How instruction placement affects compliance
  - Few-shot examples for grounding behavior
  - Anti-hallucination patterns (explicit rules + examples)
  - Citation formats and when to use each
  - How temperature=0 enforces context faithfulness

KEY PRINCIPLE:
  In RAG, the prompt is not just a question — it is a CONTRACT:
    "Here is the evidence. Answer using ONLY this evidence.
     If you cannot answer from it, say so."
  The system prompt defines the rules of that contract.

INSTALL:
  pip install anthropic python-dotenv
"""

import os
import time
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()


# ─── RAG System Prompt Templates ─────────────────────────────────────────────

# TEMPLATE 1: Minimal (common mistake)
# WHY this is WRONG: No grounding rule, no "don't hallucinate" instruction,
#   no format guidance. The model WILL hallucinate if context is insufficient.
MINIMAL_SYSTEM = """You are a helpful assistant. Answer questions based on the provided context."""


# TEMPLATE 2: Strict Grounding (production recommended)
# WHY each rule exists — annotated below
STRICT_GROUNDING_SYSTEM = """\
You are a precise enterprise knowledge assistant for Criterion Networks.

RULES (MANDATORY):
1. Answer using ONLY the provided CONTEXT DOCUMENTS below.
   WHY: Prevents the model from mixing training knowledge with retrieved docs.
2. Cite every factual claim using [Doc N] notation.
   WHY: Enables downstream audit — users can verify every claim.
3. If the answer is not in the context, respond EXACTLY: "NOT IN PROVIDED CONTEXT"
   Do NOT attempt to answer from general knowledge.
   WHY: Explicit fallback prevents hallucination on missing information.
4. Do not add opinions, recommendations, or inferences beyond what the documents state.
   WHY: Keeps the system in "retrieval mode," not "reasoning mode."
5. If multiple documents contradict each other, say so and cite both.
   WHY: Forces transparency on conflicting sources.
"""

# TEMPLATE 3: Structured Output (for downstream parsing)
# WHY structured output in RAG:
#   If the RAG response feeds into another system (dashboard, ticket, alert),
#   you need deterministic output format — not prose. Use JSON output.
STRUCTURED_OUTPUT_SYSTEM = """\
You are a technical infrastructure analyst. Respond in valid JSON only.

For each question, output:
{
  "answer": "<concise answer from documents>",
  "evidence": [
    { "doc_id": "Doc N", "quote": "<verbatim quote from doc>" }
  ],
  "confidence": "high|medium|low",
  "missing_info": "<what information was NOT in the context, if any>"
}

If the answer is not in the context, set answer to "NOT IN PROVIDED CONTEXT"
and evidence to an empty array.
"""


# ─── Citation Format Comparison ──────────────────────────────────────────────

CITATION_FORMATS = {
    "inline_numeric": {
        "description": "Inline [1] notation — most common for RAG",
        "example":     "ACI uses a Leaf-Spine topology [1]. The APIC controller manages the fabric [1][2].",
        "best_for":    "Knowledge base Q&A, support bots",
        "pro":         "Compact, readable, unambiguous",
        "con":         "Requires doc list to be numbered in context",
    },
    "inline_named": {
        "description": "Inline [source.pdf] notation — preserves source names",
        "example":     "Hypershield uses eBPF [hypershield.pdf]. ISE supports SGT [ise_admin.pdf].",
        "best_for":    "Technical documentation assistants",
        "pro":         "Source name visible without looking up index",
        "con":         "Verbose for long filenames",
    },
    "footnote": {
        "description": "Footnote-style — answer first, sources at end",
        "example":     "ACI uses a Leaf-Spine topology.\n\nSources:\n1. aci_guide.pdf, p.12",
        "best_for":    "Report generation, executive summaries",
        "pro":         "Clean main text, full source details at end",
        "con":         "Ambiguous which fact maps to which source",
    },
    "verbatim_quote": {
        "description": "Verbatim quote from document, then interpretation",
        "example":     "Per Doc 1: 'ACI uses a Leaf-Spine topology' — this means all compute connects through leaf switches.",
        "best_for":    "High-stakes compliance, legal, audit contexts",
        "pro":         "Zero ambiguity about what was retrieved vs interpreted",
        "con":         "Very verbose, expensive in tokens",
    },
}


# ─── Build Annotated RAG Prompt ───────────────────────────────────────────────

def build_rag_prompt(
    system_template:   str,
    retrieved_docs:    list[dict],
    conversation_hist: list[dict],
    user_query:        str,
    include_few_shot:  bool = True,
) -> tuple[str, list[dict]]:
    """
    Build a complete RAG prompt with all components in optimal order.

    WHY this specific order:
      1. SYSTEM: Grounding rules are in the system prompt — highest priority.
      2. FEW-SHOT: Show the model exactly what "correct" behavior looks like.
      3. RETRIEVED DOCS: Main payload. Static, cacheable.
      4. CONVERSATION HISTORY: Prior context. Dynamic, sliding window.
      5. USER QUERY: Last — directly before the expected response.

    Args:
        system_template:   System prompt string (use STRICT_GROUNDING_SYSTEM)
        retrieved_docs:    List of dicts with 'id', 'source', 'content', 'score'
        conversation_hist: List of previous turns [{"role": ..., "content": ...}]
        user_query:        The current user question
        include_few_shot:  Whether to prepend few-shot examples in the context

    Returns:
        (system_prompt, messages_list) — ready to pass to client.messages.create()
    """

    # ── Build context block from retrieved documents ──────────────────────────
    # WHY explicit [Doc N] labeling:
    #   Maps directly to the [Doc N] citation format in the system prompt.
    #   Model follows the format because the docs use it first.
    context_lines = ["CONTEXT DOCUMENTS:"]
    context_lines.append("─" * 50)

    for doc in retrieved_docs:
        context_lines.append(
            f"[{doc['id']}] Source: {doc['source']}  |  Relevance: {doc['score']:.2f}"
        )
        context_lines.append(doc["content"])
        context_lines.append("")

    context_block = "\n".join(context_lines)

    # ── Optional few-shot examples ────────────────────────────────────────────
    # WHY few-shot:
    #   Even with strict rules, models can drift in edge cases.
    #   A few-shot example of "correct behavior for NOT IN CONTEXT" prevents
    #   the model from hedging ("I'm not sure, but maybe...") instead of
    #   using the exact required phrase.
    few_shot_block = ""
    if include_few_shot:
        few_shot_block = """\

EXAMPLE INTERACTIONS:

Example 1 — Answer found in context:
User: What is Cisco ACI?
Assistant: Cisco ACI (Application Centric Infrastructure) is a software-defined
networking solution that uses a policy-driven model to automate provisioning [Doc 1].
It uses a Leaf-Spine topology where all endpoint groups communicate through
contracts, managed by the APIC controller [Doc 1].

Example 2 — Answer NOT in context:
User: What is the default VLAN on a Nexus 9000?
Assistant: NOT IN PROVIDED CONTEXT

Example 3 — Contradictory sources:
User: Does ACI support multi-site deployment?
Assistant: The documents present different information on this. [Doc 2] states
ACI Multi-Pod connects geographic locations via IPN with VXLAN. However, [Doc 4]
refers to Multi-Site as a separate product requiring Nexus Dashboard Orchestrator.
Both answers may be correct for different product versions.

"""

    # ── Assemble messages list ────────────────────────────────────────────────
    # WHY context + few-shot in the FIRST user message (not system):
    #   Anthropic's API takes system prompt separately.
    #   The context documents go into the first user message so the model
    #   sees them as "input to process," not "background knowledge."
    messages = []

    # Add conversation history first (oldest turns)
    for turn in conversation_hist:
        messages.append(turn)

    # Add the current query with context
    user_content = (
        f"{context_block}"
        f"{few_shot_block}"
        f"\n{'─'*50}\n"
        f"QUESTION: {user_query}"
    )
    messages.append({"role": "user", "content": user_content})

    return system_template, messages


# ─── Live Comparison: Minimal vs Strict Grounding ─────────────────────────────

def compare_prompt_strategies():
    """
    Compare minimal vs strict grounding prompts on the same RAG context.
    Shows how the system prompt controls hallucination vs faithfulness.
    """

    # Simulated retrieved docs — intentionally missing some info
    # so we can test what happens when answer is NOT in context
    retrieved_docs = [
        {
            "id":      "Doc 1",
            "source":  "cisco_aci_guide.pdf, p.12",
            "score":   0.91,
            "content": (
                "Cisco ACI uses a Leaf-Spine topology. Every endpoint group (EPG) "
                "communicates through contracts that define which groups can talk. "
                "The APIC (Application Policy Infrastructure Controller) is the "
                "centralized management plane for the entire ACI fabric."
            ),
        },
        {
            "id":      "Doc 2",
            "source":  "readyops_spec.pdf, p.3",
            "score":   0.78,
            "content": (
                "ReadyOps operates four AI agent classes: Health & Posture, "
                "Validation, Operational, and Stress & Adversarial. "
                "Changes are promoted from the Production-Representative environment "
                "to Live Operations only after passing validation gates."
            ),
        },
    ]

    queries = [
        {
            "label":       "Answer IS in context",
            "query":       "What are the four ReadyOps agent classes?",
        },
        {
            "label":       "Answer is NOT in context",
            "query":       "What is the maximum scale of ISE deployments?",
        },
    ]

    for query_info in queries:
        query = query_info["query"]
        label = query_info["label"]

        print(f"\n{'='*65}")
        print(f"TEST: {label}")
        print(f"Query: {query}")
        print(f"{'='*65}")

        for name, system_tmpl in [
            ("MINIMAL (bad practice)", MINIMAL_SYSTEM),
            ("STRICT GROUNDING (recommended)", STRICT_GROUNDING_SYSTEM),
        ]:
            system_prompt, messages = build_rag_prompt(
                system_template=system_tmpl,
                retrieved_docs=retrieved_docs,
                conversation_hist=[],
                user_query=query,
                include_few_shot=(name != "MINIMAL (bad practice)"),
            )

            # Count tokens before calling
            token_check = client.messages.count_tokens(
                model="claude-sonnet-4-6",
                system=system_prompt,
                messages=messages,
            )

            print(f"\n  [{name}]")
            print(f"  Tokens: {token_check.input_tokens}")
            print(f"  Response:")

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=250,
                temperature=0,   # WHY 0: maximizes faithfulness to retrieved context
                system=system_prompt,
                messages=messages,
            )

            text = response.content[0].text
            # Indent response lines
            for line in text.strip().split("\n"):
                print(f"    {line}")

        print()


# ─── Structured Output RAG ────────────────────────────────────────────────────

def structured_output_rag_demo():
    """
    Demonstrate RAG with JSON output for downstream parsing.
    Shows how tool_use can enforce structure (covered more in Lesson 2).
    """

    print("\n" + "=" * 65)
    print("STRUCTURED OUTPUT RAG (JSON response format)")
    print("=" * 65)

    retrieved_docs = [
        {
            "id":      "Doc 1",
            "source":  "nexus_switching.pdf, p.4",
            "score":   0.88,
            "content": (
                "The Cisco Nexus 9000 series supports VXLAN EVPN fabric for "
                "multi-tenant data center deployments. It uses BGP EVPN as the "
                "control plane and VXLAN as the data plane overlay. Spine switches "
                "run BGP route reflectors while leaf switches peer to the spines."
            ),
        }
    ]

    query = "How does Nexus 9000 implement multi-tenant networking?"

    system_prompt, messages = build_rag_prompt(
        system_template=STRUCTURED_OUTPUT_SYSTEM,
        retrieved_docs=retrieved_docs,
        conversation_hist=[],
        user_query=query,
        include_few_shot=False,   # JSON system already defines output format
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        temperature=0,
        system=system_prompt,
        messages=messages,
    )

    print(f"\n  Query: {query}")
    print(f"\n  Raw JSON response:")
    for line in response.content[0].text.strip().split("\n"):
        print(f"    {line}")

    # Try to parse — in production, you'd validate this
    import json
    try:
        parsed = json.loads(response.content[0].text)
        print(f"\n  Parsed confidence: {parsed.get('confidence')}")
        print(f"  Evidence count:    {len(parsed.get('evidence', []))}")
        print(f"  Missing info:      {parsed.get('missing_info') or 'None reported'}")
    except json.JSONDecodeError:
        print("\n  ⚠ Model did not return valid JSON — add retry logic in production")


# ─── Citation Format Comparison (no API needed) ───────────────────────────────

def show_citation_formats():
    """
    Print the citation format comparison table.
    """
    print("\n" + "=" * 65)
    print("CITATION FORMAT COMPARISON")
    print("=" * 65)

    for fmt_name, fmt in CITATION_FORMATS.items():
        print(f"\n  [{fmt_name}]")
        print(f"  {fmt['description']}")
        print(f"  Best for: {fmt['best_for']}")
        print(f"  Pro: {fmt['pro']}")
        print(f"  Con: {fmt['con']}")
        print(f"  Example:")
        for line in fmt["example"].split("\n"):
            print(f"    {line}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("PROMPT ENGINEERING FOR RAG")
    print("=" * 65)

    show_citation_formats()

    print("\n\n" + "─" * 65)
    print("Comparing Minimal vs Strict Grounding System Prompts...")
    print("─" * 65)
    compare_prompt_strategies()

    structured_output_rag_demo()
