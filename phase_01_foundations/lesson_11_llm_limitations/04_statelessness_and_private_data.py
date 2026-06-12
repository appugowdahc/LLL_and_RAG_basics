"""
FILE: 04_statelessness_and_private_data.py
LESSON: Phase 1 - Lesson 11 - LLM Limitations
TOPIC: Statelessness and private data — why LLMs need external memory

WHAT THIS FILE TEACHES:
  - What statelessness means: each LLM call has zero memory of prior calls
  - The four types of "memory" an AI system needs
  - How to implement each memory type for a production RAG system
  - Why private data (internal docs, configs, incidents) is permanently missing
  - Session context management: how to pass history without blowing the context window
  - WHY RAG is the private data solution, not fine-tuning

INSTALL: no external dependencies
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import hashlib
import json


# ─── The Four Types of AI Memory ──────────────────────────────────────────────

def explain_memory_types():
    """
    The four types of memory in AI systems. LLMs provide only one natively.
    Understanding this clarifies which external systems you need to build.
    """

    print("=" * 72)
    print("FOUR TYPES OF AI MEMORY: What LLMs Have vs What RAG Adds")
    print("=" * 72)

    memory_types = [
        {
            "type":    "Semantic memory",
            "what":    "General world knowledge and concepts",
            "llm":     "YES (parametric) — but stale and cannot be updated",
            "rag":     "EXTENDS — retrieval adds current, domain-specific knowledge",
            "example": "What is BGP? How does VXLAN work?",
            "storage": "LLM weights (frozen) + knowledge base (updateable)",
        },
        {
            "type":    "Episodic memory",
            "what":    "History of past events, conversations, incidents",
            "llm":     "NO — each call is stateless; prior conversations are forgotten",
            "rag":     "YES — retrieve from conversation history store or incident log",
            "example": "What did we discuss last week? What caused the June 5 outage?",
            "storage": "Conversation database + indexed incident reports",
        },
        {
            "type":    "Procedural memory",
            "what":    "How to do things: runbooks, SOPs, playbooks",
            "llm":     "PARTIAL — general procedures from training; org-specific: NO",
            "rag":     "YES — retrieve org-specific runbooks from private knowledge base",
            "example": "What is our change window procedure? How do we escalate P1?",
            "storage": "Private knowledge base (indexed runbooks and SOPs)",
        },
        {
            "type":    "Working memory",
            "what":    "Current task context: active conversation, live system state",
            "llm":     "YES — the current context window IS working memory",
            "rag":     "EXTENDS — inject retrieved chunks into active context",
            "example": "Current EPG state, live alert details, active ticket context",
            "storage": "Context window (volatile) + real-time retrieval from live systems",
        },
    ]

    for mt in memory_types:
        print(f"\n  ── {mt['type'].upper()} ──")
        print(f"  What it is:  {mt['what']}")
        print(f"  LLM native:  {mt['llm']}")
        print(f"  With RAG:    {mt['rag']}")
        print(f"  Example:     {mt['example']}")
        print(f"  Storage:     {mt['storage']}")

    print(f"""
  ARCHITECTURE IMPLICATION:
    A production RAG system is NOT just a vector database.
    It is a full memory infrastructure:

      ┌─────────────────────────────────────────────────────────┐
      │  Query                                                  │
      │    │                                                    │
      │    ├──► Vector DB (semantic memory)                     │
      │    ├──► Conversation store (episodic memory)            │
      │    ├──► Runbook index (procedural memory)               │
      │    └──► Live system API (working memory refresh)        │
      │                    │                                    │
      │              Context assembly                           │
      │                    │                                    │
      │               LLM call                                  │
      └─────────────────────────────────────────────────────────┘
""")


# ─── Stateless Call Demonstration ─────────────────────────────────────────────

@dataclass
class Message:
    role:    str   # "user" | "assistant"
    content: str


class StatelessConversationManager:
    """
    Demonstrates statelessness and the explicit history management required.

    WHY this class:
      Every LLM call is independent. "Memory" is the history you pass in.
      This class shows exactly what gets sent to the API per turn.
    """

    def __init__(self, max_history_chars: int = 4000):
        self.history:              list[Message] = []
        self.max_history_chars     = max_history_chars
        self._chars_sent_total:    int = 0

    def add_user(self, content: str):
        self.history.append(Message("user", content))

    def add_assistant(self, content: str):
        self.history.append(Message("assistant", content))

    def get_context_for_api(self) -> list[dict]:
        """
        Build the messages list to send to the API.
        Trims old messages if history exceeds the character budget.

        WHY this matters:
          Without trimming, a long conversation eventually overflows the context window.
          We must choose WHAT to keep: recent messages (recency), or most relevant (semantic).
          Here: simplest approach — keep most recent until budget is met.
        """
        messages = []
        total    = 0
        # WHY reversed: keep most recent messages; drop oldest when budget is full
        for msg in reversed(self.history):
            chars = len(msg.content)
            if total + chars > self.max_history_chars:
                break
            messages.insert(0, {"role": msg.role, "content": msg.content})
            total += chars

        self._chars_sent_total += total
        return messages

    def display_state(self, turn: int):
        context = self.get_context_for_api()
        total_h = sum(len(m.content) for m in self.history)
        total_c = sum(len(m["content"]) for m in context)

        print(f"\n  Turn {turn}: history={len(self.history)} msgs ({total_h} chars), "
              f"sent to API={len(context)} msgs ({total_c} chars)")
        for m in context:
            preview = m["content"][:50].replace("\n", " ")
            print(f"    [{m['role']:<10}] '{preview}...'")


def stateless_demo():
    """
    Walk through a multi-turn conversation, showing what gets sent each turn.
    """

    print("=" * 72)
    print("STATELESSNESS: What the LLM Actually Receives Per Turn")
    print("=" * 72)

    mgr = StatelessConversationManager(max_history_chars=400)

    turns = [
        ("user",      "What is the minimum APIC node count for ACI HA?"),
        ("assistant", "The APIC cluster requires a minimum of 3 nodes for HA."),
        ("user",      "And what is the maximum number of leaf switches per pod in ACI 6.0?"),
        ("assistant", "ACI 6.0 supports up to 200 leaf switches per pod."),
        ("user",      "What about spine switch count?"),
        ("assistant", "ACI 6.0 supports up to 20 spine switches per pod."),
        ("user",      "Can you summarize the scale limits you mentioned?"),
        # WHY this last query is interesting: the word "mentioned" assumes memory.
        # But the model only has what we EXPLICITLY pass. If history was trimmed,
        # it no longer "mentioned" those facts — it doesn't remember them.
    ]

    print("\n  Adding messages to conversation history...")
    for i, (role, content) in enumerate(turns):
        if role == "user":
            mgr.add_user(content)
            mgr.display_state(turn=i // 2 + 1)
        else:
            mgr.add_assistant(content)

    print(f"""
  CRITICAL OBSERVATION:
    Turn 4 ("summarize what you mentioned") assumes the LLM remembers
    what it said in Turns 1 and 2. But if history was trimmed due to budget,
    the model only has Turns 3+ in its context. It will:
      a) Hallucinate the earlier facts (most common), OR
      b) Say "I don't have that information available"

    SOLUTION: Never rely on implicit model "memory".
    If you need Turn 1 data at Turn 4, either:
      1. Keep ALL history in context (expensive).
      2. Summarize past turns (Lesson 6: context compression).
      3. Store key facts in a session KV store and retrieve them explicitly.
""")


# ─── Private Data — Always Missing ───────────────────────────────────────────

def private_data_gap_analysis():
    """
    Show why private/internal data is permanently missing from LLMs
    and why RAG (not fine-tuning) is the right solution.
    """

    print("=" * 72)
    print("PRIVATE DATA: Permanently Missing and Why RAG > Fine-Tuning")
    print("=" * 72)

    print(f"""
  WHAT IS PRIVATE DATA?
    Any information that was never in the public training corpus:
      - Your ACI fabric topology and naming conventions
      - Your customers' device inventories
      - Your internal runbooks (change management, escalation)
      - Your incident history and post-mortem reports
      - Your SLAs and contractual obligations
      - Your proprietary configurations and templates

  WHY NOT FINE-TUNING?
    Fine-tuning is training the model on your private data.
    This seems like the obvious solution. It is NOT.

    ┌──────────────────────────────────────────────────────────────────┐
    │  Comparison: Fine-Tuning vs RAG for Private Data                 │
    │                                                                  │
    │                    FINE-TUNING     RAG                           │
    │  Update latency:   Days-weeks      Minutes                       │
    │  Update cost:      $100-$10K/run   $0 (just index)               │
    │  Data freshness:   Stale at cutoff Always current                │
    │  Forgetting risk:  YES (catastrophic forgetting) NO              │
    │  Citation/audit:   NO (baked in)   YES (chunk attribution)       │
    │  Scope control:    NO (leaked)     YES (per-query filtering)     │
    │  Setup complexity: High            Low                           │
    └──────────────────────────────────────────────────────────────────┘

  FINE-TUNING TEACHES STYLE, NOT FACTS:
    Fine-tuning is best for teaching the model HOW to respond
    (tone, format, domain vocabulary) — not WHAT to know.
    Facts injected via fine-tuning decay rapidly due to:
      1. Catastrophic forgetting: new training overwrites old weights.
      2. Compression: exact facts are not stored verbatim in weights.
      3. Confabulation: model blends fine-tuned facts with training data.

  CORRECT ARCHITECTURE:
    Fine-tuning (optional):  Teach model to follow your response format,
                             use your naming conventions, understand domain vocab.
    RAG (required):          Provide ALL private, current, and internal knowledge.

  EXAMPLE FOR CRITERION NETWORKS:
    Fine-tune: "When asked about ReadyOps, respond with the framing:
                Validate Before You Operate. Use 'Production-Representative'
                not 'test environment'. Keep responses executive-ready."
    RAG:       Index all ReadyOps documentation, customer configs, incident
               reports, and runbooks into the knowledge base.
               Every factual claim is retrieved, not memorized.
""")


# ─── Real-Time Data ───────────────────────────────────────────────────────────

def realtime_data_gap():
    """
    Show the difference between static knowledge base and real-time data,
    and when you need real-time retrieval pipelines.
    """

    print("=" * 72)
    print("REAL-TIME DATA: When Static Knowledge Base Is Not Enough")
    print("=" * 72)

    print(f"""
  STATIC KNOWLEDGE BASE (RAG default):
    Documents indexed once and retrieved at query time.
    Content is current as of last index run.
    Freshness: depends on indexing schedule (hourly/daily/weekly).

  REAL-TIME RETRIEVAL (agentic RAG):
    Instead of indexed documents, the system calls live APIs.
    Examples:
      - APIC REST API: live fabric state (EPGs, contracts, faults)
      - Cisco PSIRT: fresh CVE advisories
      - ServiceNow: open incidents and change requests
      - Prometheus: current metric values

  WHICH DO YOU NEED?

  ┌─────────────────────────────────────────────────────────────────┐
  │  Query type                    Static KB  Real-time API         │
  │  ─────────────────────────────────────────────────────────────  │
  │  What does ACI EPG policy mean?      YES  NO                    │
  │  What faults are active right now?   NO   YES                   │
  │  What is the latest CVE?             Partial  YES               │
  │  What is our change runbook?         YES  NO                    │
  │  Is fabric leaf-101 up?              NO   YES                   │
  │  What happened in last P1 incident?  YES  NO (if indexed)       │
  └─────────────────────────────────────────────────────────────────┘

  PATTERN: ReadyOps RAG Architecture
    Static KB:   Policy docs, runbooks, historical incidents, specs.
    Real-time:   APIC faults, live EPG state, current metric values.
    Both:        Combined in context for the final LLM answer.

    "Leaf-101 has 3 active faults. [real-time]
     According to the ACI troubleshooting guide [static KB], fault F0532
     indicates a port misconfiguration. Recommended action: verify port
     channel configuration on the connected device."
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    explain_memory_types()
    print()
    stateless_demo()
    print()
    private_data_gap_analysis()
    print()
    realtime_data_gap()
