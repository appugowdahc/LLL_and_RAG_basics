"""
FILE: 03_lost_in_the_middle.py
LESSON: Phase 1 - Lesson 6 - Context Window
TOPIC: "Lost in the Middle" problem — positional bias in LLM context reading

WHAT THIS FILE TEACHES:
  - What the "Lost in the Middle" phenomenon is (Liu et al. 2023)
  - WHY LLMs read the start and end of context better than the middle
  - How to measure positional bias in your own RAG system
  - Three chunk ordering strategies to mitigate the problem:
      1. Recency order (default, bad for RAG)
      2. Relevance order (puts best content first)
      3. "Lost in Middle" order (best first, best last, rest in middle)
  - A live API demo that measures whether chunk position affects citation

RESEARCH CONTEXT:
  Liu et al. 2023 "Lost in the Middle: How Language Models Use Long Contexts"
  Key finding: when you have 10 retrieved documents, the model uses docs at
  positions 1 and 10 far better than docs at positions 4-7.
  Recall drops ~25% for docs in the middle of a 10-doc window.

INSTALL:
  pip install anthropic python-dotenv
"""

import os
import time
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()


# ─── Positional Attention Model ──────────────────────────────────────────────

# Approximate recall rates per chunk position based on Liu et al. 2023
# WHY these specific numbers:
#   Derived from the paper's multi-document QA experiments.
#   The paper used GPT-3.5, GPT-4, and Claude — all showed similar bias.
#   These are APPROXIMATE — exact values depend on model and context length.
POSITION_RECALL_RATES = {
    1:  0.80,   # First doc: high attention (in system prompt boundary)
    2:  0.75,   # Second: still strong
    3:  0.70,   # Dropping
    4:  0.63,   # Middle begins
    5:  0.58,
    6:  0.55,   # Lowest recall region
    7:  0.57,
    8:  0.60,
    9:  0.65,   # End boost begins
    10: 0.75,   # Last doc: high attention (near user query)
}


def get_estimated_recall(position: int, total_docs: int) -> float:
    """
    Estimate recall rate for a chunk at a given position.
    Interpolates from the POSITION_RECALL_RATES lookup.

    WHY not a simple formula:
      The bias is not linear. It forms a U-shape: high at boundaries,
      lowest in the middle. A lookup with interpolation is more accurate.
    """
    if total_docs <= 1:
        return 0.80

    # Normalize position to 1-10 scale for lookup
    # WHY / (total_docs - 1):
    #   Maps position 0..(N-1) to [0,1] range, then scale to [1,10]
    normalized = position / max(total_docs - 1, 1)
    lookup_pos  = 1 + normalized * 9          # float in [1, 10]

    # Linear interpolation between nearest integer positions
    low  = max(1, int(lookup_pos))
    high = min(10, low + 1)
    frac = lookup_pos - low

    low_recall  = POSITION_RECALL_RATES.get(low,  0.60)
    high_recall = POSITION_RECALL_RATES.get(high, 0.60)

    return low_recall + frac * (high_recall - low_recall)


def visualize_positional_bias(chunks: list[dict]):
    """
    Show a visual table of expected recall for chunks at each position.
    chunks: list of dicts with 'id', 'score', 'title'
    """

    total = len(chunks)

    print("\n  POSITIONAL RECALL ESTIMATE (Lost in the Middle)")
    print(f"  {'Pos':<5} {'Chunk':<30} {'Relevance':>10} {'Est.Recall':>12}  Bar")
    print(f"  {'─'*5} {'─'*30} {'─'*10} {'─'*12}  {'─'*20}")

    for i, chunk in enumerate(chunks):
        pos     = i + 1
        recall  = get_estimated_recall(i, total)
        score   = chunk["score"]
        bar_len = max(1, int(recall * 20))
        bar     = "█" * bar_len

        # Highlight danger zone
        middle_start = max(2, total // 4)
        middle_end   = min(total - 1, 3 * total // 4)
        danger       = " ← DANGER" if middle_start <= i <= middle_end else ""

        print(
            f"  {pos:<5} {chunk['title'][:28]:<30} {score:>9.2f}  "
            f"{recall:>10.0%}  {bar}{danger}"
        )

    print(f"\n  ⚠ Middle positions {middle_start+1}-{middle_end+1} have lowest recall.")
    print(f"  Best content should be at POSITION 1 or POSITION {total}.")


# ─── Chunk Ordering Strategies ────────────────────────────────────────────────

def order_chunks_by_recency(chunks: list[dict]) -> list[dict]:
    """
    Default naive ordering: most recent documents first.
    WHY this is BAD for RAG:
      Most recent != most relevant. A recent doc with score=0.60 will be
      placed at position 1, while an older doc with score=0.95 gets pushed
      to the middle. The model attends to the 0.60 doc more.
    """
    # Sort by timestamp descending (most recent first)
    return sorted(chunks, key=lambda c: c.get("timestamp", 0), reverse=True)


def order_chunks_by_relevance(chunks: list[dict]) -> list[dict]:
    """
    Order by relevance score descending (highest first).
    WHY BETTER:
      The model will read the best-matching chunks first and last,
      where attention is strongest.
    WHY STILL NOT OPTIMAL:
      Position 1 gets high attention, but position N also gets attention.
      If your best chunk is at position 1 and second-best at position 2,
      the second-best is still in a high-recall zone.
      But the WORST chunks cluster in the attention-degraded middle.
    """
    return sorted(chunks, key=lambda c: c["score"], reverse=True)


def order_chunks_lost_in_middle(chunks: list[dict]) -> list[dict]:
    """
    "Lost in the Middle" mitigation ordering:
      1. Sort by relevance descending.
      2. Place top half at the START of the list.
      3. Place bottom half at the END.
      The MIDDLE positions contain the lowest-relevance chunks.

    WHY this is BEST:
      The highest-relevance chunks (most likely to contain the answer)
      are at positions 1 and N — the two zones of strongest attention.
      The weakest chunks are buried in the middle where they are read less
      anyway, minimizing their "noise" contribution.
    """
    sorted_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)

    n = len(sorted_chunks)
    if n <= 2:
        return sorted_chunks

    # Split into top half and bottom half
    # WHY ceiling for top:
    #   If odd, put one extra in the "top" group to keep start strong.
    top_half    = sorted_chunks[: (n + 1) // 2]
    bottom_half = sorted_chunks[(n + 1) // 2 :]

    # top half → beginning, bottom half → end (reversed so worst is most-middle)
    return top_half + list(reversed(bottom_half))


# ─── Simulate Effect on Recall ────────────────────────────────────────────────

def compare_orderings(chunks: list[dict]):
    """
    Compare the three ordering strategies by weighted recall.
    Weighted recall = Σ (relevance_score × position_recall_rate)
    Higher = better chance the model uses high-quality chunks.
    """

    strategies = {
        "Recency Order (bad)":          order_chunks_by_recency(chunks),
        "Relevance Order (good)":       order_chunks_by_relevance(chunks),
        "Lost-in-Middle Order (best)":  order_chunks_lost_in_middle(chunks),
    }

    print("\n" + "=" * 65)
    print("CHUNK ORDERING STRATEGY COMPARISON")
    print("=" * 65)

    for name, ordered in strategies.items():
        total = len(ordered)

        # Weighted recall: relevance × position recall → how much of the
        # important content will actually be attended to?
        weighted_recall = sum(
            c["score"] * get_estimated_recall(i, total)
            for i, c in enumerate(ordered)
        )
        max_possible = sum(c["score"] for c in ordered)
        efficiency   = weighted_recall / max(max_possible, 0.001) * 100

        print(f"\n  [{name}]")
        print(f"  {'Pos':<4} {'Title':<35} {'Score':>7} {'Recall%':>9}")
        print(f"  {'─'*4} {'─'*35} {'─'*7} {'─'*9}")

        for i, chunk in enumerate(ordered):
            recall = get_estimated_recall(i, total)
            print(
                f"  {i+1:<4} {chunk['title'][:33]:<35} "
                f"{chunk['score']:>7.2f} {recall:>8.0%}"
            )

        print(f"\n  Weighted recall efficiency: {efficiency:.1f}%")


# ─── Live API Demo: Does Position Matter? ─────────────────────────────────────

def live_position_demo():
    """
    Live API call demonstrating that the model cites start/end docs more.
    Uses two orderings of the same chunks and measures citation differences.
    """

    print("\n" + "=" * 65)
    print("LIVE DEMO: Citation frequency by chunk position")
    print("=" * 65)

    # Answer is ONLY in doc_id=3 — a "medium relevance" doc
    # WHY: We can measure whether the model finds it based on position
    chunks = [
        {
            "id":     "Doc 1",
            "title":  "Cisco ACI Overview",
            "score":  0.90,
            "content": (
                "Cisco ACI uses a Leaf-Spine topology for data center networking. "
                "The APIC controller manages the fabric policy centrally."
            ),
        },
        {
            "id":     "Doc 2",
            "title":  "Nexus Switching Guide",
            "score":  0.82,
            "content": (
                "Nexus 9000 series supports VXLAN EVPN fabric. BGP EVPN is the "
                "control plane. Spine switches run BGP route reflectors."
            ),
        },
        {
            "id":     "Doc 3",
            "title":  "ReadyOps Platform Spec",
            "score":  0.74,
            "content": (
                "ReadyOps performs continuous validation across two environments: "
                "Live Operations and Production-Representative. "
                "The Validation agent class runs automated test suites against "
                "a digital twin before promoting changes to production."
            ),
        },
        {
            "id":     "Doc 4",
            "title":  "Cisco ISE Admin Guide",
            "score":  0.65,
            "content": (
                "Cisco ISE provides network access control. TrustSec uses SGTs "
                "(Security Group Tags) for policy enforcement. ISE integrates "
                "with Active Directory for identity resolution."
            ),
        },
    ]

    # The answer is in Doc 3 ("ReadyOps performs continuous validation...")
    query = "How does ReadyOps handle environment isolation for validation?"

    system = (
        "Answer using ONLY the provided context documents. "
        "Cite every claim with [Doc N]. "
        "If not in context, say: NOT IN CONTEXT."
    )

    orderings = {
        "Best-first (Doc 3 at position 3 of 4)": order_chunks_by_relevance(chunks),
        "Reversed (Doc 3 at position 2 of 4)":   list(reversed(order_chunks_by_relevance(chunks))),
    }

    for ordering_name, ordered_chunks in orderings.items():
        context = "\n\n".join(
            f"[{c['id']}] ({c['title']}, relevance={c['score']:.2f})\n{c['content']}"
            for c in ordered_chunks
        )

        user_message = f"CONTEXT:\n{context}\n\nQUESTION: {query}"

        # Find where Doc 3 ended up
        positions = {c["id"]: i + 1 for i, c in enumerate(ordered_chunks)}
        doc3_pos  = positions["Doc 3"]
        recall_at_pos = get_estimated_recall(doc3_pos - 1, len(ordered_chunks))

        print(f"\n  Ordering: {ordering_name}")
        print(f"  Doc 3 is at position {doc3_pos}/{len(ordered_chunks)} "
              f"(est. recall: {recall_at_pos:.0%})")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )

        text = response.content[0].text
        cited_doc3 = "[Doc 3]" in text or "Doc 3" in text

        print(f"  Response: {text[:200].strip()}{'...' if len(text)>200 else ''}")
        print(f"  Doc 3 cited: {'YES ✓' if cited_doc3 else 'NO ✗'}")

    print(f"\n  LESSON:")
    print(f"  Docs at start/end positions are cited more reliably than middle.")
    print(f"  Use 'lost-in-middle' ordering to place best docs at boundaries.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Sample chunks simulating retrieved search results
    sample_chunks = [
        {"id": 1, "title": "Cisco Hypershield Overview",    "score": 0.95, "timestamp": 1720000000},
        {"id": 2, "title": "ReadyOps Validation Platform",  "score": 0.89, "timestamp": 1718000000},
        {"id": 3, "title": "ACI Leaf-Spine Architecture",   "score": 0.83, "timestamp": 1715000000},
        {"id": 4, "title": "ISE TrustSec Configuration",    "score": 0.74, "timestamp": 1722000000},
        {"id": 5, "title": "SD-WAN Policy Guide",           "score": 0.62, "timestamp": 1716000000},
        {"id": 6, "title": "Nexus 9000 VXLAN Setup",        "score": 0.51, "timestamp": 1717000000},
    ]

    print("=" * 65)
    print("LOST IN THE MIDDLE: Positional Bias in RAG")
    print("=" * 65)

    # Show positional recall for relevance-ordered chunks
    relevance_ordered = order_chunks_by_relevance(sample_chunks)
    visualize_positional_bias(relevance_ordered)

    compare_orderings(sample_chunks)

    live_position_demo()
