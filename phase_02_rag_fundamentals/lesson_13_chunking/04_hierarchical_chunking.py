"""
FILE: 04_hierarchical_chunking.py
LESSON: Phase 2 - Lesson 13 - Document Processing and Chunking
TOPIC: Hierarchical (Parent-Child) chunking — precision retrieval, full context

WHAT THIS FILE TEACHES:
  - The parent-child chunking pattern
  - WHY small child chunks retrieve better (precise embedding)
  - WHY large parent chunks answer better (full context for LLM)
  - How to build a two-level index
  - Summary-chunk variant: index a summary embedding, return full parent
  - Context-aware chunk: inject parent context as a prefix into child chunks
  - Real production tradeoff table: storage vs quality

INSTALL: pip install numpy
"""

import re
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─── Token Approximation ─────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def mock_embed(text: str, dims: int = 32) -> np.ndarray:
    """Deterministic mock embedding. Replace with Voyage AI in production."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.RandomState(seed)
    v    = rng.randn(dims).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ParentChunk:
    """
    A large parent chunk sent to the LLM as context.
    The parent is NOT directly indexed for retrieval.
    WHY not indexed: a 1000-token chunk embedding blurs too many topics —
    the similarity signal is weak. We use children for retrieval precision.
    """
    parent_id:   str
    content:     str
    metadata:    dict
    token_count: int = 0

    def __post_init__(self):
        self.token_count = approx_tokens(self.content)


@dataclass
class ChildChunk:
    """
    A small child chunk used for retrieval.
    The child carries a reference to its parent so that after retrieval
    we can look up the full parent context.
    """
    child_id:    str
    parent_id:   str        # WHY parent_id: the retrieval-to-context bridge
    content:     str
    metadata:    dict
    token_count: int = 0
    embedding:   Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        self.token_count = approx_tokens(self.content)
        self.embedding   = mock_embed(self.content)


# ─── Hierarchical Chunker ─────────────────────────────────────────────────────

def hierarchical_chunk(
    document_text:  str,
    source:         str,
    metadata:       dict,
    parent_tokens:  int = 800,   # WHY 800: large enough to contain a full procedure step
    child_tokens:   int = 150,   # WHY 150: small enough for precise embedding
    child_overlap:  int = 20,    # WHY 20: minimal overlap at child level
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """
    Create a two-level chunk hierarchy.

    Level 1 (parents): Large chunks covering complete logical units.
      - Sized to give the LLM enough context to answer comprehensively.
      - Not directly embedded/retrieved.
      - Stored in a lookup table keyed by parent_id.

    Level 2 (children): Small sub-chunks of each parent.
      - Sized for precise embedding (one focused topic per chunk).
      - Embedded and stored in the vector index.
      - Each carries its parent_id so the full context can be fetched.

    Retrieval flow:
      query → embed → child search → get parent_ids → fetch parents → LLM

    Args:
        document_text: Full document content.
        source:        Document identifier.
        metadata:      Base metadata dict (shared by all chunks from this doc).
        parent_tokens: Target tokens per parent chunk.
        child_tokens:  Target tokens per child chunk.
        child_overlap: Token overlap between consecutive children.

    Returns:
        (parents, children) lists.
    """

    def split_into_chunks(text: str, target_toks: int, overlap_toks: int) -> list[str]:
        """Split text into chunks of approximately target_toks tokens."""
        sentences  = re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", text)
        chunks_out = []
        current    = []
        cur_toks   = 0
        for sent in sentences:
            st = approx_tokens(sent)
            if cur_toks + st > target_toks and current:
                chunks_out.append(" ".join(current))
                # carry-over for overlap
                overlap_sents = []
                carry_toks    = 0
                for s in reversed(current):
                    carry_toks += approx_tokens(s)
                    overlap_sents.insert(0, s)
                    if carry_toks >= overlap_toks:
                        break
                current  = overlap_sents
                cur_toks = carry_toks
            current.append(sent)
            cur_toks += st
        if current:
            chunks_out.append(" ".join(current))
        return chunks_out

    # Build parents
    parent_texts = split_into_chunks(document_text, parent_tokens, overlap_toks=0)
    parents      = []
    children     = []

    for p_idx, p_text in enumerate(parent_texts):
        parent_id = f"{source}:parent:{p_idx}"
        parent    = ParentChunk(
            parent_id   = parent_id,
            content     = p_text,
            metadata    = {**metadata, "source": source, "parent_idx": p_idx},
        )
        parents.append(parent)

        # Build children from this parent's text
        child_texts = split_into_chunks(p_text, child_tokens, child_overlap)

        for c_idx, c_text in enumerate(child_texts):
            child_id = f"{source}:child:{p_idx}:{c_idx}"
            child    = ChildChunk(
                child_id  = child_id,
                parent_id = parent_id,   # WHY: this is the critical link
                content   = c_text,
                metadata  = {**metadata, "source": source, "parent_idx": p_idx, "child_idx": c_idx},
            )
            children.append(child)

    return parents, children


# ─── Hierarchical Index ───────────────────────────────────────────────────────

class HierarchicalIndex:
    """
    Two-level retrieval index implementing parent-child RAG.

    Retrieval: search children (small, precise embeddings).
    Context:   look up parent (large, full-context text).
    """

    def __init__(self):
        self._parents:  dict[str, ParentChunk] = {}
        self._children: list[ChildChunk]       = []
        self._matrix:   Optional[np.ndarray]   = None

    def build(self, parents: list[ParentChunk], children: list[ChildChunk]):
        """Index parents for lookup and children for retrieval."""
        for p in parents:
            self._parents[p.parent_id] = p

        self._children = children
        self._matrix   = np.vstack([c.embedding for c in children])
        print(f"  [Index] {len(parents)} parents, {len(children)} children indexed.")

    def search(
        self,
        query:         str,
        top_k_children: int = 5,
        deduplicate:    bool = True,
    ) -> list[tuple[ParentChunk, ChildChunk, float]]:
        """
        Search children → fetch unique parents.

        Args:
            query:           User query string.
            top_k_children:  Number of child chunks to retrieve.
            deduplicate:     If True, return each parent only once
                             (from its highest-scoring child).

        Returns:
            List of (parent, best_child, score) tuples sorted by score.
        """
        q_vec = mock_embed(query)
        qn    = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        scores= np.dot(self._matrix, qn)
        top   = np.argsort(-scores)[:top_k_children]

        results: dict[str, tuple[ParentChunk, ChildChunk, float]] = {}

        for idx in top:
            child = self._children[idx]
            score = float(scores[idx])
            pid   = child.parent_id

            # WHY deduplicate by parent:
            #   Multiple children may map to the same parent.
            #   We only need the parent once — send it to the LLM once.
            #   Keep the highest-scoring child as the "match evidence".
            if deduplicate:
                if pid not in results or score > results[pid][2]:
                    parent = self._parents.get(pid)
                    if parent:
                        results[pid] = (parent, child, score)
            else:
                parent = self._parents.get(pid)
                if parent:
                    results[f"{pid}:{idx}"] = (parent, child, score)

        return sorted(results.values(), key=lambda x: -x[2])

    def stats(self):
        child_toks  = [c.token_count for c in self._children]
        parent_toks = [p.token_count for p in self._parents.values()]
        print(f"  Children: {len(self._children)} | avg {np.mean(child_toks):.0f} tokens")
        print(f"  Parents:  {len(self._parents)}  | avg {np.mean(parent_toks):.0f} tokens")
        print(f"  Storage ratio: {len(self._children)/max(len(self._parents),1):.1f} children per parent")


# ─── Context-Aware Chunk Variant ──────────────────────────────────────────────

def make_context_aware_chunks(
    parents:  list[ParentChunk],
    children: list[ChildChunk],
) -> list[ChildChunk]:
    """
    Context-aware chunking: prefix each child with its parent's opening text.

    WHY context-aware:
      A child chunk might be: "This requires 100% pass rate."
      Without context, the LLM doesn't know what "this" refers to.
      With the parent prefix: "ReadyOps validation gate [context]. This requires 100% pass rate."
      The child now carries its own semantic content PLUS a context anchor.

    TRADEOFF:
      Each child is now larger (adding ~30–50 tokens of prefix).
      The embedding is still dominated by the child's own content.
      But the LLM, which receives the child-with-prefix, has full context.

    Note: This is an alternative to returning the full parent. Use when:
      - Storage is constrained (parents are very large).
      - You want the embedding to remain close to the child's content.
    """
    parent_map = {p.parent_id: p for p in parents}
    result     = []

    for child in children:
        parent = parent_map.get(child.parent_id)
        if parent:
            # Use first sentence of parent as context prefix
            first_sent = re.split(r"(?<=[.!?])\s+", parent.content.strip())[0]
            # WHY first sentence: establishes the topic without overwhelming the child content
            context_prefix = f"[Context: {first_sent}]\n"
            new_content    = context_prefix + child.content
        else:
            new_content = child.content

        result.append(ChildChunk(
            child_id  = child.child_id + ":ctx",
            parent_id = child.parent_id,
            content   = new_content,
            metadata  = child.metadata,
        ))

    return result


# ─── Demo ─────────────────────────────────────────────────────────────────────

SAMPLE_DOC = """
ReadyOps is Criterion Networks' continuous validation platform for network infrastructure.
It operates across two deliberately isolated environments: Production-Representative and
Live Operations. These environments share one intent model but never cross the wire.

The Production-Representative environment serves as the validation sandbox. It can be
a digital twin, a physical lab, or a hybrid combination. All validation tests run here
first. The environment must faithfully replicate the production fabric topology, policy
model, and connected workloads.

A formal promotion gate separates Production-Representative from Live Operations.
The gate opens only when all Validation agent tests pass at 100%. Partial pass rates
are not accepted — even for emergency changes. This strict gate is the core of the
ReadyOps design philosophy: Validate Before You Operate.

ReadyOps agent classes each have a distinct role. Health and Posture agents continuously
monitor baseline state and detect configuration drift. Validation agents run
pre-change compliance and connectivity tests. Operational agents execute approved
runbooks automatically. Stress and Adversarial agents perform resilience testing
including fault injection and load simulation.

Integration with Cisco ACI: ReadyOps consumes APIC policy snapshots to build the
digital twin. It validates that EPG contracts, tenant policies, and VLAN configurations
are consistent between Production-Representative and Live Operations. Any discrepancy
blocks promotion until resolved.
""".strip()


def run_hierarchical_demo():
    """Demonstrate hierarchical chunking and retrieval."""

    print("=" * 70)
    print("HIERARCHICAL CHUNKING: Precision Retrieval + Full Context")
    print("=" * 70)

    parents, children = hierarchical_chunk(
        document_text = SAMPLE_DOC,
        source        = "readyops_guide",
        metadata      = {"product": "ReadyOps", "date": "2025-06-01"},
        parent_tokens = 250,
        child_tokens  = 80,
        child_overlap = 10,
    )

    print(f"\n  Document: ~{approx_tokens(SAMPLE_DOC)} tokens")
    print(f"\n  Parents ({len(parents)} total):")
    for p in parents:
        print(f"    [{p.parent_id}] {p.token_count} tokens: '{p.content[:70]}...'")

    print(f"\n  Children ({len(children)} total):")
    for c in children:
        print(f"    [{c.child_id}] parent={c.parent_id} {c.token_count} tok: '{c.content[:60]}...'")

    # Build and query the index
    print(f"\n  {'─'*60}")
    print(f"  RETRIEVAL DEMO")
    print(f"  {'─'*60}")

    index = HierarchicalIndex()
    index.build(parents, children)
    index.stats()

    queries = [
        "What is the validation pass rate requirement?",
        "How does ReadyOps integrate with ACI?",
    ]

    for q in queries:
        print(f"\n  Query: '{q}'")
        results = index.search(q, top_k_children=3)

        for i, (parent, child, score) in enumerate(results[:2], 1):
            print(f"\n    [{i}] score={score:.3f}")
            print(f"    Match (child, {child.token_count} tok): '{child.content[:70]}...'")
            print(f"    → Returns parent ({parent.token_count} tok): '{parent.content[:100]}...'")

    # Show context-aware variant
    print(f"\n  {'─'*60}")
    print(f"  CONTEXT-AWARE CHUNKS (child + parent prefix)")
    print(f"  {'─'*60}")
    ctx_children = make_context_aware_chunks(parents, children[:3])
    for c in ctx_children:
        print(f"\n  [{c.child_id}]")
        print(f"  '{c.content[:150]}'")


def tradeoff_table():
    """Print a tradeoff comparison of chunking strategies."""

    print("\n" + "=" * 70)
    print("CHUNKING STRATEGY SELECTION GUIDE")
    print("=" * 70)

    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Strategy           Retrieval  Context  Speed   Storage  Best for   │
  ├─────────────────────────────────────────────────────────────────────┤
  │ Fixed-size char    ★★☆☆☆     ★★☆☆☆   ★★★★★  ★★★★★  Logs, baseline │
  │ Fixed-size token   ★★★☆☆     ★★★☆☆   ★★★★☆  ★★★★★  Multilingual   │
  │ Sentence-aware     ★★★★☆     ★★★★☆   ★★★★☆  ★★★★☆  Prose docs     │
  │ Paragraph-aware    ★★★★☆     ★★★★★   ★★★★☆  ★★★★☆  Guides, FAQs   │
  │ Recursive          ★★★★☆     ★★★★☆   ★★★☆☆  ★★★★☆  Markdown,code  │
  │ Semantic           ★★★★★     ★★★★☆   ★★☆☆☆  ★★★☆☆  Topic-diverse  │
  │ Hierarchical       ★★★★★     ★★★★★   ★★★☆☆  ★★☆☆☆  Long tech docs │
  └─────────────────────────────────────────────────────────────────────┘

  DECISION FLOW:
    Is the document < 500 tokens?              → No chunking needed
    Is it a code file?                         → Recursive (Python separators)
    Is it YAML/config?                         → Recursive (YAML separators)
    Is it Markdown with headings?              → Recursive (Markdown separators)
    Is it long prose (>10 pages)?              → Hierarchical
    Is it short prose (1–5 pages)?             → Paragraph-aware
    Is it a log or transcript?                 → Fixed-size token
    Is content topic-diverse within sections?  → Semantic
    Need a fast baseline?                      → Fixed-size char
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_hierarchical_demo()
    tradeoff_table()
