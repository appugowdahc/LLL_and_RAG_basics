"""
FILE: 02_semantic_chunking.py
LESSON: Phase 2 - Lesson 13 - Document Processing and Chunking
TOPIC: Sentence-aware and semantic chunking

WHAT THIS FILE TEACHES:
  - Sentence-aware chunking: group sentences until token budget is reached
  - Paragraph-aware chunking: respect double-newline boundaries
  - Semantic chunking: embed adjacent sentences, cut where similarity drops
  - WHY semantic chunking produces topic-coherent chunks
  - The tradeoff: semantic chunking requires embedding at index time (slow)
  - Choosing between sentence-aware and semantic chunking

INSTALL: pip install numpy  (voyageai optional for real semantic chunking)
"""

import re
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─── Token Approximation ─────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ─── Sentence Splitter ────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.
    WHY not split("."):
      "APIC v6.0 supports..." would split at the period in "v6.0".
      Our regex requires at least two characters before the delimiter.
      Also handles !? and abbreviations reasonably well.
    """
    # WHY (?<=[.!?])\s+:
    #   Lookbehind: split AFTER .!? followed by whitespace.
    #   This keeps the punctuation with the sentence it ends.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", text)
    return [s.strip() for s in sentences if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    """
    Split text at paragraph boundaries (double newlines).
    WHY paragraphs first:
      Paragraphs are the highest-level semantic unit in most documents.
      Crossing a paragraph boundary in a chunk is worse than crossing
      a sentence boundary — paragraphs represent complete thoughts.
    """
    # WHY strip each: trailing whitespace in paragraphs is noise
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


# ─── Strategy 1: Sentence-Aware Chunking ──────────────────────────────────────

def chunk_by_sentences(
    text:           str,
    target_tokens:  int = 300,
    overlap_sents:  int = 1,       # number of sentences to repeat in next chunk
    source:         str = "doc",
) -> list[dict]:
    """
    Group sentences into chunks that fit within target_tokens.

    WHY sentence-level grouping:
      Each sentence is a complete thought. Grouping N complete thoughts into
      a chunk produces coherent context for the LLM — far better than cutting
      mid-sentence at a character boundary.

    WHY overlap_sents (not overlap_chars):
      A sentence is the natural unit to overlap. Repeating 1 sentence from
      the previous chunk gives the new chunk context without wasted tokens.

    Args:
        text:          Full document text.
        target_tokens: Approximate maximum tokens per chunk.
        overlap_sents: Number of trailing sentences to carry into next chunk.
        source:        Source document identifier.

    Returns:
        List of chunk dicts with content, token_count, sentence_count.
    """
    sentences = split_sentences(text)
    chunks    = []
    current   = []        # list of sentences in current chunk
    cur_toks  = 0
    idx       = 0

    for sent in sentences:
        sent_toks = approx_tokens(sent)

        # If adding this sentence would overflow the budget AND we already have content,
        # finalize the current chunk before starting a new one.
        if cur_toks + sent_toks > target_tokens and current:
            chunks.append({
                "chunk_id":      f"{source}:sent:{idx}",
                "content":       " ".join(current),
                "token_count":   approx_tokens(" ".join(current)),
                "sentence_count": len(current),
                "source":        source,
            })
            idx += 1

            # WHY carry-over: the last overlap_sents sentences form the bridge
            # to the next chunk so boundary context is not lost.
            current  = current[-overlap_sents:] if overlap_sents > 0 else []
            cur_toks = sum(approx_tokens(s) for s in current)

        current.append(sent)
        cur_toks += sent_toks

    # Flush the last chunk
    if current:
        chunks.append({
            "chunk_id":       f"{source}:sent:{idx}",
            "content":        " ".join(current),
            "token_count":    approx_tokens(" ".join(current)),
            "sentence_count": len(current),
            "source":         source,
        })

    return chunks


# ─── Strategy 2: Paragraph-Aware Chunking ─────────────────────────────────────

def chunk_by_paragraphs(
    text:           str,
    target_tokens:  int = 400,
    max_tokens:     int = 600,    # hard ceiling — long paragraphs get split
    source:         str = "doc",
) -> list[dict]:
    """
    Group paragraphs into chunks, respecting max_tokens as a hard ceiling.

    WHY paragraph-aware:
      Paragraphs are pre-existing semantic units created by the author.
      Respecting them produces chunks that each cover one complete idea.
      A runbook step or a FAQ entry typically fits in one paragraph.

    WHY max_tokens hard ceiling:
      Some paragraphs are very long (multi-page legal text, verbose guides).
      If a paragraph exceeds max_tokens, fall back to sentence splitting
      within that paragraph rather than producing an oversized chunk.
    """
    paragraphs = split_paragraphs(text)
    chunks     = []
    current    = []
    cur_toks   = 0
    idx        = 0

    for para in paragraphs:
        para_toks = approx_tokens(para)

        # If the paragraph alone exceeds max_tokens, split it by sentences
        if para_toks > max_tokens:
            # Flush current before splitting the long paragraph
            if current:
                chunks.append({
                    "chunk_id":    f"{source}:para:{idx}",
                    "content":     "\n\n".join(current),
                    "token_count": approx_tokens("\n\n".join(current)),
                    "source":      source,
                })
                idx    += 1
                current = []
                cur_toks = 0

            # WHY recursive sentence split: reuse sentence chunker for oversized paragraphs
            sub_chunks = chunk_by_sentences(para, target_tokens=target_tokens, source=f"{source}:p{idx}")
            chunks.extend(sub_chunks)
            continue

        # If adding this paragraph would overflow, finalize current chunk
        if cur_toks + para_toks > target_tokens and current:
            chunks.append({
                "chunk_id":    f"{source}:para:{idx}",
                "content":     "\n\n".join(current),
                "token_count": approx_tokens("\n\n".join(current)),
                "source":      source,
            })
            idx     += 1
            current  = []
            cur_toks = 0

        current.append(para)
        cur_toks += para_toks

    if current:
        chunks.append({
            "chunk_id":    f"{source}:para:{idx}",
            "content":     "\n\n".join(current),
            "token_count": approx_tokens("\n\n".join(current)),
            "source":      source,
        })

    return chunks


# ─── Strategy 3: Semantic Chunking ───────────────────────────────────────────

def mock_sentence_embedding(sentence: str, dims: int = 32) -> np.ndarray:
    """
    Deterministic mock embedding for a sentence.
    WHY SHA-256: reproducible without API keys.
    In production: voyageai.Client().embed([sentence], input_type="document")[0]
    """
    seed = int(hashlib.md5(sentence.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.RandomState(seed)
    v    = rng.randn(dims).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit vectors."""
    return float(np.dot(a, b))


def chunk_semantically(
    text:              str,
    threshold:         float = 0.4,   # WHY 0.4: similarity below this = topic change
    min_chunk_tokens:  int   = 80,    # WHY min: don't create tiny semantic fragments
    max_chunk_tokens:  int   = 500,   # WHY max: prevent oversized chunks
    source:            str   = "doc",
) -> list[dict]:
    """
    Semantic chunking: cut when adjacent sentence embeddings diverge.

    Algorithm:
      1. Split into sentences.
      2. Embed each sentence.
      3. Compute cosine similarity between sentence[i] and sentence[i+1].
      4. Where similarity < threshold, insert a chunk boundary.
      5. Merge fragments that are too small; split groups that are too large.

    WHY similarity-based boundaries:
      The author's topic transitions are the natural chunk boundaries.
      A paragraph about EPGs followed by a paragraph about contracts has
      low embedding similarity → semantic chunking inserts a boundary there.
      Fixed-size chunking might cut mid-EPG-paragraph instead.

    WHY threshold=0.4 (with mock embeddings):
      With real embeddings (Voyage AI, OpenAI), 0.3–0.5 is the typical range.
      Lower = fewer, larger chunks. Higher = many tiny chunks.
      In production: tune on a held-out set of your documents.

    NOTE: This implementation uses mock embeddings.
      Real semantic chunking requires calling an embedding API per sentence —
      which means indexing is slower (N sentences × embedding latency).
      The quality payoff is significant for topic-diverse documents.
    """
    sentences   = split_sentences(text)
    if not sentences:
        return []

    # Embed every sentence
    embeddings  = [mock_sentence_embedding(s) for s in sentences]

    # Compute similarity between adjacent sentences
    similarities = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # Find boundary positions: indices where similarity < threshold
    boundaries = {0}   # always start a chunk at position 0
    for i, sim in enumerate(similarities):
        if sim < threshold:
            boundaries.add(i + 1)   # WHY i+1: sentence i+1 starts a new topic

    # Build chunks from boundaries
    boundary_list = sorted(boundaries)
    boundary_list.append(len(sentences))   # sentinel end

    chunks = []
    idx    = 0

    for b_start, b_end in zip(boundary_list[:-1], boundary_list[1:]):
        content   = " ".join(sentences[b_start:b_end])
        tok_count = approx_tokens(content)

        # Enforce min/max token constraints
        if tok_count < min_chunk_tokens and chunks:
            # Merge with previous chunk if too small
            prev          = chunks[-1]
            merged        = prev["content"] + " " + content
            prev["content"]     = merged
            prev["token_count"] = approx_tokens(merged)
            continue

        if tok_count > max_chunk_tokens:
            # Fall back to sentence chunking for oversized semantic groups
            sub = chunk_by_sentences(content, target_tokens=max_chunk_tokens // 2, source=f"{source}:sem:{idx}")
            chunks.extend(sub)
            idx += len(sub)
            continue

        sim_at_boundary = similarities[b_start - 1] if b_start > 0 else 1.0

        chunks.append({
            "chunk_id":          f"{source}:sem:{idx}",
            "content":           content,
            "token_count":       tok_count,
            "sentence_count":    b_end - b_start,
            "topic_break_score": 1.0 - sim_at_boundary,  # higher = stronger topic change at boundary
            "source":            source,
        })
        idx += 1

    return chunks


# ─── Comparison Demo ──────────────────────────────────────────────────────────

SAMPLE_DOC = """
Cisco ACI uses a Leaf-Spine topology for its data center fabric. Every leaf switch connects
to every spine switch, providing full-mesh redundancy without Spanning Tree Protocol.
The fabric uses VXLAN for the overlay protocol, which allows Layer 2 extension across Layer 3 boundaries.

The APIC controller is the policy management system for ACI. It stores the entire fabric policy model
and distributes policy to leaf switches using OpFlex. The APIC cluster requires three nodes for HA.
When one APIC node fails, the remaining two maintain quorum and continue managing the fabric.

ReadyOps is Criterion Networks' continuous validation platform. It operates across two isolated
environments: Production-Representative and Live Operations. Changes must pass a 100% validation
gate before being promoted to the production fabric.

The ReadyOps Validation agent class runs pre-change tests against the Production-Representative
environment. Only after all tests pass is the promotion gate opened. This ensures that every
change to the Live Operations fabric has been tested against a faithful digital twin.

Security Group Tags (SGTs) are assigned by Cisco ISE at authentication time using TrustSec.
SGTs travel with the packet and are enforced at the ingress of each network device. This provides
identity-based microsegmentation that follows the user, not the network topology.
""".strip()


def run_comparison():
    """Compare all three semantic-aware strategies on the same document."""

    print("=" * 70)
    print("SEMANTIC CHUNKING STRATEGIES: Sentence, Paragraph, Semantic")
    print("=" * 70)
    print(f"\n  Document: ~{approx_tokens(SAMPLE_DOC)} tokens")

    strategies = [
        ("Sentence-aware (300 tok target)", chunk_by_sentences(SAMPLE_DOC, target_tokens=300, source="aci")),
        ("Paragraph-aware (400 tok target)", chunk_by_paragraphs(SAMPLE_DOC, target_tokens=400, source="aci")),
        ("Semantic (threshold=0.4)",         chunk_semantically(SAMPLE_DOC, threshold=0.4, source="aci")),
    ]

    for name, chunks in strategies:
        tok_counts = [c["token_count"] for c in chunks]
        print(f"\n  [{name}]")
        print(f"    Chunks: {len(chunks)}")
        print(f"    Tokens — mean: {np.mean(tok_counts):.0f}, std: {np.std(tok_counts):.0f}, "
              f"range: {min(tok_counts)}–{max(tok_counts)}")
        print(f"    First chunk preview:")
        print(f"      '{chunks[0]['content'][:120]}...'")

    # Show semantic similarity profile
    print(f"\n  SEMANTIC SIMILARITY PROFILE (adjacent sentences):")
    sentences   = split_sentences(SAMPLE_DOC)
    embeddings  = [mock_sentence_embedding(s) for s in sentences]
    similarities = [cosine_similarity(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]

    print(f"  {'Sent pair':<12} {'Similarity':>12}  {'Signal'}")
    print(f"  {'─'*12} {'─'*12}  {'─'*30}")
    for i, sim in enumerate(similarities):
        signal = "▓▓▓ TOPIC BREAK" if sim < 0.4 else ("▒▒  transition" if sim < 0.6 else "░   same topic")
        print(f"  {i+1}→{i+2:<9}  {sim:>12.3f}  {signal}")

    print(f"""
  INSIGHT:
    Semantic chunking identifies topic breaks automatically.
    With real embeddings (not mock), low-similarity pairs correspond to
    actual topic transitions (e.g., switching from ACI topology to APIC policy).
    This produces chunks that are coherent single-topic units — the ideal
    retrieval target for an embedding-based search system.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_comparison()
