"""
FILE: 01_fixed_size_chunking.py
LESSON: Phase 2 - Lesson 13 - Document Processing and Chunking
TOPIC: Fixed-size chunking — the baseline strategy

WHAT THIS FILE TEACHES:
  - Character-based vs token-based splitting (and why they differ)
  - Why overlap is necessary and how to compute it
  - How fixed-size chunking breaks sentence boundaries
  - Measuring chunk quality: size distribution, boundary quality score
  - WHY fixed-size is a useful baseline even though it's not optimal
  - When fixed-size chunking is actually the right choice

INSTALL: pip install numpy  (tiktoken optional for accurate token counts)
"""

import re
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─── Token Count Approximation ────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    """
    Approximate token count without tiktoken.
    WHY approximate: tiktoken is accurate but adds a dependency.
    Rule: 1 token ≈ 4 chars for English. For code: 1 token ≈ 3 chars.
    For production: use tiktoken.encode(text) and len() the result.
    """
    return max(1, len(text) // 4)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single document chunk with provenance metadata."""
    chunk_id:      str
    content:       str
    start_char:    int    # character offset in original document
    end_char:      int
    token_count:   int

    @property
    def ends_at_sentence_boundary(self) -> bool:
        """
        Check if this chunk ends at a natural sentence boundary.
        WHY this metric: chunks ending mid-sentence degrade LLM comprehension.
        A high score here is a proxy for chunking quality.
        """
        stripped = self.content.strip()
        return bool(re.search(r"[.!?]$", stripped))

    @property
    def starts_at_sentence_boundary(self) -> bool:
        """Check if this chunk starts at a new sentence (capital letter after space/start)."""
        stripped = self.content.strip()
        return bool(re.match(r"^[A-Z\d]", stripped))


# ─── Fixed-Size Chunkers ──────────────────────────────────────────────────────

def chunk_by_characters(
    text:         str,
    chunk_size:   int = 1600,   # WHY 1600 chars ≈ 400 tokens for English
    overlap:      int = 160,    # WHY 160 chars ≈ 40 tokens (10% overlap)
    source:       str = "doc",
) -> list[Chunk]:
    """
    Split text into fixed-size character chunks with overlap.

    WHY character-based:
      Simple and fast. No dependency on a tokenizer.
      Consistent chunk sizes in characters — predictable storage.
    WHY overlap:
      Without overlap, an answer spanning a chunk boundary would be split.
      Overlap ensures the bridge between chunks is always present in one of them.

    Args:
        text:       Full document text.
        chunk_size: Maximum characters per chunk.
        overlap:    Characters repeated from the previous chunk.
        source:     Source document identifier for metadata.

    Returns:
        List of Chunk objects.
    """
    if not text.strip():
        return []

    chunks = []
    start  = 0
    idx    = 0

    while start < len(text):
        end     = min(start + chunk_size, len(text))
        content = text[start:end]

        chunks.append(Chunk(
            chunk_id    = f"{source}:char:{idx}",
            content     = content,
            start_char  = start,
            end_char    = end,
            token_count = approx_tokens(content),
        ))

        # WHY start + chunk_size - overlap:
        #   The next chunk starts overlap chars before the current end.
        #   This ensures the overlap window is always at the END of the previous
        #   chunk and BEGINNING of the next — exactly where boundary context is needed.
        start += chunk_size - overlap
        idx   += 1

    return chunks


def chunk_by_tokens(
    text:         str,
    chunk_tokens: int = 400,
    overlap_tokens: int = 40,
    source:       str = "doc",
) -> list[Chunk]:
    """
    Split text into fixed-size token chunks with overlap.

    WHY token-based over character-based:
      LLM context windows are measured in tokens, not characters.
      Token-based splitting guarantees your chunks respect the context budget.
      A 400-character Chinese string ≈ 400 tokens; same string in English ≈ 100 tokens.
      For multilingual corpora, character-based chunking produces wildly inconsistent sizes.

    Implementation note:
      Without tiktoken: approximate with chars // 4.
      With tiktoken: enc = tiktoken.get_encoding("cl100k_base"); tokens = enc.encode(text).
    """
    # Build approximate token boundaries by splitting on whitespace
    # WHY split on whitespace: each word is approximately 1–2 tokens; gives us
    # token-granularity control without requiring tiktoken.
    words  = text.split()
    chunks = []
    start  = 0     # word index
    idx    = 0

    while start < len(words):
        # Accumulate words until we hit the token budget
        end        = start
        tok_count  = 0

        while end < len(words) and tok_count < chunk_tokens:
            tok_count += max(1, len(words[end]) // 4)   # WHY //4: 1 word ≈ 1.3 tokens on avg
            end       += 1

        content    = " ".join(words[start:end])
        start_char = text.find(" ".join(words[start:start+1]))   # approximate

        chunks.append(Chunk(
            chunk_id    = f"{source}:tok:{idx}",
            content     = content,
            start_char  = 0,    # WHY 0: word-based reconstruction loses exact char offset
            end_char    = len(content),
            token_count = approx_tokens(content),
        ))

        # WHY move by (chunk_tokens - overlap_tokens) words:
        #   This is an approximation — actual word count per token varies.
        #   In production use tiktoken to find the exact word that ends at token N.
        words_per_chunk = max(1, end - start)
        overlap_words   = max(1, int(words_per_chunk * overlap_tokens / chunk_tokens))
        start           = end - overlap_words
        idx            += 1

    return chunks


# ─── Quality Metrics ──────────────────────────────────────────────────────────

@dataclass
class ChunkQualityReport:
    """Summary statistics for a list of chunks."""
    total_chunks:           int
    mean_tokens:            float
    std_tokens:             float
    min_tokens:             int
    max_tokens:             int
    pct_sentence_end:       float   # % of chunks that end at a sentence boundary
    pct_sentence_start:     float   # % of chunks that start at a sentence boundary
    boundary_quality_score: float   # combined metric (higher = better)

    def display(self, title: str = ""):
        if title:
            print(f"\n  ── {title} ──")
        print(f"  Chunks:          {self.total_chunks}")
        print(f"  Tokens (mean):   {self.mean_tokens:.1f}")
        print(f"  Tokens (std):    {self.std_tokens:.1f}")
        print(f"  Range:           {self.min_tokens}–{self.max_tokens}")
        print(f"  Sentence ends:   {self.pct_sentence_end:.0%}")
        print(f"  Sentence starts: {self.pct_sentence_start:.0%}")
        print(f"  Boundary score:  {self.boundary_quality_score:.2f} / 1.00")


def evaluate_chunks(chunks: list[Chunk]) -> ChunkQualityReport:
    """Compute quality metrics for a chunking strategy."""
    if not chunks:
        return ChunkQualityReport(0, 0, 0, 0, 0, 0, 0, 0)

    tokens     = [c.token_count for c in chunks]
    sent_ends  = sum(1 for c in chunks if c.ends_at_sentence_boundary) / len(chunks)
    sent_starts= sum(1 for c in chunks if c.starts_at_sentence_boundary) / len(chunks)
    quality    = (sent_ends + sent_starts) / 2

    return ChunkQualityReport(
        total_chunks           = len(chunks),
        mean_tokens            = float(np.mean(tokens)),
        std_tokens             = float(np.std(tokens)),
        min_tokens             = min(tokens),
        max_tokens             = max(tokens),
        pct_sentence_end       = sent_ends,
        pct_sentence_start     = sent_starts,
        boundary_quality_score = quality,
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────

SAMPLE_DOCUMENT = """
Cisco ACI uses a Leaf-Spine topology where all traffic flows through the fabric. The APIC controller
manages the entire fabric policy and provides a centralized REST API for automation. Each leaf switch
connects to every spine switch, providing full-mesh connectivity without Spanning Tree Protocol.

The APIC cluster requires a minimum of three nodes for high availability. When one APIC node fails,
the remaining two nodes maintain quorum and continue to manage fabric policy. APIC nodes communicate
over an in-band management network using the fabric itself.

Endpoint Groups (EPGs) are the fundamental unit of policy in ACI. An EPG is a logical grouping of
endpoints that share the same policy requirements. Contracts define which EPGs can communicate with
each other and what traffic is permitted. Without a contract, two EPGs cannot exchange traffic.

ACI version 6.0 introduced several improvements including enhanced Multi-Pod support, up to 200 leaf
switches per pod, and improved BGP Route Reflector scalability. The new version also includes
enhanced microsegmentation capabilities that integrate with Cisco Hypershield for workload-level
policy enforcement using eBPF technology in the host kernel.

ReadyOps validates ACI configuration changes in a Production-Representative environment before those
changes are promoted to the Live Operations fabric. This validation gate requires 100% of all
Validation agent tests to pass before promotion is allowed. The Production-Representative environment
can be a digital twin, a physical lab, or a hybrid combination of both.
""".strip()


def run_comparison():
    """
    Compare character-based and token-based fixed-size chunking on the same document.
    Shows chunk count, size distribution, and boundary quality.
    """

    print("=" * 70)
    print("FIXED-SIZE CHUNKING: Character vs Token Splitting")
    print("=" * 70)

    # Character-based
    char_chunks = chunk_by_characters(SAMPLE_DOCUMENT, chunk_size=400, overlap=40, source="aci_guide")
    char_report = evaluate_chunks(char_chunks)

    # Token-based
    tok_chunks = chunk_by_tokens(SAMPLE_DOCUMENT, chunk_tokens=100, overlap_tokens=10, source="aci_guide")
    tok_report = evaluate_chunks(tok_chunks)

    print(f"\n  Document: {approx_tokens(SAMPLE_DOCUMENT)} approx tokens, {len(SAMPLE_DOCUMENT)} chars")

    char_report.display("Character-based (400 chars, 40 overlap)")
    tok_report.display("Token-based (100 tokens, 10 overlap)")

    # Show example of a bad boundary (where fixed-size cuts mid-sentence)
    print(f"\n  EXAMPLE: Where fixed-size cutting breaks sentences")
    print(f"  {'─'*60}")
    for i, chunk in enumerate(char_chunks[:3]):
        end_ok  = "✓ sentence end"  if chunk.ends_at_sentence_boundary  else "✗ MID-SENTENCE"
        start_ok= "✓ sentence start" if chunk.starts_at_sentence_boundary else "✗ mid-sentence"
        print(f"\n  Chunk {i+1} ({chunk.token_count} tokens)  [{start_ok}] [{end_ok}]")
        print(f"  First 80 chars: '{chunk.content[:80]}'")
        print(f"  Last  80 chars: '...{chunk.content[-80:]}'")


def overlap_impact_demo():
    """
    Show why overlap prevents information loss at chunk boundaries.
    """

    print("\n" + "=" * 70)
    print("OVERLAP IMPACT: Information Loss at Boundaries")
    print("=" * 70)

    # A document where critical info spans the boundary
    doc = (
        "The APIC cluster requires three nodes for HA. "
        "Node failure handling: when node 1 fails, nodes 2 and 3 maintain quorum. "
        "This quorum mechanism ensures continuous policy management even during maintenance. "
        "The quorum threshold is N/2 + 1, so a 3-node cluster tolerates exactly 1 failure."
    )

    chunk_size = 160

    no_overlap = chunk_by_characters(doc, chunk_size=chunk_size, overlap=0,   source="test")
    w_overlap  = chunk_by_characters(doc, chunk_size=chunk_size, overlap=40,  source="test")

    print(f"\n  Query: 'What happens when an APIC node fails?'")
    print(f"  Answer lives at the boundary between chunk 1 and chunk 2.")

    print(f"\n  WITHOUT overlap ({len(no_overlap)} chunks):")
    for i, c in enumerate(no_overlap[:3]):
        print(f"    [{i+1}] '{c.content[:90]}'")

    print(f"\n  WITH 40-char overlap ({len(w_overlap)} chunks):")
    for i, c in enumerate(w_overlap[:3]):
        print(f"    [{i+1}] '{c.content[:90]}'")

    print(f"""
  INSIGHT:
    Without overlap, chunk 1 ends at "HA." and chunk 2 starts at "Node failure...".
    If retrieval returns only chunk 1, the model sees the HA claim but not the details.
    With overlap, chunk 2 starts with the end of chunk 1 — the context bridge is present.
    Either chunk retrieved alone is sufficient to answer the question.
""")


def when_to_use_fixed_size():
    """
    Fixed-size chunking is a valid choice in specific scenarios.
    Enumerate them to prevent the misconception that it is always wrong.
    """

    print("=" * 70)
    print("WHEN FIXED-SIZE IS THE RIGHT CHOICE")
    print("=" * 70)

    print(f"""
  1. HOMOGENEOUS UNSTRUCTURED TEXT
     Log files, transcripts, continuous narrative with no paragraph breaks.
     Semantic boundaries don't exist to respect — fixed-size is fine.

  2. BASELINE / BENCHMARKING
     Always start with fixed-size chunking to establish a baseline.
     Compare other strategies against it to prove they add value.
     If fixed-size achieves 85% faithfulness and semantic chunking achieves 87%,
     the complexity of semantic chunking may not be worth the 2% gain.

  3. LARGE-SCALE INITIAL INDEXING
     When indexing millions of documents quickly, fixed-size is 10–100× faster
     than semantic chunking (no embedding at chunk time).
     You can always re-chunk later once you've identified quality issues.

  4. VERY SHORT DOCUMENTS
     If every document is < 500 tokens, no chunking is needed at all.
     Fixed-size with chunk_size = document_size = 1 chunk per document.

  5. WHEN CONTENT-TYPE DETECTION FAILS
     If you cannot reliably detect whether a document is prose, code, or YAML,
     use fixed-size as a safe fallback — better than misclassifying and
     applying the wrong semantic chunking strategy.

  ANTI-PATTERN: Using fixed-size for structured technical documentation.
    A 400-char chunk that splits "The APIC requires 3 nodes. When a node fails,..."
    at the period produces one chunk saying "requires 3 nodes" and another saying
    "When a node fails..." — the second is uninterpretable without context.
    For structured docs: use paragraph-aware or recursive chunking.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_comparison()
    overlap_impact_demo()
    when_to_use_fixed_size()
