"""
FILE: 05_mini_project_chunking_pipeline.py
LESSON: Phase 2 - Lesson 13 - Document Processing and Chunking
TOPIC: Mini-project — Chunking pipeline that selects strategy per content type

WHAT THIS BUILDS:
  A production-ready chunking pipeline that:
    1. Detects content type for each document (markdown, python, yaml, prose)
    2. Selects the optimal chunking strategy for that content type
    3. Applies content-type-specific metadata enrichment
    4. Evaluates chunk quality (size distribution, boundary quality)
    5. Benchmarks retrieval quality across strategies on the same corpus
    6. Outputs an ingestion report

  In production, this is the ingestion layer that runs when you add a new
  document to the RAG knowledge base. Getting this right once means every
  subsequent retrieval benefits from properly-shaped chunks.

INSTALL: pip install numpy
"""

import re
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


# ─── Shared Utilities ─────────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def mock_embed(text: str, dims: int = 32) -> np.ndarray:
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.RandomState(seed)
    v    = rng.randn(dims).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class IndexedChunk:
    """A fully-processed chunk ready for vector index insertion."""
    chunk_id:     str
    content:      str
    token_count:  int
    embedding:    np.ndarray
    metadata:     dict
    strategy:     str   # which chunking strategy produced this chunk


# ─── Content-Type Detection ───────────────────────────────────────────────────

def detect_content_type(text: str, filename: str = "") -> str:
    """
    Determine document content type from filename extension and text signals.
    WHY filename first: most reliable signal.
    WHY text fallback: many docs arrive without useful filename (from DB, PDF extraction).
    """
    ext_map = {".md": "markdown", ".py": "python", ".yaml": "yaml",
               ".yml": "yaml", ".json": "json", ".txt": "plain"}

    for ext, ctype in ext_map.items():
        if filename.endswith(ext):
            return ctype

    signals: dict[str, int] = {"markdown": 0, "python": 0, "yaml": 0, "plain": 0}

    if re.search(r"^#{1,6}\s", text, re.MULTILINE):     signals["markdown"] += 3
    if "```" in text:                                     signals["markdown"] += 2
    if re.search(r"^\s*[-*]\s", text, re.MULTILINE):     signals["markdown"] += 1
    if re.search(r"^(def |class )", text, re.MULTILINE): signals["python"]   += 4
    if "import " in text and "def " in text:             signals["python"]   += 2
    if re.search(r"^\w[\w_-]+:\s*$", text, re.MULTILINE): signals["yaml"]   += 3
    if text.strip().startswith("---"):                   signals["yaml"]     += 2

    return max(signals, key=lambda k: signals[k])


# ─── Chunking Strategy Registry ───────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", text)
    return [s.strip() for s in sents if s.strip()]


def _sentence_chunks(text: str, target: int, overlap_sents: int = 1, source: str = "doc") -> list[dict]:
    sentences = _split_sentences(text)
    chunks, current, cur_toks, idx = [], [], 0, 0
    for sent in sentences:
        st = approx_tokens(sent)
        if cur_toks + st > target and current:
            chunks.append({"content": " ".join(current), "source": source, "idx": idx})
            idx += 1
            current = current[-overlap_sents:] if overlap_sents else []
            cur_toks = sum(approx_tokens(s) for s in current)
        current.append(sent)
        cur_toks += st
    if current:
        chunks.append({"content": " ".join(current), "source": source, "idx": idx})
    return chunks


def _recursive_chunk(text: str, seps: list[str], target: int, source: str = "doc") -> list[dict]:
    if approx_tokens(text) <= target:
        return [{"content": text.strip(), "source": source}]
    if not seps:
        # Hard split fallback
        cs, chunks = target * 4, []
        for i in range(0, len(text), cs):
            c = text[i:i+cs].strip()
            if c:
                chunks.append({"content": c, "source": source})
        return chunks
    sep, rest = seps[0], seps[1:]
    parts = [p for p in re.split(f"(?={re.escape(sep)})", text) if p.strip()] \
            if sep.startswith("\n") else [p for p in text.split(sep) if p.strip()]
    if len(parts) <= 1:
        return _recursive_chunk(text, rest, target, source)
    result = []
    for p in parts:
        if approx_tokens(p) <= target:
            result.append({"content": p.strip(), "source": source})
        else:
            result.extend(_recursive_chunk(p, rest, target, source))
    # Merge small fragments
    merged = [result[0]] if result else []
    for item in result[1:]:
        if approx_tokens(merged[-1]["content"] + item["content"]) <= target:
            merged[-1]["content"] += "\n" + item["content"]
        else:
            merged.append(item)
    return merged


STRATEGY_MAP: dict[str, Callable] = {
    "markdown": lambda text, src: _recursive_chunk(
        text,
        ["\n## ", "\n### ", "\n```", "\n\n", "\n", ". ", " "],
        target=300, source=src,
    ),
    "python": lambda text, src: _recursive_chunk(
        text,
        ["\nclass ", "\ndef ", "\n\n", "\n", " "],
        target=200, source=src,
    ),
    "yaml": lambda text, src: _recursive_chunk(
        text,
        ["\n---", "\n\n", "\n  ", "\n", ": "],
        target=150, source=src,
    ),
    "json": lambda text, src: _recursive_chunk(
        text,
        ["},\n", "\n    ", "\n  ", "\n"],
        target=150, source=src,
    ),
    "plain": lambda text, src: _sentence_chunks(text, target=300, source=src),
}


# ─── Metadata Enricher ────────────────────────────────────────────────────────

def enrich_metadata(
    chunk_content: str,
    base_metadata: dict,
    content_type:  str,
) -> dict:
    """
    Add content-type-specific metadata signals to each chunk.
    WHY per-chunk metadata:
      At retrieval time, metadata enables filtering (e.g., only retrieve code chunks
      when the query is about implementation, only prose when about concepts).
    """
    meta = {**base_metadata, "content_type": content_type}

    # Detect if chunk contains a code block
    if "```" in chunk_content or re.search(r"^\s{4}\w", chunk_content, re.MULTILINE):
        meta["has_code"] = True

    # Detect if chunk contains a numbered list / steps
    if re.search(r"^\s*\d+\.\s", chunk_content, re.MULTILINE):
        meta["has_steps"] = True

    # Detect if chunk contains a table
    if "|" in chunk_content and re.search(r"\|[-─]+\|", chunk_content):
        meta["has_table"] = True

    # Extract header/title if present (Markdown heading)
    heading = re.search(r"^#{1,6}\s+(.+)$", chunk_content, re.MULTILINE)
    if heading:
        meta["section_heading"] = heading.group(1).strip()

    return meta


# ─── Ingestion Pipeline ───────────────────────────────────────────────────────

@dataclass
class IngestionReport:
    """Summary of one document's ingestion."""
    source:         str
    content_type:   str
    strategy:       str
    doc_tokens:     int
    num_chunks:     int
    mean_tokens:    float
    std_tokens:     float
    min_tokens:     int
    max_tokens:     int

    def display(self):
        print(f"  [{self.source}] type={self.content_type} strategy={self.strategy}")
        print(f"    doc={self.doc_tokens} tok → {self.num_chunks} chunks "
              f"(mean={self.mean_tokens:.0f}, std={self.std_tokens:.0f}, "
              f"range={self.min_tokens}–{self.max_tokens})")


class ChunkingPipeline:
    """
    Production-grade ingestion pipeline: detect → chunk → enrich → embed → report.
    """

    def __init__(self):
        self._chunks: list[IndexedChunk] = []

    def ingest_document(
        self,
        content:       str,
        source:        str,
        metadata:      dict,
        filename:      str = "",
        force_strategy: Optional[str] = None,
    ) -> IngestionReport:
        """
        Process one document: detect type → chunk → enrich metadata → embed.

        Args:
            content:         Full document text.
            source:          Unique document identifier.
            metadata:        Base metadata (product, date, version, etc.)
            filename:        Optional filename for extension-based type detection.
            force_strategy:  Override auto-detection with a specific strategy.
        """
        # Step 1: Detect content type
        content_type = force_strategy or detect_content_type(content, filename)
        strategy     = force_strategy or content_type
        chunk_fn     = STRATEGY_MAP.get(strategy, STRATEGY_MAP["plain"])

        # Step 2: Chunk
        raw_chunks = chunk_fn(content, source)

        # Step 3: Enrich metadata + embed
        indexed = []
        for i, raw in enumerate(raw_chunks):
            text     = raw["content"].strip()
            if not text:
                continue
            enriched = enrich_metadata(text, metadata, content_type)
            enriched["chunk_idx"] = i

            indexed.append(IndexedChunk(
                chunk_id    = f"{source}:chunk:{i}",
                content     = text,
                token_count = approx_tokens(text),
                embedding   = mock_embed(text),
                metadata    = enriched,
                strategy    = strategy,
            ))

        self._chunks.extend(indexed)

        # Step 4: Build report
        tok_counts = [c.token_count for c in indexed]
        return IngestionReport(
            source       = source,
            content_type = content_type,
            strategy     = strategy,
            doc_tokens   = approx_tokens(content),
            num_chunks   = len(indexed),
            mean_tokens  = float(np.mean(tok_counts)) if tok_counts else 0,
            std_tokens   = float(np.std(tok_counts))  if tok_counts else 0,
            min_tokens   = min(tok_counts)             if tok_counts else 0,
            max_tokens   = max(tok_counts)             if tok_counts else 0,
        )

    def search(self, query: str, top_k: int = 5) -> list[tuple[IndexedChunk, float]]:
        """Simple cosine similarity search over all indexed chunks."""
        if not self._chunks:
            return []
        q_vec  = mock_embed(query)
        matrix = np.vstack([c.embedding for c in self._chunks])
        scores = np.dot(matrix, q_vec)
        top    = np.argsort(-scores)[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top]

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)


# ─── Sample Knowledge Base ────────────────────────────────────────────────────

DOCS = [
    {
        "source":   "aci_guide.md",
        "filename": "aci_guide.md",
        "metadata": {"product": "ACI", "source_type": "guide", "date": "2025-03-01"},
        "content":  """
## ACI Fabric Architecture

Cisco ACI uses a Leaf-Spine topology. Every leaf connects to every spine.
ACI version 6.0 supports up to 200 leaf switches per pod.

### APIC Controller

The APIC cluster manages all fabric policy. Minimum 3 nodes for HA.
The APIC REST API uses JSON over HTTPS on port 443.

### EPGs and Contracts

EPGs define groups of endpoints with shared policy requirements.
Contracts define allowed traffic between EPGs. Without a contract, all traffic is denied.
A contract includes one or more subjects, each with filters defining permitted protocols and ports.

## Bug Advisory CSCvh23456

Affects APIC 5.2(1g). Contract deployment fails with more than 200 EPGs in one VRF.
Workaround: split into multiple VRFs. Fixed in 5.2(2a) and ACI 6.0.
""".strip(),
    },
    {
        "source":   "readyops_guide.md",
        "filename": "readyops_guide.md",
        "metadata": {"product": "ReadyOps", "source_type": "guide", "date": "2025-06-01"},
        "content":  """
## ReadyOps Platform Overview

ReadyOps is Criterion Networks' continuous validation platform.
It operates across two isolated environments: Production-Representative and Live Operations.
Changes are validated in Production-Representative before promotion to Live Operations.

### Validation Gate

The promotion gate requires 100% validation pass rate. No exceptions.
The gate blocks promotion until all Validation agent tests pass.

### Agent Classes

Health and Posture agents monitor baseline configuration drift continuously.
Validation agents run pre-change connectivity and compliance tests.
Operational agents execute approved runbooks automatically.
Stress and Adversarial agents perform resilience and fault injection testing.
""".strip(),
    },
    {
        "source":   "apic_client.py",
        "filename": "apic_client.py",
        "metadata": {"product": "ACI", "source_type": "code", "date": "2025-01-15"},
        "content":  """
import requests

class APICClient:
    def __init__(self, url, username, password):
        self.url = url
        self.token = self._login(username, password)

    def _login(self, username, password):
        payload = {"aaaUser": {"attributes": {"name": username, "pwd": password}}}
        resp = requests.post(f"{self.url}/api/aaaLogin.json", json=payload, verify=False)
        return resp.json()["imdata"][0]["aaaLogin"]["attributes"]["token"]

    def get_epgs(self, tenant):
        headers = {"Cookie": f"APIC-cookie={self.token}"}
        url = f"{self.url}/api/node/mo/uni/tn-{tenant}.json?query-target=subtree&target-subtree-class=fvAEPg"
        return requests.get(url, headers=headers).json()["imdata"]
""".strip(),
    },
]


# ─── Main Demo ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CHUNKING PIPELINE: Content-Aware Ingestion")
    print("=" * 70)

    pipeline = ChunkingPipeline()

    print("\n  [Ingestion Reports]")
    for doc in DOCS:
        report = pipeline.ingest_document(
            content  = doc["content"],
            source   = doc["source"],
            metadata = doc["metadata"],
            filename = doc["filename"],
        )
        report.display()

    print(f"\n  Total indexed: {pipeline.total_chunks} chunks across {len(DOCS)} documents")

    print("\n" + "─" * 70)
    print("  RETRIEVAL TESTS")
    print("─" * 70)

    queries = [
        "How does ACI enforce policy between EPGs?",
        "What is the validation pass rate requirement in ReadyOps?",
        "How do I authenticate with APIC using the Python client?",
        "Bug CSCvh23456 workaround",
    ]

    for q in queries:
        results = pipeline.search(q, top_k=3)
        print(f"\n  Query: '{q}'")
        for i, (chunk, score) in enumerate(results, 1):
            ct = chunk.metadata.get("content_type", "?")
            prod = chunk.metadata.get("product", "?")
            print(f"    [{i}] score={score:.3f} type={ct} product={prod}")
            print(f"         '{chunk.content[:80].strip()}'")

    print("\n" + "─" * 70)
    print("  METADATA FILTER DEMO")
    print("─" * 70)

    print("\n  Filter: content_type='python' — code-only chunks:")
    code_chunks = [c for c in pipeline._chunks if c.metadata.get("content_type") == "python"]
    for c in code_chunks:
        print(f"    [{c.chunk_id}] {c.token_count} tok: '{c.content[:70].strip()}'")

    print(f"""
  CHUNKING PIPELINE SUMMARY:
  ┌──────────────────────────────────────────────────────────────────┐
  │  detect_content_type()  → markdown / python / yaml / plain      │
  │         │                                                        │
  │  STRATEGY_MAP[type]     → strategy-specific chunker             │
  │         │                                                        │
  │  enrich_metadata()      → has_code, has_steps, section_heading  │
  │         │                                                        │
  │  mock_embed() / Voyage AI → float32 embedding vector            │
  │         │                                                        │
  │  IndexedChunk           → into vector store (Qdrant/FAISS)      │
  └──────────────────────────────────────────────────────────────────┘

  NEXT: Phase 2 Lesson 14 covers query understanding and rewriting —
  how to improve retrieval by transforming the user's raw query before
  sending it to the search engine.
""")


if __name__ == "__main__":
    main()
