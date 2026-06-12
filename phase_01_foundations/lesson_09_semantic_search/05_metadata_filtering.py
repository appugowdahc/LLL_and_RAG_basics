"""
FILE: 05_metadata_filtering.py
LESSON: Phase 1 - Lesson 9 - Semantic Search
TOPIC: Metadata filtering — narrowing vector search with structured constraints

WHAT THIS FILE TEACHES:
  - What metadata fields are and why every RAG chunk should have them
  - Pre-filter vs post-filter vs in-filter strategies
  - Building a filtered in-memory vector index
  - Combining metadata filters with hybrid BM25 + dense search
  - Common production metadata schemas for enterprise RAG
  - WHY wrong filter strategy causes recall collapse

PRODUCTION CONTEXT:
  A knowledge base for Criterion Networks would tag each chunk with:
    source_type: "spec" | "guide" | "config" | "advisory"
    product:     "ACI" | "Hypershield" | "ISE" | "ReadyOps" | ...
    date:        "2025-01-15"
    tier:        "core" | "supporting" | "general"
    language:    "en" | "zh" | "de" | ...

  Without metadata filtering, a query about "ACI configuration" would
  also retrieve chunks about Hypershield and ISE — wasting context tokens.

INSTALL: pip install numpy
"""

import re
import math
import hashlib
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Document:
    """A document chunk with content and metadata."""
    doc_id:   str
    content:  str
    metadata: dict[str, Any]   # arbitrary key-value metadata fields

    @property
    def embedding(self) -> np.ndarray:
        """Deterministic mock embedding derived from content."""
        seed = int(hashlib.md5(self.content.encode()).hexdigest(), 16) % (2**32)
        rng  = np.random.RandomState(seed)
        v    = rng.randn(64).astype(np.float32)
        return v / np.linalg.norm(v)


# ─── Filter Engine ────────────────────────────────────────────────────────────

class MetadataFilter:
    """
    A composable metadata filter.
    Supports: exact match, list membership, range, boolean, and logical AND/OR.

    WHY composable filters:
      In production, users don't just filter by one field.
      Common pattern: product="ACI" AND date > "2024-01-01" AND language="en"
      A composable filter object allows building arbitrary filter trees.
    """

    def __init__(self, conditions: list[dict]):
        """
        Args:
            conditions: List of filter conditions. Each is a dict:
              {"field": "product",   "op": "eq",  "value": "ACI"}
              {"field": "date",      "op": "gte", "value": "2025-01-01"}
              {"field": "source_type","op": "in", "value": ["spec","guide"]}
              {"field": "tier",      "op": "ne",  "value": "general"}
        """
        self.conditions = conditions

    def matches(self, doc: Document) -> bool:
        """Return True if the document passes ALL filter conditions (AND logic)."""
        for cond in self.conditions:
            field  = cond["field"]
            op     = cond["op"]
            value  = cond["value"]
            actual = doc.metadata.get(field)

            if actual is None:
                return False   # WHY: missing field = fail (strict match)

            if op == "eq"  and actual != value:            return False
            if op == "ne"  and actual == value:            return False
            if op == "in"  and actual not in value:        return False
            if op == "nin" and actual in value:            return False
            if op == "gte" and not (actual >= value):      return False
            if op == "lte" and not (actual <= value):      return False
            if op == "gt"  and not (actual >  value):      return False
            if op == "lt"  and not (actual <  value):      return False

        return True   # WHY all conditions must pass: AND logic


# ─── Filtered Vector Index ────────────────────────────────────────────────────

class FilteredVectorIndex:
    """
    In-memory vector index with support for metadata pre-filter and post-filter.

    Demonstrates both strategies so you can understand the recall implications.
    """

    def __init__(self, dims: int = 64):
        self.dims     = dims
        self._docs:   list[Document] = []
        self._matrix: Optional[np.ndarray] = None

    def add(self, docs: list[Document]):
        """Add documents to the index."""
        self._docs.extend(docs)
        embeddings = np.vstack([d.embedding for d in docs])

        if self._matrix is None:
            self._matrix = embeddings
        else:
            self._matrix = np.vstack([self._matrix, embeddings])

    def search_prefilter(
        self,
        query_embedding: np.ndarray,
        filter_:         Optional[MetadataFilter],
        top_k:           int = 5,
    ) -> list[tuple[Document, float]]:
        """
        PRE-FILTER: Apply metadata filter BEFORE vector search.

        Strategy:
          1. Select only documents matching the metadata filter.
          2. Build a subset matrix from those documents.
          3. Run exact dot product on the subset.

        WHY pre-filter can hurt recall:
          If the filter is very restrictive (returns only 20 docs out of 10K),
          and the true answer is in those 20, great. But if you set ef_search=100
          in an HNSW index, pre-filtering to 20 docs means HNSW has only 20 nodes
          to traverse — it can't use the multi-layer graph effectively.

          IN PRACTICE: pre-filter works well when filter returns >= 1000 docs.
          For tighter filters, use in-filter (Qdrant's approach) or post-filter.
        """

        if filter_ is None:
            filtered_docs = self._docs
            filtered_mat  = self._matrix
        else:
            filtered_indices = [i for i, d in enumerate(self._docs) if filter_.matches(d)]

            if not filtered_indices:
                return []

            filtered_docs = [self._docs[i] for i in filtered_indices]
            filtered_mat  = self._matrix[filtered_indices]   # WHY index: numpy fancy indexing

        qn     = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        scores = np.dot(filtered_mat, qn)
        top    = np.argsort(-scores)[:top_k]

        return [(filtered_docs[i], float(scores[i])) for i in top]

    def search_postfilter(
        self,
        query_embedding: np.ndarray,
        filter_:         Optional[MetadataFilter],
        top_k:           int = 5,
        overretrieve:    int = 4,   # multiplier: search top_k × overretrieve before filtering
    ) -> list[tuple[Document, float]]:
        """
        POST-FILTER: Run full vector search, then apply metadata filter.

        Strategy:
          1. Retrieve top_k × overretrieve candidates via vector search (ignoring metadata).
          2. Apply metadata filter to the retrieved candidates.
          3. Return top_k from the filtered candidates.

        WHY the overretrieve multiplier:
          If you need top-5 after filtering, and only 40% of the corpus passes the filter,
          you need to retrieve top-5/0.40 ≈ top-13 before filtering to reliably get 5.
          overretrieve=4 → retrieve 20 for top-5 → works unless filter is very tight (<20%).

        WHY post-filter preserves recall:
          Vector search sees ALL vectors → can find the globally nearest neighbor.
          Then we just discard results that fail the metadata check.
          Downside: wastes vector search compute on filtered-out results.
        """

        if filter_ is None or self._matrix is None:
            return self.search_prefilter(query_embedding, None, top_k)

        # Search a larger set before filtering
        search_k  = min(top_k * overretrieve, len(self._docs))
        qn        = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        scores    = np.dot(self._matrix, qn)
        top_large = np.argsort(-scores)[:search_k]

        # Apply metadata filter to candidates
        results = []
        for idx in top_large:
            doc = self._docs[idx]
            if filter_ is None or filter_.matches(doc):
                results.append((doc, float(scores[idx])))
            if len(results) >= top_k:
                break

        return results


# ─── Production Metadata Schema ──────────────────────────────────────────────

def build_sample_corpus() -> list[Document]:
    """
    Build a sample corpus with rich metadata.
    Represents a realistic Criterion Networks knowledge base.
    """

    return [
        Document("doc_001", "Cisco ACI uses Leaf-Spine topology. APIC manages fabric policy.", {
            "product": "ACI", "source_type": "guide", "tier": "core",
            "date": "2025-03-01", "language": "en", "version": "6.0"
        }),
        Document("doc_002", "APIC REST API uses JSON over HTTPS for fabric management.", {
            "product": "ACI", "source_type": "spec", "tier": "core",
            "date": "2025-01-15", "language": "en", "version": "5.2"
        }),
        Document("doc_003", "ReadyOps validates ACI changes before promoting to production.", {
            "product": "ReadyOps", "source_type": "guide", "tier": "core",
            "date": "2025-06-01", "language": "en", "version": "2.0"
        }),
        Document("doc_004", "Hypershield uses eBPF for kernel-level policy without appliances.", {
            "product": "Hypershield", "source_type": "guide", "tier": "core",
            "date": "2025-02-20", "language": "en", "version": "1.0"
        }),
        Document("doc_005", "ISE TrustSec assigns SGTs at authentication for microsegmentation.", {
            "product": "ISE", "source_type": "guide", "tier": "core",
            "date": "2024-11-10", "language": "en", "version": "3.3"
        }),
        Document("doc_006", "Nexus 9336C-FX2 supports ACI and NX-OS modes. 36-port 100G.", {
            "product": "ACI", "source_type": "spec", "tier": "supporting",
            "date": "2024-09-05", "language": "en", "version": "6.0"
        }),
        Document("doc_007", "Bug CSCvh23456: APIC 5.2(1g) contract deployment failures.", {
            "product": "ACI", "source_type": "advisory", "tier": "general",
            "date": "2024-06-15", "language": "en", "version": "5.2"
        }),
        Document("doc_008", "ReadyOps agent classes: Health Posture, Validation, Operational.", {
            "product": "ReadyOps", "source_type": "spec", "tier": "core",
            "date": "2025-06-01", "language": "en", "version": "2.0"
        }),
        Document("doc_009", "ACI Multi-Pod connects geographic locations via VXLAN IPN.", {
            "product": "ACI", "source_type": "guide", "tier": "supporting",
            "date": "2025-01-20", "language": "en", "version": "6.0"
        }),
        Document("doc_010", "Old ACI guide: APIC cluster requires 3 nodes minimum.", {
            "product": "ACI", "source_type": "guide", "tier": "core",
            "date": "2022-05-01", "language": "en", "version": "4.0"
        }),
        Document("doc_011", "Cisco ISE profiling identifies device type at network access.", {
            "product": "ISE", "source_type": "guide", "tier": "supporting",
            "date": "2025-03-15", "language": "en", "version": "3.3"
        }),
        Document("doc_012", "SD-WAN provides cloud-managed WAN with zero-touch provisioning.", {
            "product": "SD-WAN", "source_type": "guide", "tier": "core",
            "date": "2025-04-10", "language": "en", "version": "20.12"
        }),
    ]


# ─── Filtering Demo ───────────────────────────────────────────────────────────

def run_filter_demo():
    """
    Demonstrate metadata filtering patterns and their recall implications.
    """

    print("=" * 65)
    print("METADATA FILTERING: Precision without Recall Loss")
    print("=" * 65)

    docs   = build_sample_corpus()
    index  = FilteredVectorIndex(dims=64)
    index.add(docs)

    query     = "How does ACI manage policy enforcement?"
    query_vec = docs[0].embedding  # use a known vector for determinism

    # ── Test 1: No filter ─────────────────────────────────────────────────────
    print(f"\n  Query: '{query}'")

    print(f"\n  [1] No filter (all {len(docs)} docs searchable)")
    results = index.search_prefilter(query_vec, filter_=None, top_k=5)
    for doc, score in results:
        print(f"    {score:.4f}  {doc.doc_id}  {doc.metadata['product']:<12} "
              f"{doc.metadata['source_type']:<10} {doc.content[:50]}...")

    # ── Test 2: Narrow filter (ACI product only) ──────────────────────────────
    aci_filter = MetadataFilter([
        {"field": "product", "op": "eq", "value": "ACI"},
    ])
    aci_docs = [d for d in docs if aci_filter.matches(d)]
    print(f"\n  [2] Pre-filter: product='ACI'  ({len(aci_docs)} docs pass filter)")

    results_pre = index.search_prefilter(query_vec, filter_=aci_filter, top_k=5)
    for doc, score in results_pre:
        print(f"    {score:.4f}  {doc.doc_id}  {doc.metadata['product']:<12} "
              f"{doc.metadata['version']:<6} {doc.content[:50]}...")

    # ── Test 3: Version filter (recent only) ──────────────────────────────────
    recent_filter = MetadataFilter([
        {"field": "product",  "op": "eq",  "value": "ACI"},
        {"field": "date",     "op": "gte", "value": "2025-01-01"},
        {"field": "source_type", "op": "in", "value": ["guide", "spec"]},
    ])
    recent_docs = [d for d in docs if recent_filter.matches(d)]
    print(f"\n  [3] Complex filter: product='ACI' AND date>='2025-01-01' AND type in [guide,spec]")
    print(f"      ({len(recent_docs)} docs pass filter)")

    results_recent = index.search_prefilter(query_vec, filter_=recent_filter, top_k=5)
    for doc, score in results_recent:
        print(f"    {score:.4f}  {doc.doc_id}  version={doc.metadata['version']:<6} "
              f"{doc.metadata['date']}  {doc.content[:50]}...")

    # ── Test 4: Pre vs Post filter recall comparison ──────────────────────────
    tight_filter = MetadataFilter([
        {"field": "product",     "op": "eq", "value": "ACI"},
        {"field": "source_type", "op": "eq", "value": "advisory"},
    ])
    matching = [d for d in docs if tight_filter.matches(d)]
    print(f"\n  [4] Tight filter: product='ACI' AND source_type='advisory'")
    print(f"      ({len(matching)} docs pass filter — very tight!)")

    print(f"\n  Pre-filter results:")
    pre = index.search_prefilter(query_vec, filter_=tight_filter, top_k=5)
    for doc, score in pre:
        print(f"    {score:.4f}  {doc.doc_id}  {doc.content[:60]}...")
    if not pre:
        print(f"    (no results)")

    print(f"\n  Post-filter results (overretrieve=4):")
    post = index.search_postfilter(query_vec, filter_=tight_filter, top_k=5, overretrieve=4)
    for doc, score in post:
        print(f"    {score:.4f}  {doc.doc_id}  {doc.content[:60]}...")
    if not post:
        print(f"    (no results — advisory docs not semantically close to query)")


def metadata_schema_guide():
    """
    Best practices for metadata schema design in enterprise RAG.
    """

    print("\n" + "=" * 65)
    print("ENTERPRISE METADATA SCHEMA GUIDE")
    print("=" * 65)

    print(f"""
  RECOMMENDED METADATA FIELDS FOR ENTERPRISE RAG:

  MANDATORY (always include):
    doc_id        : Unique chunk identifier (for citation tracking)
    source        : Filename or URL of the source document
    chunk_idx     : Position within the source document (0-based)
    created_at    : When the chunk was indexed (ISO 8601)
    content_hash  : SHA-256 of the content (for change detection)

  RETRIEVAL QUALITY (add these):
    source_type   : "spec"|"guide"|"config"|"advisory"|"blog"|"faq"
                    WHY: Filter to authoritative docs; exclude blog posts for compliance
    product       : "ACI"|"ISE"|"Hypershield"|"ReadyOps"|"SD-WAN"|...
                    WHY: Scope queries to relevant product; avoid cross-product noise
    date          : "YYYY-MM-DD" of the source document publication
                    WHY: Filter to recent versions; exclude stale configuration guides
    version       : "6.0"|"5.2"|"3.3" — product version this content applies to
                    WHY: "Is this step valid for ACI 6.0?" → filter to version="6.0"

  CONTEXT WINDOW MANAGEMENT:
    tier          : "core"|"supporting"|"general" — for hierarchical budget allocation
    token_count   : Actual token count of this chunk (computed at index time)
                    WHY: Use this to fill context window without re-counting

  MULTILINGUAL:
    language      : ISO 639-1 code "en"|"zh"|"de"|...
                    WHY: Route to language-specific index or filter for user's language

  COMPLIANCE/AUDIT:
    classification: "public"|"internal"|"confidential"|"restricted"
                    WHY: Enforce data access controls in the retrieval layer

  ANTI-PATTERNS:
    ✗ Don't store full content in metadata (use vector content instead)
    ✗ Don't use free-form text fields as filter targets (use enums/categories)
    ✗ Don't skip doc_id — you NEED it for citation attribution
    ✗ Don't skip created_at — you NEED it for freshness filtering

  FILTER STRATEGY SELECTION:
    Filter returns > 10% of corpus : pre-filter (fast, no recall loss)
    Filter returns 1-10% of corpus : post-filter with overretrieve=10
    Filter returns < 1% of corpus  : in-filter (Qdrant payload indexing)
    Dynamic, per-query filters      : always benchmark recall vs unfiltered
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_filter_demo()
    metadata_schema_guide()
