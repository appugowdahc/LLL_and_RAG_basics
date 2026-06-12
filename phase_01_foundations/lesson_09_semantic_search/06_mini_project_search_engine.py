"""
FILE: 06_mini_project_search_engine.py
LESSON: Phase 1 - Lesson 9 - Semantic Search
TOPIC: Mini-project — Full hybrid search engine

WHAT THIS FILE BUILDS:
  A production-grade search engine combining ALL Lesson 9 components:
    1. BM25 keyword search (exact term matching)
    2. Dense vector search (semantic matching)
    3. Reciprocal Rank Fusion (RRF) to combine results
    4. Metadata filtering (pre-filter + post-filter)
    5. Evaluation against ground truth queries

  This is the search layer that sits between the user's query and the
  LLM in a production RAG system. Getting this layer right is the
  difference between a RAG system that works and one that doesn't.

ARCHITECTURE:
  Query
    │
    ├──► [BM25 Index]  ──► keyword results
    │
    ├──► [Metadata Filter] ──► filtered doc set
    │
    └──► [Dense Index]  ──► semantic results
              │
              ▼
        [RRF Fusion]
              │
              ▼
        [Reranked Results]
              │
              ▼
        [Context Builder]  ──► RAG prompt

INSTALL: pip install numpy
"""

import math
import re
import hashlib
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


# ─── Shared Utilities ─────────────────────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "at", "by", "for", "with", "as", "this", "that", "it", "its",
    "from", "into", "has", "have", "had", "will", "can", "should", "not",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanum, remove stopwords and single chars."""
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def mock_embedding(text: str, dims: int = 64) -> np.ndarray:
    """
    Deterministic mock embedding from text content.
    WHY SHA-256 seed: same text → same vector every run (reproducible demos).
    In production this would call Voyage AI or similar.
    """
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.RandomState(seed)
    v    = rng.randn(dims).astype(np.float32)
    return v / np.linalg.norm(v)   # WHY normalize: cosine sim requires unit vectors


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A document chunk ready for indexing."""
    chunk_id:  str
    content:   str
    metadata:  dict[str, Any]
    source:    str = ""

    @property
    def embedding(self) -> np.ndarray:
        return mock_embedding(self.content)


@dataclass
class SearchResult:
    """A single search result with score and provenance."""
    chunk:       Chunk
    score:       float
    source:      str   # "bm25", "dense", "hybrid"
    bm25_rank:   Optional[int] = None
    dense_rank:  Optional[int] = None


# ─── BM25 Index ───────────────────────────────────────────────────────────────

class BM25Index:
    """BM25 keyword index (k1=1.5, b=0.75 defaults per original paper)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1       = k1
        self.b        = b
        self._chunks: list[Chunk] = []
        self._corpus: list[list[str]] = []
        self._avgdl:  float = 0.0
        self._idf:    dict[str, float] = {}
        self._inv:    dict[str, list[int]] = defaultdict(list)

    def build(self, chunks: list[Chunk]):
        """Index all chunks."""
        self._chunks = chunks
        self._corpus = [tokenize(c.content) for c in chunks]
        N            = len(self._corpus)
        self._avgdl  = sum(len(d) for d in self._corpus) / max(N, 1)

        doc_freq: dict[str, set] = defaultdict(set)
        for idx, tokens in enumerate(self._corpus):
            for t in set(tokens):
                doc_freq[t].add(idx)
                self._inv[t].append(idx)

        for term, docs in doc_freq.items():
            df             = len(docs)
            self._idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Return top-k (chunk, score) pairs for the query."""
        terms      = tokenize(query)
        candidates = set()
        for t in terms:
            candidates.update(self._inv.get(t, []))

        scores = []
        for idx in candidates:
            tokens = self._corpus[idx]
            dl     = len(tokens)
            score  = 0.0
            for t in terms:
                tf       = tokens.count(t)
                if tf == 0:
                    continue
                idf      = self._idf.get(t, 0.0)
                ln       = 1 - self.b + self.b * (dl / max(self._avgdl, 1))
                tf_sat   = (tf * (self.k1 + 1)) / (tf + self.k1 * ln)
                score   += idf * tf_sat
            scores.append((idx, score))

        scores.sort(key=lambda x: -x[1])
        return [(self._chunks[i], s) for i, s in scores[:top_k]]


# ─── Dense Vector Index ───────────────────────────────────────────────────────

class DenseIndex:
    """Exact nearest-neighbor search via numpy matrix multiplication."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._matrix: Optional[np.ndarray] = None

    def build(self, chunks: list[Chunk]):
        """Index all chunks by embedding them into a matrix."""
        self._chunks = chunks
        self._matrix = np.vstack([c.embedding for c in chunks])

    def search(
        self,
        query_vec: np.ndarray,
        top_k:     int = 10,
        candidates: Optional[list[int]] = None,   # WHY: supports pre-filter
    ) -> list[tuple[Chunk, float]]:
        """Return top-k (chunk, score) pairs for the query vector."""
        if self._matrix is None or not self._chunks:
            return []

        qn = query_vec / (np.linalg.norm(query_vec) + 1e-10)

        if candidates is not None:
            # WHY subset search: pre-filter passes only matching doc indices
            mat    = self._matrix[candidates]
            scores = np.dot(mat, qn)
            top    = np.argsort(-scores)[:top_k]
            return [(self._chunks[candidates[i]], float(scores[i])) for i in top]
        else:
            scores = np.dot(self._matrix, qn)
            top    = np.argsort(-scores)[:top_k]
            return [(self._chunks[i], float(scores[i])) for i in top]


# ─── Metadata Filter ──────────────────────────────────────────────────────────

class MetadataFilter:
    """Evaluates per-chunk metadata against a list of filter conditions."""

    def __init__(self, conditions: list[dict]):
        # conditions: [{"field": "product", "op": "eq", "value": "ACI"}, ...]
        self.conditions = conditions

    def matches(self, chunk: Chunk) -> bool:
        for cond in self.conditions:
            actual = chunk.metadata.get(cond["field"])
            if actual is None:
                return False
            op, value = cond["op"], cond["value"]
            if op == "eq"  and actual != value:      return False
            if op == "ne"  and actual == value:      return False
            if op == "in"  and actual not in value:  return False
            if op == "nin" and actual in value:      return False
            if op == "gte" and not actual >= value:  return False
            if op == "lte" and not actual <= value:  return False
            if op == "gt"  and not actual >  value:  return False
            if op == "lt"  and not actual <  value:  return False
        return True

    def filter_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        return [c for c in chunks if self.matches(c)]

    def filter_indices(self, chunks: list[Chunk]) -> list[int]:
        """Return indices of matching chunks (for pre-filter in DenseIndex)."""
        return [i for i, c in enumerate(chunks) if self.matches(c)]


# ─── RRF Fusion ───────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[tuple[Chunk, float]]],
    k:            int = 60,
) -> list[tuple[Chunk, float]]:
    """
    Combine multiple ranked result lists via Reciprocal Rank Fusion.

    WHY k=60: Bendersky et al. (2022) showed k=60 minimizes variance in RRF
      across different ranking systems. Higher k reduces the impact of top ranks;
      lower k amplifies top-rank signals but destabilizes with rank swaps.

    RRF score for a document d:
      score(d) = Σ over lists: 1 / (k + rank(d, list))

    WHY RRF over weighted scores:
      1. No score normalization needed — works with any scoring scale.
      2. Robust: outlier scores don't blow up the fusion.
      3. A document ranked 1st in BM25 and 10th in dense still aggregates well.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map:  dict[str, Chunk] = {}

    for result_list in result_lists:
        for rank, (chunk, _) in enumerate(result_list, start=1):
            rrf_scores[chunk.chunk_id] += 1.0 / (k + rank)
            chunk_map[chunk.chunk_id]   = chunk

    combined = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [(chunk_map[cid], score) for cid, score in combined]


# ─── Full Hybrid Search Engine ────────────────────────────────────────────────

class HybridSearchEngine:
    """
    Production-grade hybrid search engine combining BM25, dense, metadata filter, and RRF.

    This is the search layer of a RAG system. Its job is to find the most
    relevant chunks given a user query, within a token budget and optional
    metadata constraints.
    """

    def __init__(self):
        self._chunks:    list[Chunk] = []
        self._bm25:      BM25Index   = BM25Index()
        self._dense:     DenseIndex  = DenseIndex()
        self._built:     bool        = False

    def build(self, chunks: list[Chunk]):
        """
        Index all chunks in both BM25 and dense indexes.
        Call this once after loading your document corpus.
        """
        self._chunks = chunks
        self._bm25.build(chunks)
        self._dense.build(chunks)
        self._built = True
        print(f"  [Engine] Indexed {len(chunks)} chunks into BM25 + dense indexes.")

    def search(
        self,
        query:       str,
        top_k:       int = 5,
        filter_:     Optional[MetadataFilter] = None,
        bm25_k:      int = 10,    # WHY > top_k: retrieve more for RRF to combine
        dense_k:     int = 10,
        rrf_k:       int = 60,
        prefilter:   bool = False,  # WHY flag: caller can choose pre vs post filter
    ) -> list[SearchResult]:
        """
        Run full hybrid search: BM25 + dense + metadata filter + RRF fusion.

        Args:
            query:      User query string.
            top_k:      Final number of results to return.
            filter_:    Optional metadata filter to apply.
            bm25_k:     Top-K from BM25 before fusion.
            dense_k:    Top-K from dense before fusion.
            rrf_k:      RRF k parameter (default 60).
            prefilter:  If True, apply filter before dense search (pre-filter).
                        If False, apply filter after fusion (post-filter).

        Returns:
            List of SearchResult ordered by fused RRF score.
        """
        assert self._built, "Call build() before search()."

        query_vec = mock_embedding(query)   # WHY mock: API-key-free demo

        # ── BM25 Search ────────────────────────────────────────────────────────
        bm25_results = self._bm25.search(query, top_k=bm25_k)

        # ── Dense Search (with optional pre-filter) ────────────────────────────
        if prefilter and filter_ is not None:
            candidate_indices = filter_.filter_indices(self._chunks)
            dense_results     = self._dense.search(query_vec, dense_k, candidates=candidate_indices)
        else:
            dense_results = self._dense.search(query_vec, dense_k)

        # ── RRF Fusion ─────────────────────────────────────────────────────────
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=rrf_k)

        # ── Post-filter (if not pre-filtering) ────────────────────────────────
        if filter_ is not None and not prefilter:
            fused = [(chunk, score) for chunk, score in fused if filter_.matches(chunk)]

        # ── Annotate with search provenance ───────────────────────────────────
        bm25_ranks  = {c.chunk_id: r + 1 for r, (c, _) in enumerate(bm25_results)}
        dense_ranks = {c.chunk_id: r + 1 for r, (c, _) in enumerate(dense_results)}

        results = []
        for chunk, score in fused[:top_k]:
            results.append(SearchResult(
                chunk      = chunk,
                score      = score,
                source     = "hybrid",
                bm25_rank  = bm25_ranks.get(chunk.chunk_id),
                dense_rank = dense_ranks.get(chunk.chunk_id),
            ))

        return results

    def build_rag_context(
        self,
        results:     list[SearchResult],
        token_budget: int = 4000,   # approximate character budget (no tiktoken in demo)
    ) -> str:
        """
        Build the context string to insert into the RAG prompt.

        WHY token_budget:
          The context window is finite. We must respect it.
          In production, use tiktoken to count tokens exactly.
          Here we approximate: 1 token ≈ 4 chars for English text.
        """
        char_budget = token_budget * 4   # WHY *4: rough chars-per-token approximation
        context_parts = []
        used = 0

        for i, result in enumerate(results, 1):
            chunk     = result.chunk
            header    = f"[{i}] Source: {chunk.metadata.get('source_type', 'doc')} | "
            header   += f"Product: {chunk.metadata.get('product', 'unknown')} | "
            header   += f"Date: {chunk.metadata.get('date', 'unknown')}"
            entry     = f"{header}\n{chunk.content}\n"
            used     += len(entry)

            if used > char_budget:
                break   # WHY hard stop: never overflow the context window

            context_parts.append(entry)

        return "\n".join(context_parts)


# ─── Sample Corpus ────────────────────────────────────────────────────────────

def build_corpus() -> list[Chunk]:
    """
    Build a realistic Criterion Networks knowledge base corpus.
    In production, this would be loaded from Qdrant, Weaviate, or Pinecone.
    """

    raw = [
        # ACI – core docs
        ("c001", "Cisco ACI uses Leaf-Spine topology. The APIC cluster manages all fabric policy via a centralized model.", {"product": "ACI", "source_type": "guide",    "tier": "core",       "date": "2025-03-01", "version": "6.0"}),
        ("c002", "In ACI, an EPG (Endpoint Group) defines a collection of endpoints with the same policy requirements. Contracts define communication between EPGs.", {"product": "ACI", "source_type": "guide",    "tier": "core",       "date": "2025-03-01", "version": "6.0"}),
        ("c003", "APIC REST API uses JSON over HTTPS. All operations use: POST /api/mo/<dn>.json with a body containing the Managed Object.", {"product": "ACI", "source_type": "spec",     "tier": "core",       "date": "2025-01-15", "version": "5.2"}),
        ("c004", "ACI Multi-Pod architecture connects geographic locations using a VXLAN IPN (Inter-Pod Network). BGP serves as the overlay control plane.", {"product": "ACI", "source_type": "guide",    "tier": "supporting", "date": "2025-01-20", "version": "6.0"}),
        ("c005", "Bug CSCvh23456 in APIC 5.2(1g): contract deployment fails when more than 200 EPGs are in a single VRF. Workaround: split VRFs.", {"product": "ACI", "source_type": "advisory", "tier": "general",    "date": "2024-06-15", "version": "5.2"}),
        ("c006", "Nexus 9336C-FX2: 36-port 100G QSFP28, supports ACI fabric mode and NX-OS standalone mode. Hardware VTEP for VXLAN termination.", {"product": "ACI", "source_type": "spec",     "tier": "supporting", "date": "2024-09-05", "version": "6.0"}),
        ("c007", "Old ACI guide (v4.0): The APIC cluster requires minimum 3 nodes for HA. Maximum supported scale: 180 leaf switches per pod.", {"product": "ACI", "source_type": "guide",    "tier": "core",       "date": "2022-05-01", "version": "4.0"}),

        # ReadyOps – core docs
        ("c008", "ReadyOps is Criterion Networks' continuous validation platform. It runs AI agents across Production-Representative and Live Operations environments.", {"product": "ReadyOps", "source_type": "guide", "tier": "core",       "date": "2025-06-01", "version": "2.0"}),
        ("c009", "ReadyOps agent classes: Health and Posture agents monitor baseline drift. Validation agents run pre-change tests. Operational agents enforce runbooks.", {"product": "ReadyOps", "source_type": "spec",  "tier": "core",       "date": "2025-06-01", "version": "2.0"}),
        ("c010", "ReadyOps validates ACI changes in a digital twin before promoting to Live Operations. The promotion gate requires 100% validation pass rate.", {"product": "ReadyOps", "source_type": "guide", "tier": "core",       "date": "2025-06-01", "version": "2.0"}),

        # Hypershield
        ("c011", "Cisco Hypershield uses eBPF for kernel-level policy enforcement without dedicated appliances. Policies are enforced at the workload, not at the perimeter.", {"product": "Hypershield", "source_type": "guide", "tier": "core",       "date": "2025-02-20", "version": "1.0"}),
        ("c012", "Hypershield integrates with ACI using the ACI Hypershield integration API. EPG membership is propagated to Hypershield policy engines.", {"product": "Hypershield", "source_type": "spec",  "tier": "supporting", "date": "2025-02-20", "version": "1.0"}),

        # ISE
        ("c013", "Cisco ISE TrustSec assigns Security Group Tags (SGTs) at authentication. SXP propagates SGT-to-IP bindings to non-TrustSec network devices.", {"product": "ISE", "source_type": "guide", "tier": "core",       "date": "2024-11-10", "version": "3.3"}),
        ("c014", "ISE profiling identifies device type using RADIUS, DHCP, HTTP, and SNMP probes. Profiling feeds into posture assessment and policy assignment.", {"product": "ISE", "source_type": "guide", "tier": "supporting", "date": "2025-03-15", "version": "3.3"}),

        # SD-WAN
        ("c015", "Cisco SD-WAN (Catalyst SD-WAN) provides cloud-managed WAN with zero-touch provisioning. vManage is the centralized management plane.", {"product": "SD-WAN", "source_type": "guide", "tier": "core",       "date": "2025-04-10", "version": "20.12"}),
    ]

    return [Chunk(chunk_id=cid, content=content, metadata=meta, source=meta["product"])
            for cid, content, meta in raw]


# ─── Evaluation ───────────────────────────────────────────────────────────────

@dataclass
class EvalQuery:
    query:           str
    relevant_ids:    list[str]   # ground truth chunk IDs that should be retrieved
    filter_:         Optional[MetadataFilter] = None
    description:     str = ""


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    top_k = retrieved[:k]
    hits  = sum(1 for r in top_k if r in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of relevant docs that appear in top-k retrieved."""
    top_k = retrieved[:k]
    hits  = sum(1 for r in top_k if r in relevant)
    return hits / max(len(relevant), 1)


def evaluate_engine(engine: HybridSearchEngine, eval_queries: list[EvalQuery]) -> dict:
    """
    Run evaluation queries and compute P@5, R@5 for each query.
    Returns aggregated scores.
    """
    p_scores, r_scores = [], []

    for eq in eval_queries:
        results = engine.search(eq.query, top_k=5, filter_=eq.filter_)
        retrieved_ids = [r.chunk.chunk_id for r in results]

        p = precision_at_k(retrieved_ids, eq.relevant_ids, k=5)
        r = recall_at_k(retrieved_ids,    eq.relevant_ids, k=5)

        p_scores.append(p)
        r_scores.append(r)

    return {
        "mean_p5": sum(p_scores) / len(p_scores),
        "mean_r5": sum(r_scores) / len(r_scores),
        "per_query": list(zip(eval_queries, p_scores, r_scores)),
    }


# ─── Main Demo ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HYBRID SEARCH ENGINE: BM25 + Dense + RRF + Metadata Filter")
    print("=" * 70)

    # Build corpus and index
    chunks = build_corpus()
    engine = HybridSearchEngine()
    engine.build(chunks)

    # ── Demo 1: Semantic query ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("DEMO 1: Semantic query — no metadata filter")
    print("─" * 70)

    query = "How does ACI manage network policy?"
    results = engine.search(query, top_k=5)

    print(f"  Query: '{query}'")
    print(f"\n  {'Rank':<5} {'RRF':>7} {'BM25↑':>7} {'Dense↑':>7}  Chunk content (first 60 chars)")
    print(f"  {'─'*5} {'─'*7} {'─'*7} {'─'*7}  {'─'*60}")

    for i, r in enumerate(results, 1):
        bm25_r  = f"#{r.bm25_rank}"  if r.bm25_rank  else "—"
        dense_r = f"#{r.dense_rank}" if r.dense_rank else "—"
        print(f"  {i:<5} {r.score:>7.4f} {bm25_r:>7} {dense_r:>7}  {r.chunk.content[:60]}...")

    # ── Demo 2: Exact term query (BM25 advantage) ─────────────────────────────
    print("\n" + "─" * 70)
    print("DEMO 2: Exact term query — BM25 advantage for bug ID")
    print("─" * 70)

    query = "CSCvh23456"
    results = engine.search(query, top_k=5)

    print(f"  Query: '{query}'")
    print(f"\n  {'Rank':<5} {'RRF':>7} {'BM25↑':>7} {'Dense↑':>7}  Chunk ID / Content")
    print(f"  {'─'*5} {'─'*7} {'─'*7} {'─'*7}  {'─'*60}")

    for i, r in enumerate(results, 1):
        bm25_r  = f"#{r.bm25_rank}"  if r.bm25_rank  else "—"
        dense_r = f"#{r.dense_rank}" if r.dense_rank else "—"
        print(f"  {i:<5} {r.score:>7.4f} {bm25_r:>7} {dense_r:>7}  [{r.chunk.chunk_id}] {r.chunk.content[:60]}...")

    print(f"\n  KEY INSIGHT: BM25 finds the exact bug ID; dense has no signal for a random code.")

    # ── Demo 3: Filtered search ───────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("DEMO 3: Metadata filtered search — ReadyOps docs only")
    print("─" * 70)

    readyops_filter = MetadataFilter([
        {"field": "product", "op": "eq", "value": "ReadyOps"},
    ])
    query   = "How does ReadyOps validate changes before production?"
    results = engine.search(query, top_k=5, filter_=readyops_filter)

    matching = len(readyops_filter.filter_chunks(chunks))
    print(f"  Query:  '{query}'")
    print(f"  Filter: product='ReadyOps'  ({matching}/{len(chunks)} docs pass filter)")

    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.chunk.chunk_id}] {r.chunk.metadata['product']:<12} {r.chunk.content[:65]}...")

    # ── Demo 4: Complex filter — ACI + recent + core tier ─────────────────────
    print("\n" + "─" * 70)
    print("DEMO 4: Complex filter — ACI, v6.0, date >= 2025-01-01, tier=core")
    print("─" * 70)

    aci_recent_filter = MetadataFilter([
        {"field": "product", "op": "eq",  "value": "ACI"},
        {"field": "version", "op": "eq",  "value": "6.0"},
        {"field": "date",    "op": "gte", "value": "2025-01-01"},
        {"field": "tier",    "op": "eq",  "value": "core"},
    ])
    query   = "ACI leaf spine topology overview"
    results = engine.search(query, top_k=5, filter_=aci_recent_filter)

    matching = len(aci_recent_filter.filter_chunks(chunks))
    print(f"  Query:  '{query}'")
    print(f"  Filter: ACI + v6.0 + date>=2025 + tier=core  ({matching}/{len(chunks)} docs pass)")

    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.chunk.chunk_id}] date={r.chunk.metadata['date']} {r.chunk.content[:60]}...")

    if not results:
        print("  (no results — filter too tight for corpus)")

    # ── Demo 5: Build RAG context ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("DEMO 5: RAG context builder output")
    print("─" * 70)

    query   = "How does Cisco ACI handle multi-site deployments?"
    results = engine.search(query, top_k=4)
    context = engine.build_rag_context(results, token_budget=1000)

    print(f"  Query: '{query}'")
    print(f"\n  Built context ({len(context)} chars):\n")
    for line in context.split("\n"):
        print(f"    {line}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("EVALUATION: Precision@5 and Recall@5 on ground truth queries")
    print("─" * 70)

    eval_queries = [
        EvalQuery(
            query        = "ACI policy enforcement and EPG contracts",
            relevant_ids = ["c001", "c002", "c003"],
            description  = "ACI core policy",
        ),
        EvalQuery(
            query        = "ReadyOps validation before production deployment",
            relevant_ids = ["c008", "c009", "c010"],
            description  = "ReadyOps validation",
        ),
        EvalQuery(
            query        = "CSCvh23456 bug workaround",
            relevant_ids = ["c005"],
            description  = "Exact bug ID lookup",
        ),
        EvalQuery(
            query        = "Hypershield eBPF kernel policy",
            relevant_ids = ["c011", "c012"],
            description  = "Hypershield architecture",
        ),
        EvalQuery(
            query        = "ISE TrustSec SGT assignment",
            relevant_ids = ["c013"],
            description  = "ISE TrustSec",
        ),
    ]

    eval_results = evaluate_engine(engine, eval_queries)

    print(f"\n  {'Query':<35} {'P@5':>6}  {'R@5':>6}")
    print(f"  {'─'*35} {'─'*6}  {'─'*6}")
    for eq, p, r in eval_results["per_query"]:
        status = "PASS" if p >= 0.60 else "FAIL"
        print(f"  {eq.description:<35} {p:>6.2f}  {r:>6.2f}  {status}")

    print(f"\n  {'MEAN':>35} {eval_results['mean_p5']:>6.2f}  {eval_results['mean_r5']:>6.2f}")

    print(f"""
  ARCHITECTURE SUMMARY:
  ┌─────────────────────────────────────────────────────────────┐
  │  HybridSearchEngine                                         │
  │                                                             │
  │  query ──► BM25Index.search()     ──► ranked BM25 list     │
  │        ──► MetadataFilter         ──► filtered candidates   │
  │        ──► DenseIndex.search()    ──► ranked dense list     │
  │                    │                                        │
  │                    ▼                                        │
  │          reciprocal_rank_fusion() ──► fused RRF list        │
  │                    │                                        │
  │          build_rag_context()      ──► context string        │
  │                    │                                        │
  │         [Attach to Claude prompt] ──► LLM answer           │
  └─────────────────────────────────────────────────────────────┘

  PRODUCTION REPLACEMENTS:
    BM25Index    →  Elasticsearch (BM25 built-in) or OpenSearch
    DenseIndex   →  Qdrant, Weaviate, Pinecone, Milvus
    mock_embed() →  voyageai.Client().embed() with input_type="query"/"document"
    MetadataFilter → Qdrant payload filters / Weaviate where clauses
""")


if __name__ == "__main__":
    main()
