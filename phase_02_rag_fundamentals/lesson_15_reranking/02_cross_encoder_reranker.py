"""
FILE: 02_cross_encoder_reranker.py
LESSON: Phase 2 - Lesson 15 - Reranking
TOPIC: Cross-encoder reranking — joint query+document scoring

WHAT THIS FILE TEACHES:
  - Cross-encoder architecture: [CLS] query [SEP] document [SEP] → score
  - WHY cross-attention between query and document beats independent embeddings
  - Implementing a mock cross-encoder that simulates joint scoring
  - How to integrate cross-encoder into the two-stage retrieval pipeline
  - Batch scoring for latency efficiency
  - sentence-transformers CrossEncoder API reference

INSTALL: numpy (core); sentence-transformers (optional, real cross-encoder)
"""

import re
import math
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from sentence_transformers import CrossEncoder as STCrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# ─── Mock Cross-Encoder ───────────────────────────────────────────────────────

def mock_embed(text: str, dims: int = 64) -> np.ndarray:
    keywords  = sorted(set(re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())))[:8]
    topic_key = " ".join(keywords)
    seed      = int(hashlib.md5(topic_key.encode()).hexdigest(), 16) % (2**32)
    rng       = np.random.RandomState(seed)
    full_rng  = np.random.RandomState(int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31))
    v = rng.randn(dims).astype(np.float32) + full_rng.randn(dims).astype(np.float32) * 0.15
    return v / (np.linalg.norm(v) + 1e-10)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                        b / (np.linalg.norm(b) + 1e-10)))


def _term_overlap(query: str, document: str) -> float:
    """
    Fraction of meaningful query terms found in the document.
    WHY meaningful terms only: stop words ('the', 'is') add noise.
    This simulates a key signal in cross-encoder scoring: lexical presence.
    """
    stop  = {"what", "how", "does", "the", "and", "for", "with", "are", "can",
             "will", "is", "in", "of", "to", "a", "an", "that", "this", "be"}
    q_tok = {t.lower() for t in re.findall(r"\b\w{3,}\b", query)   if t.lower() not in stop}
    d_tok = {t.lower() for t in re.findall(r"\b\w{3,}\b", document) if t.lower() not in stop}
    if not q_tok:
        return 0.5
    return len(q_tok & d_tok) / len(q_tok)


def mock_cross_encoder_score(query: str, document: str) -> float:
    """
    Simulate cross-encoder scoring.

    Real cross-encoders compute a single relevance score by passing
    [CLS] query [SEP] document [SEP] through a transformer with FULL
    cross-attention — every query token can attend to every document token.

    This mock approximates that with three signals:
      1. Semantic similarity (embedding cosine) — topic match
      2. Term overlap — direct keyword presence
      3. Length penalty — very short docs rarely fully answer complex queries

    WHY three signals:
      A real cross-encoder also implicitly captures all three of these
      (plus syntactic relationships, question-answer span alignment, etc.)
      This is a simplified but directionally correct approximation.

    Returns a score in [0.0, 1.0].
    """
    semantic  = cosine_sim(mock_embed(query), mock_embed(document))
    term_ov   = _term_overlap(query, document)

    # Length penalty: documents < 20 chars rarely contain a full answer
    length_factor = min(1.0, len(document) / 80.0)

    # Weighted combination
    score = 0.50 * semantic + 0.35 * term_ov + 0.15 * length_factor
    return float(np.clip(score, 0.0, 1.0))


# ─── CrossEncoderReranker ────────────────────────────────────────────────────

@dataclass
class RerankResult:
    """One document's reranked position and scores."""
    original_rank:  int
    new_rank:       int
    bi_score:       float    # stage-1 cosine similarity
    ce_score:       float    # cross-encoder relevance score
    doc:            dict
    rank_changed:   bool = field(init=False)

    def __post_init__(self):
        self.rank_changed = self.original_rank != self.new_rank


class CrossEncoderReranker:
    """
    Two-stage reranker using a cross-encoder.

    Stage 1: Bi-encoder retrieval (provided externally as top-K candidates).
    Stage 2: Cross-encoder re-scores each (query, document) pair and re-ranks.

    WHY this class exists:
      Encapsulates the reranking logic so the RAG pipeline only calls:
        candidates = retriever.search(query, top_k=100)
        reranked   = reranker.rerank(query, candidates, top_m=5)
        context    = build_context(reranked)
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: If sentence-transformers is installed, use this model.
                        None falls back to mock scoring (no ML library needed).
        """
        self.model_name = model_name
        self._model = None

        if model_name and HAS_SENTENCE_TRANSFORMERS:
            self._model = STCrossEncoder(model_name)  # WHY lazy load: ~200MB model

    def score_pair(self, query: str, document: str) -> float:
        """Score a single (query, document) pair."""
        if self._model:
            return float(self._model.predict([(query, document)])[0])
        return mock_cross_encoder_score(query, document)

    def score_batch(self, query: str, documents: list[str]) -> list[float]:
        """
        Score all (query, doc) pairs in one batch.
        WHY batch: real cross-encoders run on GPU; batching amortizes overhead.
        """
        if self._model:
            pairs = [(query, doc) for doc in documents]
            return [float(s) for s in self._model.predict(pairs)]
        return [mock_cross_encoder_score(query, doc) for doc in documents]

    def rerank(
        self,
        query:         str,
        candidates:    list[tuple[dict, float]],  # (doc, bi_score) from stage 1
        top_m:         int = 5,
    ) -> list[RerankResult]:
        """
        Re-rank stage-1 candidates using cross-encoder scoring.

        Args:
            query:      Original user query.
            candidates: List of (doc, bi_encoder_score) from retrieval.
            top_m:      How many top candidates to return to the LLM.

        Returns:
            List of RerankResult, sorted by ce_score descending, capped at top_m.
        """
        docs     = [doc["content"] for doc, _ in candidates]
        ce_scores = self.score_batch(query, docs)

        # Zip original rank, doc, bi_score, ce_score
        scored = [
            (orig_rank + 1, doc, bi_score, ce_score)
            for orig_rank, ((doc, bi_score), ce_score)
            in enumerate(zip(candidates, ce_scores))
        ]

        # Sort by ce_score
        scored.sort(key=lambda x: -x[3])

        results = [
            RerankResult(
                original_rank = orig_rank,
                new_rank      = new_rank + 1,
                bi_score      = bi_score,
                ce_score      = ce_score,
                doc           = doc,
            )
            for new_rank, (orig_rank, doc, bi_score, ce_score) in enumerate(scored)
        ]
        return results[:top_m]


# ─── Demo ─────────────────────────────────────────────────────────────────────

CORPUS = [
    {"id": "c001", "content": "A minimum of 3 APIC nodes form a cluster to maintain quorum and HA."},
    {"id": "c002", "content": "APIC communicates with leaf and spine switches using OpFlex protocol."},
    {"id": "c003", "content": "For ACI HA, deploy 3 APIC nodes in separate failure domains."},
    {"id": "c004", "content": "APIC nodes are managed through the ACI GUI or REST API on port 443."},
    {"id": "c005", "content": "High availability in ACI requires an odd number of APIC nodes to avoid split-brain."},
    {"id": "c006", "content": "The APIC cluster health is visible in the Fault Manager dashboard."},
    {"id": "c007", "content": "Cisco recommends deploying APIC on dedicated hardware or VMware vSphere."},
    {"id": "c008", "content": "BGP route reflection is used as the ACI Multi-Pod IPN control plane."},
    {"id": "c009", "content": "A 3-node APIC cluster maintains policy even when one physical APIC fails."},
    {"id": "c010", "content": "ACI leaf switches connect to endpoints; spine switches provide fabric links."},
]


def run_reranking_demo():
    """Show before/after ranking: bi-encoder retrieval vs cross-encoder reranking."""

    print("=" * 70)
    print("CROSS-ENCODER RERANKING: Before and After")
    print("=" * 70)

    query    = "How many APIC nodes are required for high availability?"
    reranker = CrossEncoderReranker()

    # Stage 1: Bi-encoder retrieval
    q_vec      = mock_embed(query)
    candidates = sorted(
        [(doc, cosine_sim(q_vec, mock_embed(doc["content"]))) for doc in CORPUS],
        key=lambda x: -x[1],
    )

    print(f"\n  Query: '{query}'")
    print(f"\n  STAGE 1 — Bi-encoder ranking (top-10):")
    print(f"  {'Rank':<5} {'Score':>6}  Document")
    print(f"  {'─'*4} {'─'*6}  {'─'*55}")
    for rank, (doc, score) in enumerate(candidates, 1):
        print(f"  {rank:<5} {score:>6.3f}  '{doc['content'][:58]}'")

    # Stage 2: Cross-encoder reranking
    results = reranker.rerank(query, candidates, top_m=5)

    print(f"\n  STAGE 2 — Cross-encoder re-ranked (top-5):")
    print(f"  {'New':<5} {'Old':<5} {'Moved':>6}  {'CE':>6}  Document")
    print(f"  {'─'*4} {'─'*4} {'─'*6}  {'─'*6}  {'─'*50}")
    for r in results:
        delta  = r.original_rank - r.new_rank
        arrow  = f"▲{delta}" if delta > 0 else (f"▼{abs(delta)}" if delta < 0 else "  =")
        print(f"  {r.new_rank:<5} {r.original_rank:<5} {arrow:>6}  {r.ce_score:>6.3f}  '{r.doc['content'][:50]}'")

    # Show rank changes
    moved_up   = sum(1 for r in results if r.original_rank > r.new_rank)
    moved_down = sum(1 for r in results if r.original_rank < r.new_rank)
    print(f"\n  Summary: {moved_up} docs moved up, {moved_down} moved down in top-5.")


def model_reference():
    """Production cross-encoder models reference with trade-offs."""

    print("\n" + "=" * 70)
    print("CROSS-ENCODER MODELS: Production Reference")
    print("=" * 70)
    print(f"""
  Model                                   Size    MRR@10  Latency/query
  ─────────────────────────────────────── ─────── ─────── ─────────────
  cross-encoder/ms-marco-MiniLM-L-6-v2   22MB    0.388   ~5ms (CPU)
  cross-encoder/ms-marco-MiniLM-L-12-v2  33MB    0.395   ~10ms (CPU)
  cross-encoder/ms-marco-electra-base    445MB   0.409   ~25ms (GPU)
  BAAI/bge-reranker-base                 278MB   0.415   ~15ms (GPU)
  BAAI/bge-reranker-large                560MB   0.430   ~30ms (GPU)

  HOW TO USE (sentence-transformers):

    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Score single pair
    score = model.predict([("query text", "document text")])

    # Score batch (always preferred — GPU parallelism)
    scores = model.predict([
        ("query text", "doc1"),
        ("query text", "doc2"),
        ("query text", "doc3"),
    ])
    # scores is a numpy array of shape (3,)

  PERFORMANCE TIPS:
    - Always batch all K candidates in one .predict() call.
    - Use GPU: 100 docs batch ≈ 30ms on RTX 3090, ≈ 500ms on CPU.
    - For very low latency: distil models (L-6) are 4× faster than base.
    - Max input length: 512 tokens for most BERT-based models.
      Truncate document to 400 tokens before passing to cross-encoder.

  TRUNCATION STRATEGY:
    - Truncate from the end (tail) — not the beginning.
    - Keep the first 400 tokens: intro sentences usually have the key fact.
    - For passage ranking: each chunk should already be ≤ 256 tokens.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_reranking_demo()
    model_reference()
