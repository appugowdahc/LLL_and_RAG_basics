"""
FILE: 03_cohere_rerank.py
LESSON: Phase 2 - Lesson 15 - Reranking
TOPIC: Cohere Rerank API — managed cross-encoder reranking as a service

WHAT THIS FILE TEACHES:
  - Cohere Rerank API: no model hosting, production-grade quality
  - rerank-v3.5 model: state-of-the-art relevance scoring
  - How to integrate Cohere Rerank into the two-stage pipeline
  - JSON mode and multi-field document reranking
  - Score normalization and score_threshold filtering
  - WHY managed reranking often outperforms self-hosted small cross-encoders

INSTALL: pip install cohere python-dotenv
API KEY: https://cohere.com — free tier: 100 API calls/min, 5M tokens/month
"""

import os
import re
import math
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    import cohere
    HAS_COHERE = bool(os.environ.get("COHERE_API_KEY"))
except ImportError:
    HAS_COHERE = False


# ─── Mock Utilities ───────────────────────────────────────────────────────────

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


def _mock_relevance_score(query: str, document: str) -> float:
    """
    Mock Cohere-like relevance score in [0, 1].
    Uses semantic similarity + term overlap, similar to 02_cross_encoder_reranker.py.
    """
    stop  = {"what", "how", "does", "the", "and", "for", "with", "are", "is"}
    q_tok = {t.lower() for t in re.findall(r"\b\w{3,}\b", query) if t.lower() not in stop}
    d_tok = {t.lower() for t in re.findall(r"\b\w{3,}\b", document) if t.lower() not in stop}
    term_overlap = len(q_tok & d_tok) / max(len(q_tok), 1)
    semantic     = cosine_sim(mock_embed(query), mock_embed(document))
    length_f     = min(1.0, len(document) / 100.0)
    score        = 0.45 * semantic + 0.40 * term_overlap + 0.15 * length_f
    return float(np.clip(score, 0.0, 1.0))


# ─── CohereReranker ───────────────────────────────────────────────────────────

@dataclass
class CohereRerankResult:
    """One document's reranked position from Cohere Rerank."""
    original_index: int     # position in the input candidates list
    new_rank:       int     # 1-based rank after reranking
    relevance_score: float  # Cohere relevance score in [0, 1]
    doc:            dict
    content:        str     # text that was ranked


class CohereReranker:
    """
    Wraps the Cohere Rerank API (or mock) in the two-stage pipeline interface.

    WHY Cohere over self-hosted:
      1. No infrastructure: no GPU server, no model download, no version management.
      2. Model quality: rerank-v3.5 is trained on 100s of millions of (query, doc) pairs —
         far more data than most self-hosted cross-encoder fine-tunes.
      3. Multi-lingual: rerank-v3.5 handles 100+ languages natively.
      4. JSON document support: rank by a specific field, not just raw text.
      5. Pricing: ~$0.001 per 1K results — cheaper than GPU hosting at low volume.
    """

    def __init__(self, model: str = "rerank-v3.5"):
        """
        Args:
            model: Cohere rerank model name.
                   "rerank-v3.5" — latest, best quality, multilingual.
                   "rerank-english-v3.0" — English only, slightly faster.
        """
        self.model = model
        self._client = cohere.Client() if HAS_COHERE else None

    def rerank(
        self,
        query:        str,
        candidates:   list[tuple[dict, float]],   # (doc, bi_score)
        top_m:        int   = 5,
        score_threshold: float = 0.0,             # drop results below this relevance
    ) -> list[CohereRerankResult]:
        """
        Call Cohere Rerank API (or mock) to re-order stage-1 candidates.

        Args:
            query:      User's query string.
            candidates: List of (doc, bi_score) from stage-1 retrieval.
            top_m:      Maximum number of results to return.
            score_threshold: Drop candidates with relevance below this value.
                             Use 0.1–0.3 to filter obviously irrelevant docs.
        """
        docs     = [doc["content"] for doc, _ in candidates]
        orig_docs = [doc for doc, _ in candidates]

        if HAS_COHERE:
            response = self._client.rerank(
                model     = self.model,
                query     = query,
                documents = docs,
                top_n     = top_m,
            )
            scored = [
                (r.index, r.relevance_score)
                for r in response.results
                if r.relevance_score >= score_threshold
            ]
        else:
            raw = [(i, _mock_relevance_score(query, doc)) for i, doc in enumerate(docs)]
            raw.sort(key=lambda x: -x[1])
            scored = [(i, s) for i, s in raw if s >= score_threshold][:top_m]

        results = []
        for new_rank, (orig_idx, relevance) in enumerate(scored, 1):
            results.append(CohereRerankResult(
                original_index  = orig_idx,
                new_rank        = new_rank,
                relevance_score = relevance,
                doc             = orig_docs[orig_idx],
                content         = docs[orig_idx],
            ))
        return results


# ─── Multi-Field Document Reranking ───────────────────────────────────────────

def rerank_structured_docs(
    query:      str,
    candidates: list[dict],
    rank_field: str = "content",
) -> list[dict]:
    """
    Rerank documents by a specific field when docs have multiple text fields.

    WHY multi-field: Production docs often have:
      - "title": short heading
      - "content": main body
      - "section": section heading
    Ranking only on "content" ignores the title signal.
    Cohere supports passing dicts directly — the model sees all fields.

    For the mock, we concatenate all string fields.
    """
    if HAS_COHERE:
        client   = cohere.Client()
        response = client.rerank(
            model     = "rerank-v3.5",
            query     = query,
            documents = candidates,     # WHY pass dicts: Cohere ranks on all text fields
            top_n     = min(5, len(candidates)),
        )
        return [candidates[r.index] for r in response.results]
    else:
        # Mock: build text from all string fields and score
        def doc_text(doc):
            return " ".join(str(v) for v in doc.values() if isinstance(v, str))

        scored = sorted(
            [(doc, _mock_relevance_score(query, doc_text(doc))) for doc in candidates],
            key=lambda x: -x[1]
        )
        return [doc for doc, _ in scored[:5]]


# ─── Demo ─────────────────────────────────────────────────────────────────────

CORPUS = [
    {"id": "c01", "content": "A minimum of 3 APIC nodes form a cluster to maintain quorum and HA.",          "source": "ACI Design Guide"},
    {"id": "c02", "content": "APIC communicates with leaf and spine switches using OpFlex policy protocol.",  "source": "ACI Architecture"},
    {"id": "c03", "content": "For ACI HA, deploy 3 APIC nodes in separate failure domains.",                 "source": "ACI Best Practices"},
    {"id": "c04", "content": "APIC nodes are managed through the ACI GUI or REST API on port 443.",          "source": "ACI Operations"},
    {"id": "c05", "content": "High availability requires an odd number of APIC nodes to avoid split-brain.", "source": "ACI Design Guide"},
    {"id": "c06", "content": "The APIC cluster health is visible in the Fault Manager dashboard.",           "source": "ACI Monitoring"},
    {"id": "c07", "content": "Cisco recommends deploying APIC on dedicated hardware or VMware vSphere.",     "source": "ACI Install Guide"},
    {"id": "c08", "content": "BGP route reflection is used as the ACI Multi-Pod IPN control plane.",         "source": "Multi-Pod Guide"},
    {"id": "c09", "content": "A 3-node APIC cluster maintains policy even when one physical APIC fails.",    "source": "ACI Design Guide"},
    {"id": "c10", "content": "ACI leaf switches connect to endpoints; spine switches provide fabric links.",  "source": "ACI Architecture"},
]


def run_cohere_rerank_demo():
    """Compare bi-encoder ranking vs Cohere Rerank on ACI queries."""

    print("=" * 70)
    print("COHERE RERANK: Managed Reranking Service")
    print("=" * 70)

    reranker = CohereReranker(model="rerank-v3.5")
    query    = "Minimum APIC nodes required for cluster high availability"

    # Stage 1: Bi-encoder retrieval
    q_vec      = mock_embed(query)
    candidates = sorted(
        [(doc, cosine_sim(q_vec, mock_embed(doc["content"]))) for doc in CORPUS],
        key=lambda x: -x[1],
    )

    print(f"\n  Query: '{query}'")
    print(f"\n  BEFORE (bi-encoder top-10):")
    print(f"  {'#':<4} {'Score':>6}  Document")
    print(f"  {'─'*3} {'─'*6}  {'─'*60}")
    for i, (doc, score) in enumerate(candidates, 1):
        print(f"  {i:<4} {score:>6.3f}  '{doc['content'][:60]}'")

    # Stage 2: Cohere reranking
    results = reranker.rerank(query, candidates, top_m=5, score_threshold=0.1)

    print(f"\n  AFTER (Cohere Rerank top-5):")
    print(f"  {'New':<4} {'Old':<4} {'Relevance':>9}  Document")
    print(f"  {'─'*3} {'─'*3} {'─'*9}  {'─'*55}")
    for r in results:
        orig_bi = candidates[r.original_index][1]
        print(f"  {r.new_rank:<4} {r.original_index+1:<4} {r.relevance_score:>9.4f}  '{r.content[:55]}'")

    used_api = "LIVE Cohere API" if HAS_COHERE else "MOCK (no COHERE_API_KEY)"
    print(f"\n  [{used_api}]")


def score_threshold_demo():
    """Show how score_threshold filters irrelevant documents."""

    print("\n" + "=" * 70)
    print("SCORE THRESHOLD: Filtering Irrelevant Candidates")
    print("=" * 70)

    reranker = CohereReranker()
    query    = "APIC node count for high availability"
    q_vec    = mock_embed(query)
    candidates = sorted(
        [(doc, cosine_sim(q_vec, mock_embed(doc["content"]))) for doc in CORPUS],
        key=lambda x: -x[1],
    )[:10]

    for threshold in [0.0, 0.15, 0.30]:
        results = reranker.rerank(query, candidates, top_m=10, score_threshold=threshold)
        print(f"\n  score_threshold={threshold}  →  {len(results)} docs pass filter")
        for r in results:
            print(f"    [{r.new_rank}] {r.relevance_score:.3f}  '{r.content[:55]}'")

    print(f"""
  HOW TO CHOOSE score_threshold:
    0.0:  No filter — pass all to LLM (safest if context budget allows)
    0.1:  Remove obvious noise — docs that are completely off-topic
    0.2:  Remove weakly relevant docs — use when top-M is small (M ≤ 3)
    0.3+: Aggressive — only use if you've validated on your eval set first

  RISK of too-high threshold:
    On edge-case queries, all documents may score < 0.3 → empty context → LLM refuses.
    Always have a fallback: if len(results) == 0, use top-3 regardless of threshold.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_cohere_rerank_demo()
    score_threshold_demo()
