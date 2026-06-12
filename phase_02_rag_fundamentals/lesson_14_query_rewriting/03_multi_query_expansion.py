"""
FILE: 03_multi_query_expansion.py
LESSON: Phase 2 - Lesson 14 - Query Understanding and Rewriting
TOPIC: Multi-query expansion — broaden retrieval coverage with diverse paraphrases

WHAT THIS FILE TEACHES:
  - Multi-query expansion algorithm (generate N paraphrases, retrieve for each, RRF)
  - WHY each paraphrase covers a different region of the semantic space
  - Paraphrase generation with Claude (or mock)
  - How to measure recall improvement from multi-query
  - Deduplication and RRF fusion of multiple ranked lists
  - Cost/recall tradeoff: N=3 vs N=5 vs N=10

INSTALL: pip install anthropic python-dotenv numpy
"""

import os
import re
import hashlib
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    HAS_ANTHROPIC = False


# ─── Utilities ────────────────────────────────────────────────────────────────

def mock_embed(text: str, dims: int = 64) -> np.ndarray:
    keywords  = sorted(set(re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())))[:8]
    topic_key = " ".join(keywords)
    seed      = int(hashlib.md5(topic_key.encode()).hexdigest(), 16) % (2**32)
    rng       = np.random.RandomState(seed)
    full_seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
    full_rng  = np.random.RandomState(full_seed)
    v = rng.randn(dims).astype(np.float32) + full_rng.randn(dims).astype(np.float32) * 0.15
    return v / (np.linalg.norm(v) + 1e-10)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                        b / (np.linalg.norm(b) + 1e-10)))


# ─── Multi-Query Generator ────────────────────────────────────────────────────

MULTI_QUERY_SYSTEM = """You are a search query optimizer.
Given a user question, generate {n} semantically diverse paraphrases of that question.
Each paraphrase should:
  - Use different vocabulary (synonyms, related terms, alternative phrasings)
  - Cover the same information need from a slightly different angle
  - Be standalone (not reference "the above" or other paraphrases)
Output ONLY a numbered list, one paraphrase per line. No explanations."""

MULTI_QUERY_USER = """Original question: {question}

Generate {n} diverse paraphrases of this question for retrieval augmentation."""

# Mock paraphrases for demo without API key
MOCK_PARAPHRASES: dict[str, list[str]] = {
    "apic ha": [
        "What is the minimum APIC cluster size for high availability?",
        "How many APIC nodes are required to maintain HA in ACI?",
        "APIC controller redundancy requirements and node count",
        "ACI fabric management quorum configuration",
    ],
    "readyops": [
        "How does the Criterion Networks validation platform work?",
        "What is the ReadyOps promotion gate requirement?",
        "Continuous validation process for network changes",
        "Production-Representative environment testing before deployment",
    ],
    "epg contract": [
        "How are EPG permissions configured in ACI policy?",
        "What allows traffic between endpoint groups in ACI?",
        "ACI inter-EPG communication policy configuration",
        "Cisco ACI contract definition between EPGs",
    ],
}


def generate_paraphrases(query: str, n: int = 3) -> list[str]:
    """
    Generate N semantically diverse paraphrases of a query.

    WHY diverse paraphrases:
      A query like "APIC HA?" uses very different vocabulary from the indexed
      documentation which says "cluster requires minimum 3 nodes" and "quorum".
      Paraphrase 1 might use "cluster size" and "HA".
      Paraphrase 2 might use "node count" and "redundancy".
      Paraphrase 3 might use "quorum" and "availability".
      Together they cover the semantic region where the answer lives.

    Returns N paraphrases. Always includes the original query as paraphrase 0.
    """
    if HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model    = "claude-haiku-4-5-20251001",
            max_tokens = 300,
            system   = MULTI_QUERY_SYSTEM.format(n=n),
            messages = [{"role": "user", "content": MULTI_QUERY_USER.format(question=query, n=n)}],
        )
        text  = resp.content[0].text.strip()
        lines = re.findall(r"^\d+[.)]\s*(.+)$", text, re.MULTILINE)
        return lines[:n] if lines else [query]

    else:
        # Find closest mock
        q_low = query.lower()
        for key, paraphrases in MOCK_PARAPHRASES.items():
            if any(w in q_low for w in key.split()):
                return paraphrases[:n]
        # Generic fallback
        return [
            query,
            f"Details about {' '.join(query.split()[:4])}",
            f"Configuration and requirements for {query.split()[0]}",
        ][:n]


# ─── Multi-Query Retrieval ────────────────────────────────────────────────────

def rrf_fusion(
    result_lists: list[list[tuple[dict, float]]],
    k:            int = 60,
) -> list[tuple[dict, float]]:
    """
    Reciprocal Rank Fusion over multiple retrieval result lists.
    WHY k=60: see Lesson 9. Robust default that minimizes rank-flip sensitivity.
    """
    scores: dict[str, float] = defaultdict(float)
    docs:   dict[str, dict]  = {}

    for lst in result_lists:
        for rank, (doc, _) in enumerate(lst, start=1):
            doc_id = doc["id"]
            scores[doc_id] += 1.0 / (k + rank)
            docs[doc_id]    = doc

    combined = sorted(scores.items(), key=lambda x: -x[1])
    return [(docs[did], score) for did, score in combined]


@dataclass
class MultiQueryResult:
    """Result of multi-query expanded retrieval."""
    original_query:    str
    paraphrases:       list[str]
    per_query_results: list[list[tuple[dict, float]]]
    fused_results:     list[tuple[dict, float]]
    unique_docs:       int

    def display(self):
        print(f"\n  Original: '{self.original_query}'")
        print(f"  Paraphrases ({len(self.paraphrases)}):")
        for i, p in enumerate(self.paraphrases, 1):
            print(f"    {i}. '{p}'")
        print(f"\n  Per-query top result:")
        for i, (results, para) in enumerate(zip(self.per_query_results, [self.original_query] + self.paraphrases)):
            if results:
                doc, score = results[0]
                print(f"    [{i}] {score:.3f}  '{doc['content'][:60]}'")
        print(f"\n  Fused results ({self.unique_docs} unique docs):")
        for i, (doc, score) in enumerate(self.fused_results[:3], 1):
            print(f"    [{i}] rrf={score:.4f}  '{doc['content'][:65]}'")


def multi_query_retrieve(
    query:       str,
    corpus:      list[dict],
    n_paraphrases: int = 3,
    top_k_each:  int  = 5,
) -> MultiQueryResult:
    """
    Run multi-query retrieval: generate paraphrases, retrieve for each, fuse.

    Args:
        query:          Original user query.
        corpus:         List of documents with "id" and "content".
        n_paraphrases:  Number of paraphrases to generate.
        top_k_each:     Documents to retrieve per query before fusion.
    """
    paraphrases = generate_paraphrases(query, n=n_paraphrases)
    all_queries = [query] + paraphrases

    # Retrieve for each query variant
    all_results = []
    for q in all_queries:
        q_vec   = mock_embed(q)
        results = sorted(
            [(doc, cosine_sim(q_vec, mock_embed(doc["content"]))) for doc in corpus],
            key=lambda x: -x[1]
        )[:top_k_each]
        all_results.append(results)

    # Fuse all result lists
    fused   = rrf_fusion(all_results)
    unique  = len({doc["id"] for results in all_results for doc, _ in results})

    return MultiQueryResult(
        original_query    = query,
        paraphrases       = paraphrases,
        per_query_results = all_results,
        fused_results     = fused,
        unique_docs       = unique,
    )


# ─── Recall Measurement ───────────────────────────────────────────────────────

def measure_recall(
    results:      list[tuple[dict, float]],
    relevant_ids: list[str],
    k:            int = 5,
) -> float:
    """Recall@K: fraction of relevant docs in top-K results."""
    top_ids = {doc["id"] for doc, _ in results[:k]}
    hits    = sum(1 for rid in relevant_ids if rid in top_ids)
    return hits / max(len(relevant_ids), 1)


# ─── Demo ─────────────────────────────────────────────────────────────────────

CORPUS = [
    {"id": "c001", "content": "The APIC cluster requires minimum 3 nodes for high availability quorum."},
    {"id": "c002", "content": "EPGs define policy groups. Contracts permit inter-EPG traffic in ACI."},
    {"id": "c003", "content": "ReadyOps validates changes with 100% validation pass rate requirement."},
    {"id": "c004", "content": "Hypershield uses eBPF for kernel-level microsegmentation policy."},
    {"id": "c005", "content": "ACI Multi-Pod extends fabric across geographies using VXLAN IPN."},
    {"id": "c006", "content": "ISE TrustSec assigns SGTs at network authentication for microsegmentation."},
    {"id": "c007", "content": "APIC REST API authenticates via aaaLogin endpoint on port 443."},
    {"id": "c008", "content": "ReadyOps agent classes: Health Posture, Validation, Operational, Stress."},
    {"id": "c009", "content": "ACI 6.0 supports up to 200 leaf switches per pod in the fabric."},
    {"id": "c010", "content": "Promotion gate in ReadyOps blocks changes until 100% tests pass."},
]


def run_multi_query_demo():
    """Show multi-query expansion improving recall over single-query retrieval."""

    print("=" * 70)
    print("MULTI-QUERY EXPANSION: Broader Retrieval Coverage via Paraphrases")
    print("=" * 70)

    test_cases = [
        {
            "query":        "APIC controller redundancy",
            "relevant_ids": ["c001", "c007"],
            "description":  "Short query with vocabulary mismatch",
        },
        {
            "query":        "ReadyOps promotion requirements",
            "relevant_ids": ["c003", "c008", "c010"],
            "description":  "Multi-doc query across ReadyOps",
        },
    ]

    for tc in test_cases:
        query   = tc["query"]
        rel_ids = tc["relevant_ids"]

        # Single-query baseline
        q_vec    = mock_embed(query)
        baseline = sorted([(doc, cosine_sim(q_vec, mock_embed(doc["content"]))) for doc in CORPUS],
                          key=lambda x: -x[1])
        recall_baseline = measure_recall(baseline, rel_ids, k=5)

        # Multi-query
        mq_result       = multi_query_retrieve(query, CORPUS, n_paraphrases=3, top_k_each=5)
        recall_mq       = measure_recall(mq_result.fused_results, rel_ids, k=5)

        print(f"\n  [{tc['description']}]")
        mq_result.display()
        print(f"\n  Recall@5 — Baseline: {recall_baseline:.0%} → Multi-query: {recall_mq:.0%}")
        gain = recall_mq - recall_baseline
        print(f"  Improvement: {gain:+.0%}")


def cost_recall_tradeoff():
    """Show how N paraphrases trades off cost vs recall improvement."""

    print("\n" + "=" * 70)
    print("COST vs RECALL TRADEOFF: How Many Paraphrases?")
    print("=" * 70)

    print(f"""
  Setup: 1 query + N paraphrases, each requiring 1 embedding call + retrieval.
  Assumes each paraphrase recovers ~15% of previously missed relevant docs.

  N  Queries  Extra LLM calls  Approx recall gain   Latency (est.)
  ─  ───────  ───────────────  ──────────────────   ───────────────
  1  1        0                Baseline             50ms
  2  2        1 (gen call)     +10–15%             100ms
  3  3        1 (gen call)     +15–25%             150ms
  5  5        1 (gen call)     +20–30%             200ms
  10 10       1 (gen call)     +25–35%             350ms

  WHY diminishing returns:
    First 3 paraphrases cover the main synonyms and phrasings.
    Paraphrases 4–10 increasingly overlap with earlier ones.
    The marginal recall gain per additional query drops rapidly.

  RECOMMENDATION: N=3 for most production use cases.
    - One generation call (Haiku, ~0.1ms, ~$0.001) produces 3 paraphrases.
    - 3× retrieval cost (fast: <50ms with HNSW index).
    - Covers the most common vocabulary mismatch cases.
    - Use N=5 only for high-recall use cases (compliance search, incident triage).

  DO NOT use multi-query for:
    - Exact-term queries (bug IDs, CVEs): BM25 handles these without paraphrases.
    - Very long queries (>20 words): already has enough vocabulary.
    - Low-latency requirements (<100ms): the extra retrieval cost may be unacceptable.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_multi_query_demo()
    cost_recall_tradeoff()
