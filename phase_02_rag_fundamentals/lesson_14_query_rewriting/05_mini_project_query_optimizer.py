"""
FILE: 05_mini_project_query_optimizer.py
LESSON: Phase 2 - Lesson 14 - Query Understanding and Rewriting
TOPIC: Complete query optimization pipeline combining all four techniques

WHAT THIS FILE TEACHES:
  - How to orchestrate analysis → rewrite selection → retrieval → synthesis
  - Decision tree: which rewriting technique based on query analysis
  - Unified QueryOptimizer class with all four technique integrations
  - Measuring improvement in retrieval coverage (recall@K)
  - Production patterns: cost budgeting, latency, fallback
  - WHY query rewriting is the cheapest per-query improvement in RAG

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


# ─── Utilities (identical across lesson files) ────────────────────────────────

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


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ─── Query Analysis (inline from 01) ─────────────────────────────────────────

QUERY_TYPES = {
    "exact_term": {
        "signals":      [r"CVE-\d{4}-\d+", r"CSC[a-z]{2}\d+", r"\b\d+\.\d+\.\d+\b",
                         r"\bN[0-9]K-[A-Z0-9-]+\b"],
        "best_rewrite": "NONE",
    },
    "negation": {
        "signals":      [r"\bnot\s+\w+\b", r"\bexcept\b", r"\bexcluding\b"],
        "best_rewrite": "FILTER",
    },
    "comparative": {
        "signals":      [r"\bdifference between\b", r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b"],
        "best_rewrite": "DECOMPOSE",
    },
    "multi_hop": {
        "signals":      [r"\band (what|how|why)\b", r"\bwhich\b.{0,30}\band\b"],
        "best_rewrite": "DECOMPOSE",
    },
    "procedural": {
        "signals":      [r"\bhow (do|to|can)\b", r"\bsteps? to\b", r"\bconfigure\b"],
        "best_rewrite": "HYDE+EXPAND",
    },
    "factual": {
        "signals":      [r"\bwhat is\b", r"\bhow many\b", r"\bwhat are the\b", r"\bdefine\b"],
        "best_rewrite": "HYDE",
    },
    "general": {
        "signals":      [r"\bexplain\b", r"\bhow does .{0,20} work\b"],
        "best_rewrite": "MULTI_QUERY",
    },
}

TYPE_ORDER = ["exact_term", "negation", "comparative", "multi_hop", "procedural", "factual", "general"]


@dataclass
class QueryAnalysis:
    query:              str
    query_type:         str
    complexity:         float
    technique:          str       # NONE / HYDE / MULTI_QUERY / DECOMPOSE / FILTER
    exact_terms:        list[str]
    word_count:         int


def analyze_query(query: str) -> QueryAnalysis:
    q_low = query.lower()

    exact_terms = []
    for p in QUERY_TYPES["exact_term"]["signals"]:
        exact_terms.extend(re.findall(p, query, re.IGNORECASE))

    detected = "general"
    for qt in TYPE_ORDER:
        for p in QUERY_TYPES[qt]["signals"]:
            if re.search(p, q_low):
                detected = qt
                break
        if detected != "general" or qt == "general":
            break

    complexity = 0.0
    if len(query.split()) <= 4:            complexity += 0.3
    if q_low.count(" and ") >= 1:         complexity += 0.3
    if re.search(r"\?.*\?", query):       complexity += 0.4
    if detected in ("multi_hop", "comparative"): complexity += 0.2
    complexity = min(1.0, complexity)

    # Map best_rewrite → canonical technique name for optimizer router
    technique_map = {
        "NONE":        "PASSTHROUGH",
        "FILTER":      "PASSTHROUGH",
        "DECOMPOSE":   "DECOMPOSE",
        "HYDE+EXPAND": "HYDE",
        "HYDE":        "HYDE",
        "MULTI_QUERY": "MULTI_QUERY",
    }
    raw_technique = QUERY_TYPES[detected]["best_rewrite"]
    technique     = technique_map.get(raw_technique, "MULTI_QUERY")

    return QueryAnalysis(
        query       = query,
        query_type  = detected,
        complexity  = complexity,
        technique   = technique,
        exact_terms = exact_terms,
        word_count  = len(query.split()),
    )


# ─── Corpus ───────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    {"id": "kb001", "content": "The APIC cluster requires a minimum of 3 nodes for high availability quorum.", "tags": ["apic", "ha"]},
    {"id": "kb002", "content": "EPGs define policy groups in ACI. Contracts permit inter-EPG traffic.", "tags": ["epg", "contract"]},
    {"id": "kb003", "content": "ReadyOps is Criterion Networks' continuous validation platform for enterprise network infrastructure.", "tags": ["readyops"]},
    {"id": "kb004", "content": "ReadyOps validates changes in Production-Representative environment; 100% pass rate required for promotion.", "tags": ["readyops", "promotion"]},
    {"id": "kb005", "content": "ReadyOps agent classes: Health and Posture, Validation, Operational, Stress and Adversarial.", "tags": ["readyops", "agents"]},
    {"id": "kb006", "content": "Cisco Hypershield uses eBPF for kernel-level microsegmentation at the workload.", "tags": ["hypershield", "ebpf"]},
    {"id": "kb007", "content": "Hypershield integrates with ACI EPG membership propagated via APIC.", "tags": ["hypershield", "aci"]},
    {"id": "kb008", "content": "ACI Multi-Pod extends the fabric across geographies using a VXLAN IPN.", "tags": ["aci", "multipod"]},
    {"id": "kb009", "content": "ISE TrustSec assigns Security Group Tags (SGTs) at authentication for microsegmentation.", "tags": ["ise", "trustsec"]},
    {"id": "kb010", "content": "The ReadyOps promotion gate opens only when all validation tests pass at 100%.", "tags": ["readyops", "promotion"]},
    {"id": "kb011", "content": "Cisco ACI 6.0 supports up to 200 leaf switches per pod with VXLAN fabric.", "tags": ["aci", "scale"]},
    {"id": "kb012", "content": "APIC REST API exposes aaaLogin on port 443 for authentication.", "tags": ["apic", "api"]},
]


# ─── Retrieval (baseline) ─────────────────────────────────────────────────────

def vector_search(query_vec: np.ndarray, top_k: int = 5) -> list[tuple[dict, float]]:
    """Simple brute-force dense retrieval over KNOWLEDGE_BASE."""
    scored = sorted(
        [(doc, cosine_sim(query_vec, mock_embed(doc["content"]))) for doc in KNOWLEDGE_BASE],
        key=lambda x: -x[1]
    )
    return scored[:top_k]


def rrf_fusion(result_lists: list[list[tuple[dict, float]]], k: int = 60) -> list[tuple[dict, float]]:
    """Reciprocal Rank Fusion from Lesson 9/14."""
    scores: dict[str, float] = defaultdict(float)
    docs:   dict[str, dict]  = {}
    for lst in result_lists:
        for rank, (doc, _) in enumerate(lst, start=1):
            scores[doc["id"]] += 1.0 / (k + rank)
            docs[doc["id"]]    = doc
    return sorted([(docs[did], sc) for did, sc in scores.items()], key=lambda x: -x[1])


# ─── Rewriting Implementations ────────────────────────────────────────────────

MOCK_HYDE_RESPONSES = {
    "apic":       "The APIC controller cluster in Cisco ACI requires a minimum of 3 nodes to maintain quorum and HA. A single APIC node failure is tolerated without disrupting fabric policy management.",
    "readyops":   "ReadyOps by Criterion Networks validates changes in a Production-Representative environment before promotion. All agent tests including Health, Validation, and Operational must pass at 100%.",
    "hypershield":"Cisco Hypershield leverages eBPF to enforce microsegmentation policy at kernel level within workloads. Integration with ACI EPG membership is delivered via APIC for consistent policy.",
    "epg":        "Endpoint Groups (EPGs) are the policy boundary in Cisco ACI. Traffic between EPGs is governed by contracts defining permitted communication.",
}

MOCK_PARAPHRASES = {
    "apic ha":        ["What is the APIC cluster HA node requirement?", "Minimum controller nodes for ACI fabric management redundancy", "APIC quorum size for high availability"],
    "readyops":       ["How does the Criterion Networks validation platform work?", "Production-Representative validation process", "Continuous validation before change promotion"],
    "hypershield":    ["Cisco Hypershield eBPF microsegmentation", "Kernel-level workload policy enforcement", "Hypershield ACI EPG integration"],
    "epg contract":   ["ACI endpoint group policy configuration", "How inter-EPG traffic is permitted in ACI", "Cisco ACI contract and EPG relationship"],
}

MOCK_DECOMPOSITIONS = {
    "readyops aci":   ["What is ReadyOps?", "How does ReadyOps integrate with Cisco ACI?"],
    "hypershield aci":["What is Cisco Hypershield?", "How does Hypershield integrate with ACI EPGs?"],
    "compare":        ["What is {a}?", "What is {b}?", "What are the differences between {a} and {b}?"],
}


def rewrite_hyde(query: str) -> np.ndarray:
    """
    Generate hypothetical document embedding and blend with query (alpha=0.3).
    WHY blend: anchor with original query to guard against HyDE hallucination.
    """
    if HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 120,
            system     = "Write a 2-3 sentence technical documentation passage that directly answers the question. Use domain-specific vocabulary.",
            messages   = [{"role": "user", "content": query}],
        )
        hyde_text = resp.content[0].text.strip()
    else:
        q_low = query.lower()
        hyde_text = next(
            (v for k, v in MOCK_HYDE_RESPONSES.items() if k in q_low),
            f"This topic involves {query.split()[0]} configuration in enterprise infrastructure.",
        )

    q_vec     = mock_embed(query)
    hyde_vec  = mock_embed(hyde_text)
    fused     = 0.3 * q_vec + 0.7 * hyde_vec
    return fused / (np.linalg.norm(fused) + 1e-10)


def rewrite_multi_query(query: str, n: int = 3) -> list[np.ndarray]:
    """
    Generate N paraphrases and return their embeddings for RRF.
    WHY N=3: sweet spot of recall gain vs latency per cost_recall_tradeoff() in 03_.
    """
    if HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 200,
            system     = f"Generate {n} diverse paraphrases. Output numbered list only.",
            messages   = [{"role": "user", "content": query}],
        )
        text  = resp.content[0].text.strip()
        lines = re.findall(r"^\d+[.)]\s*(.+)$", text, re.MULTILINE)[:n]
        paras = lines if lines else [query]
    else:
        q_low = query.lower()
        paras = next(
            (v[:n] for k, v in MOCK_PARAPHRASES.items() if any(w in q_low for w in k.split())),
            [query, f"Technical details for {query.split()[0]}", f"Requirements for {query.split()[0]}"],
        )[:n]

    return [mock_embed(p) for p in [query] + paras]


def rewrite_decompose(query: str) -> list[str]:
    """
    Break query into atomic sub-questions for separate retrieval.
    WHY separate retrieval: one chunk can't answer both parts of a compound query.
    """
    if HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 150,
            system     = "Break the complex question into 2-4 atomic sub-questions. Output numbered list only.",
            messages   = [{"role": "user", "content": query}],
        )
        text  = resp.content[0].text.strip()
        lines = re.findall(r"^\d+[.)]\s*(.+)$", text, re.MULTILINE)
        return lines if lines else [query]
    else:
        q_low = query.lower()
        for key, subs in MOCK_DECOMPOSITIONS.items():
            if all(w in q_low for w in key.split()):
                return subs
        if " and " in q_low:
            parts = [p.strip() for p in query.split(" and ", maxsplit=1)]
            return [parts[0] + "?", "And " + parts[1] + "?"]
        return [query]


# ─── QueryOptimizer ────────────────────────────────────────────────────────────

@dataclass
class OptimizedQuery:
    """Full result of query optimization: analysis, rewriting, and retrieval."""
    original_query:  str
    analysis:        QueryAnalysis
    technique_used:  str
    retrieved_docs:  list[tuple[dict, float]]
    extra_queries:   list[str]         # HyDE/paraphrases/sub-questions used
    recall_at_5:     Optional[float]   # set if relevant_ids provided to optimize()

    def display(self):
        print(f"\n  Query:     '{self.original_query}'")
        print(f"  Type:       {self.analysis.query_type:<15} Complexity: {self.analysis.complexity:.2f}")
        print(f"  Technique:  {self.technique_used}")
        if self.extra_queries:
            print(f"  Rewrites used:")
            for eq in self.extra_queries[:3]:
                print(f"    - '{eq[:70]}'")
        print(f"  Top-3 retrieved:")
        for i, (doc, score) in enumerate(self.retrieved_docs[:3], 1):
            print(f"    [{i}] {score:.4f}  '{doc['content'][:65]}'")
        if self.recall_at_5 is not None:
            print(f"  Recall@5: {self.recall_at_5:.0%}")


class QueryOptimizer:
    """
    Routes each query to the best rewriting technique based on analysis,
    executes retrieval, and returns deduplicated fused results.

    Decision tree:
      exact_term  → PASSTHROUGH (BM25 in production; dense here)
      negation    → PASSTHROUGH (filter-first in production)
      comparative → DECOMPOSE   (one retrieval per subject)
      multi_hop   → DECOMPOSE   (one retrieval per hop)
      procedural  → HYDE        (step docs use specific vocab)
      factual     → HYDE        (short factual queries embed weakly)
      general     → MULTI_QUERY (broaden coverage with paraphrases)
    """

    def optimize(
        self,
        query:        str,
        top_k:        int = 5,
        relevant_ids: Optional[list[str]] = None,
    ) -> OptimizedQuery:

        analysis = analyze_query(query)

        if analysis.technique == "PASSTHROUGH":
            # No rewriting — send query as-is
            q_vec   = mock_embed(query)
            results = vector_search(q_vec, top_k=top_k)
            extras  = []

        elif analysis.technique == "HYDE":
            # HyDE embedding replaces/augments the query embedding
            fused   = rewrite_hyde(query)
            results = vector_search(fused, top_k=top_k)
            extras  = ["[HyDE hypothetical document — blend alpha=0.3]"]

        elif analysis.technique == "MULTI_QUERY":
            # Generate N paraphrases, retrieve for each, fuse via RRF
            vecs         = rewrite_multi_query(query, n=3)
            all_lists    = [vector_search(v, top_k=top_k) for v in vecs]
            results      = rrf_fusion(all_lists)[:top_k]
            extras       = [f"paraphrase_{i}" for i in range(len(vecs))]

        elif analysis.technique == "DECOMPOSE":
            # Break into sub-questions, retrieve for each, merge via RRF
            sub_qs    = rewrite_decompose(query)
            all_lists = [vector_search(mock_embed(sq), top_k=top_k) for sq in sub_qs]
            results   = rrf_fusion(all_lists)[:top_k]
            extras    = sub_qs

        else:
            q_vec   = mock_embed(query)
            results = vector_search(q_vec, top_k=top_k)
            extras  = []

        # Optional recall measurement
        recall = None
        if relevant_ids:
            top_ids = {doc["id"] for doc, _ in results[:5]}
            recall  = sum(1 for rid in relevant_ids if rid in top_ids) / max(len(relevant_ids), 1)

        return OptimizedQuery(
            original_query = query,
            analysis       = analysis,
            technique_used = analysis.technique,
            retrieved_docs = results,
            extra_queries  = extras,
            recall_at_5    = recall,
        )


# ─── Demo ─────────────────────────────────────────────────────────────────────

EVAL_CASES = [
    {
        "query":        "APIC HA?",
        "relevant_ids": ["kb001"],
        "description":  "Ultra-short factual → HYDE",
    },
    {
        "query":        "What is ReadyOps and how does it integrate with ACI?",
        "relevant_ids": ["kb003", "kb004", "kb010"],
        "description":  "Compound query → DECOMPOSE",
    },
    {
        "query":        "Compare Cisco Hypershield vs ISE for microsegmentation",
        "relevant_ids": ["kb006", "kb007", "kb009"],
        "description":  "Comparative query → DECOMPOSE",
    },
    {
        "query":        "ReadyOps validation requirements for network changes",
        "relevant_ids": ["kb004", "kb010"],
        "description":  "Vocabulary mismatch → MULTI_QUERY",
    },
    {
        "query":        "CSCvh23456 EPG contract issue",
        "relevant_ids": ["kb002"],
        "description":  "Exact bug ID → PASSTHROUGH",
    },
]


def run_optimizer_demo():
    """Show the full optimizer routing and retrieval for each case."""

    print("=" * 70)
    print("QUERY OPTIMIZER: Full Pipeline Demo")
    print("=" * 70)

    optimizer = QueryOptimizer()
    total_recall_baseline = 0.0
    total_recall_optimized = 0.0

    for i, tc in enumerate(EVAL_CASES, 1):
        print(f"\n  ─── Case {i}: {tc['description']} ───")

        # Baseline: no rewriting
        q_vec    = mock_embed(tc["query"])
        baseline = vector_search(q_vec, top_k=5)
        top_ids  = {doc["id"] for doc, _ in baseline[:5]}
        recall_b = sum(1 for rid in tc["relevant_ids"] if rid in top_ids) / max(len(tc["relevant_ids"]), 1)

        # Optimized
        result   = optimizer.optimize(tc["query"], top_k=5, relevant_ids=tc["relevant_ids"])
        result.display()

        gain = result.recall_at_5 - recall_b
        print(f"  Baseline recall@5: {recall_b:.0%}  →  Optimized: {result.recall_at_5:.0%}  (Δ{gain:+.0%})")

        total_recall_baseline  += recall_b
        total_recall_optimized += result.recall_at_5

    n = len(EVAL_CASES)
    print(f"\n  {'═'*65}")
    print(f"  AGGREGATE  Baseline avg recall@5: {total_recall_baseline/n:.0%}")
    print(f"             Optimized avg recall@5: {total_recall_optimized/n:.0%}")
    print(f"             Average gain:           {(total_recall_optimized-total_recall_baseline)/n:+.0%}")


def technique_selection_guide():
    """Print the full decision table for technique selection."""

    print("\n" + "=" * 70)
    print("TECHNIQUE SELECTION GUIDE: When to Use What")
    print("=" * 70)
    print(f"""
  ┌───────────────────────────┬──────────────┬────────────────────────────────────┐
  │ Query characteristic      │ Technique    │ Rationale                          │
  ├───────────────────────────┼──────────────┼────────────────────────────────────┤
  │ Short (≤4 words)          │ HYDE         │ Short queries embed poorly          │
  │ Factual ("What is X?")    │ HYDE         │ Hypothetical answer ≈ doc vocab     │
  │ Procedural ("How to X?")  │ HYDE         │ Step-by-step docs use specific lang │
  │ High vocab mismatch       │ MULTI_QUERY  │ Paraphrases cover more space        │
  │ General knowledge         │ MULTI_QUERY  │ Broad coverage preferred            │
  │ Multi-part ("X and Y?")   │ DECOMPOSE    │ One retrieval per part needed       │
  │ Comparative ("X vs Y?")   │ DECOMPOSE    │ Two subject retrievals + synthesis  │
  │ Multi-hop ("X → Y → Z")   │ DECOMPOSE    │ Each hop = separate retrieval       │
  │ Exact ID (CVE/CSC/model)  │ PASSTHROUGH  │ BM25 handles exact match perfectly  │
  │ Negation ("NOT X")        │ PASSTHROUGH  │ Filter logic, not embedding         │
  └───────────────────────────┴──────────────┴────────────────────────────────────┘

  PRODUCTION COST TABLE (per query, Haiku pricing, June 2026):
  ┌──────────────┬────────────────────┬──────────────────┬──────────────────┐
  │ Technique    │ Extra LLM calls    │ Extra cost       │ Latency add      │
  ├──────────────┼────────────────────┼──────────────────┼──────────────────┤
  │ PASSTHROUGH  │ 0                  │ $0               │ +0ms             │
  │ HYDE         │ 1 (gen hypothetical│ ~$0.001          │ +30–80ms         │
  │ MULTI_QUERY  │ 1 (gen paraphrases)│ ~$0.001          │ +40–100ms        │
  │ DECOMPOSE    │ 2 (decompose+synth)│ ~$0.002          │ +60–150ms        │
  └──────────────┴────────────────────┴──────────────────┴──────────────────┘

  KEY INSIGHT:
    All rewriting costs < $0.003/query and add < 200ms.
    Recall improvement ranges from +10% (HYDE on factual) to +40% (DECOMPOSE on multi-part).
    This is the highest ROI optimization available in RAG — cheap, immediate, measurable.

  PHASE 2 REMAINING:
    Lesson 15: Reranking — re-score top-K chunks with a cross-encoder
    Lesson 16: RAG Evaluation — measure faithfulness, relevance, recall with RAGAS
    Lesson 17: Generation Patterns — prompt design for RAG generation
    Lesson 18: Production RAG — monitoring, latency, cost, deployment
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_optimizer_demo()
    technique_selection_guide()
