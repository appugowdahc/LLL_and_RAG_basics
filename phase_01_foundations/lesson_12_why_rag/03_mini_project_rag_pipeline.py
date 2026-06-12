"""
FILE: 03_mini_project_rag_pipeline.py
LESSON: Phase 1 - Lesson 12 - Why RAG Was Invented
TOPIC: Mini-project — End-to-end RAG pipeline integrating all Phase 1 concepts

WHAT THIS BUILDS:
  A complete, working RAG pipeline that demonstrates every concept from Phase 1:
    - Tokenization (Lesson 7): chunk sizing by token count
    - Embeddings (Lesson 8): mock deterministic embeddings
    - BM25 (Lesson 9): keyword retrieval
    - Dense search (Lesson 9): semantic retrieval
    - Hybrid + RRF (Lesson 9): fused ranking
    - Metadata filtering (Lesson 9): scoped retrieval
    - Context window management (Lesson 6): token budget enforcement
    - Query router (Lesson 11): route by limitation signals
    - Hallucination detection (Lesson 10): faithfulness scoring
    - Citation attribution (Lesson 10): [ChunkN] in every answer

  This is the Phase 1 capstone — everything connects here.
  You can run this file without any API keys using mock components.
  With ANTHROPIC_API_KEY set, you get real LLM answers.

INSTALL: pip install anthropic python-dotenv numpy  (all optional — runs without)
"""

import os
import re
import math
import hashlib
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Optional

try:
    import anthropic
    HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    HAS_ANTHROPIC = False


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: DOCUMENT INGESTION
# (Lesson 7: tokenization, chunk sizing)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """
    A document chunk ready for indexing.
    WHY chunk_id: needed for citation attribution and deduplication.
    WHY metadata: needed for metadata filtering and citation display.
    """
    chunk_id:  str
    content:   str
    metadata:  dict[str, Any]

    @property
    def token_count(self) -> int:
        """Approximate token count: 1 token ≈ 4 chars for English text."""
        return max(1, len(self.content) // 4)   # WHY //4: rough approximation, tiktoken preferred

    @property
    def embedding(self) -> np.ndarray:
        """
        Deterministic mock embedding.
        WHY SHA-256 seed: same content → same vector across runs (reproducible).
        In production: voyageai.Client().embed(texts, input_type="document")
        """
        seed = int(hashlib.md5(self.content.encode()).hexdigest(), 16) % (2**32)
        rng  = np.random.RandomState(seed)
        v    = rng.randn(64).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-10)


def chunk_document(
    source:   str,
    content:  str,
    metadata: dict,
    target_tokens: int = 300,
) -> list[Chunk]:
    """
    Split a document into chunks of approximately target_tokens each.
    WHY sentences as split unit: preserves semantic coherence within chunks.
    WHY not character split: character splits break sentences mid-thought.
    """
    sentences = re.split(r"(?<=[.!?])\s+", content.strip())
    chunks    = []
    current   = []
    current_tokens = 0
    idx       = 0

    for sent in sentences:
        sent_tokens = max(1, len(sent) // 4)

        if current_tokens + sent_tokens > target_tokens and current:
            text = " ".join(current)
            chunks.append(Chunk(
                chunk_id = f"{source}:{idx}",
                content  = text,
                metadata = {**metadata, "chunk_idx": idx, "source": source},
            ))
            idx     += 1
            current  = []
            current_tokens = 0

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        text = " ".join(current)
        chunks.append(Chunk(
            chunk_id = f"{source}:{idx}",
            content  = text,
            metadata = {**metadata, "chunk_idx": idx, "source": source},
        ))

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: SEARCH ENGINE (BM25 + Dense + Hybrid)
# (Lesson 9: BM25, HNSW, RRF, metadata filtering)
# ═══════════════════════════════════════════════════════════════════════════════

STOPWORDS = {"the","a","an","is","are","was","were","to","of","and","or","in",
             "on","at","by","for","with","as","this","that","it","its","from",
             "into","has","have","had","will","can","should","not","be","been"}

def tokenize(text: str) -> list[str]:
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


class BM25:
    """BM25 keyword index (k1=1.5, b=0.75)."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._corpus: list[list[str]] = []
        self._avgdl:  float = 0.0
        self._idf:    dict[str, float] = {}
        self._inv:    dict[str, list[int]] = defaultdict(list)

    def build(self, chunks: list[Chunk]):
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
            df = len(docs)
            self._idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        k1, b   = 1.5, 0.75
        terms   = tokenize(query)
        cands   = set()
        for t in terms:
            cands.update(self._inv.get(t, []))
        scores = []
        for idx in cands:
            tokens = self._corpus[idx]
            dl     = len(tokens)
            score  = 0.0
            for t in terms:
                tf   = tokens.count(t)
                if tf == 0: continue
                idf  = self._idf.get(t, 0.0)
                ln   = 1 - b + b * (dl / max(self._avgdl, 1))
                score += idf * (tf * (k1 + 1)) / (tf + k1 * ln)
            scores.append((idx, score))
        scores.sort(key=lambda x: -x[1])
        return [(self._chunks[i], s) for i, s in scores[:top_k]]


class DenseIndex:
    """Brute-force dense search using numpy dot product."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._matrix: Optional[np.ndarray] = None

    def build(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._matrix = np.vstack([c.embedding for c in chunks])

    def search(
        self,
        query_vec:  np.ndarray,
        top_k:      int = 10,
        candidates: Optional[list[int]] = None,
    ) -> list[tuple[Chunk, float]]:
        if self._matrix is None: return []
        qn = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        if candidates is not None:
            mat    = self._matrix[candidates]
            scores = np.dot(mat, qn)
            top    = np.argsort(-scores)[:top_k]
            return [(self._chunks[candidates[i]], float(scores[i])) for i in top]
        scores = np.dot(self._matrix, qn)
        top    = np.argsort(-scores)[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top]


class MetadataFilter:
    """Simple AND-filter over chunk metadata fields."""

    def __init__(self, conditions: list[dict]):
        self.conditions = conditions

    def matches(self, chunk: Chunk) -> bool:
        for cond in self.conditions:
            actual = chunk.metadata.get(cond["field"])
            if actual is None: return False
            op, val = cond["op"], cond["value"]
            if op == "eq"  and actual != val:      return False
            if op == "ne"  and actual == val:      return False
            if op == "in"  and actual not in val:  return False
            if op == "gte" and not actual >= val:  return False
            if op == "lte" and not actual <= val:  return False
        return True

    def matching_indices(self, chunks: list[Chunk]) -> list[int]:
        return [i for i, c in enumerate(chunks) if self.matches(c)]


def rrf_fusion(
    result_lists: list[list[tuple[Chunk, float]]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion — combines multiple ranked lists."""
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for lst in result_lists:
        for rank, (chunk, _) in enumerate(lst, start=1):
            scores[chunk.chunk_id] += 1.0 / (k + rank)
            chunks[chunk.chunk_id]  = chunk
    combined = sorted(scores.items(), key=lambda x: -x[1])
    return [(chunks[cid], score) for cid, score in combined]


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: QUERY ROUTING
# (Lesson 11: limitation detection)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_query(query: str) -> dict:
    """
    Detect which limitations apply to this query and recommend routing.
    Returns dict with needs_retrieval, needs_realtime, needs_tool flags.
    """
    q = query.lower()
    return {
        "needs_retrieval": any(w in q for w in [
            "our", "internal", "runbook", "latest", "current", "version",
            "cve-", "csc", "customer", "tenant"
        ]),
        "needs_realtime":  any(w in q for w in [
            "right now", "active", "currently", "live", "fault", "alert"
        ]),
        "needs_tool":      any(w in q for w in [
            "calculate", "total", "sum", "how many", "count"
        ]) or bool(re.search(r"\d+\s*[×x\*]\s*\d+|\d+\s*[+\-]\s*\d+", q)),
        "is_general":      not any(w in q for w in [
            "our", "internal", "specific", "latest", "cve", "current", "customer"
        ]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4: CONTEXT ASSEMBLY
# (Lesson 6: context window, token budget)
# ═══════════════════════════════════════════════════════════════════════════════

def build_context(chunks: list[Chunk], token_budget: int = 4000) -> tuple[str, list[Chunk]]:
    """
    Build the context block for the LLM prompt.
    Returns (context_string, included_chunks).
    WHY return included_chunks: needed for citation verification.
    """
    used     = 0
    included = []
    parts    = []

    for i, chunk in enumerate(chunks, 1):
        if used + chunk.token_count > token_budget:
            break   # WHY hard stop: never exceed context budget
        header = (
            f"[Chunk{i}] Source: {chunk.metadata.get('source', 'unknown')} | "
            f"Product: {chunk.metadata.get('product', '?')} | "
            f"Date: {chunk.metadata.get('date', '?')} | "
            f"Type: {chunk.metadata.get('source_type', '?')}"
        )
        parts.append(f"{header}\n{chunk.content}")
        included.append(chunk)
        used     += chunk.token_count

    return "\n\n".join(parts), included


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 5: LLM CALL WITH CITATION PROMPT
# (Lessons 1-5: the LLM, Lesson 10: attribution)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a precise technical assistant for Criterion Networks.

Answer questions using ONLY the provided context chunks.
Cite every sentence with [ChunkN] matching the chunk number in the context.
If the answer cannot be found in the provided chunks, say:
"This information is not in the provided documents. [unsupported]"
Never answer from prior knowledge. Never guess. Keep answers concise."""


def call_llm(question: str, context: str) -> str:
    """
    Call Claude with the question and retrieved context.
    Falls back to a mock answer if no API key is available.
    """
    if HAS_ANTHROPIC:
        client   = anthropic.Anthropic()
        response = client.messages.create(
            model      = "claude-haiku-4-5-20251001",   # WHY Haiku: fast and cheap for demo
            max_tokens = 300,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        )
        return response.content[0].text
    else:
        # Mock answer for demo without API key
        return (
            f"[MOCK ANSWER — set ANTHROPIC_API_KEY for real LLM responses]\n"
            f"Based on the provided context, the answer addresses '{question[:50]}...'. [Chunk1]"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 6: FAITHFULNESS CHECK
# (Lesson 10: hallucination detection)
# ═══════════════════════════════════════════════════════════════════════════════

def score_faithfulness(answer: str, context: str) -> float:
    """
    Approximate faithfulness: fraction of answer sentences with lexical overlap to context.
    Score 0.0–1.0. Production: replace with NLI model or LLM-as-judge.
    """
    def tokenize_content(t: str) -> set[str]:
        return {w for w in re.findall(r"\b\w+\b", t.lower()) if len(w) > 3}

    clean_ans = re.sub(r"\[Chunk\d+\]|\[unsupported\]", "", answer)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_ans) if len(s.strip()) > 15]
    ctx_toks  = tokenize_content(context)

    if not sentences:
        return 1.0

    supported = sum(
        1 for s in sentences
        if len(tokenize_content(s) & ctx_toks) / max(len(tokenize_content(s)), 1) >= 0.25
    )
    return supported / len(sentences)


# ═══════════════════════════════════════════════════════════════════════════════
# THE FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    End-to-end RAG pipeline integrating all Phase 1 components.
    This is the capstone of Phase 1 — all concepts in one class.
    """

    def __init__(self, token_budget: int = 3000):
        self._chunks:       list[Chunk] = []
        self._bm25:         BM25        = BM25()
        self._dense:        DenseIndex  = DenseIndex()
        self._token_budget: int         = token_budget
        self._built:        bool        = False

    def ingest(self, documents: list[dict]):
        """
        Ingest and index all documents.
        Args: list of {"source", "content", "metadata"} dicts.
        """
        all_chunks = []
        for doc in documents:
            chunks = chunk_document(
                source   = doc["source"],
                content  = doc["content"],
                metadata = doc["metadata"],
            )
            all_chunks.extend(chunks)

        self._chunks = all_chunks
        self._bm25.build(all_chunks)
        self._dense.build(all_chunks)
        self._built  = True
        print(f"  [Ingest] {len(documents)} docs → {len(all_chunks)} chunks indexed.")

    def query(
        self,
        question:    str,
        filter_:     Optional[MetadataFilter] = None,
        top_k:       int = 5,
        verbose:     bool = True,
    ) -> dict:
        """
        Run the full RAG pipeline for one query.
        Returns structured result with answer, citations, faithfulness score.
        """
        assert self._built, "Call ingest() before query()."

        if verbose:
            print(f"\n  ══ QUERY: '{question}' ══")

        # Step 1: Route the query
        routing = classify_query(question)
        if verbose:
            print(f"  [Router] needs_retrieval={routing['needs_retrieval']}, "
                  f"needs_tool={routing['needs_tool']}")

        # Step 2: Retrieve (BM25 + Dense + Hybrid)
        query_vec = self._chunks[0].embedding.__class__  # just get mock query vec
        # WHY re-derive query embedding: in production, embed the query text
        q_seed  = int(hashlib.md5(question.encode()).hexdigest(), 16) % (2**32)
        q_rng   = np.random.RandomState(q_seed)
        q_vec   = q_rng.randn(64).astype(np.float32)
        q_vec  /= np.linalg.norm(q_vec) + 1e-10

        if filter_ is not None:
            candidate_idx = filter_.matching_indices(self._chunks)
            dense_results = self._dense.search(q_vec, top_k * 2, candidates=candidate_idx)
        else:
            dense_results = self._dense.search(q_vec, top_k * 2)

        bm25_results = self._bm25.search(question, top_k * 2)

        # Apply post-filter to BM25 if metadata filter set
        if filter_ is not None:
            bm25_results = [(c, s) for c, s in bm25_results if filter_.matches(c)]

        fused = rrf_fusion([bm25_results, dense_results])[:top_k]

        if verbose:
            print(f"  [Search] BM25: {len(bm25_results)}, Dense: {len(dense_results)}, "
                  f"Fused: {len(fused)}")

        # Step 3: Build context
        context, included = build_context([c for c, _ in fused], self._token_budget)

        if verbose:
            print(f"  [Context] {len(included)} chunks, ~{sum(c.token_count for c in included)} tokens")

        # Step 4: LLM call
        answer = call_llm(question, context)

        # Step 5: Faithfulness check
        faith_score = score_faithfulness(answer, context)
        verdict     = "PASS" if faith_score >= 0.70 else ("WARN" if faith_score >= 0.40 else "FAIL")

        if verbose:
            print(f"  [Faith]  score={faith_score:.0%}  verdict={verdict}")

        return {
            "question":    question,
            "answer":      answer,
            "chunks":      included,
            "faithfulness": faith_score,
            "verdict":     verdict,
            "routing":     routing,
        }

    def display_result(self, result: dict):
        """Pretty-print a query result."""
        print(f"\n  Answer:")
        for line in result["answer"].split("\n"):
            print(f"    {line}")

        print(f"\n  Sources used ({len(result['chunks'])} chunks):")
        for i, chunk in enumerate(result["chunks"], 1):
            m = chunk.metadata
            print(f"    [Chunk{i}] {m.get('source','?')} | {m.get('product','?')} | "
                  f"{m.get('date','?')} | {chunk.content[:50]}...")

        faith_icon = "✓" if result["faithfulness"] >= 0.70 else "⚠"
        print(f"\n  {faith_icon} Faithfulness: {result['faithfulness']:.0%} [{result['verdict']}]")


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

def build_knowledge_base() -> list[dict]:
    return [
        {
            "source": "aci_guide_v6.md",
            "content": (
                "Cisco ACI uses a Leaf-Spine topology. The APIC cluster is the policy controller. "
                "A minimum of 3 APIC nodes are required for high availability. "
                "ACI 6.0 supports up to 200 leaf switches per pod. "
                "ACI uses VXLAN as the overlay protocol for the fabric. "
                "The APIC REST API uses JSON over HTTPS on port 443. "
                "EPGs (Endpoint Groups) define collections of endpoints sharing the same policy. "
                "Contracts define allowed communication between EPGs."
            ),
            "metadata": {"product": "ACI", "source_type": "guide", "date": "2025-03-01", "version": "6.0"},
        },
        {
            "source": "readyops_guide_v2.md",
            "content": (
                "ReadyOps is Criterion Networks' continuous validation platform. "
                "It operates across two isolated environments: Production-Representative and Live Operations. "
                "AI agent classes include Health and Posture, Validation, Operational, and Stress and Adversarial. "
                "Changes must pass a 100% validation gate before promotion to Live Operations. "
                "The Production-Representative environment is a digital twin, physical lab, or hybrid. "
                "ReadyOps validates ACI changes before they reach the production fabric."
            ),
            "metadata": {"product": "ReadyOps", "source_type": "guide", "date": "2025-06-01", "version": "2.0"},
        },
        {
            "source": "hypershield_guide.md",
            "content": (
                "Cisco Hypershield uses eBPF for kernel-level policy enforcement. "
                "It enforces policy at the workload without dedicated network appliances. "
                "Hypershield integrates with ACI by consuming EPG membership from APIC. "
                "Policy is enforced at the process level inside the host operating system. "
                "This enables microsegmentation without perimeter-based network controls."
            ),
            "metadata": {"product": "Hypershield", "source_type": "guide", "date": "2025-02-20", "version": "1.0"},
        },
        {
            "source": "aci_advisory.md",
            "content": (
                "Bug CSCvh23456 affects APIC version 5.2(1g). "
                "Symptom: contract deployment fails when more than 200 EPGs exist in a single VRF. "
                "Workaround: split the VRF into multiple smaller VRFs with fewer EPGs. "
                "Fixed in APIC version 5.2(2a) and later. "
                "This issue does not affect ACI versions 6.0 and above."
            ),
            "metadata": {"product": "ACI", "source_type": "advisory", "date": "2024-06-15", "version": "5.2"},
        },
        {
            "source": "ise_guide.md",
            "content": (
                "Cisco ISE TrustSec assigns Security Group Tags at authentication time. "
                "SXP propagates SGT-to-IP bindings to non-TrustSec devices. "
                "ISE profiling identifies device type using RADIUS, DHCP, and HTTP probes. "
                "Policy assignment uses profiling results to determine access level. "
                "ISE integrates with ACI to propagate SGT policy into the fabric."
            ),
            "metadata": {"product": "ISE", "source_type": "guide", "date": "2025-03-15", "version": "3.3"},
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("PHASE 1 CAPSTONE: End-to-End RAG Pipeline")
    print("Integrating all concepts from Lessons 1–12")
    print("=" * 72)

    # Build and index the knowledge base
    print("\n  [Phase: Ingestion]")
    pipeline = RAGPipeline(token_budget=2000)
    pipeline.ingest(build_knowledge_base())

    print("\n" + "─" * 72)
    print("  DEMO 1: Semantic query (dense wins)")
    result_1 = pipeline.query("How does ACI enforce network policy between workloads?", top_k=3)
    pipeline.display_result(result_1)

    print("\n" + "─" * 72)
    print("  DEMO 2: Exact term query (BM25 advantage)")
    result_2 = pipeline.query("CSCvh23456 bug details and workaround", top_k=3)
    pipeline.display_result(result_2)

    print("\n" + "─" * 72)
    print("  DEMO 3: Metadata filtered query (ACI core docs only)")
    aci_filter = MetadataFilter([
        {"field": "product",     "op": "eq", "value": "ACI"},
        {"field": "source_type", "op": "in", "value": ["guide", "spec"]},
    ])
    result_3 = pipeline.query(
        "What is the APIC cluster HA requirement?",
        filter_  = aci_filter,
        top_k    = 3,
    )
    pipeline.display_result(result_3)

    print("\n" + "─" * 72)
    print("  DEMO 4: Cross-product query (ReadyOps + ACI)")
    result_4 = pipeline.query(
        "How does ReadyOps validate ACI changes before production?",
        top_k = 4,
    )
    pipeline.display_result(result_4)

    # Final architecture summary
    print("\n\n" + "=" * 72)
    print("PHASE 1 COMPLETE: What You've Learned")
    print("=" * 72)

    curriculum = [
        ("Lesson 1",  "Generative AI",          "LLM = next-token prediction at scale"),
        ("Lesson 2",  "LLM Fundamentals",        "Temperature, sampling, system prompts"),
        ("Lesson 3",  "LLM Internals",           "Transformer layers, residual stream, KV cache"),
        ("Lesson 4",  "Transformer Architecture","Encoder/decoder, positional encoding, attention heads"),
        ("Lesson 5",  "Attention Mechanism",      "Scaled dot-product, multi-head, causal masking"),
        ("Lesson 6",  "Context Window",           "Token budget, Lost in the Middle, compression"),
        ("Lesson 7",  "Tokens",                   "BPE, cost arithmetic, chunk sizing"),
        ("Lesson 8",  "Embeddings",               "Cosine similarity, Voyage AI, asymmetric encoding"),
        ("Lesson 9",  "Semantic Search",          "BM25, HNSW, RRF, metadata filtering"),
        ("Lesson 10", "Hallucinations",           "Types, detection, faithfulness scoring"),
        ("Lesson 11", "LLM Limitations",          "Cutoff, context, reasoning, state, privacy"),
        ("Lesson 12", "Why RAG Was Invented",     "5 gaps → RAG architecture → 3 generations"),
    ]

    for lesson, topic, summary in curriculum:
        print(f"  {lesson:<12} {topic:<28} → {summary}")

    print(f"""
  You now have the complete foundation to build production RAG systems.
  Phase 2 onwards covers: query rewriting, reranking, evaluation,
  chunking strategies, production deployment, and agentic patterns.
""")


if __name__ == "__main__":
    main()
