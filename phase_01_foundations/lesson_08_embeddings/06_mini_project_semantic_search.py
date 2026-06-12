"""
FILE: 06_mini_project_semantic_search.py
LESSON: Phase 1 - Lesson 8 - Embeddings
TOPIC: Mini-Project — Full semantic search pipeline: embed → index → query → RAG

WHAT THIS PROJECT BUILDS:
  A complete in-memory semantic search engine that:
    1. Takes a corpus of documents
    2. Chunks them into token-sized pieces
    3. Embeds each chunk with Voyage AI (or mock)
    4. Builds an in-memory vector index (numpy)
    5. Accepts a user query, embeds it with input_type="query"
    6. Returns ranked results using cosine similarity
    7. Passes top-K results to Claude for grounded answer generation
    8. Reports retrieval quality metrics

ARCHITECTURE:
  ┌───────────────────────────────────────────────────────────┐
  │  INGEST                                                   │
  │    Documents → Chunk (by token count) → Embed → Index     │
  │                                                           │
  │  QUERY                                                    │
  │    User query → Embed (query mode) → Cosine search        │
  │    → Top-K chunks → Claude → Grounded answer              │
  └───────────────────────────────────────────────────────────┘

TIES TOGETHER:
  - Lesson 7:  Token-based chunking
  - Lesson 8:  Voyage embeddings, cosine similarity, evaluation metrics

INSTALL:
  pip install anthropic voyageai python-dotenv numpy tiktoken
"""

import os
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
import anthropic
import numpy as np

load_dotenv()

client = anthropic.Anthropic()

# ── Voyage AI setup ──────────────────────────────────────────────────────────
try:
    import voyageai
    vo = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY", ""))
    HAS_VOYAGE = bool(os.environ.get("VOYAGE_API_KEY"))
except ImportError:
    HAS_VOYAGE = False
    vo = None

# ── Local token counting ──────────────────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def tok_count(text: str) -> int:
        return len(_enc.encode(text))
    def tok_split(text: str, max_tokens: int) -> list[str]:
        ids = _enc.encode(text)
        return [_enc.decode(ids[i:i+max_tokens]) for i in range(0, len(ids), max_tokens)]
except ImportError:
    def tok_count(text: str) -> int:
        return int(len(text.split()) / 0.75)
    def tok_split(text: str, max_tokens: int) -> list[str]:
        words = text.split()
        n = int(max_tokens * 0.75)
        return [" ".join(words[i:i+n]) for i in range(0, len(words), n)]


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:   str
    source:     str
    content:    str
    chunk_idx:  int   # position within source document
    embedding:  Optional[list[float]] = None

    @property
    def token_count(self) -> int:
        return tok_count(self.content)


@dataclass
class SearchResult:
    chunk:      Chunk
    score:      float
    rank:       int


# ─── Chunker ──────────────────────────────────────────────────────────────────

def chunk_document(
    source:       str,
    content:      str,
    chunk_tokens: int = 300,
    overlap:      int = 30,
) -> list[Chunk]:
    """
    Split a document into token-sized chunks with optional overlap.

    WHY token-based not character-based:
      Ensures consistent token counts for reliable context window budgeting.
      Character splits produce unpredictable token counts by content type.

    WHY overlap:
      Prevents answer sentences from falling right at a chunk boundary.
      overlap=30 means each chunk shares 30 tokens with the previous chunk.
      Cost: ~10% extra tokens stored. Benefit: ~15% better retrieval on boundary content.

    Args:
        source:       Document name/filename for attribution.
        content:      Full document text.
        chunk_tokens: Target token count per chunk.
        overlap:      Tokens of overlap with the previous chunk.
    """

    chunks  = []
    tokens  = _enc.encode(content) if HAS_VOYAGE or True else None
    step    = max(1, chunk_tokens - overlap)   # WHY chunk - overlap: sliding window step

    if tokens is not None:
        for i, start in enumerate(range(0, len(tokens), step)):
            end          = min(start + chunk_tokens, len(tokens))
            chunk_tokens_slice = tokens[start:end]

            try:
                chunk_text = _enc.decode(chunk_tokens_slice)
            except Exception:
                chunk_text = content[start*4 : end*4]   # fallback: character estimate

            chunk_id = f"{source.replace(' ','_')}_{i:04d}"
            chunks.append(Chunk(
                chunk_id  = chunk_id,
                source    = source,
                content   = chunk_text.strip(),
                chunk_idx = i,
            ))
            if end >= len(tokens):
                break
    else:
        # Fallback: word-based chunking
        raw_chunks = tok_split(content, chunk_tokens)
        for i, text in enumerate(raw_chunks):
            chunks.append(Chunk(
                chunk_id  = f"{source.replace(' ','_')}_{i:04d}",
                source    = source,
                content   = text.strip(),
                chunk_idx = i,
            ))

    return [c for c in chunks if len(c.content) > 20]   # discard tiny trailing chunks


# ─── Embedding Functions ──────────────────────────────────────────────────────

def _mock_embed(text: str, dims: int = 1024) -> list[float]:
    """Deterministic mock embedding for when Voyage API is not available."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.RandomState(seed)
    vec  = rng.randn(dims).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


def embed_chunks(chunks: list[Chunk], model: str = "voyage-3", batch_size: int = 128):
    """
    Embed all chunks in batches.
    Updates chunk.embedding in-place.
    """

    texts = [c.content for c in chunks]

    if not HAS_VOYAGE:
        embeddings = [_mock_embed(t) for t in texts]
    else:
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            result = vo.embed(texts=batch, model=model, input_type="document")
            embeddings.extend(result.embeddings)

    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb


def embed_query_text(query: str, model: str = "voyage-3") -> list[float]:
    """Embed a user query with input_type='query'."""
    if not HAS_VOYAGE:
        return _mock_embed(query)
    result = vo.embed(texts=[query], model=model, input_type="query")
    return result.embeddings[0]


# ─── In-Memory Vector Index ───────────────────────────────────────────────────

class VectorIndex:
    """
    Simple in-memory vector index using numpy for cosine search.

    WHY in-memory (not a real vector DB):
      Teaches the math directly — FAISS, Qdrant, Pinecone do the same operations
      but with HNSW indexing for sub-linear search time.
      For < 50,000 chunks, numpy exhaustive search is fast enough.

    In production: swap this for Qdrant, FAISS, Pinecone, or Weaviate.
    The embed_chunks() and embed_query_text() calls are identical — only the
    index.search() call changes to the vector DB's query method.
    """

    def __init__(self):
        self._chunks:    list[Chunk] = []
        self._matrix:   Optional[np.ndarray] = None   # n_chunks × dims matrix

    def add(self, chunks: list[Chunk]):
        """
        Add pre-embedded chunks to the index.
        Builds the numpy matrix for fast batch cosine search.
        """
        assert all(c.embedding is not None for c in chunks), \
            "All chunks must be embedded before adding to index"

        self._chunks.extend(chunks)

        # Stack all embeddings into a 2D matrix: shape (n_chunks, dims)
        # WHY matrix: allows batch cosine similarity with one matrix multiply
        all_embeddings = np.array([c.embedding for c in self._chunks], dtype=np.float32)

        # Unit-normalize each row so dot product = cosine similarity
        # WHY normalize ONCE at index time (not at every query):
        #   Query time normalization would add overhead on every search call.
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # WHY: prevent div-by-zero
        self._matrix = all_embeddings / norms

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """
        Return top-K chunks by cosine similarity to the query.

        MATH:
          scores = matrix @ query_vec   (n_chunks dot products in one operation)
          This works because both matrix rows AND query are unit-normalized:
          dot_product(norm_a, norm_b) = cosine_similarity(a, b)

        WHY np.dot (not a loop):
          np.dot leverages BLAS matrix multiply → typically 100-1000× faster than
          a Python loop over individual cosine_similarity() calls.
          For 10,000 chunks × 1024 dims: ~0.5ms with numpy vs ~500ms with a loop.
        """

        if self._matrix is None:
            return []

        # Unit-normalize the query vector
        qvec = np.array(query_embedding, dtype=np.float32)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-10)

        # Batch cosine similarity: (n_chunks,) scores vector
        scores = np.dot(self._matrix, qvec)   # WHY matmul: single fast BLAS call

        # Top-K indices (argsort descending)
        # WHY -scores: np.argsort returns ascending by default; negate to get descending
        top_k_indices = np.argsort(-scores)[:top_k]

        return [
            SearchResult(
                chunk = self._chunks[i],
                score = float(scores[i]),
                rank  = rank + 1,
            )
            for rank, i in enumerate(top_k_indices)
        ]

    @property
    def size(self) -> int:
        return len(self._chunks)


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

def ask_rag(
    query:        str,
    index:        VectorIndex,
    top_k:        int = 5,
    min_score:    float = 0.0,
    model:        str = "claude-sonnet-4-6",
    embedding_model: str = "voyage-3",
) -> dict:
    """
    Full RAG pipeline: semantic search → Claude answer generation.

    Returns:
        Dict with 'answer', 'results', 'citations', and 'metrics'.
    """

    # ── 1. Embed query ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    query_embedding = embed_query_text(query, model=embedding_model)
    embed_ms        = int((time.perf_counter() - t0) * 1000)

    # ── 2. Vector search ──────────────────────────────────────────────────────
    results = index.search(query_embedding, top_k=top_k)
    results = [r for r in results if r.score >= min_score]

    if not results:
        return {
            "answer":    "NOT IN CONTEXT — no relevant documents found.",
            "results":   [],
            "citations": [],
            "metrics":   {"embed_ms": embed_ms, "total_ms": embed_ms},
        }

    # ── 3. Build RAG prompt ───────────────────────────────────────────────────
    context_parts = []
    for r in results:
        context_parts.append(
            f"[{r.chunk.chunk_id}] Source: {r.chunk.source} | Score: {r.score:.3f}\n"
            f"{r.chunk.content}"
        )
    context_block = "\n\n".join(context_parts)

    user_message = (
        f"CONTEXT DOCUMENTS:\n{context_block}\n\n"
        f"QUESTION: {query}"
    )

    # ── 4. Count tokens before calling ────────────────────────────────────────
    system = (
        "You are a precise technical assistant. "
        "Answer using ONLY the provided CONTEXT DOCUMENTS. "
        "Cite every claim with [chunk_id]. "
        "If not in context, respond: NOT IN PROVIDED CONTEXT."
    )

    token_count = client.messages.count_tokens(
        model    = model,
        system   = system,
        messages = [{"role": "user", "content": user_message}],
    ).input_tokens

    # ── 5. Generate answer with streaming ────────────────────────────────────
    t1 = time.perf_counter()
    first_tok = None
    answer    = ""

    print(f"\n  ANSWER: ", end="", flush=True)

    with client.messages.stream(
        model       = model,
        max_tokens  = 400,
        temperature = 0,
        system      = system,
        messages    = [{"role": "user", "content": user_message}],
    ) as stream:
        for delta in stream.text_stream:
            if first_tok is None:
                first_tok = time.perf_counter()
            answer += delta
            print(delta, end="", flush=True)

        final_usage = stream.get_final_message().usage

    print()

    total_ms = int((time.perf_counter() - t0) * 1000)
    ttft_ms  = int((first_tok - t1) * 1000) if first_tok else 0

    # ── 6. Extract citations ──────────────────────────────────────────────────
    citations = [r.chunk.chunk_id for r in results if f"[{r.chunk.chunk_id}]" in answer]

    return {
        "answer":    answer,
        "results":   results,
        "citations": citations,
        "metrics": {
            "input_tokens":   token_count,
            "output_tokens":  final_usage.output_tokens,
            "embed_ms":       embed_ms,
            "ttft_ms":        ttft_ms,
            "total_ms":       total_ms,
            "chunks_found":   len(results),
            "chunks_cited":   len(citations),
        },
    }


# ─── Demo Corpus ──────────────────────────────────────────────────────────────

CORPUS = {
    "readyops_overview.txt": """\
ReadyOps is Criterion Networks' continuous validation platform powered by AI agent classes.
It operates across two deliberately isolated environments: Live Operations and Production-Representative.
The Production-Representative environment can be a digital twin, physical lab, or hybrid of both.
Environments share one intent model but never cross the wire.
Operational changes execute in Live Operations ONLY after validation and formal promotion from
the Production-Representative environment. This ensures that no untested change ever reaches production.

ReadyOps supports four agent classes:
- Health & Posture: Continuously monitors network health and compliance posture across both environments.
- Validation: Runs automated test suites that simulate real traffic and configuration scenarios
  against the Production-Representative environment before any change is promoted.
- Operational: Executes approved changes with full audit logging and rollback capability.
- Stress & Adversarial: Tests resilience under failure conditions and adversarial scenarios.

The ReadyOps infrastructure lifecycle coverage spans: PoV, Validation, Production, and Operations.
Criterion Networks tagline: "Validate Before You Operate. Validate While You Operate."
""",

    "aci_guide.txt": """\
Cisco ACI (Application Centric Infrastructure) is a software-defined networking solution
that uses a policy-driven model to automate network provisioning and management.

Architecture: ACI uses a Leaf-Spine topology. Every leaf switch connects to every spine switch,
providing no more than two hops between any two endpoints. This gives predictable latency
and linear scalability.

The APIC (Application Policy Infrastructure Controller) is the centralized management plane.
It translates business intent expressed as policies into the physical configuration of the fabric.
The APIC cluster typically has three nodes for high availability.

Endpoint Groups (EPGs) are logical groups of endpoints with similar policy requirements.
EPGs communicate with each other only through contracts. A contract defines which protocols
and ports are allowed between two EPGs. This model enforces microsegmentation by default.

ACI Multi-Pod extends the fabric across geographic locations while maintaining a single
policy domain. The Inter-Pod Network (IPN) connects pods using VXLAN encapsulation.
All pods share the same APIC cluster for unified management.

ReadyOps can create a digital twin of an ACI fabric for validation purposes.
Changes to ACI policy are validated against the digital twin before being promoted
to the production ACI fabric via the ReadyOps Operational agent class.
""",

    "security_overview.txt": """\
Cisco Hypershield is an AI-native security architecture that embeds security enforcement
directly into the network fabric and compute infrastructure. It uses eBPF (Extended Berkeley
Packet Filter) technology to enforce policy at the kernel level on every workload endpoint,
without requiring dedicated security appliances.

Hypershield supports:
- Distributed Exploit Protection: patches vulnerabilities at the network level while
  OS-level patches are being deployed, eliminating the remediation window.
- Autonomous Segmentation: automatically creates and enforces microsegmentation policies
  based on observed workload behavior.
- Dual Data Plane: runs two instances of every policy — one enforcing, one shadowing —
  so new policies can be validated before enforcement. This is similar to ReadyOps'
  Production-Representative environment concept applied at the workload level.

Cisco ISE (Identity Services Engine) provides network access control (NAC) and policy enforcement.
ISE integrates with Active Directory for identity resolution. When a device authenticates,
ISE assigns a TrustSec Security Group Tag (SGT) that follows the workload as it moves.
SGTs are propagated via Cisco TrustSec inline tagging or the SGT Exchange Protocol (SXP).
ISE and Hypershield work together: ISE assigns identity, Hypershield enforces policy.
""",
}


# ─── Main Demo ────────────────────────────────────────────────────────────────

def run_demo():
    """
    Build the index, then run semantic search + RAG on 3 queries.
    """

    print("=" * 65)
    print("SEMANTIC SEARCH MINI-PROJECT: End-to-End Pipeline")
    print("=" * 65)

    if not HAS_VOYAGE:
        print("\n  ⚠ VOYAGE_API_KEY not set — using mock embeddings.")
        print("  Similarity scores are random but pipeline behavior is real.\n")

    # ── 1. Ingest and chunk ───────────────────────────────────────────────────
    print("\n  STEP 1: Chunking documents")
    all_chunks = []

    for source, content in CORPUS.items():
        chunks = chunk_document(source=source, content=content, chunk_tokens=150, overlap=20)
        print(f"    {source}: {len(content):,} chars → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"  Total: {len(all_chunks)} chunks across {len(CORPUS)} documents")

    # ── 2. Embed ──────────────────────────────────────────────────────────────
    print(f"\n  STEP 2: Embedding {len(all_chunks)} chunks")
    t0 = time.perf_counter()
    embed_chunks(all_chunks, model="voyage-3")
    embed_ms = int((time.perf_counter() - t0) * 1000)
    print(f"  Done in {embed_ms}ms  ({len(all_chunks)/embed_ms*1000:.0f} chunks/sec)")

    # ── 3. Build index ────────────────────────────────────────────────────────
    print(f"\n  STEP 3: Building vector index")
    index = VectorIndex()
    index.add(all_chunks)
    print(f"  Index size: {index.size} chunks × {len(all_chunks[0].embedding)} dims")

    # ── 4. Search + RAG ───────────────────────────────────────────────────────
    queries = [
        "How does ReadyOps validate changes before production?",
        "What is the relationship between ISE and Hypershield for security enforcement?",
        "How does ACI Multi-Pod extend the fabric across geographic locations?",
    ]

    for query in queries:
        print("\n" + "═" * 65)
        print(f"  QUERY: {query}")
        print("═" * 65)

        # Show top retrieval results before answering
        test_embedding = embed_query_text(query)
        search_results = index.search(test_embedding, top_k=5)

        print(f"\n  Top-5 retrieved chunks:")
        for r in search_results:
            print(f"    [{r.rank}] score={r.score:.4f}  {r.chunk.source}  "
                  f"— {r.chunk.content[:60]}...")

        result = ask_rag(
            query   = query,
            index   = index,
            top_k   = 4,
            min_score = 0.0,
        )

        m = result["metrics"]
        print(f"\n  Metrics: embed={m['embed_ms']}ms  ttft={m['ttft_ms']}ms  "
              f"total={m['total_ms']}ms")
        print(f"  Tokens: {m['input_tokens']} in / {m['output_tokens']} out")
        print(f"  Citations: {result['citations'] or 'none detected'}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_demo()
