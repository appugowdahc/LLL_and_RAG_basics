"""
FILE: 02_hyde.py
LESSON: Phase 2 - Lesson 14 - Query Understanding and Rewriting
TOPIC: HyDE — Hypothetical Document Embeddings (Gao et al., 2022)

WHAT THIS FILE TEACHES:
  - The HyDE algorithm step-by-step
  - Why short queries embed weakly compared to full hypothetical answers
  - How to generate hypothetical documents with Claude
  - Query + HyDE fusion (blend both embeddings for safety)
  - HyDE failure modes and when NOT to use it
  - Recall comparison: raw query vs HyDE vs fusion

REFERENCE: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
           Gao et al., 2022 — arXiv:2212.10496

INSTALL: pip install anthropic python-dotenv numpy
"""

import os
import re
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    HAS_ANTHROPIC = False


# ─── Utilities ────────────────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def mock_embed(text: str, dims: int = 64) -> np.ndarray:
    """
    Deterministic mock embedding.
    Simulates the key property of real embeddings: similar text → similar vector.
    We seed with a hash of the first 200 chars so short queries and longer answers
    on the same topic have similar (but not identical) seeds.
    WHY first 200 chars: captures the topic without long-tail noise.
    """
    # Use topic keywords to create a consistent seed for related texts
    keywords = sorted(set(re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())))[:8]
    topic_key = " ".join(keywords)
    seed = int(hashlib.md5(topic_key.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.RandomState(seed)
    # Add small noise based on full text to differentiate docs on same topic
    full_seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
    full_rng  = np.random.RandomState(full_seed)
    base  = rng.randn(dims).astype(np.float32)
    noise = full_rng.randn(dims).astype(np.float32) * 0.15   # WHY 0.15: small noise, same direction
    v     = base + noise
    return v / (np.linalg.norm(v) + 1e-10)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                        b / (np.linalg.norm(b) + 1e-10)))


# ─── HyDE Prompt Templates ────────────────────────────────────────────────────

HYDE_SYSTEM_PROMPT = """You are a technical documentation writer for enterprise network infrastructure.
Given a question, write a concise 2-4 sentence paragraph that would appear in technical documentation
and directly answers the question. Write as if you are the document, not as a person answering.
Use precise technical vocabulary. Do not preface with "The answer is" — write the document text directly."""

HYDE_PROMPT_TEMPLATE = """Write a technical documentation passage that directly answers this question:

{question}

Write 2-4 sentences of technical documentation prose. Use the same vocabulary
that would appear in Cisco product guides, operational runbooks, or technical specifications."""


# ─── HyDE Generator ───────────────────────────────────────────────────────────

def generate_hyde_document(query: str) -> str:
    """
    Generate a hypothetical document that would answer the query.

    WHY generate with an LLM:
      The LLM has been trained on millions of technical documents and knows
      the vocabulary, phrasing, and structure used in indexed documentation.
      The hypothetical doc uses THIS vocabulary → its embedding aligns
      with real document embeddings in the vector space.

    WHY 2-4 sentences (not 1, not 10):
      Too short (1 sentence): embedding too similar to raw query — no benefit.
      Too long (10+ sentences): embedding blurs across too many facts.
      2-4 sentences: enough vocabulary enrichment without topical dilution.
    """
    if HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model      = "claude-haiku-4-5-20251001",   # WHY Haiku: fast and cheap for rewriting
            max_tokens = 150,
            system     = HYDE_SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": HYDE_PROMPT_TEMPLATE.format(question=query)}],
        )
        return resp.content[0].text.strip()
    else:
        # Mock HyDE: return a plausible technical answer
        mock_responses = {
            "apic": "The APIC cluster requires a minimum of 3 nodes for high availability. "
                    "When one APIC node becomes unavailable, the remaining two nodes maintain "
                    "quorum and continue managing fabric policy without interruption.",
            "epg": "Endpoint Groups (EPGs) are the fundamental policy unit in Cisco ACI. "
                   "An EPG represents a collection of endpoints with identical policy requirements. "
                   "Communication between EPGs is governed by contracts that define permitted traffic.",
            "readyops": "ReadyOps validates network changes in a Production-Representative environment "
                        "before promoting them to Live Operations. All Validation agent tests must pass "
                        "at 100% before the promotion gate opens.",
            "hypershield": "Cisco Hypershield uses eBPF technology for kernel-level policy enforcement "
                           "at the workload level. It integrates with ACI EPG membership from APIC "
                           "to enforce microsegmentation without dedicated perimeter appliances.",
        }
        q_low = query.lower()
        for key, response in mock_responses.items():
            if key in q_low:
                return response
        return (f"This topic involves {query.split()[0]} configuration in enterprise network infrastructure. "
                f"The relevant specification covers requirements, configuration steps, and operational best practices.")


# ─── HyDE Retrieval ───────────────────────────────────────────────────────────

@dataclass
class HyDEResult:
    """Result of a HyDE-augmented retrieval."""
    query:         str
    hyde_doc:      str
    query_embed:   np.ndarray
    hyde_embed:    np.ndarray
    fusion_embed:  np.ndarray   # weighted blend of query + HyDE
    query_tokens:  int
    hyde_tokens:   int


def build_hyde_embedding(
    query:      str,
    alpha:      float = 0.3,   # WHY 0.3: query is a weaker signal; HyDE dominates
) -> HyDEResult:
    """
    Build a HyDE-augmented embedding for retrieval.

    Returns both the raw query embedding and the fused (query + HyDE) embedding.
    Use the fused embedding for retrieval in production.

    Args:
        query: User's raw query string.
        alpha: Weight of original query in fusion (0.0 = pure HyDE, 1.0 = raw query).
               Default 0.3: HyDE carries 70% of the weight.
    """
    hyde_doc     = generate_hyde_document(query)
    query_embed  = mock_embed(query)
    hyde_embed   = mock_embed(hyde_doc)

    # WHY weighted fusion:
    #   Pure HyDE: if the LLM hallucinates, we retrieve wrong documents.
    #   Pure query: we lose the vocabulary enrichment from HyDE.
    #   Blend: HyDE enriches, query anchors. Robust to both failure modes.
    fusion = alpha * query_embed + (1 - alpha) * hyde_embed
    fusion = fusion / (np.linalg.norm(fusion) + 1e-10)

    return HyDEResult(
        query        = query,
        hyde_doc     = hyde_doc,
        query_embed  = query_embed,
        hyde_embed   = hyde_embed,
        fusion_embed = fusion,
        query_tokens = approx_tokens(query),
        hyde_tokens  = approx_tokens(hyde_doc),
    )


# ─── Corpus for Demo ─────────────────────────────────────────────────────────

CORPUS = [
    {
        "id":      "c001",
        "content": "The APIC cluster requires a minimum of 3 nodes for high availability. When one node fails, the remaining two maintain quorum and continue managing fabric policy.",
        "topic":   "apic",
    },
    {
        "id":      "c002",
        "content": "EPGs in ACI define groups of endpoints sharing the same policy. Contracts permit traffic between EPGs. Without a contract, inter-EPG traffic is denied.",
        "topic":   "epg",
    },
    {
        "id":      "c003",
        "content": "ReadyOps validates changes in a Production-Representative environment. The promotion gate requires 100% pass rate from all Validation agents.",
        "topic":   "readyops",
    },
    {
        "id":      "c004",
        "content": "Cisco Hypershield uses eBPF for kernel-level policy enforcement. It integrates with ACI EPG membership to enforce microsegmentation at the workload.",
        "topic":   "hypershield",
    },
    {
        "id":      "c005",
        "content": "ACI Multi-Pod connects geographic locations using a VXLAN IPN. Each pod maintains a local APIC cluster and BGP serves as the control plane.",
        "topic":   "multipod",
    },
    {
        "id":      "c006",
        "content": "Cisco ISE assigns Security Group Tags at authentication using TrustSec. SGTs travel with packets and are enforced at network ingress.",
        "topic":   "ise",
    },
]


def search_corpus(query_vec: np.ndarray, top_k: int = 3) -> list[tuple[dict, float]]:
    """Search corpus by cosine similarity."""
    scores = [(doc, cosine_sim(query_vec, mock_embed(doc["content"]))) for doc in CORPUS]
    return sorted(scores, key=lambda x: -x[1])[:top_k]


# ─── Demo ─────────────────────────────────────────────────────────────────────

def run_hyde_demo():
    """
    Compare retrieval quality: raw query vs HyDE vs fused embedding.
    """

    print("=" * 70)
    print("HyDE: Hypothetical Document Embedding Retrieval")
    print("=" * 70)

    queries = [
        "APIC HA?",
        "What is the minimum controller node count?",
        "ReadyOps validation requirements",
    ]

    for q in queries:
        result = build_hyde_embedding(q, alpha=0.3)

        print(f"\n  {'─'*65}")
        print(f"  QUERY:    '{q}'  ({result.query_tokens} tokens)")
        print(f"  HyDE doc: '{result.hyde_doc[:100]}...'  ({result.hyde_tokens} tokens)")

        # Retrieve with each embedding
        raw_results    = search_corpus(result.query_embed)
        hyde_results   = search_corpus(result.hyde_embed)
        fusion_results = search_corpus(result.fusion_embed)

        print(f"\n  Retrieval comparison (top-3):")
        print(f"  {'Method':<12} {'Top result':<60} {'Score':>6}")
        print(f"  {'─'*12} {'─'*60} {'─'*6}")

        for method, results in [("Raw query", raw_results), ("HyDE only", hyde_results), ("Fused", fusion_results)]:
            doc, score = results[0]
            print(f"  {method:<12} '{doc['content'][:58]}'  {score:>6.3f}")


def hyde_failure_modes():
    """
    Show when HyDE fails and how fusion mitigates the risk.
    """

    print("\n" + "=" * 70)
    print("HyDE FAILURE MODES: When Hallucination Hurts Retrieval")
    print("=" * 70)

    print(f"""
  FAILURE MODE 1: Hallucinated HyDE document

    Query:    "What is the APIC node count for a 6-pod Multi-Pod?"
    HyDE gen: "A 6-pod ACI Multi-Pod requires 18 APIC nodes — 3 per pod.
               Each pod maintains a local APIC cluster."
    Problem:  "3 APIC nodes per pod" is correct, BUT "18 APIC nodes total"
              conflates per-pod (3) with global. The HyDE embedding drifts
              toward the wrong numerical context.
    Result:   Retrieves Multi-Pod docs correctly (topic right)
              but may miss the per-pod HA requirement chunk (embedding diluted).

  MITIGATION: alpha=0.3 (query keeps 30% weight)
    Even if HyDE hallucinates specifics, the query embedding anchors
    retrieval to the correct topic. The hallucinated quantity doesn't
    dominate the fused vector.

  FAILURE MODE 2: HyDE vocabulary divergence

    Query:    "Redundancy requirements for the network controller"
    HyDE gen: "High availability for the network management system requires
               redundant controllers in an active-passive cluster."
    Problem:  HyDE uses "network management system" and "active-passive" —
              vocabulary NOT in the ACI corpus (which uses "APIC", "quorum").
    Result:   Poor retrieval — vocabulary mismatch in BOTH query and HyDE.
    Fix:      Domain-specific HyDE prompt with explicit terminology.

  BEST PRACTICE:
    Always use BOTH query + HyDE (fused).
    Never use pure HyDE alone in production.
    Test HyDE quality on a held-out eval set before deploying.
    For exact-term queries (bug IDs, model numbers): skip HyDE entirely.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_hyde_demo()
    hyde_failure_modes()
