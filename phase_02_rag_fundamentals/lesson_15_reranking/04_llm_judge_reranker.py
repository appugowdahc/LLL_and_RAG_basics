"""
FILE: 04_llm_judge_reranker.py
LESSON: Phase 2 - Lesson 15 - Reranking
TOPIC: LLM-as-judge reranking — using the generation LLM as a relevance scorer

WHAT THIS FILE TEACHES:
  - LLM-as-judge scoring: ask the LLM to rate document relevance (1–10)
  - Pointwise vs listwise LLM ranking
  - Prompt design for reliable numeric relevance scoring
  - WHY LLM judge is the most accurate but most expensive reranking option
  - When to use LLM judge vs cross-encoder vs Cohere Rerank
  - Latency and cost model for LLM reranking

INSTALL: pip install anthropic python-dotenv
"""

import os
import re
import math
import hashlib
import json
import numpy as np
from dataclasses import dataclass, field
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
    full_rng  = np.random.RandomState(int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31))
    v = rng.randn(dims).astype(np.float32) + full_rng.randn(dims).astype(np.float32) * 0.15
    return v / (np.linalg.norm(v) + 1e-10)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                        b / (np.linalg.norm(b) + 1e-10)))


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ─── Pointwise LLM Judge ──────────────────────────────────────────────────────

POINTWISE_SYSTEM = """You are a relevance judge for a retrieval system.
Rate how relevant the given passage is to the query on a scale of 1–10.

Scoring rubric:
  10: Passage directly and completely answers the query.
  7–9: Passage is highly relevant and contains most of the answer.
  4–6: Passage is somewhat relevant — related topic but incomplete or tangential.
  1–3: Passage is off-topic or only loosely related.

Output ONLY a JSON object in this exact format (no other text):
{"score": <integer 1-10>, "reason": "<one sentence>"}"""

POINTWISE_USER = """Query: {query}

Passage:
{passage}

Rate this passage's relevance to the query (1–10):"""

# Mock scores for demo without API
MOCK_SCORES: dict[tuple[str, str], int] = {
    # (query_keyword, doc_keyword): score
}


def _mock_pointwise_score(query: str, document: str) -> tuple[int, str]:
    """Mock a pointwise LLM judge score (1–10) + reason."""
    stop  = {"what", "how", "does", "the", "and", "for", "with", "are", "is", "this"}
    q_tok = {t.lower() for t in re.findall(r"\b\w{3,}\b", query) if t.lower() not in stop}
    d_tok = {t.lower() for t in re.findall(r"\b\w{3,}\b", document) if t.lower() not in stop}
    overlap = len(q_tok & d_tok) / max(len(q_tok), 1)
    semantic = cosine_sim(mock_embed(query), mock_embed(document))
    raw = 0.5 * semantic + 0.5 * overlap
    score = max(1, min(10, int(raw * 10 + 1)))

    if score >= 8:
        reason = "Passage directly addresses the query with specific relevant information."
    elif score >= 5:
        reason = "Passage is related to the topic but does not fully answer the query."
    else:
        reason = "Passage is off-topic or only tangentially related."
    return score, reason


@dataclass
class PointwiseJudgment:
    """Score and reason from a pointwise LLM judge for one document."""
    doc:             dict
    content:         str
    score:           int         # 1–10
    reason:          str
    normalized:      float       # score / 10.0 for comparison with other methods
    tokens_used:     int


def pointwise_judge(
    query:    str,
    document: str,
) -> tuple[int, str, int]:
    """
    Ask the LLM to rate one (query, document) pair on 1–10.

    WHY pointwise vs listwise:
      Pointwise: score each doc independently → parallelizable, easy to threshold.
      Listwise: give all docs at once, ask for ranking → single LLM call but
                limited by context window and ordering biases.

    Returns: (score, reason, tokens_used)
    """
    if HAS_ANTHROPIC:
        client   = anthropic.Anthropic()
        prompt   = POINTWISE_USER.format(query=query, passage=document)
        resp     = client.messages.create(
            model      = "claude-haiku-4-5-20251001",  # WHY Haiku: cheap for scoring
            max_tokens = 60,
            system     = POINTWISE_SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
        )
        text  = resp.content[0].text.strip()
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
        try:
            parsed = json.loads(text)
            return int(parsed["score"]), str(parsed.get("reason", "")), tokens
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to extract number from text
            nums = re.findall(r"\b([1-9]|10)\b", text)
            score = int(nums[0]) if nums else 5
            return score, text[:100], tokens
    else:
        score, reason = _mock_pointwise_score(query, document)
        tokens        = approx_tokens(POINTWISE_USER.format(query=query, passage=document)) + 30
        return score, reason, tokens


# ─── Listwise LLM Judge ───────────────────────────────────────────────────────

LISTWISE_SYSTEM = """You are a relevance judge for a retrieval system.
Given a query and a numbered list of passages, output a JSON object with a ranked list of passage indices from most to least relevant.

Output format (JSON only, no other text):
{"ranking": [<index1>, <index2>, ...], "reason": "<one sentence>"}"""

LISTWISE_USER = """Query: {query}

Passages:
{passages}

Rank the passage indices from most to least relevant to the query.
Output JSON with "ranking" (list of 0-based indices) and "reason":"""


def listwise_judge(query: str, documents: list[str]) -> list[int]:
    """
    Ask the LLM to rank all documents in one call.

    WHY listwise is sometimes better than pointwise:
      - One LLM call for K documents (vs K calls for pointwise).
      - The model can compare documents directly — relative judgments can be
        more consistent than absolute 1–10 scores.

    WHY listwise has limits:
      - Context window: K > 20 documents may exceed the useful attention range.
      - Ordering bias: the LLM may prefer documents that appear earlier
        in the prompt (primacy bias). Mitigate with shuffled ordering.
      - Output parsing: harder than a simple integer.

    Returns: list of original indices sorted best-to-worst.
    """
    if HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        passages_text = "\n".join(
            f"[{i}] {doc[:300]}" for i, doc in enumerate(documents)
        )
        prompt = LISTWISE_USER.format(query=query, passages=passages_text)
        resp   = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 100,
            system     = LISTWISE_SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        try:
            parsed = json.loads(text)
            return [int(i) for i in parsed["ranking"]]
        except (json.JSONDecodeError, KeyError, ValueError):
            return list(range(len(documents)))   # fallback: original order
    else:
        # Mock: sort by mock pointwise score
        scores = [_mock_pointwise_score(query, doc)[0] for doc in documents]
        return sorted(range(len(documents)), key=lambda i: -scores[i])


# ─── LLMJudgeReranker ────────────────────────────────────────────────────────

class LLMJudgeReranker:
    """
    Reranks stage-1 candidates using LLM-as-judge scoring.
    Supports both pointwise (parallel, per-doc) and listwise (single-call) modes.
    """

    def rerank_pointwise(
        self,
        query:      str,
        candidates: list[tuple[dict, float]],  # (doc, bi_score)
        top_m:      int   = 5,
        threshold:  int   = 4,                 # drop docs with score < threshold
    ) -> list[PointwiseJudgment]:
        """
        Score each candidate independently and return top-M by score.
        WHY threshold=4: scores 1–3 are clearly irrelevant; 4+ has some value.
        """
        judgments = []
        for doc, bi_score in candidates:
            score, reason, tokens = pointwise_judge(query, doc["content"])
            if score >= threshold:
                judgments.append(PointwiseJudgment(
                    doc         = doc,
                    content     = doc["content"],
                    score       = score,
                    reason      = reason,
                    normalized  = score / 10.0,
                    tokens_used = tokens,
                ))

        judgments.sort(key=lambda j: -j.score)
        return judgments[:top_m]

    def rerank_listwise(
        self,
        query:      str,
        candidates: list[tuple[dict, float]],
        top_m:      int = 5,
    ) -> list[dict]:
        """
        Rank all candidates in one LLM call (listwise approach).
        Returns top-M docs in ranked order.
        """
        docs    = [doc for doc, _ in candidates]
        texts   = [doc["content"] for doc in docs]
        ranking = listwise_judge(query, texts)
        return [docs[i] for i in ranking[:top_m]]


# ─── Cost Comparison ──────────────────────────────────────────────────────────

def cost_comparison_demo():
    """Show the cost and latency tradeoffs for the three reranking approaches."""

    print("=" * 70)
    print("RERANKING COST & LATENCY COMPARISON")
    print("=" * 70)
    print(f"""
  Scenario: Reranking 50 candidates per query at 10K queries/day.

  ┌──────────────────────┬──────────────────┬────────────────┬───────────────┐
  │ Method               │ Latency/query    │ Cost/query     │ Cost/day      │
  ├──────────────────────┼──────────────────┼────────────────┼───────────────┤
  │ Cross-encoder (CPU)  │ 200–500ms        │ ~$0.0001       │ ~$1/day       │
  │   ms-marco-MiniLM    │ (50 × 4ms/doc)   │ (inference)    │ (cloud VM)    │
  ├──────────────────────┼──────────────────┼────────────────┼───────────────┤
  │ Cross-encoder (GPU)  │ 30–80ms          │ ~$0.0005       │ ~$5/day       │
  │   bge-reranker-large │ (50 × 0.5ms/doc) │ (GPU amortized)│ (GPU server)  │
  ├──────────────────────┼──────────────────┼────────────────┼───────────────┤
  │ Cohere Rerank v3.5   │ 100–200ms        │ ~$0.001        │ ~$10/day      │
  │   (managed API)      │ (network + model)│ (per 1K docs)  │               │
  ├──────────────────────┼──────────────────┼────────────────┼───────────────┤
  │ LLM Judge (Haiku)    │ 500–2000ms       │ ~$0.005        │ ~$50/day      │
  │   pointwise (50×)    │ (50 LLM calls)   │ (50 × $0.0001) │               │
  ├──────────────────────┼──────────────────┼────────────────┼───────────────┤
  │ LLM Judge (Haiku)    │ 200–600ms        │ ~$0.002        │ ~$20/day      │
  │   listwise (1 call)  │ (1 LLM call)     │ (1 × $0.002)   │               │
  └──────────────────────┴──────────────────┴────────────────┴───────────────┘

  QUALITY RANKING (approximate, on BEIR benchmark):
    LLM Judge (Claude Opus/Sonnet) > Cohere rerank-v3.5 > bge-reranker-large
    > ms-marco-electra > ms-marco-MiniLM

  WHEN TO USE EACH:

    Cross-encoder (self-hosted):
      - High query volume (>100K/day) where API costs become significant.
      - Low-latency requirements (<100ms P99).
      - On-premises deployments (data privacy, no API egress).

    Cohere Rerank:
      - Best default for most production RAG systems.
      - No ML infrastructure team needed.
      - Multilingual corpora.
      - Up to ~10M queries/day before self-hosting becomes cheaper.

    LLM Judge:
      - Regulated industries requiring explainability (the 'reason' field).
      - Custom relevance criteria that a cross-encoder wasn't trained on.
        Example: "Rank documents by policy-compliance risk, not just topic match."
      - Offline evaluation of retrieval quality (not online query path).
      - Use listwise to cut cost when K is manageable (K ≤ 20).
""")


# ─── Demo ─────────────────────────────────────────────────────────────────────

CORPUS = [
    {"id": "c01", "content": "A minimum of 3 APIC nodes form a cluster for quorum and HA."},
    {"id": "c02", "content": "APIC communicates with leaf and spine switches using OpFlex."},
    {"id": "c03", "content": "For ACI HA, deploy 3 APIC nodes in separate failure domains."},
    {"id": "c04", "content": "APIC REST API exposes aaaLogin on port 443 for authentication."},
    {"id": "c05", "content": "High availability requires an odd number of APIC nodes to avoid split-brain."},
    {"id": "c06", "content": "The APIC cluster health is visible in the Fault Manager dashboard."},
    {"id": "c07", "content": "BGP route reflection is the control plane for ACI Multi-Pod IPN."},
    {"id": "c08", "content": "A 3-node APIC cluster maintains policy even when one physical node fails."},
]


def run_llm_judge_demo():
    """Compare pointwise and listwise LLM judge reranking."""

    print("\n" + "=" * 70)
    print("LLM-AS-JUDGE RERANKING DEMO")
    print("=" * 70)

    reranker = LLMJudgeReranker()
    query    = "APIC cluster minimum node count for high availability"
    q_vec    = mock_embed(query)
    candidates = sorted(
        [(doc, cosine_sim(q_vec, mock_embed(doc["content"]))) for doc in CORPUS],
        key=lambda x: -x[1]
    )

    print(f"\n  Query: '{query}'")

    # Pointwise
    print(f"\n  POINTWISE judge results (top-5):")
    print(f"  {'Score':<7} {'Reason':<50} Document")
    print(f"  {'─'*6} {'─'*50} {'─'*45}")
    pwise = reranker.rerank_pointwise(query, candidates, top_m=5, threshold=3)
    for j in pwise:
        print(f"  {j.score}/10   '{j.reason[:47]}'  '{j.content[:42]}'")

    # Listwise
    print(f"\n  LISTWISE judge results (top-5, single LLM call):")
    lwise = reranker.rerank_listwise(query, candidates[:8], top_m=5)
    for i, doc in enumerate(lwise, 1):
        print(f"  [{i}]  '{doc['content'][:65]}'")

    used_api = "LIVE Claude API" if HAS_ANTHROPIC else "MOCK (no ANTHROPIC_API_KEY)"
    print(f"\n  [{used_api}]")

    cost_comparison_demo()


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_llm_judge_demo()
