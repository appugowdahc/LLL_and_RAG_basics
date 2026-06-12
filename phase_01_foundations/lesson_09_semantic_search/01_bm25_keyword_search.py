"""
FILE: 01_bm25_keyword_search.py
LESSON: Phase 1 - Lesson 9 - Semantic Search
TOPIC: BM25 keyword search — the algorithm behind Elasticsearch and Solr

WHAT THIS FILE TEACHES:
  - TF-IDF intuition (the foundation BM25 improves on)
  - The two BM25 improvements: TF saturation and document length normalization
  - Full BM25 implementation from scratch
  - WHY BM25 beats dense search for exact technical terms
  - Inverted index: the data structure that makes BM25 fast
  - When to use BM25 vs dense vs hybrid

PRODUCTION RELEVANCE:
  Even in 2025, BM25 is NOT obsolete. Every production RAG system should use
  BM25 as a COMPONENT (not replacement) for dense search.
  Cases where BM25 wins: exact product names, error codes, config parameters,
  serial numbers, model numbers — anything that is rare and exact.

INSTALL: pip install numpy
"""

import math
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─── Text Preprocessing ───────────────────────────────────────────────────────

# Simple English stop words — filter these out to reduce noise
# WHY filter stopwords: "the", "is", "in" appear in every document and have
# zero discriminative power. IDF penalizes them but filtering is cleaner.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "in", "on", "at", "by", "for", "with",
    "as", "this", "that", "it", "its", "from", "into", "has", "have",
    "had", "will", "can", "should", "may", "might", "not", "no",
}


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Simple tokenizer: lowercase, split on non-alphanumeric, filter stopwords.

    WHY lowercase: "ACI" and "aci" should match as the same term.
    WHY split on punctuation: "ACI.fabric" → ["aci", "fabric"] not ["aci.fabric"].
    WHY NOT stemming here: stems like "validat" for "validation" can be added
      (e.g., NLTK PorterStemmer) but complicates this demo.

    In production: use a proper tokenizer (spaCy, NLTK) with stemming/lemmatization
    for better recall on morphological variants.
    """
    # WHY re.split on non-alphanum: handles "ACI/fabric", "VXLAN-EVPN", "BGP:route"
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    tokens = [t for t in tokens if len(t) > 1]  # WHY > 1: single letters are noise

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    return tokens


# ─── Step 1: TF-IDF (The Foundation) ─────────────────────────────────────────

def compute_tfidf(
    query_terms: list[str],
    doc_terms:   list[str],
    all_docs_terms: list[list[str]],
) -> float:
    """
    TF-IDF: Term Frequency × Inverse Document Frequency.
    This is what BM25 improves upon.

    TF(t, d)  = count of t in d / total terms in d
    IDF(t)    = log(N / df(t))
      N     = total documents
      df(t) = documents containing t

    WHY TF:   Rewards docs that mention the query term more often.
    WHY IDF:  Penalizes terms that appear in EVERY doc (low discriminative power).
    WHY log:  Prevents IDF from growing unboundedly for very rare terms.

    PROBLEM (fixed by BM25):
      TF grows linearly — a doc mentioning a term 100× scores 100× a doc mentioning it 1×.
      This is unrealistic: the 2nd mention adds value, but the 100th adds almost nothing.
    """
    N      = len(all_docs_terms)
    score  = 0.0

    for term in set(query_terms):
        tf  = doc_terms.count(term) / max(len(doc_terms), 1)
        df  = sum(1 for d in all_docs_terms if term in d)
        idf = math.log((N + 1) / (df + 1)) + 1   # WHY +1: smoothed IDF avoids div-by-zero

        score += tf * idf

    return score


# ─── Step 2: BM25 (The Production Standard) ──────────────────────────────────

class BM25:
    """
    BM25 (Best Matching 25) retrieval model.

    Improvements over TF-IDF:
      1. TF SATURATION: TF score grows but flattens out (controlled by k1).
         A term appearing 10× doesn't score 10× more than appearing 1× —
         more like 2-3×. This reflects real relevance more accurately.

      2. DOCUMENT LENGTH NORMALIZATION: Short docs with a matching term beat
         long docs with many terms. Controlled by b.
         b=0: no normalization. b=1: full normalization to average length.
         b=0.75 is the BM25 default.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: TF saturation. Higher → less saturation (closer to raw TF).
                Range: 1.2-2.0. Default 1.5.
                WHY 1.5: empirically optimal on TREC benchmarks.
            b:  Length normalization. 0=off, 1=full.
                Default 0.75: partial normalization.
                WHY not 1.0: over-normalizing hurts long, comprehensive documents.
        """
        self.k1     = k1
        self.b      = b
        self.corpus  = []           # list of tokenized documents
        self.doc_ids = []           # original document IDs
        self.avgdl   = 0.0          # average document length
        self.N       = 0            # corpus size
        self.idf_cache: dict[str, float] = {}    # term → IDF value
        self.inv_index: dict[str, list[int]] = defaultdict(list)  # term → doc indices

    def fit(self, documents: list[str], doc_ids: Optional[list[str]] = None):
        """
        Build the BM25 index from a list of documents.

        Steps:
          1. Tokenize all documents
          2. Compute average document length (avgdl)
          3. Build inverted index: term → [doc_idx, ...]
          4. Pre-compute IDF for every unique term

        Args:
            documents: List of document strings.
            doc_ids:   Optional list of IDs. Defaults to 0, 1, 2...
        """

        self.N       = len(documents)
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(self.N)]

        # WHY store tokenized corpus:
        #   BM25 score needs TF at query time. Pre-tokenizing avoids re-tokenizing
        #   for every query. Trade-off: more memory, faster queries.
        self.corpus  = [tokenize(doc) for doc in documents]

        # WHY avgdl:
        #   The BM25 formula normalizes each document's length against the average.
        #   Documents shorter than average get a boost; longer documents get penalized.
        self.avgdl   = sum(len(d) for d in self.corpus) / max(self.N, 1)

        # Build inverted index: term → set of document indices containing that term
        # WHY inverted index: at query time, we only score documents that contain
        # at least one query term. Without an inverted index, we'd score ALL documents.
        doc_freq: dict[str, set] = defaultdict(set)
        for idx, doc_tokens in enumerate(self.corpus):
            for token in set(doc_tokens):   # WHY set: count each term once per doc
                doc_freq[token].add(idx)
                self.inv_index[token].append(idx)

        # Pre-compute IDF for every term in the vocabulary
        # WHY pre-compute: same IDF is reused across all queries
        for term, docs_containing_term in doc_freq.items():
            df = len(docs_containing_term)
            # WHY this IDF formula (not the basic log(N/df)):
            #   Adds smoothing (+0.5 to df) to prevent IDF=∞ for terms in 1 doc.
            #   The "+1" outside prevents negative IDF for very common terms.
            self.idf_cache[term] = math.log(
                (self.N - df + 0.5) / (df + 0.5) + 1
            )

    def score(self, doc_idx: int, query_terms: list[str]) -> float:
        """
        Compute BM25 score for one document against a tokenized query.

        BM25 formula per query term t:
          score += IDF(t) × (TF(t,d) × (k1+1)) / (TF(t,d) + k1×(1 - b + b×|d|/avgdl))

        The denominator is the saturation function:
          - As TF(t,d) → ∞, the fraction → (k1+1) → constant upper bound.
          - At TF=1: score = (k1+1)/(1 + k1×length_norm)
          - At TF=10: score ≈ (k1+1) if k1 is small (fast saturation)
        """
        doc_tokens = self.corpus[doc_idx]
        dl         = len(doc_tokens)         # document length (token count)

        total_score = 0.0

        for term in query_terms:
            tf  = doc_tokens.count(term)     # raw term frequency in this document
            if tf == 0:
                continue                     # term not in this document → skip

            idf = self.idf_cache.get(term, 0.0)

            # WHY (1 - b + b × dl/avgdl):
            #   When dl = avgdl: this = 1.0 (no normalization effect)
            #   When dl < avgdl: this < 1.0 → denominator smaller → higher score
            #   When dl > avgdl: this > 1.0 → denominator larger → lower score
            length_norm = 1 - self.b + self.b * (dl / max(self.avgdl, 1))

            # The saturation factor: grows with TF but flattens
            tf_sat = (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)

            total_score += idf * tf_sat

        return total_score

    def search(
        self,
        query:    str,
        top_k:    int = 10,
    ) -> list[tuple[str, float]]:
        """
        Search for top-K documents matching the query.

        Args:
            query: Raw query string (will be tokenized).
            top_k: Number of results to return.

        Returns:
            List of (doc_id, score) sorted by score descending.
        """

        query_terms = tokenize(query)

        if not query_terms:
            return []

        # WHY inverted index lookup:
        #   Only score documents that contain at least ONE query term.
        #   If corpus has 100K docs but only 50 contain any query term,
        #   we score 50 docs not 100K.
        candidate_indices = set()
        for term in query_terms:
            candidate_indices.update(self.inv_index.get(term, []))

        if not candidate_indices:
            return []

        scores = [
            (self.doc_ids[idx], self.score(idx, query_terms))
            for idx in candidate_indices
        ]

        # Sort by score descending, return top-K
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ─── TF Saturation Visualization ─────────────────────────────────────────────

def visualize_tf_saturation():
    """
    Show how BM25's TF saturation behaves compared to raw TF.
    Demonstrates WHY BM25 doesn't over-reward repeated terms.
    """

    print("=" * 65)
    print("TF SATURATION: BM25 vs Raw TF")
    print("=" * 65)
    print("(k1=1.5, b=0 for isolation, avgdl=100, dl=100)")

    k1     = 1.5
    avgdl  = 100
    dl     = 100
    b      = 0.0   # WHY b=0: isolate TF saturation effect (no length normalization)
    idf    = 1.0   # fixed for comparison

    print(f"\n  {'TF count':>10} {'Raw TF':>12} {'BM25 TF':>12} {'Saturation%':>12}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*12}")

    raw_at_1 = 1.0
    bm25_max = (k1 + 1)  # theoretical maximum BM25 TF contribution

    for tf in [1, 2, 3, 5, 10, 20, 50, 100]:
        raw_tf   = tf
        # WHY this formula: see score() method above
        bm25_tf  = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        sat_pct  = bm25_tf / bm25_max * 100

        print(f"  {tf:>10} {raw_tf:>12.2f} {bm25_tf:>12.3f} {sat_pct:>11.1f}%")

    print(f"""
  INSIGHT:
    At TF=1:   BM25 gives full initial value, raw TF=1.
    At TF=10:  BM25 has only {(10*(k1+1))/(10+k1)/(k1+1)*100:.0f}% of max while raw TF=10.
    At TF=100: BM25 has only {(100*(k1+1))/(100+k1)/(k1+1)*100:.0f}% of max while raw TF=100.

    A document saying "ACI" 100 times doesn't score 100× a doc mentioning it once.
    BM25 caps the benefit — which better models real-world relevance.
""")


# ─── Demo: BM25 on Technical Corpus ──────────────────────────────────────────

def run_bm25_demo():
    """
    Run BM25 on a corpus of infrastructure documents.
    Shows what BM25 is good at and where it fails.
    """

    documents = {
        "doc_001": "Cisco ACI uses Leaf-Spine topology with APIC controller managing fabric policy and EPG contracts.",
        "doc_002": "The APIC REST API uses JSON over HTTPS. Authentication requires a session token from aaaLogin.",
        "doc_003": "ReadyOps validates ACI changes in a digital twin before promoting to Live Operations.",
        "doc_004": "Cisco Hypershield uses eBPF for kernel-level policy enforcement without dedicated appliances.",
        "doc_005": "ISE TrustSec assigns SGTs at authentication. SXP propagates SGTs to non-TrustSec devices.",
        "doc_006": "VXLAN EVPN provides multi-tenant fabric using BGP as the control plane on Nexus 9000.",
        "doc_007": "The Nexus 9336C-FX2 is a 36-port 100G QSFP28 switch supporting ACI and NX-OS modes.",
        "doc_008": "Bug CSCvh23456 affects APIC version 5.2(1g) causing contract deployment failures.",
        "doc_009": "ReadyOps agent classes: Health and Posture, Validation, Operational, Stress and Adversarial.",
        "doc_010": "My cat enjoys sitting near warm network switches in the data center.",
    }

    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(list(documents.values()), doc_ids=list(documents.keys()))

    queries = [
        ("Semantic query (dense wins)",   "How does ACI manage policy enforcement?"),
        ("Exact term (BM25 wins)",        "CSCvh23456"),
        ("Model number (BM25 wins)",      "Nexus 9336C-FX2"),
        ("Mixed query",                   "ReadyOps validation ACI"),
        ("Off-topic query",               "cat sitting on switches"),
    ]

    print("\n" + "=" * 65)
    print("BM25 SEARCH DEMO: Technical infrastructure corpus")
    print("=" * 65)

    for label, query in queries:
        results = bm25.search(query, top_k=3)

        print(f"\n  [{label}]")
        print(f"  Query: '{query}'")

        if results:
            for doc_id, score in results:
                print(f"    {score:>6.3f}  {doc_id}  — {documents[doc_id][:65]}...")
        else:
            print(f"    No results (no matching terms)")

    print(f"""
  KEY OBSERVATIONS:
    - "CSCvh23456" (a bug ID): ONLY BM25 finds this. A dense embedding model
      has never seen this specific bug ID in training data → near-zero similarity.
    - "Nexus 9336C-FX2" (a model number): Dense embeddings handle "Nexus" well
      but the specific model number is a rare token combination → BM25 is reliable.
    - Semantic query: BM25 finds "ACI", "EPG", "policy" but misses "Hypershield"
      which is also about policy enforcement. Dense wins here.
    - This is WHY hybrid search exists: combine both signals.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    visualize_tf_saturation()
    run_bm25_demo()
