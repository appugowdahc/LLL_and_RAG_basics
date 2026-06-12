# Phase 2 — Lesson 14: Query Understanding and Rewriting

## Why Query Rewriting Matters

The user's raw query is rarely the optimal retrieval input. It is short, ambiguous, may contain typos, uses different vocabulary than the indexed documents, or asks a complex multi-part question that no single chunk can answer.

**Query rewriting is the gap between what the user says and what the retriever needs to hear.**

```
┌─────────────────────────────────────────────────────────────────────┐
│  RAW QUERY PROBLEMS                                                 │
│                                                                     │
│  Too short:      "APIC HA?"                                         │
│  Too vague:      "What are the requirements?"                       │
│  Wrong vocab:    "cluster size" → docs say "node count"             │
│  Multi-intent:   "What is ACI and how does ReadyOps validate it?"   │
│  Negation:       "Which versions are NOT affected?"                 │
│  Misspelling:    "Hypersheld eBPF policy"                           │
│                                                                     │
│  Each problem reduces retrieval recall. Query rewriting fixes them. │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Four Query Rewriting Techniques

### 1. Query Expansion / Synonym Injection
Add related terms, synonyms, and domain-specific variations to the query before retrieval.

```
Input:  "APIC cluster size"
Output: "APIC cluster size minimum nodes HA high availability node count"
```

**When to use**: Short queries with domain vocabulary that may differ from indexed docs.

---

### 2. HyDE — Hypothetical Document Embeddings (Gao et al., 2022)
Instead of embedding the query, generate a hypothetical document that *would answer* the query, then embed that document for retrieval.

```
Query:    "What is the minimum APIC node count for HA?"
HyDE doc: "The APIC cluster in Cisco ACI requires a minimum of 3 nodes for
           high availability. When one node fails, the remaining two maintain
           quorum. APIC nodes communicate over the in-band management network."

Embed HyDE doc → retrieve similar real documents
```

**Why this works**: The hypothetical document uses the same vocabulary and structure as real indexed documents — its embedding is much closer to the target chunk than the short query embedding alone.

**When to use**: Any query where the query embedding is weaker than a full-sentence answer embedding. Almost always beneficial for factual Q&A.

---

### 3. Multi-Query Expansion
Generate N semantically different paraphrases of the query. Retrieve for each. Merge results via RRF.

```
Query:  "How does ReadyOps validate ACI changes?"
↓
Paraphrase 1: "ReadyOps validation process for ACI configuration changes"
Paraphrase 2: "Criterion Networks platform testing ACI before deployment"
Paraphrase 3: "Production-Representative environment ACI validation"
↓
Retrieve for each → RRF fusion → deduplicated merged results
```

**Why this works**: Each paraphrase covers a different part of the semantic space. Documents that are near any paraphrase get retrieved — increasing recall.

**When to use**: Queries where the vocabulary mismatch between query and docs is high.

---

### 4. Sub-Question Decomposition
Break complex multi-hop queries into atomic sub-questions. Answer each sub-question with its own retrieval. Synthesize.

```
Complex query: "What is ReadyOps and how does it integrate with ACI?"
↓
Sub-questions:
  1. "What is ReadyOps?"
  2. "How does ReadyOps integrate with Cisco ACI?"
↓
Retrieve and answer each → combine into final answer
```

**Why this works**: A single chunk cannot answer both sub-questions. Decomposition ensures each retrieval step fetches the right content for that specific sub-question.

**When to use**: Multi-part queries, comparison queries ("compare X and Y"), causal queries ("why does X happen when Y is configured?").

---

## When to Apply Each Technique

| Query characteristic | Technique | Rationale |
|---|---|---|
| Very short (≤ 5 words) | HyDE + expansion | Short queries embed poorly |
| Domain vocabulary mismatch | Multi-query | Different phrasings cover more space |
| Multiple questions in one | Sub-question decomposition | One retrieval per sub-question |
| Specific version/date in query | No rewriting needed | Exact match is already good |
| Ambiguous pronoun ("it", "this") | Coreference expansion | Replace pronoun with referent |
| Typo or misspelling | Query correction | Fix before embedding |
| General knowledge question | No rewriting needed | Parametric memory handles it |

---

## Interview Questions

**Q: What is HyDE and why does it improve retrieval?**
A: HyDE (Hypothetical Document Embeddings) generates a hypothetical answer to the query using the LLM, then embeds that answer instead of (or in addition to) the original query. The key insight is that a well-written hypothetical answer uses the same vocabulary, structure, and terminology as real indexed documents — so its embedding is much closer to the target chunk than the short query text. HyDE is particularly effective when the query is short and abstract (e.g., "APIC HA requirements?") and the indexed documents are long and specific.

**Q: What is the risk of HyDE?**
A: The hypothetical document generated by the LLM may hallucinate — it may use incorrect terminology or describe a capability that doesn't exist. If the hallucination is close to a real chunk, it retrieves the wrong document. Mitigation: use both the original query embedding AND the HyDE embedding (query + HyDE fusion), so a wrong HyDE doesn't fully derail retrieval.

**Q: How does multi-query expansion improve recall?**
A: A single query embedding covers a limited region of the semantic space. If the relevant document uses different vocabulary, the query embedding may not be close to the document embedding. Generating N paraphrases (e.g., 3-5) broadens the search to N different regions. Documents near any paraphrase are retrieved. RRF fusion deduplicates and re-ranks. This directly addresses vocabulary mismatch between query phrasing and document phrasing.

**Q: When should you NOT rewrite queries?**
A: For exact-term queries (bug IDs, CVE numbers, model numbers), rewriting adds noise and latency. BM25 already handles exact matches well — no rewriting needed. Also avoid rewriting when the query already matches the document vocabulary (e.g., the query uses the same technical terms as the knowledge base), as additional paraphrases introduce unnecessary processing.

**Q: What is sub-question decomposition and when should you use it?**
A: Decomposition breaks a complex multi-part query into atomic sub-questions, answers each independently (with its own retrieval), and synthesizes the results. Use it when: (1) the query contains "and" connecting two independent facts, (2) the query is comparative ("compare X vs Y"), (3) the query requires multi-hop reasoning (fact A leads to fact B leads to the answer). Without decomposition, a complex query's embedding is a blend of multiple topics — and no single chunk addresses all of them.

---

## Quiz

1. HyDE improves retrieval because:
   a) It bypasses the embedding step
   b) It uses BM25 for exact matching
   **c) A hypothetical answer embeds closer to real documents than a short query**
   d) It reduces token costs

2. Multi-query expansion increases:
   a) Precision
   **b) Recall (more diverse retrieval coverage)**
   c) Faithfulness
   d) Chunk quality

3. Sub-question decomposition is most useful when:
   a) The query is very short
   b) The query contains a typo
   **c) The query contains multiple independent parts**
   d) The query is about code

4. Query rewriting should be skipped when:
   a) The query is long
   **b) The query is an exact technical ID (bug ID, CVE, model number)**
   c) The query is ambiguous
   d) The user is non-technical

5. The typical cost of multi-query expansion (3 paraphrases):
   a) Same as baseline (no extra calls)
   b) 2× baseline cost
   **c) ~1 extra LLM call for paraphrase generation + 3× retrieval cost**
   d) 10× baseline cost
