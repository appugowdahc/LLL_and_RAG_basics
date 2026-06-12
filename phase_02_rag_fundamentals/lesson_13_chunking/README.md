# Phase 2 — Lesson 13: Document Processing and Chunking Strategies

## Why Chunking Matters

The quality of a RAG system is bounded by the quality of its chunks. Too large: the embedding averages over too many concepts and becomes a poor retrieval signal. Too small: a single retrieved chunk lacks enough context for the LLM to answer. Wrong boundary: the LLM receives semantically incoherent context.

**Chunking is the first and most impactful engineering decision in a RAG system.**

```
┌──────────────────────────────────────────────────────────────────────┐
│  CHUNKING QUALITY IMPACT                                             │
│                                                                      │
│  Too large (>1K tokens):   Embedding blurs — low retrieval precision │
│  Too small (<50 tokens):   No context — LLM can't answer             │
│  Wrong boundary:           Incoherent chunk — model misreads it      │
│                                                                      │
│  Sweet spot:  200–500 tokens prose │ 50–200 tokens code/YAML        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Four Chunking Strategies

### 1. Fixed-Size Chunking
Split every N tokens with optional overlap.
- **Pros**: Simple, deterministic, no dependencies.
- **Cons**: Cuts sentences at arbitrary positions.
- **Use when**: Homogeneous unstructured text, baseline.

### 2. Sentence/Paragraph-Aware Chunking
Group sentences until token budget is met.
- **Pros**: Chunks respect semantic units — coherent prose.
- **Cons**: Variable chunk sizes; tables/lists may not split well.
- **Use when**: Prose documents (guides, runbooks, articles).

### 3. Recursive Chunking
Try `\n\n` first, then `\n`, then `.`, then ` ` — fall back only when chunk still exceeds budget.
- **Pros**: Preserves document structure where it exists.
- **Cons**: Can produce very uneven sizes in poorly-structured docs.
- **Use when**: Mixed-structure (Markdown, HTML, log files).

### 4. Hierarchical (Parent-Child) Chunking
Index small child chunks for retrieval precision; return parent chunk to LLM for full context.
```
Parent (1000 tokens) → sent to LLM
├── Child A (200 tokens) → used for retrieval (precise embedding)
├── Child B (200 tokens)
└── Child C (200 tokens)
```
- **Pros**: Best recall AND precision. Best context quality.
- **Cons**: More complex index; higher storage.
- **Use when**: Long technical documents needing both precision and context.

---

## Content-Type Chunk Size Guide

| Content type | Target tokens | Why |
|---|---|---|
| Prose / guides | 300–500 | Paragraphs vary; 400 is safe average |
| Technical specs | 200–400 | Dense info; smaller = more precise |
| Code blocks | 50–200 | One function per chunk |
| YAML / JSON | 50–150 | One config block per chunk |
| Tables | 100–300 | Keep headers with data rows |
| Q&A / FAQ | 100–200 | One Q+A pair per chunk |
| Incident reports | 300–500 | Timeline + context together |

---

## Overlap: Why and How Much

```
Without overlap:  [...sentence A ends.] [Sentence B starts...]
  → Answer spanning A-B boundary: neither chunk is sufficient alone.

With 50-token overlap: [...sentence A ends.] [sentence A ends. B starts...]
  → Either chunk contains the A-B context.
```

**Rule**: 10–15% of chunk size. For 400-token chunks → 40–60 token overlap.

---

## Interview Questions

**Q: What is the most common chunking mistake in production RAG?**
A: Fixed-size character splitting without respect for document structure. Cuts mid-sentence, breaks tables, splits code blocks. Second most common: same chunk size for all content types — code needs smaller chunks than prose.

**Q: What is hierarchical chunking and when should you use it?**
A: Small child chunks for retrieval precision + large parent chunk returned to LLM for full context. Use when documents are long and the relevant answer is one specific step, but that step only makes sense with surrounding context — e.g., a 10-page runbook.

**Q: How does chunk size affect retrieval precision vs recall?**
A: Small chunks → high precision (focused embedding), low recall (answer may span chunks). Large chunks → high recall, low precision (embedding blurs across topics). Optimal depends on how atomic your facts are — FAQs: small; procedures: medium; policy docs: hierarchical.

**Q: What is semantic chunking?**
A: Embeds consecutive sentences, computes cosine similarity between adjacent sentence embeddings, inserts a chunk boundary when similarity drops below a threshold. Each chunk covers one coherent topic. Slower (requires embedding at index time) but produces higher-quality retrieval targets.

**Q: How should code be chunked differently from prose?**
A: Split at function/class boundaries, not character counts. Keep docstrings with their function. Use smaller targets (one function per chunk). Mixed files (Markdown + code) need content-type detection to avoid splitting code mid-function.

---

## Quiz

1. A 400-token chunk with 40-token overlap — consecutive chunks share:
   a) 400 tokens  **b) 40 tokens**  c) 10%  d) Last sentence

2. Best strategy for a long runbook needing precision AND context:
   a) Fixed-size  b) Sentence-aware  c) Recursive  **d) Hierarchical**

3. Fixed-size character splitting is problematic for code because:
   **a) It splits functions mid-body, producing incoherent chunks**
   b) Code tokens are longer  c) Code can't be embedded

4. Sweet-spot chunk size for prose:
   a) 50–100  **b) 200–500**  c) 1,000–2,000  d) 50–200

5. Semantic chunking uses:
   a) Fixed counts  b) Regex paragraphs
   **c) Embedding similarity drops between adjacent sentences**
