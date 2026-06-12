# Phase 1 — Lesson 7: Tokens Deep-Dive

## Definition

A **token** is the atomic unit of text that an LLM processes.
Not a word. Not a character. A token is a _learned subword unit_ — a chunk of text
that appears frequently enough in training data to deserve its own vocabulary entry.

```
"tokenization" → ["token", "ization"]         → [3,642, 1,634]
"RAG"          → ["R", "AG"]                  → [49, 1360]
"hello"        → ["hello"]                    → [15339]
" hello"       → [" hello"]                   → [24748]   ← leading space = different token!
```

The leading space in " hello" maps to a **different** token than "hello".
This surprises almost every developer the first time.

---

## How BPE (Byte-Pair Encoding) Works

```
BPE is the most common tokenization algorithm (GPT-4, Claude, Llama, Mistral all use it).

TRAINING PHASE (done once, before model training):

Step 1: Start with individual characters as the vocabulary
  ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]

Step 2: Count all adjacent character pairs in the corpus
  Pair ("l","l") appears 3 times → merge into "ll"
  Pair ("h","e") appears 5 times → merge into "he"
  ...

Step 3: Merge the most frequent pair → add to vocabulary
  ["he", "l", "l", "o", " ", "w", "o", "r", "l", "d"]

Step 4: Repeat until vocabulary reaches target size (e.g. 100,000 tokens)

INFERENCE PHASE (at runtime):
  Apply the same learned merge rules in priority order.
  "tokenization" → merge "t"+"o"="to" → merge "to"+"k"="tok" → ... → ["token","ization"]
```

---

## Token Counts for Common Text Types

```
Text Type                   Tokens / Character   Example
─────────────────────────────────────────────────────────────────
English prose               ~0.25 (4 chars/tok)  "The quick brown fox" = 5 tokens
English code (Python)       ~0.30 (3.3 chars/tok) import statements, keywords
JSON / YAML config          ~0.40 (2.5 chars/tok) punctuation-heavy
Chinese / Japanese          ~1.0  (1 char/tok)    each character often = 1 token
Arabic / Korean             ~0.5  (2 chars/tok)   partial word merges
Numbers (integers)          ~0.33 (3 digits/tok)  "12345" → 2-3 tokens
URLs                        ~0.6  (1.7 chars/tok) slashes, dots are their own tokens
Code with whitespace        ~0.35                 4-space indents = multiple tokens
```

---

## The Surprising Rules of Tokenization

```
RULE 1: Leading whitespace matters
  "hello"   → 1 token  (15339)
  " hello"  → 1 token  (24748)  ← different token!
  "  hello" → 2 tokens (220, 24748)

  WHY IT MATTERS FOR RAG:
    When you concatenate chunks: "...end of chunk A" + "Start of chunk B..."
    The join character(s) affect the token count and meaning.

RULE 2: Capitalization creates different tokens
  "Cisco"  → 1 token  (34,296)
  "cisco"  → 1 token  (22,745)
  "CISCO"  → 2 tokens (34,296, 18,539)  ← uppercase letters may split differently!

RULE 3: Numbers tokenize unpredictably
  "100"    → 1 token
  "1000"   → 1 token
  "10000"  → 2 tokens  (depends on frequency in training data)
  "12345"  → 2-3 tokens

  WHY IT MATTERS: Financial reports, IP addresses, config values
  cost more tokens than you'd expect.

RULE 4: Programming symbols are their own tokens
  "{"  → 1 token    "}"   → 1 token
  "()" → 1 token    "[]"  → 1 token
  "::" → 1 token    "=>"  → 1 token
  Code is ~20-30% more expensive per character than English prose.

RULE 5: Unicode and emojis are expensive
  "😀" → 2-4 tokens  (UTF-8 encoded as bytes, each a token)
  Chinese character → 1 token (after BPE merges)
  Arabic character  → 1-2 tokens

RULE 6: Context does NOT affect tokenization
  Tokenization is a pure preprocessing step.
  "bank" in "river bank" and "bank" in "bank account" → same tokens.
  Meaning disambiguation happens in the transformer, not the tokenizer.
```

---

## Token Cost Arithmetic

```
COST FORMULA:
  cost = (input_tokens / 1_000_000) × input_price_per_million
        + (output_tokens / 1_000_000) × output_price_per_million

CLAUDE SONNET PRICING (as of 2025):
  Input:  $3.00 / 1M tokens
  Output: $15.00 / 1M tokens
  Prompt Cache write: $3.75 / 1M tokens
  Prompt Cache read:  $0.30 / 1M tokens  ← 90% cheaper than input

EXAMPLE: Enterprise RAG at scale
  10,000 queries/day
  Each query: 5,000 input tokens + 300 output tokens

  Daily input cost:  10,000 × 5,000 / 1M × $3.00 = $150
  Daily output cost: 10,000 × 300 / 1M × $15.00 = $45
  Daily total: $195

  With Prompt Caching (system prompt = 500 tokens, cached):
    Cache savings: 10,000 × 500 / 1M × ($3.00 - $0.30) = $13.50/day
    Monthly savings: ~$405
```

---

## Token Budget for RAG Chunks

```
CHUNK SIZING RULE OF THUMB:
  If you want chunks of ~500 words → budget 667 tokens (500 / 0.75)
  If you budget 500 tokens per chunk → expect ~375 words

  But TEXT TYPE matters:
    500 tokens of English prose  = ~375 words   ≈ 1.5 paragraphs
    500 tokens of Python code    = ~300 words   ≈ 20 lines of code
    500 tokens of JSON config    = ~225 words   ≈ 30 key-value pairs

  This affects how many chunks fit in your context window budget:
    doc_budget = 60,000 tokens
    chunk_size = 500 tokens
    max_chunks = 60,000 / 500 = 120 chunks

  BUT if your chunks are code-heavy:
    Actual chunk size ≈ 650 tokens (30% overhead)
    max_chunks = 60,000 / 650 = 92 chunks  ← 23% fewer!
```

---

## Files in This Lesson

| File | What It Teaches |
|------|-----------------|
| 01_bpe_deep_dive.py | BPE algorithm from scratch, merge rules, vocab building |
| 02_tokenization_edge_cases.py | Whitespace, capitalization, numbers, code, Unicode |
| 03_token_cost_arithmetic.py | Cost formulas, scale projections, prompt cache savings |
| 04_multilingual_tokens.py | Token efficiency across languages, implications for global RAG |
| 05_rag_chunk_tokenization.py | Chunk sizing by content type, boundary detection |
| 06_mini_project_token_budget_planner.py | Full token budget planner for a RAG deployment |

---

## Interview Questions

Q1: What is a token and how does BPE create the vocabulary?
A: A token is a subword unit learned via Byte-Pair Encoding. BPE iteratively merges
   the most frequent character pairs in the training corpus, building up from individual
   characters to common multi-character subwords and full words. The vocabulary
   (typically 50K-100K entries) is fixed before model training begins.

Q2: Why does " hello" tokenize differently from "hello"?
A: BPE learns tokens from raw text including whitespace. The space before a word is
   part of the token's training context. "hello" at the start of a string and " hello"
   in the middle of a sentence appear differently in training data, so they get different
   vocabulary entries. This means word boundaries affect token IDs.

Q3: How does token count affect RAG system design?
A: Token count directly determines cost (price × tokens), latency (more tokens = more
   compute per forward pass), and how many chunks fit in the context window. Different
   content types (prose vs code vs JSON) have different tokens-per-character ratios,
   so chunk size budgets must account for content type. Also, output tokens cost 5× more
   than input tokens on most models — keep responses concise.

Q4: What is the "tokens per word" ratio for English vs Chinese text?
A: English prose: ~0.75 tokens/word (4 chars/token average). Chinese text: ~1 token/
   character because Chinese characters represent morphemes that each need their own
   token — there are no multi-character merges as common as English. Practical impact:
   a Chinese RAG system uses 3-4× more tokens per equivalent information content.

Q5: How would you reduce token costs in a high-volume RAG system?
A: (1) Anthropic Prompt Caching: cache static system prompt + base docs, pay $0.30/1M
   on cache reads vs $3.00/1M on regular input. (2) Model routing: use Haiku for
   classification/simple queries, Sonnet for complex reasoning. (3) Compression: LLM
   chunk compression reduces doc tokens by 40-60%. (4) Smaller chunks: 250-token chunks
   vs 1000-token chunks allow more precise retrieval → fewer chunks needed per query.

---

## Quiz

1. What algorithm does Claude use for tokenization, and what are its two phases?
2. Why does "CISCO" often tokenize into more tokens than "Cisco"?
3. A RAG system serves 50,000 queries/day with 3,000 input tokens and 200 output tokens
   per query. Calculate the monthly cost at Claude Sonnet pricing ($3/$15 per 1M).
4. You have a 60,000 token doc budget. Your chunks average 800 tokens (code-heavy).
   How many chunks can you inject? How does this compare to 500-token prose chunks?
5. What is the minimum token length for Anthropic Prompt Caching to activate?
