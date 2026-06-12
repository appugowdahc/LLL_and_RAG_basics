# Phase 1 — Lesson 6: Context Window

## Definition

The **context window** is the maximum number of tokens an LLM can process
in a single forward pass — combining ALL inputs AND the generated output.

```
context_window = system_prompt + conversation_history
               + retrieved_documents + user_query
               + generated_response

If this sum exceeds the limit → API error or silent truncation.
```

---

## Current Context Windows (2025)

```
Model                     Context Window   Practical RAG Limit
────────────────────────────────────────────────────────────────
GPT-3.5                       4,096 tk       ~3,000 tk input
GPT-4                       128,000 tk     ~100,000 tk input
Claude Haiku/Sonnet/Opus    200,000 tk     ~160,000 tk input
Gemini 1.5 Pro            1,000,000 tk     ~800,000 tk input
Llama-3.1 (405B)            128,000 tk     ~100,000 tk input
Mistral Large               128,000 tk     ~100,000 tk input
```

"Practical RAG Limit" = context window minus ~20% reserved for output tokens.

---

## The "Lost in the Middle" Problem

```
Research finding (Liu et al. 2023 — "Lost in the Middle"):
  LLMs are BETTER at using information at the START and END of the context.
  Information in the MIDDLE is retrieved less reliably.

  Accuracy by chunk position (approximate):
    Position 1 (first):   ~80% accuracy
    Position 5 (middle):  ~55% accuracy
    Position 10 (last):   ~75% accuracy

IMPLICATION FOR RAG CHUNK PLACEMENT:
  ┌─────────────────────────────────────────────────────┐
  │  SYSTEM PROMPT                                      │ ← always seen well
  │  ─────────────────────────────────────────────────  │
  │  Most relevant chunk   [rank 1]                     │ ← strong attention
  │  Second most relevant  [rank 2]                     │ ← good attention
  │  ─────────────────────────────────────────────────  │
  │  Less relevant chunks [rank 3..N-1]  ← DANGER ZONE  │ ← attention degrades
  │  ─────────────────────────────────────────────────  │
  │  Last relevant chunk   [rank N]                     │ ← strong attention
  │  ─────────────────────────────────────────────────  │
  │  USER QUERY                                         │ ← always seen well
  └─────────────────────────────────────────────────────┘

  STRATEGY: Place most important chunks at START and END.
  Or: Use fewer, higher-quality chunks instead of many mediocre ones.
```

---

## Context Window Budget Allocation

```
RAG PROMPT ANATOMY (recommended allocation for Claude Sonnet, 200K window):

  Component               Tokens    % of 200K   Notes
  ────────────────────────────────────────────────────────────
  System instruction      200-500    0.25%       Persona, grounding rules
  Few-shot examples       0-2000     0-1%        Optional, for format guidance
  Retrieved documents     60,000    30%          Main RAG payload
  Conversation history    10,000     5%          Last N turns (sliding window)
  User query              50-200     0.1%        Current question
  Output reservation      30,000    15%          max_tokens ceiling
  Safety buffer           5,000      2.5%        Never fill 100%
  ────────────────────────────────────────────────────────────
  Total used              ~105,000   52.5%
  Remaining headroom       95,000    47.5%

  WHY leave 47.5% free?
    RAG queries vary in complexity. Some questions need
    5 chunks. Others need 20. Headroom lets you scale dynamically.
```

---

## Context Window Architecture Patterns

```
PATTERN 1: FIXED ALLOCATION (simple, safe)
  Always inject K chunks of fixed size.
  K=5, chunk_size=500 → always 2,500 doc tokens.
  PRO: Predictable cost. CON: Wastes budget for simple queries.

PATTERN 2: DYNAMIC ALLOCATION (efficient)
  Inject as many chunks as fit under a token budget.
  token_budget = 60,000 (30% of window)
  Fill with top-scored chunks until budget is reached.
  PRO: Uses budget fully. CON: Response size varies.

PATTERN 3: SCORE-THRESHOLD ALLOCATION (quality-focused)
  Only inject chunks with similarity score > threshold.
  threshold = 0.70
  PRO: No irrelevant content injected. CON: May get 0 chunks.

PATTERN 4: HIERARCHICAL ALLOCATION (advanced)
  Different context types get separate fixed budgets:
    Core documents:   30,000 tokens
    Supporting docs:  15,000 tokens
    History:          10,000 tokens
  Ensures core context is never displaced by history.
```

---

## Context Extension Techniques

```
PROBLEM: Model trained on 4K context, you need 128K context.
  Simple approach: just increase max position id → breaks
  because the model has never seen positions > 4096.

SOLUTIONS:

1. YaRN (Yet another RoPE extensioN):
   Scales RoPE θ values to spread existing frequencies
   across a longer context. Fine-tuned on long docs.
   Used by: Mistral long context, Yi-200K.

2. NTK-Aware Scaling:
   Changes the base frequency (10000) of RoPE proportionally.
   No fine-tuning needed — works out of the box.
   Formula: new_base = base × (new_ctx/trained_ctx)^(d/(d-2))

3. ALiBi (already relative):
   Naturally generalizes to longer sequences — no modification needed.
   The penalty just applies to larger distances.

4. Position Interpolation (Chen et al. 2023):
   Scale all position indices: pos → pos × (trained_ctx / target_ctx)
   Requires brief fine-tuning on longer sequences.
   Used by: LongLlama, LongAlpaca.

RAG TAKEAWAY:
  Always check what context length the model was FINE-TUNED on,
  not just what it CLAIMS to support.
  Quality degrades beyond fine-tuning length even with extension.
```

---

## Files in This Lesson

| File                              | What It Teaches                                    |
|-----------------------------------|----------------------------------------------------|
| 01_context_anatomy.py             | Token budget breakdown, section analysis           |
| 02_prompt_engineering_for_rag.py  | Optimal RAG prompt patterns, few-shot, grounding   |
| 03_lost_in_the_middle.py          | Positional bias, chunk ordering strategies         |
| 04_window_filling_strategies.py   | Fixed, dynamic, threshold, hierarchical allocation |
| 05_context_compression.py         | History compression, chunk summarization           |
| 06_mini_project_context_optimizer.py | Full context window optimizer + RAG call        |

---

## Interview Questions

Q1: What is the context window and what does it include?
A: The maximum tokens processed in one LLM call. It includes system prompt,
   conversation history, retrieved documents, user query, AND the model's
   generated response. All must fit within the limit simultaneously.

Q2: What is the "Lost in the Middle" problem and how do you mitigate it?
A: LLMs attend better to context at the START and END of the window. Middle
   information is less reliably used. Mitigation: place most critical chunks
   at the beginning (or end), use reranking to select only top chunks,
   or use fewer but higher-quality retrieved documents.

Q3: How would you implement a dynamic context window filler for RAG?
A: Sort retrieved chunks by relevance score (descending). Maintain a running
   token count. Add chunks until count exceeds the budget allocation for
   retrieved docs (e.g., 30% of context window). Stop when budget is full
   or chunks fall below a quality threshold.

Q4: What is YaRN and why would you use it in a RAG system?
A: YaRN (Yet another RoPE extensioN) scales RoPE frequencies to allow a model
   trained at 4K/8K tokens to process 64K-128K+ tokens. Use it when you need
   long RAG contexts but only have access to a smaller base model.

Q5: How does Anthropic's Prompt Caching interact with context window management?
A: Prompt Caching caches computed key-value states for the first N tokens.
   In RAG: cache the static system prompt + base documents (unchanged across calls).
   Dynamic parts (user query, turn-specific docs) are appended after the cache point.
   This reduces latency and cost for repeated context prefixes.

---

## Quiz

1. What is included in the context window token count?
2. Describe the "Lost in the Middle" finding and its implication for RAG.
3. What is the difference between fixed and dynamic context allocation?
4. How does NTK-Aware scaling extend a model's context window?
5. Where should you place your most important retrieved chunk — start, middle, or end?
