# Phase 1 — Lesson 1: What is Generative AI?

## Definition

Generative AI is a class of AI systems that can **create** new content
(text, images, audio, video, code) by learning statistical patterns from
large training datasets.

Unlike discriminative AI (which classifies existing data),
generative AI **produces** novel outputs that did not exist before.

> Core idea: Given a context (a prompt), produce a plausible continuation.

---

## Why It Exists

Traditional software is explicitly programmed — every rule is hand-written.
This fails when:
- Rules are too complex to enumerate (natural language is infinite)
- The domain changes faster than engineers can update rules
- You need human-like fluency, creativity, or reasoning

Generative AI shifts the paradigm:
  Instead of writing rules → show the model examples → it learns the pattern

---

## Problem It Solves

| Old Approach         | Problem                    | Generative AI Solution          |
|----------------------|----------------------------|---------------------------------|
| Rule-based chatbots  | Can't handle novel phrasing| LLMs understand intent          |
| Template-based text  | Rigid, robotic output      | Fluid, contextual generation    |
| Keyword search       | Misses semantics           | Semantic understanding          |
| Single-task models   | One model per task         | General-purpose reasoning       |

---

## The Generative Process (Autoregressive)

Input:  "The capital of France is"

Step 1: Tokenize → ["The", "capital", "of", "France", "is"]
Step 2: Encode   → Convert tokens to numbers
Step 3: Predict  → Probability distribution over all next tokens
                   "Paris" → 94.2%
                   "Lyon"  →  1.1%
Step 4: Sample   → Pick "Paris"
Step 5: Repeat   → Feed "Paris" back, predict next token

This is called autoregressive generation.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  GENERATIVE AI SYSTEM                   │
│                                                         │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐  │
│  │  Input   │───▶│  Model (LLM)  │───▶│   Output     │  │
│  │ (Prompt) │    │               │    │  (Generated  │  │
│  │          │    │  Parameters   │    │   Content)   │  │
│  └──────────┘    │  learned from │    └──────────────┘  │
│                  │  training data│                      │
│                  └───────────────┘                      │
│                                                         │
│  KEY: Model knowledge is FROZEN after training.         │
│  It cannot learn from new events after cutoff date.     │
│  → This is WHY RAG was invented (Lesson 13).            │
└─────────────────────────────────────────────────────────┘
```

---

## Files in This Lesson

| File                        | What It Teaches                        |
|-----------------------------|----------------------------------------|
| 01_basic_generation.py      | First API call, understand response    |
| 02_temperature_demo.py      | How temperature controls randomness    |
| 03_token_counter.py         | Understand tokens and cost             |
| 04_mini_project_cli_chat.py | CLI chatbot (stateless Q&A tool)       |

---

## Best Practices

- Never treat LLM output as ground truth — always validate
- Model knowledge is frozen at training cutoff — don't rely on it for recent facts
- Temperature=0 for deterministic tasks (extraction, classification)
- Temperature>0 for creative tasks (drafting, brainstorming)
- Always track token usage — cost adds up in production

---

## Common Mistakes

| Mistake                        | Fix                                    |
|--------------------------------|----------------------------------------|
| Treating LLM output as fact    | Add retrieval + grounding (RAG)        |
| Ignoring context window limits | Chunking + retrieval (Phase 5)         |
| No output validation           | Schema validation + retry logic        |
| Logging raw LLM calls with PII | Redact sensitive fields before logging |

---

## Interview Questions

Q1: What is the difference between discriminative and generative AI?
A: Discriminative models learn the boundary between classes (spam vs not spam).
   Generative models learn the full data distribution and produce new samples.

Q2: Why can't a raw LLM answer questions about your company's internal docs?
A: LLMs are trained on public data up to a cutoff date.
   They have no knowledge of internal documents unless you inject context — RAG does this.

Q3: What is autoregressive generation?
A: Each token is predicted conditioned on all previous tokens, one at a time.

Q4: What does temperature=0 do?
A: Always picks the highest-probability next token (greedy/deterministic output).

---

## Quiz

1. What is the difference between generative and discriminative AI?
2. What does autoregressive mean in the context of LLMs?
3. Name two problems with using a raw LLM (no RAG) for a company knowledge base.
4. What does temperature=0 do to LLM output?
5. Why is the model's knowledge "frozen"?
