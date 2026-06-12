# Phase 1 — Lesson 2: What are LLMs?

## Definition

A **Large Language Model (LLM)** is a deep neural network trained on massive
amounts of text data to understand and generate human language.

"Large" refers to TWO things:
  1. Training data size  → trillions of tokens (books, web, code, papers)
  2. Model parameters    → billions of learnable weights (GPT-4 ~1.8T, Llama3 ~70B)

An LLM learns to predict the next token given all previous tokens.
That single objective, applied at massive scale, produces general intelligence.

---

## Why It Exists

Before LLMs, NLP required:
  - Separate models for each task (translation, summarization, Q&A)
  - Hand-crafted feature engineering
  - Thousands of labeled examples per task
  - Different architectures per language

LLMs change this:
  - ONE model handles ALL language tasks
  - Zero-shot or few-shot (no task-specific training data)
  - Emergent capabilities arise from scale alone
  - Language-agnostic (one model, many languages)

---

## The Core Training Objective

Given text: "The cat sat on the"
Predict:                            "mat"

Repeat this for TRILLIONS of examples.
The model must learn:
  - Grammar        (to predict grammatically correct continuations)
  - Facts          (to predict factually accurate continuations)
  - Logic          (to predict logically consistent continuations)
  - Style          (to match the tone of the surrounding context)
  - Code patterns  (when trained on code)

This is called: Self-Supervised Learning
  WHY self-supervised: The "labels" (next tokens) come from the data itself.
  No human labeling needed. Any text corpus becomes training data automatically.

---

## Three Stages of LLM Creation

```
STAGE 1: PRE-TRAINING
─────────────────────
Raw text from internet, books, code, papers
        │
        ▼
  Train on "predict next token"
  Learns language, facts, reasoning
        │
        ▼
  BASE MODEL (knows language, but unruly)
  Example: Llama-3-70B-base, GPT-4-base


STAGE 2: SUPERVISED FINE-TUNING (SFT)
───────────────────────────────────────
Human-written (prompt, ideal response) pairs
        │
        ▼
  Fine-tune base model on these pairs
  Learns to follow instructions
        │
        ▼
  INSTRUCTION-TUNED MODEL (helpful, but may still be unsafe)


STAGE 3: RLHF (Reinforcement Learning from Human Feedback)
────────────────────────────────────────────────────────────
Human rankers compare two model responses: "A is better than B"
        │
        ▼
  Train a Reward Model on human preferences
        │
        ▼
  Use PPO (reinforcement learning) to optimize the LLM
  to produce outputs the reward model scores highly
        │
        ▼
  ALIGNED MODEL (helpful, harmless, honest)
  Example: Claude 3, GPT-4, Gemini Pro
```

---

## LLM Architecture: The Transformer Stack

```
Input Text
    │
    ▼
┌─────────────────┐
│   Tokenizer     │  Text → Token IDs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Token Embedding │  Token IDs → Dense Vectors (e.g., 4096-dim)
└────────┬────────┘
         │
         ▼
┌─────────────────┐ ─┐
│ Attention Layer │  │
├─────────────────┤  │  Repeated N times
│  Feed Forward   │  │  (GPT-3: 96 layers)
│     Network     │  │  (Llama-3-70B: 80 layers)
└────────┬────────┘ ─┘
         │
         ▼
┌─────────────────┐
│   LM Head       │  Final vector → Probability over vocab
└────────┬────────┘
         │
         ▼
    Next Token Probabilities
    ["Paris": 0.94, "Lyon": 0.01, ...]
```

We cover Transformer Architecture in full detail in Lesson 4.

---

## Key LLM Properties

### 1. Context Window
Maximum tokens the model processes at once (prompt + response).
  GPT-3.5:        4,096 tokens
  GPT-4:        128,000 tokens
  Claude 3.5:   200,000 tokens
  Gemini 1.5:   1,000,000 tokens

Critical for RAG: retrieved documents must fit in the context window.

### 2. Parameters
Learnable weights in the neural network.
More parameters ≠ always better, but enables more complex patterns.
  GPT-2:        1.5B parameters
  Llama-3-8B:     8B parameters
  Llama-3-70B:   70B parameters
  GPT-4:       ~1.8T parameters (estimated)

### 3. Temperature
Controls randomness of output. Covered in Lesson 1.

### 4. Knowledge Cutoff
Training data has an end date. Events after that date are unknown.
THIS IS WHY RAG EXISTS. (Lesson 13 covers this in full.)

---

## LLM Landscape (2025)

```
CLOSED SOURCE (API only)
├── Anthropic    │ Claude Sonnet 4.6, Opus 4.8, Haiku 4.5
├── OpenAI       │ GPT-4o, o1, o3
└── Google       │ Gemini 1.5 Pro, Gemini 2.0

OPEN SOURCE / OPEN WEIGHT (self-hostable)
├── Meta         │ Llama 3.1 (8B, 70B, 405B)
├── Mistral      │ Mistral 7B, Mixtral 8x7B
├── Alibaba      │ Qwen 2.5 (0.5B–72B)
└── Microsoft    │ Phi-3 (small, efficient)

SPECIALIZED
├── Code         │ CodeLlama, DeepSeek-Coder, Starcoder2
├── Embedding    │ BGE, E5, text-embedding-ada-002
└── Multimodal   │ Claude 3 Vision, GPT-4V, LLaVA
```

---

## LLM Capabilities (Emergent at Scale)

| Capability          | Example                                     |
|---------------------|---------------------------------------------|
| Zero-shot reasoning | Solve a problem with no examples given      |
| Few-shot learning   | Learn from 3 examples in the prompt         |
| Chain of thought    | Reason step-by-step to the answer           |
| Code generation     | Write, debug, and explain code              |
| Translation         | Between 100+ languages                      |
| Summarization       | Condense documents to key points            |
| Instruction follow  | Follow complex, multi-step instructions     |
| Tool use            | Call functions/APIs (foundation of Agents)  |

---

## LLM Limitations (Why RAG is Needed)

| Limitation          | Impact                             | RAG Solution          |
|---------------------|------------------------------------|-----------------------|
| Knowledge cutoff    | No recent facts                    | Retrieve fresh docs   |
| No private data     | Can't know internal docs           | Inject via retrieval  |
| Hallucination       | Confabulates facts confidently     | Ground in retrieved   |
| Context window cap  | Can't read all docs at once        | Chunk + retrieve      |
| No citations        | Can't point to source              | Attach source metadata|
| Stateless           | Forgets between sessions           | Memory systems        |

---

## Files in This Lesson

| File                         | What It Teaches                              |
|------------------------------|----------------------------------------------|
| 01_llm_basics.py             | Model comparison, capabilities, metadata     |
| 02_llm_training_stages.py    | Simulate base vs instruct vs aligned         |
| 03_context_window_limits.py  | Hit context limits, understand truncation    |
| 04_llm_capabilities_demo.py  | Zero-shot, few-shot, CoT, tool-use patterns  |
| 05_mini_project_model_selector.py | Choose right model for right task       |

---

## Interview Questions

Q1: What does "large" mean in Large Language Model?
A: Two things: (1) trained on trillions of tokens of text data,
   (2) billions to trillions of learnable parameters in the neural network.

Q2: What is the training objective of an LLM?
A: Predict the next token given all previous tokens (autoregressive language modeling).
   This single objective at scale produces general language intelligence.

Q3: What is RLHF and why does it matter?
A: Reinforcement Learning from Human Feedback. Humans rank pairs of model outputs;
   a reward model learns these preferences; the LLM is then fine-tuned with RL
   to maximize the reward model's score. Produces helpful, harmless, honest models.

Q4: What is the difference between a base model and an instruction-tuned model?
A: Base model completes text (raw next-token prediction, unpredictable behavior).
   Instruction-tuned model follows user instructions reliably (trained on Q&A pairs).

Q5: Why does a bigger context window matter for RAG?
A: More retrieved documents can be injected into the prompt simultaneously,
   reducing the need to aggressively filter/chunk information.

---

## Quiz

1. What are the two meanings of "large" in LLM?
2. What is self-supervised learning? Why is it powerful for LLMs?
3. What are the 3 stages of LLM creation? Name each one.
4. What is the difference between a base model and an instruction-tuned model?
5. Name 3 LLM limitations that RAG directly solves.
