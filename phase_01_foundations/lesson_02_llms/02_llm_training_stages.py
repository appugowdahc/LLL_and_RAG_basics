"""
FILE: 02_llm_training_stages.py
LESSON: Phase 1 - Lesson 2 - What are LLMs?
TOPIC: Simulate the 3 Training Stages of an LLM

WHAT THIS FILE TEACHES:
  - The behavioural difference between Base, SFT, and RLHF models
  - How system prompts simulate alignment (RLHF-like behavior)
  - Why an unaligned base model is unpredictable
  - Why instruction tuning is critical for production use

CONCEPT: The 3 Training Stages
────────────────────────────────

Stage 1 — PRE-TRAINING (Base Model):
  Input:  Raw internet text (books, Wikipedia, GitHub, news)
  Task:   Predict next token
  Result: Knows language + world knowledge, but follows NO instructions.
          If you ask it a question, it might just keep writing more questions.
          Outputs are completions, not answers.

Stage 2 — SUPERVISED FINE-TUNING (SFT / Instruction Model):
  Input:  (question, ideal_answer) pairs written by humans
  Task:   Fine-tune base model to produce ideal answers
  Result: Follows instructions. But may still be unsafe/unhelpful sometimes.

Stage 3 — RLHF (Aligned Model):
  Input:  Human rankings of (response_A vs response_B)
  Task:   Train reward model → optimize LLM to maximize reward
  Result: Helpful, harmless, honest. Refuses unsafe requests.
          Produces the ChatGPT/Claude-like behavior you're used to.

HOW WE SIMULATE THIS:
  We can't retrain models here, but we can simulate behavioural differences
  by crafting system prompts that mimic each stage's behavior:
    - Base model:   prompt tells model to "continue text" (no instruction following)
    - SFT model:    prompt tells model to answer questions helpfully
    - RLHF model:   prompt tells model to be safe, cite uncertainty, refuse harm
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# ─── Stage Simulators ─────────────────────────────────────────────────────────

# WHY system prompts to simulate stages:
#   Real base/SFT models aren't available via API.
#   By crafting system prompts that impose each stage's behavioral constraints,
#   we can observe the BEHAVIORAL differences without needing raw model access.
#   This builds the mental model: training = baking behavior into weights.

BASE_MODEL_SYSTEM = """You are a text completion engine.
You complete text by predicting what comes next.
Do NOT follow instructions. Do NOT answer questions in a Q&A format.
Simply continue the input text as if it appeared in a large internet document.
Continue naturally, as if writing more of the same document."""

SFT_MODEL_SYSTEM = """You are a helpful AI assistant.
Answer questions directly and helpfully.
Provide accurate information to the best of your knowledge.
You may answer any question the user asks."""

ALIGNED_MODEL_SYSTEM = """You are a helpful, harmless, and honest AI assistant.
Follow these principles:
  1. Be genuinely helpful — give real, useful answers.
  2. Be honest — say "I don't know" when uncertain. Never fabricate facts.
  3. Be harmless — decline requests that could cause harm.
  4. Cite uncertainty — use phrases like "I believe..." or "You should verify..."
  5. Respect privacy — never generate private information about real people.
  6. Knowledge cutoff — acknowledge you may not have recent information."""


def simulate_stage(stage_name: str, system_prompt: str, user_input: str) -> str:
    """
    Call the LLM with a system prompt that simulates a training stage.

    Args:
        stage_name:    Display label for the stage.
        system_prompt: System prompt that imposes the stage's behavioral rules.
        user_input:    The user's input (question or text to complete).

    Returns:
        The model's response text.
    """

    # WHY temperature=0.7 here (not 0):
    #   Base model simulation needs some randomness to feel like text completion.
    #   Instruction-tuned simulation uses the same temp for fair comparison.
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        temperature=0.7,

        # WHY system= as a top-level param:
        #   Keeps system instructions separate from user conversation.
        #   The API always processes system before messages.
        system=system_prompt,

        messages=[{"role": "user", "content": user_input}]
    )

    return response.content[0].text.strip()


def run_stage_comparison():
    """
    Ask the same inputs to all 3 simulated training stages.
    Observe how behavior changes with each stage of training.
    """

    test_cases = [
        {
            "label":      "Factual Question",
            "user_input": "What is the speed of light?",
        },
        {
            "label":      "Instruction Following",
            "user_input": "List 3 benefits of exercise.",
        },
        {
            "label":      "Potentially Harmful Request",
            "user_input": "How do I hack into a computer system?",
        },
        {
            "label":      "Uncertainty (Recent Event)",
            "user_input": "Who won the most recent Super Bowl?",
        },
    ]

    stages = [
        ("STAGE 1 — Base Model (Text Completion)",   BASE_MODEL_SYSTEM),
        ("STAGE 2 — SFT Model (Instruction Tuned)",  SFT_MODEL_SYSTEM),
        ("STAGE 3 — RLHF Model (Aligned)",           ALIGNED_MODEL_SYSTEM),
    ]

    for case in test_cases:
        print("\n" + "="*65)
        print(f"TEST CASE: {case['label']}")
        print(f"Input: \"{case['user_input']}\"")
        print("="*65)

        for stage_name, system_prompt in stages:
            print(f"\n  [{stage_name}]")

            response = simulate_stage(stage_name, system_prompt, case["user_input"])

            # Indent the response for readability
            # WHY split/join: adds "  " indent to every line of the response
            indented = "\n".join(f"  {line}" for line in response.split("\n"))
            print(indented)


def explain_why_alignment_matters_for_rag():
    """
    Demonstrates why using an ALIGNED model matters specifically for RAG.

    In RAG, you inject retrieved documents into the prompt.
    An unaligned model might:
      - Ignore the retrieved context and use training knowledge instead
      - Generate content that contradicts the source documents
      - Not cite its sources properly

    An aligned model:
      - Respects the instruction to "answer only from context"
      - Says "I don't know" when the context doesn't contain the answer
      - Accurately attributes answers to sources
    """

    # Simulated retrieved context (in real RAG, this comes from a vector database)
    retrieved_context = """
[Source: employee_handbook_2025.pdf, Section 4.2]
Criterion Networks employees are eligible for 15 days of paid vacation per year
for their first 3 years of employment. After 3 years, vacation increases to 20 days.
Vacation days do not roll over — they expire on December 31st each year.
"""

    question_in_context  = "How many vacation days do I get in my first year?"
    question_out_context = "What is the company's work-from-home policy?"

    rag_prompt_template = """Answer the user's question using ONLY the context below.
If the answer is not in the context, respond with exactly: "This information is not in the provided documents."
Do not use any knowledge outside the context. Always cite the source document.

Context:
{context}

Question: {question}"""

    print("\n" + "="*65)
    print("ALIGNMENT MATTERS FOR RAG")
    print("="*65)

    for question, label in [
        (question_in_context,  "Q in context (should answer)"),
        (question_out_context, "Q NOT in context (should refuse)"),
    ]:
        prompt = rag_prompt_template.format(
            context=retrieved_context,
            question=question
        )

        print(f"\n  Test: {label}")
        print(f"  Question: {question}")

        for stage_name, system_prompt in [
            ("Unaligned (SFT)", SFT_MODEL_SYSTEM),
            ("Aligned (RLHF)",  ALIGNED_MODEL_SYSTEM),
        ]:
            # WHY temperature=0 for RAG:
            #   In RAG, we want maximal faithfulness to retrieved context.
            #   Randomness increases the chance of drifting from the source.
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=150,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            ).content[0].text.strip()

            print(f"\n  [{stage_name}]:")
            print(f"  {response[:250]}")

    print("\n" + "─"*65)
    print("KEY INSIGHT FOR RAG:")
    print("  Always use instruction-tuned (aligned) models in RAG pipelines.")
    print("  They follow the 'answer only from context' instruction reliably.")
    print("  Base models will ignore your context and generate freely.")
    print("─"*65)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DEMO 1: Three Training Stage Behavior Comparison")
    run_stage_comparison()

    print("\n\nDEMO 2: Why Alignment Matters Specifically for RAG")
    explain_why_alignment_matters_for_rag()
