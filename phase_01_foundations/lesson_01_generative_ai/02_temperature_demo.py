"""
FILE: 02_temperature_demo.py
LESSON: Phase 1 - Lesson 1 - What is Generative AI?
TOPIC: Temperature — How randomness controls LLM output

WHAT THIS FILE TEACHES:
  - What temperature is mathematically
  - How temperature=0 gives deterministic output
  - How higher temperature gives creative/random output
  - When to use each temperature value in RAG systems

CONCEPT: Temperature and the Softmax Distribution
─────────────────────────────────────────────────
Before a token is sampled, the model outputs raw scores (logits)
for every token in its vocabulary (50,000+ tokens).

Softmax converts logits → probabilities:
  P(token_i) = exp(logit_i / T) / sum(exp(logit_j / T))

Where T = temperature.

T = 0.0 → exp(logit / ~0) → the highest logit dominates completely
           Always picks the #1 token. Fully deterministic.

T = 1.0 → exp(logit / 1.0) → normal probabilities
           Samples proportionally. Moderate randomness.

T = 2.0 → exp(logit / 2.0) → flatter distribution
           Low-probability tokens get boosted. High randomness.

Visual:
                    T=0.0     T=0.5     T=1.0     T=2.0
Token "Paris"       100%      97%       88%       65%
Token "Lyon"          0%       2%        7%       18%
Token "Nice"          0%       1%        5%       17%
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

# WHY reuse one client instance:
#   Each Anthropic() call creates an HTTP session.
#   Reusing one client reuses the connection pool — faster and cheaper.
client = anthropic.Anthropic()


def generate_with_temperature(prompt: str, temperature: float, runs: int = 3) -> list[str]:
    """
    Generate multiple responses with the same prompt at a given temperature.
    Run multiple times to observe variance.

    Args:
        prompt:      The user prompt.
        temperature: 0.0 (deterministic) to 1.0 (creative). Max is 1.0 for Claude.
        runs:        How many times to call the model with same inputs.

    Returns:
        List of generated text strings.
    """
    results = []

    for i in range(runs):
        # WHY we call the same prompt multiple times:
        #   At temperature=0, all responses should be IDENTICAL.
        #   At temperature=1.0, responses will DIFFER each time.
        #   This demonstrates how temperature controls variance.
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,  # Short limit — we just want to see variance in phrasing

            # WHY temperature parameter:
            #   Passed inside a top-level parameter called `temperature`.
            #   Range: 0.0 to 1.0 (Claude's range; OpenAI allows up to 2.0).
            #   Default is 1.0 if not specified.
            temperature=temperature,

            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract just the text portion of the response
        results.append(response.content[0].text.strip())

    return results


def demonstrate_temperature():
    """
    Run the same creative prompt at 3 different temperatures.
    Shows how output diversity changes.
    """

    # WHY a creative prompt:
    #   For factual prompts ("What is 2+2?"), temperature barely matters
    #   because the answer is deterministic by nature.
    #   Creative prompts reveal temperature's effect most clearly.
    prompt = "Write one sentence describing the sky at sunset."

    temperature_settings = [
        (0.0, "Deterministic — same output every time. Use for: extraction, classification, JSON generation."),
        (0.5, "Moderate — some variance. Use for: Q&A, summarization, RAG responses."),
        (1.0, "Creative — high variance. Use for: brainstorming, creative writing, diverse suggestions."),
    ]

    for temperature, description in temperature_settings:
        print(f"\n{'='*65}")
        print(f"Temperature: {temperature}")
        print(f"Best for: {description}")
        print(f"{'='*65}")

        # WHY runs=3:
        #   A single call doesn't show variance.
        #   3 calls at same temperature demonstrates whether output changes.
        responses = generate_with_temperature(prompt, temperature, runs=3)

        for i, response in enumerate(responses, 1):
            print(f"\n  Run {i}: {response}")

        # Check if all responses are identical (should be at temperature=0)
        # WHY set(responses):
        #   Converting to a set removes duplicates.
        #   If len==1, all 3 runs returned the exact same string → deterministic.
        unique_count = len(set(responses))
        print(f"\n  → Unique responses out of 3 runs: {unique_count}")

        if unique_count == 1:
            print("  → FULLY DETERMINISTIC (as expected at temp=0)")
        elif unique_count == 3:
            print("  → ALL DIFFERENT (high variance at this temperature)")
        else:
            print("  → SOME VARIANCE (partial randomness)")


def temperature_for_rag():
    """
    Demonstrate recommended temperature settings for RAG use cases.

    In RAG systems, you typically want LOW temperature because:
    1. You're grounding the answer in retrieved documents (facts)
    2. Hallucination risk increases with higher temperature
    3. Users expect consistent, reliable answers
    """

    print("\n" + "="*65)
    print("RAG SYSTEM: Recommended Temperature Demonstration")
    print("="*65)

    # Simulate a RAG context (we'll build the real retriever in Phase 6)
    # WHY we embed context directly in the prompt here:
    #   This is the simplest form of RAG — manually injecting retrieved text.
    #   In a real system, the retriever fetches this from a vector database.
    retrieved_context = """
    Criterion Networks is a Cisco-aligned services and platform company.
    Their platform is called ReadyOps — a continuous validation platform.
    They are a Cisco Premier Advisor and Cisco MINT Partner.
    """

    question = "What platform does Criterion Networks offer?"

    # WHY system prompt:
    #   System prompts set the model's behavior/persona for the entire conversation.
    #   In RAG: use it to inject the retrieved context and grounding instructions.
    #   This tells the model: answer ONLY from the context, don't hallucinate.
    rag_prompt_template = f"""You are a helpful assistant. Answer the question using ONLY
the context below. If the answer is not in the context, say "I don't know."

Context:
{retrieved_context}

Question: {question}
Answer:"""

    print(f"\nQuestion: {question}")
    print(f"\nRetrieved Context (injected):\n{retrieved_context.strip()}")
    print(f"\n--- Response at temperature=0.0 (recommended for RAG) ---")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        temperature=0.0,  # WHY 0.0 for RAG: maximally faithful to retrieved facts
        messages=[
            {"role": "user", "content": rag_prompt_template}
        ]
    )
    print(response.content[0].text.strip())


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DEMO 1: Effect of Temperature on Output Diversity")
    demonstrate_temperature()

    print("\n\nDEMO 2: Temperature in a RAG Context")
    temperature_for_rag()
