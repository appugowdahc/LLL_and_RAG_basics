"""
FILE: 01_basic_generation.py
LESSON: Phase 1 - Lesson 1 - What is Generative AI?
TOPIC: Your first API call to an LLM — understand request and response structure

WHAT THIS FILE TEACHES:
  - How to initialize the Anthropic client
  - How to send a prompt to an LLM
  - What the response object looks like
  - How to extract the generated text
  - How to read token usage (important for cost control)

INSTALL:
  pip install anthropic python-dotenv

SETUP:
  Create a .env file in the project root:
    ANTHROPIC_API_KEY=sk-ant-...
"""

# ─── Imports ──────────────────────────────────────────────────────────────────

import os                       # os: access environment variables (API key)
from dotenv import load_dotenv  # load_dotenv: reads .env file into os.environ
import anthropic                # anthropic: official SDK to call Claude models


# ─── Environment Setup ────────────────────────────────────────────────────────

# WHY load_dotenv():
#   Never hardcode API keys in source code — they end up in git history.
#   .env file keeps secrets out of version control.
#   load_dotenv() reads the .env file and injects variables into os.environ.
load_dotenv()


# ─── Client Initialization ────────────────────────────────────────────────────

# WHY anthropic.Anthropic():
#   Creates a reusable HTTP client configured with your API key.
#   Automatically reads ANTHROPIC_API_KEY from environment.
#   You create this ONCE and reuse it across all calls — cheaper than
#   recreating it per request (connection pooling).
client = anthropic.Anthropic()


# ─── Core Function ────────────────────────────────────────────────────────────

def generate_response(user_question: str) -> dict:
    """
    Send a question to Claude and return the response with metadata.

    Args:
        user_question: The text prompt from the user.

    Returns:
        dict with keys: answer, input_tokens, output_tokens, model, stop_reason
    """

    # WHY client.messages.create():
    #   This is the Messages API — the standard way to interact with Claude.
    #   It follows a chat format: a list of messages with roles (user/assistant).
    #   Returns a Message object with the generated text + metadata.
    response = client.messages.create(

        # WHY model="claude-sonnet-4-6":
        #   Specifies which Claude variant to use.
        #   claude-sonnet-4-6: best balance of speed, cost, intelligence.
        #   claude-opus-4-8: most capable, more expensive.
        #   claude-haiku-4-5-20251001: fastest, cheapest, simpler tasks.
        #   Always pin the model version — model behavior changes between releases.
        model="claude-sonnet-4-6",

        # WHY max_tokens=1024:
        #   Hard cap on how many tokens the model can generate in its response.
        #   Without this cap, a verbose model might generate 10,000 tokens.
        #   In production: set this based on your use case.
        #   Q&A → 256-512, summaries → 1024, long-form → 4096+
        max_tokens=1024,

        # WHY messages=[...]:
        #   The conversation history as a list of role/content pairs.
        #   "user" role: input from the human side.
        #   "assistant" role: previous model responses (for multi-turn chat).
        #   Here we have one user message — a single-turn request.
        messages=[
            {
                "role": "user",         # Who is speaking: "user" or "assistant"
                "content": user_question  # The actual text of the message
            }
        ]
    )

    # ── Inspect the Response Object ──────────────────────────────────────────
    #
    # response is a Message object with this structure:
    #
    # Message(
    #   id='msg_01XFDUDYJgAACzvnptvVoYEL',
    #   type='message',
    #   role='assistant',
    #   content=[ContentBlock(type='text', text='...')],
    #   model='claude-sonnet-4-6',
    #   stop_reason='end_turn',   ← why generation stopped
    #   usage=Usage(input_tokens=14, output_tokens=47)
    # )
    #
    # stop_reason values:
    #   'end_turn'     → model decided it was done (normal)
    #   'max_tokens'   → hit the max_tokens limit (response may be cut off!)
    #   'stop_sequence'→ model hit a custom stop string

    # WHY response.content[0].text:
    #   content is a list because Claude can return multiple content blocks
    #   (text, tool_use, etc.). For a simple text response, [0] is the text block.
    answer = response.content[0].text

    # WHY response.usage:
    #   Token usage is how you calculate cost and monitor spend.
    #   input_tokens: tokens in your prompt (you pay for these)
    #   output_tokens: tokens in the response (you pay for these too)
    #   In production: log this to a metrics system (Datadog, CloudWatch, etc.)
    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    # WHY response.stop_reason:
    #   If stop_reason == 'max_tokens', the response was CUT OFF.
    #   In production: detect this and either retry with higher max_tokens
    #   or warn the user that the response is incomplete.
    stop_reason = response.stop_reason

    return {
        "answer":        answer,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "model":         response.model,
        "stop_reason":   stop_reason,
    }


# ─── Run Examples ─────────────────────────────────────────────────────────────

def main():
    # Example questions to demonstrate different output lengths and styles
    questions = [
        "What is Generative AI? Answer in 2 sentences.",
        "List 3 real-world use cases for Generative AI in enterprises.",
        "What is the difference between discriminative and generative AI models?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print('='*60)

        # Call the LLM
        result = generate_response(question)

        # Print the answer
        print(f"\nAnswer:\n{result['answer']}")

        # Print metadata — this is what you'd log in production
        print(f"\n--- Metadata ---")
        print(f"Model:         {result['model']}")
        print(f"Stop Reason:   {result['stop_reason']}")
        print(f"Input Tokens:  {result['input_tokens']}")
        print(f"Output Tokens: {result['output_tokens']}")
        print(f"Total Tokens:  {result['input_tokens'] + result['output_tokens']}")

        # Warn if response was cut off
        # WHY this check:
        #   A cut-off response silently gives the user incomplete information.
        #   Surfacing it allows you to increase max_tokens or re-prompt.
        if result['stop_reason'] == 'max_tokens':
            print("⚠️  WARNING: Response was cut off (hit max_tokens limit).")


# ─── Entry Point ──────────────────────────────────────────────────────────────

# WHY if __name__ == "__main__":
#   Ensures main() only runs when you execute this file directly.
#   When this file is imported as a module, main() does NOT auto-run.
#   Standard Python convention for all executable scripts.
if __name__ == "__main__":
    main()
