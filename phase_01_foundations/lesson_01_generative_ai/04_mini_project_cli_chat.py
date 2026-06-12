"""
FILE: 04_mini_project_cli_chat.py
LESSON: Phase 1 - Lesson 1 - What is Generative AI?
TOPIC: Mini-Project — CLI Question Answering Tool

WHAT THIS FILE TEACHES:
  - Putting Lessons 1-3 together into a real working tool
  - Stateless single-turn Q&A (no memory between questions)
  - Token tracking and cost estimation per session
  - Graceful error handling for production robustness
  - Why this simple design BREAKS for enterprise use → motivates RAG

HOW TO RUN:
  python 04_mini_project_cli_chat.py

  Type a question and press Enter.
  Type 'quit' or 'exit' to stop.
  Type 'stats' to see session token usage and cost.
  Type 'help' to see available commands.

WHY THIS IS THE STARTING POINT FOR RAG:
  This tool is purely LLM-based — no retrieval.
  It will FAIL when you ask about:
    - Internal company documents (not in training data)
    - Events after the model's training cutoff
    - Precise factual data that requires a source
  Try these failure cases yourself — they motivate every concept in Phase 1.
"""

import os
import sys
from dotenv import load_dotenv
import anthropic

load_dotenv()


# ─── Constants ────────────────────────────────────────────────────────────────

# WHY a constant for the model:
#   If you hardcode the string "claude-sonnet-4-6" in 10 places and Anthropic
#   releases a new model, you'd need to change 10 lines. One constant = one change.
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 512

# WHY a SYSTEM_PROMPT constant:
#   The system prompt defines the assistant's behavior for the ENTIRE session.
#   Centralizing it makes it easy to swap personas (e.g., strict RAG vs. general).
SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions concisely and accurately.

IMPORTANT LIMITATIONS TO BE HONEST ABOUT:
- You have a training knowledge cutoff and may not know recent events
- You do not have access to internal company documents or private data
- You can make mistakes — always encourage verification of important facts
- If you don't know something, say so clearly rather than guessing

When you are uncertain, say: "I'm not certain, but..." or "You should verify this."
"""


# ─── Session Tracker ─────────────────────────────────────────────────────────

class SessionTracker:
    """
    Tracks token usage and cost across the entire CLI session.

    WHY a class instead of global variables:
      Encapsulates state. Easy to reset, serialize, or pass around.
      In production: this becomes your billing/observability layer.
    """

    # WHY model pricing here:
    #   Duplicated from 03_token_counter.py intentionally —
    #   in a real project these would be in a shared config module.
    PRICING = {
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},  # per million tokens
    }

    def __init__(self):
        # WHY track turn_count:
        #   Useful for rate limiting (max N turns per session) and analytics.
        self.turn_count     = 0
        self.total_input    = 0  # cumulative input tokens this session
        self.total_output   = 0  # cumulative output tokens this session
        self.model          = DEFAULT_MODEL

    def record(self, input_tokens: int, output_tokens: int):
        """Record token usage from one API call."""
        self.turn_count  += 1
        self.total_input  += input_tokens
        self.total_output += output_tokens

    def total_cost(self) -> float:
        """Calculate total session cost in USD."""
        pricing = self.PRICING.get(self.model, {"input": 0, "output": 0})
        # WHY divide by 1_000_000:
        #   Pricing is per million tokens. Dividing converts raw count to millions.
        input_cost  = (self.total_input  / 1_000_000) * pricing["input"]
        output_cost = (self.total_output / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def report(self) -> str:
        """Return a formatted session summary string."""
        return (
            f"\n{'─'*45}\n"
            f"  SESSION STATISTICS\n"
            f"{'─'*45}\n"
            f"  Turns:           {self.turn_count}\n"
            f"  Input tokens:    {self.total_input:,}\n"
            f"  Output tokens:   {self.total_output:,}\n"
            f"  Total tokens:    {self.total_input + self.total_output:,}\n"
            f"  Estimated cost:  ${self.total_cost():.6f} USD\n"
            f"{'─'*45}"
        )


# ─── LLM Call ─────────────────────────────────────────────────────────────────

def ask_llm(client: anthropic.Anthropic, question: str) -> tuple[str, int, int]:
    """
    Send a question to the LLM and return answer + token counts.

    WHY return a tuple (answer, input_tokens, output_tokens):
      Caller needs both the content and the usage data.
      Using a tuple keeps it simple — no extra dataclass needed here.

    Returns:
        (answer_text, input_token_count, output_token_count)
    """

    # WHY try/except around API calls:
    #   Network calls can fail: timeout, rate limit, server error.
    #   In a CLI tool, we want to show the user a friendly message, not a crash.
    #   In production: add retry logic with exponential backoff.
    try:
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=DEFAULT_MAX_TOKENS,

            # WHY system parameter (not inside messages):
            #   Claude's API has a dedicated 'system' parameter for system prompts.
            #   This keeps it cleanly separate from the conversation messages.
            #   It's always prepended to the conversation context internally.
            system=SYSTEM_PROMPT,

            messages=[
                {"role": "user", "content": question}
            ]
        )

        answer        = response.content[0].text
        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # WHY check stop_reason:
        #   If 'max_tokens', the response is INCOMPLETE.
        #   Surface this so the user knows to re-ask with "continue" or "elaborate".
        if response.stop_reason == "max_tokens":
            answer += "\n\n[⚠ Response cut off — hit max_tokens limit. Ask me to continue.]"

        return answer, input_tokens, output_tokens

    # WHY catch specific exception types:
    #   Catching bare Exception hides bugs. Specific exceptions let you handle
    #   rate limits differently from auth errors or network failures.
    except anthropic.RateLimitError:
        return "Rate limit reached. Please wait a moment and try again.", 0, 0

    except anthropic.AuthenticationError:
        return "Authentication failed. Check your ANTHROPIC_API_KEY.", 0, 0

    except anthropic.APIConnectionError:
        return "Network error. Check your internet connection.", 0, 0

    except anthropic.APIError as e:
        return f"API error: {e}", 0, 0


# ─── Command Parser ───────────────────────────────────────────────────────────

def parse_command(user_input: str) -> str | None:
    """
    Check if user input is a special command.

    Returns the command name if recognized, None if it's a regular question.

    WHY separate command parsing from question handling:
      Clean separation of concerns. Commands are meta-actions (quit, stats).
      Questions are content that goes to the LLM.
    """
    normalized = user_input.strip().lower()

    # WHY a set for O(1) lookup:
    #   set.__contains__ is O(1) vs list.__contains__ O(n).
    #   For 3 commands it makes no difference — but it's the right habit.
    EXIT_COMMANDS = {"quit", "exit", "bye", "q"}
    if normalized in EXIT_COMMANDS:
        return "exit"

    if normalized in {"stats", "usage", "cost"}:
        return "stats"

    if normalized in {"help", "?", "h"}:
        return "help"

    if normalized in {"clear", "cls"}:
        return "clear"

    return None  # Not a command — treat as a question


# ─── Help Text ────────────────────────────────────────────────────────────────

HELP_TEXT = """
┌─────────────────────────────────────────────────────────┐
│              CLI QUESTION ANSWERING TOOL                │
│                  Phase 1, Lesson 1                      │
├─────────────────────────────────────────────────────────┤
│  Just type any question and press Enter.                │
│                                                         │
│  Commands:                                              │
│    stats   → Show token usage and estimated cost        │
│    clear   → Clear the screen                           │
│    help    → Show this help message                     │
│    quit    → Exit the program                           │
│                                                         │
│  TRY THESE TO SEE RAG'S MOTIVATION:                     │
│    "What happened in the news yesterday?"               │
│    "What does our employee handbook say about PTO?"     │
│    "What was our Q3 revenue?"                           │
│  → These will FAIL or HALLUCINATE → that's WHY RAG!     │
└─────────────────────────────────────────────────────────┘
"""


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    """
    Main REPL (Read-Eval-Print Loop) for the CLI chat tool.

    WHY a REPL pattern:
      Read input → Evaluate (call LLM) → Print output → Loop.
      Standard pattern for interactive CLI tools.
      In production this becomes a web API endpoint.
    """

    # WHY check for API key before entering the loop:
    #   Fail fast — if no key, every iteration of the loop would fail.
    #   Better to check once at startup and exit with a clear message.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not found in environment.")
        print("Create a .env file with: ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # WHY create client once outside the loop:
    #   Creating it inside the loop would create a new HTTP connection
    #   on every question — slow and wasteful.
    client = anthropic.Anthropic()

    # Session-level state
    tracker = SessionTracker()

    # Welcome message
    print("\n" + "="*55)
    print("  AI Question Answering Tool  |  Phase 1, Lesson 1")
    print("  Powered by Claude (Anthropic)")
    print("  Type 'help' for commands, 'quit' to exit")
    print("="*55)

    # ── REPL Loop ─────────────────────────────────────────────────────────────
    while True:

        # WHY try/except around input():
        #   On Ctrl+C (KeyboardInterrupt) or Ctrl+D (EOFError),
        #   Python raises these exceptions. Catching them lets us exit cleanly
        #   instead of showing an ugly traceback to the user.
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            print(tracker.report())
            break

        # Skip empty input — don't waste an API call on blank Enter
        if not user_input:
            continue

        # Check for commands
        command = parse_command(user_input)

        if command == "exit":
            print("\nGoodbye!")
            print(tracker.report())
            break

        elif command == "stats":
            print(tracker.report())
            continue

        elif command == "help":
            print(HELP_TEXT)
            continue

        elif command == "clear":
            # WHY \033[H\033[J: ANSI escape codes to clear terminal screen.
            # Works on Linux/Mac. On Windows use: os.system('cls')
            print("\033[H\033[J", end="")
            continue

        # ── Regular Question → LLM Call ───────────────────────────────────────
        print("\nAssistant: ", end="", flush=True)

        # WHY flush=True:
        #   By default Python buffers print output.
        #   flush=True forces it to display "Assistant: " before the API call
        #   completes — so the user sees feedback immediately (not after 2-3 seconds).
        answer, input_tokens, output_tokens = ask_llm(client, user_input)

        print(answer)

        # Record usage for this turn
        tracker.record(input_tokens, output_tokens)

        # Show per-turn token count (lightweight feedback without full stats)
        print(f"\n  [tokens: in={input_tokens}, out={output_tokens} | "
              f"session total: {tracker.total_input + tracker.total_output:,}]")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
