"""
FILE: 01_context_anatomy.py
LESSON: Phase 1 - Lesson 6 - Context Window
TOPIC: Anatomy of a RAG context window — what goes in, how much, and why

WHAT THIS FILE TEACHES:
  - The 6 components that live inside a RAG context window
  - How to measure each component's token footprint
  - Why the context window is a SHARED budget (not unlimited per component)
  - How to set safe component budgets that never overflow
  - Visual "bar chart" of context usage

CRITICAL PRODUCTION RULE:
  Never fill the context window above 85% of capacity.
  The remaining 15% is your safety margin for:
    - Token count estimation errors (actual > estimated)
    - Response overflow (output uses more tokens than reserved)
    - System prompt changes across versions
    - Unexpected query lengths from users

INSTALL:
  pip install anthropic python-dotenv tiktoken
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens_local(text: str) -> int:
        """Fast local token counting — no API call needed."""
        return len(_enc.encode(text))
    HAS_TIKTOKEN = True
except ImportError:
    def count_tokens_local(text: str) -> int:
        """Fallback: word-based approximation (1 token ≈ 0.75 words)."""
        return int(len(text.split()) / 0.75)
    HAS_TIKTOKEN = False


# ─── Context Window Specs ─────────────────────────────────────────────────────

# WHY a dict for model specs:
#   Used throughout this lesson to validate context usage.
#   Single source of truth — update once when Anthropic releases new models.
MODEL_CONTEXT_SPECS = {
    "claude-haiku-4-5-20251001": {
        "context_window": 200_000,
        "max_output":       8_192,
        "safe_input_limit": 170_000,  # 85% of 200K
    },
    "claude-sonnet-4-6": {
        "context_window": 200_000,
        "max_output":      16_000,
        "safe_input_limit": 170_000,
    },
    "claude-opus-4-8": {
        "context_window": 200_000,
        "max_output":      32_000,
        "safe_input_limit": 170_000,
    },
}


# ─── Context Component Definitions ───────────────────────────────────────────

@dataclass
class ContextComponent:
    """
    One named piece of the context window.
    Tracks its content, token count, and recommended budget.
    """
    name:             str
    content:          str
    role:             str              # "system", "user", or "assistant"
    recommended_max:  int              # soft token budget for this component
    is_cacheable:     bool = False     # can Anthropic Prompt Cache be applied?
    is_dynamic:       bool = True      # changes per query (vs static across all queries)

    @property
    def token_count(self) -> int:
        """Count tokens in this component's content."""
        return count_tokens_local(self.content)

    @property
    def exceeds_budget(self) -> bool:
        """True if this component is over its recommended max tokens."""
        return self.token_count > self.recommended_max


@dataclass
class ContextWindow:
    """
    Full context window composed of multiple ContextComponents.
    Validates total token usage and provides a visual breakdown.
    """
    model:       str
    components:  list[ContextComponent] = field(default_factory=list)

    @property
    def specs(self) -> dict:
        return MODEL_CONTEXT_SPECS.get(self.model, {
            "context_window": 200_000,
            "max_output":      16_000,
            "safe_input_limit": 170_000,
        })

    @property
    def total_input_tokens(self) -> int:
        """Sum of all component token counts."""
        return sum(c.token_count for c in self.components)

    @property
    def utilization_pct(self) -> float:
        """What % of the context window is used by input."""
        return self.total_input_tokens / self.specs["context_window"] * 100

    @property
    def is_safe(self) -> bool:
        """True if total tokens is within the safe input limit."""
        return self.total_input_tokens <= self.specs["safe_input_limit"]

    @property
    def tokens_remaining_for_output(self) -> int:
        """Tokens available for the model's response."""
        return self.specs["context_window"] - self.total_input_tokens

    def add(self, component: ContextComponent):
        """Add a component to the window."""
        self.components.append(component)

    def validate(self) -> list[str]:
        """
        Return a list of validation warnings.

        WHY separate validation from display:
          In production: run validate() before every API call.
          Log warnings as metrics. Alert if any component exceeds budget.
          This prevents surprise context overflows in production.
        """
        warnings = []

        # Check total
        if not self.is_safe:
            overflow = self.total_input_tokens - self.specs["safe_input_limit"]
            warnings.append(
                f"TOTAL exceeds safe limit by {overflow:,} tokens "
                f"({self.utilization_pct:.1f}% of context window)"
            )

        # Check individual components
        for comp in self.components:
            if comp.exceeds_budget:
                over = comp.token_count - comp.recommended_max
                warnings.append(
                    f"Component '{comp.name}' exceeds budget by {over:,} tokens "
                    f"({comp.token_count:,} > {comp.recommended_max:,})"
                )

        # Check output headroom
        if self.tokens_remaining_for_output < self.specs["max_output"]:
            shortfall = self.specs["max_output"] - self.tokens_remaining_for_output
            warnings.append(
                f"Output headroom ({self.tokens_remaining_for_output:,}) is less than "
                f"max_output ({self.specs['max_output']:,}) by {shortfall:,} tokens"
            )

        return warnings

    def display(self):
        """
        Print a detailed token budget breakdown with visual bars.
        """

        total_window = self.specs["context_window"]
        bar_width    = 40  # characters for the bar

        print(f"\n  Context Window: {self.model}")
        print(f"  Total capacity: {total_window:,} tokens")
        print(f"  Safe input limit: {self.specs['safe_input_limit']:,} tokens (85%)")
        print()
        print(f"  {'Component':<28} {'Tokens':>8}  {'Budget':>8}  {'%':>5}  Bar")
        print(f"  {'─'*28} {'─'*8}  {'─'*8}  {'─'*5}  {'─'*bar_width}")

        for comp in self.components:
            tokens   = comp.token_count
            budget   = comp.recommended_max
            pct      = tokens / total_window * 100
            bar_len  = max(1, int(pct / 100 * bar_width))

            # WHY color code:
            #   Green (█) if within budget.
            #   Warning (▓) if within safe limit but over budget.
            #   Danger  (░) if exceeds safe limit.
            if tokens <= budget:
                bar_char = "█"
            elif tokens <= self.specs["safe_input_limit"]:
                bar_char = "▓"
            else:
                bar_char = "░"

            bar      = bar_char * bar_len
            over_str = f"  ⚠ +{tokens - budget:,}" if comp.exceeds_budget else ""

            cacheable = " [cached]" if comp.is_cacheable else ""
            dynamic   = " [dyn]"   if comp.is_dynamic   else " [static]"

            print(
                f"  {comp.name:<28} {tokens:>8,}  {budget:>8,}  {pct:>4.1f}%  "
                f"{bar}{over_str}{cacheable}"
            )

        # Totals
        print(f"  {'─'*28} {'─'*8}  {'─'*8}  {'─'*5}")
        used_pct = self.total_input_tokens / total_window * 100
        print(
            f"  {'INPUT TOTAL':<28} {self.total_input_tokens:>8,}  "
            f"{self.specs['safe_input_limit']:>8,}  {used_pct:>4.1f}%"
        )
        print(
            f"  {'OUTPUT HEADROOM':<28} "
            f"{self.tokens_remaining_for_output:>8,}  "
            f"{self.specs['max_output']:>8,}"
        )

        # Validation warnings
        warnings = self.validate()
        if warnings:
            print(f"\n  ⚠ WARNINGS ({len(warnings)}):")
            for w in warnings:
                print(f"    • {w}")
        else:
            print(f"\n  ✓ All components within budget. Context window is healthy.")


# ─── Build a Real RAG Context Window ─────────────────────────────────────────

def build_enterprise_rag_context() -> ContextWindow:
    """
    Build a realistic enterprise RAG context window with all components.
    Shows a production-ready context composition.
    """

    model  = "claude-sonnet-4-6"
    window = MODEL_CONTEXT_SPECS[model]["context_window"]  # 200,000

    # Recommended budget per component (adds up to ~85% of 200K)
    budgets = {
        "system_instruction": 500,
        "few_shot_examples":  2_000,
        "retrieved_docs":    60_000,   # 30% — main payload
        "conversation_hist": 10_000,   # 5%
        "user_query":           500,
        "output_reserve":    30_000,   # reserved for response
    }

    ctx = ContextWindow(model=model)

    # ── 1. System Instruction (static, cacheable) ─────────────────────────────
    # WHY is_cacheable=True:
    #   System instruction never changes across queries in a session.
    #   Mark it for Anthropic Prompt Caching → pay only once per 5 minutes.
    ctx.add(ContextComponent(
        name             = "System Instruction",
        role             = "system",
        is_cacheable     = True,
        is_dynamic       = False,
        recommended_max  = budgets["system_instruction"],
        content          = (
            "You are a precise enterprise knowledge assistant for Criterion Networks, "
            "a Cisco-aligned infrastructure validation company. "
            "Answer questions using ONLY the provided context documents. "
            "Always cite your sources using [Doc N] notation. "
            "If information is not in the context, respond: 'NOT IN PROVIDED CONTEXT'. "
            "Never fabricate technical specifications, product names, or version numbers. "
            "Format responses in concise paragraphs with specific evidence."
        )
    ))

    # ── 2. Few-shot Examples (static, cacheable) ──────────────────────────────
    # WHY few-shot in RAG:
    #   Shows the model EXACTLY what output format you expect.
    #   Especially useful for structured outputs (JSON, tables, citations).
    ctx.add(ContextComponent(
        name             = "Few-Shot Examples",
        role             = "system",
        is_cacheable     = True,
        is_dynamic       = False,
        recommended_max  = budgets["few_shot_examples"],
        content          = (
            "EXAMPLE INTERACTION:\n"
            "User: What is Cisco ACI's fabric mode?\n"
            "Assistant: Cisco ACI supports two fabric modes: Leaf-Spine mode and "
            "Multi-Pod mode [Doc 1]. In Leaf-Spine mode, all compute is connected "
            "to leaf switches which uplink to spine switches in a 2-tier topology [Doc 1]. "
            "Multi-Pod extends this to geographically distributed pods [Doc 2].\n\n"
            "EXAMPLE WHEN ANSWER IS MISSING:\n"
            "User: What is the default VLAN on a Nexus 9000?\n"
            "Assistant: NOT IN PROVIDED CONTEXT. The provided documents do not contain "
            "information about Nexus 9000 default VLAN configuration."
        )
    ))

    # ── 3. Retrieved Documents (dynamic per query) ────────────────────────────
    # WHY is_dynamic=True:
    #   Different queries retrieve different chunks from the vector database.
    #   Cannot be cached — changes with every query.
    retrieved_content = "\n\n".join([
        "[Doc 1] Source: aci_guide.pdf, p.12 | Score: 0.94\n"
        "Cisco ACI (Application Centric Infrastructure) is a software-defined networking "
        "solution that uses a policy-driven model to automate network provisioning. "
        "ACI uses a Leaf-Spine topology where all endpoint groups (EPGs) communicate "
        "through contracts. The APIC controller manages the entire fabric.",

        "[Doc 2] Source: aci_guide.pdf, p.45 | Score: 0.89\n"
        "ACI Multi-Pod architecture extends the ACI fabric across multiple geographic "
        "locations while maintaining a single policy domain. Inter-Pod Network (IPN) "
        "connects the pods. Each pod has its own spine and leaf layer. Traffic between "
        "pods traverses the IPN using VXLAN encapsulation.",

        "[Doc 3] Source: readyops_overview.pdf, p.3 | Score: 0.82\n"
        "ReadyOps is Criterion Networks' continuous validation platform. It operates "
        "AI agent classes across two environments: Live Operations and Production-Representative. "
        "The four agent classes are: Health & Posture, Validation, Operational, and "
        "Stress & Adversarial. Operational changes only execute in Live Operations after "
        "formal promotion from the Production-Representative environment.",

        "[Doc 4] Source: cisco_hypershield.pdf, p.8 | Score: 0.71\n"
        "Cisco Hypershield is an AI-native security architecture that embeds security "
        "directly into the network fabric and compute. It uses eBPF technology to enforce "
        "policy at the kernel level without requiring dedicated security appliances. "
        "Hypershield supports distributed exploit protection and autonomous segmentation.",

        "[Doc 5] Source: ise_admin.pdf, p.22 | Score: 0.63\n"
        "Cisco ISE (Identity Services Engine) provides network access control (NAC) "
        "and policy enforcement. ISE integrates with Active Directory for identity resolution. "
        "TrustSec SGT (Security Group Tags) labels are assigned at authentication time "
        "and propagate throughout the network via inline tagging or SGT exchange protocol.",
    ])

    ctx.add(ContextComponent(
        name             = "Retrieved Documents",
        role             = "user",
        is_cacheable     = False,
        is_dynamic       = True,
        recommended_max  = budgets["retrieved_docs"],
        content          = retrieved_content
    ))

    # ── 4. Conversation History (dynamic, sliding window) ─────────────────────
    # WHY sliding window history:
    #   Keep only the LAST N turns of conversation.
    #   Old turns are either dropped or summarized (covered in Lesson 10).
    ctx.add(ContextComponent(
        name             = "Conversation History",
        role             = "user",
        is_cacheable     = False,
        is_dynamic       = True,
        recommended_max  = budgets["conversation_hist"],
        content          = (
            "Previous turns:\n"
            "[Turn 1] User: What is Cisco ACI?\n"
            "[Turn 1] Asst: ACI is a software-defined networking solution using "
            "a policy-driven model for automated provisioning [Doc 1].\n\n"
            "[Turn 2] User: How does Multi-Pod differ from standard ACI?\n"
            "[Turn 2] Asst: Multi-Pod extends the fabric across geographic locations "
            "while maintaining one policy domain, connected via IPN with VXLAN [Doc 2]."
        )
    ))

    # ── 5. User Query (dynamic, smallest component) ───────────────────────────
    ctx.add(ContextComponent(
        name             = "User Query",
        role             = "user",
        is_cacheable     = False,
        is_dynamic       = True,
        recommended_max  = budgets["user_query"],
        content          = (
            "Question: How does ReadyOps validate Cisco ACI deployments, "
            "and which agent class handles continuous health monitoring?"
        )
    ))

    return ctx


def compare_model_budgets():
    """
    Show how the same RAG scenario uses different % of context window
    across models with different window sizes.
    """

    # Simulate a fixed-size RAG input
    input_tokens = 15_000  # typical enterprise RAG prompt

    print("\n" + "=" * 65)
    print("SAME RAG PROMPT ACROSS DIFFERENT CONTEXT WINDOWS")
    print(f"Fixed input size: {input_tokens:,} tokens")
    print("=" * 65)

    print(f"\n  {'Model':<30} {'Window':>10} {'Input':>8} {'Used%':>7} {'Remaining':>12}")
    print(f"  {'─'*30} {'─'*10} {'─'*8} {'─'*7} {'─'*12}")

    models = [
        ("GPT-3.5",                      4_096),
        ("GPT-4 / Llama-3.1",          128_000),
        ("Claude Sonnet/Haiku/Opus",    200_000),
        ("Gemini 1.5 Pro",           1_000_000),
    ]

    for model_name, window in models:
        pct_used = input_tokens / window * 100
        remaining = window - input_tokens
        status = "✓" if input_tokens < window * 0.85 else "⚠" if input_tokens < window else "✗ OOM"
        print(
            f"  {model_name:<30} {window:>10,} {input_tokens:>8,} "
            f"{pct_used:>6.1f}%  {remaining:>10,} {status}"
        )

    print(f"\n  NOTE: GPT-3.5 CANNOT fit this RAG prompt (4,096 < 15,000).")
    print(f"  Claude's 200K window fits 13× this prompt size → room for 60+ chunks.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("CONTEXT WINDOW ANATOMY: Enterprise RAG Example")
    print("=" * 65)

    ctx = build_enterprise_rag_context()
    ctx.display()

    compare_model_budgets()
