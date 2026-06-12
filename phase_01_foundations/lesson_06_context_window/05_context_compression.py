"""
FILE: 05_context_compression.py
LESSON: Phase 1 - Lesson 6 - Context Window
TOPIC: Context compression — shrinking what goes into the context window
       without losing the information the model needs to answer correctly.

WHAT THIS FILE TEACHES:
  - WHY context compression is necessary (token cost at scale)
  - Conversation history compression (sliding window vs LLM summarization)
  - Chunk compression (extract only the relevant sentence)
  - Query-focused compression (keep only sentences relevant to the query)
  - Map-reduce compression for documents that don't fit at all
  - Token savings analysis

KEY INSIGHT:
  A RAG system without compression is O(N) token cost per document added.
  A RAG system with compression keeps cost roughly constant as knowledge grows.

  Without compression: 50 docs × 500 tokens = 25,000 tokens PER QUERY
  With compression:    50 docs compressed to 5 sentences each = 2,500 tokens

INSTALL:
  pip install anthropic python-dotenv tiktoken
"""

import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens_local(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens_local(text: str) -> int:
        return int(len(text.split()) / 0.75)


# ─── 1. Sliding Window History Compression ────────────────────────────────────

@dataclass
class ConversationTurn:
    role:    str    # "user" or "assistant"
    content: str

    @property
    def token_count(self) -> int:
        return count_tokens_local(self.content)


def sliding_window_history(
    history:          list[ConversationTurn],
    max_history_toks: int = 4_000,
) -> list[ConversationTurn]:
    """
    Keep only the most recent turns that fit within max_history_toks.
    Older turns are discarded.

    WHY sliding window (not just truncate from the front):
      In a conversation, the MOST RECENT turns are most relevant to the
      current query. Old context about a different topic is often wasted tokens.

    WHY discard whole turns (not partial):
      A partial turn (cut mid-sentence) confuses the model about what was said.
      Always drop complete user+assistant turn pairs.

    Args:
        history:          Full conversation history (oldest first).
        max_history_toks: Token budget for history.

    Returns:
        Trimmed history (most recent turns that fit).
    """

    # WHY reversed():
    #   We want the MOST RECENT turns. Start from the end and work backward.
    recent_turns = []
    running_toks = 0

    for turn in reversed(history):
        turn_toks = turn.token_count
        if running_toks + turn_toks > max_history_toks:
            break   # Next oldest turn would overflow — stop here
        recent_turns.insert(0, turn)  # Prepend to maintain chronological order
        running_toks += turn_toks

    return recent_turns


def summarize_history_with_llm(
    history:   list[ConversationTurn],
    max_tokens: int = 200,
) -> str:
    """
    Compress old conversation history into a brief summary using the LLM.
    Used when sliding window would discard important context.

    WHY LLM summarization instead of just dropping turns:
      If an early turn established an important constraint ("I'm asking about
      Cisco ACI specifically, not SD-WAN"), discarding it means the model
      forgets this constraint in later turns.
      A summary captures the KEY points even from old turns.

    WHY keep summary SHORT (max_tokens=200):
      The summary replaces N turns of history.
      If N=10 turns × 100 tokens = 1,000 tokens, a 200-token summary
      saves 800 tokens while preserving the gist.

    Args:
        history:    Turns to summarize (oldest, not-fitting turns).
        max_tokens: Maximum summary length in tokens.

    Returns:
        A concise summary string to prepend to the kept history.
    """

    if not history:
        return ""

    # Format conversation for summarization
    conversation_text = "\n".join(
        f"{t.role.upper()}: {t.content}" for t in history
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # WHY Haiku: cheap, fast — this is a utility call
        max_tokens=max_tokens,
        temperature=0,
        messages=[{
            "role":    "user",
            "content": (
                "Summarize this conversation in 2-3 sentences. "
                "Keep: the user's main goal, any constraints they mentioned, "
                "and key facts that were established. "
                "Discard: pleasantries, repeated questions, long explanations.\n\n"
                f"CONVERSATION:\n{conversation_text}"
            )
        }]
    )

    return response.content[0].text.strip()


def hybrid_history_compression(
    history:          list[ConversationTurn],
    max_history_toks: int = 4_000,
    summary_reserve:  int = 300,
) -> tuple[str, list[ConversationTurn]]:
    """
    Hybrid approach:
      1. Keep the most recent turns (sliding window) — verbatim, highest fidelity.
      2. Summarize older turns that were dropped — preserves key context cheaply.

    WHY hybrid beats pure sliding window:
      Pure sliding window loses early-turn constraints completely.
      Pure LLM summarization requires an extra API call every turn (expensive).
      Hybrid: only summarize when history grows past the window — less frequent.

    Returns:
        (summary_text, recent_turns) — combine as:
          [summary_system_note] + recent_turns when building message list
    """

    recent_turns = sliding_window_history(history, max_history_toks - summary_reserve)
    recent_set   = set(id(t) for t in recent_turns)
    old_turns    = [t for t in history if id(t) not in recent_set]

    if old_turns:
        summary = summarize_history_with_llm(old_turns, max_tokens=summary_reserve)
        summary_text = f"[EARLIER CONTEXT SUMMARY]\n{summary}\n[END SUMMARY]"
    else:
        summary_text = ""

    return summary_text, recent_turns


# ─── 2. Query-Focused Chunk Compression ──────────────────────────────────────

def extract_relevant_sentences(
    chunk_content: str,
    query:         str,
    max_sentences: int = 3,
) -> str:
    """
    Extract only the sentences from a chunk that are relevant to the query.
    Discards boilerplate, background, and off-topic sentences.

    WHY this matters:
      A retrieved chunk is often a paragraph from a document section.
      The section header, surrounding context, and unrelated sentences
      may be in the same paragraph but irrelevant to THIS query.
      Compressing to relevant sentences can reduce chunk size by 40-60%.

    APPROACH: Simple heuristic — score each sentence by shared content words with query.
    (In production: use a cross-encoder reranker or small LLM to score sentences.)

    Args:
        chunk_content: Full chunk text.
        query:         The user's question.
        max_sentences: Max sentences to keep.

    Returns:
        Compressed chunk with only relevant sentences.
    """

    sentences    = [s.strip() for s in chunk_content.split(".") if len(s.strip()) > 20]
    query_words  = set(query.lower().split())

    # Score each sentence: count shared content words with query
    def sentence_score(sentence: str) -> float:
        # WHY filter stopwords:
        #   Common words like "the", "is", "in" appear in every sentence.
        #   Filtering them focuses scoring on content words.
        stopwords = {"the","a","an","is","are","was","were","be","been","to","of","and",
                     "or","in","on","at","by","for","with","as","this","that","it",
                     "its","from","into","has","have","had","will","can","should"}
        sentence_words  = set(sentence.lower().split()) - stopwords
        query_content   = query_words - stopwords
        shared          = sentence_words & query_content
        return len(shared) / max(len(query_content), 1)

    scored = sorted(
        [(s, sentence_score(s)) for s in sentences],
        key=lambda x: x[1],
        reverse=True
    )

    # Keep top-N sentences, in their original ORDER (not by score order)
    # WHY restore original order:
    #   Preserving sentence order maintains coherence of the extracted text.
    top_sentences    = {s for s, _ in scored[:max_sentences]}
    ordered_selected = [s for s in sentences if s in top_sentences]

    return ". ".join(ordered_selected) + "."


def llm_chunk_compression(
    chunk_content: str,
    query:         str,
    target_tokens: int = 100,
) -> str:
    """
    Use an LLM (Haiku) to extract the query-relevant information from a chunk.
    More accurate than heuristic extraction but costs API tokens.

    WHY use LLM for compression:
      The heuristic above uses word overlap — it misses semantic similarity.
      Example: query "How does ACI handle multi-tenancy?" will score low on
      a sentence about "VRFs and Bridge Domains" even though VRF = tenant isolation.
      An LLM understands the MEANING and extracts the right sentence.

    WHY Haiku (not Sonnet):
      Compression is a utility task — cheap model is enough.
      You may compress 50 chunks per query → Haiku costs 1/10th of Sonnet.

    Args:
        chunk_content: Full chunk text.
        query:         The user's question.
        target_tokens: Target compression length.

    Returns:
        Compressed text (roughly target_tokens long).
    """

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=target_tokens,
        temperature=0,
        messages=[{
            "role":    "user",
            "content": (
                f"Extract ONLY the sentences from the DOCUMENT that are relevant to the QUERY. "
                f"Output only the extracted sentences. If nothing is relevant, output: [IRRELEVANT]\n\n"
                f"QUERY: {query}\n\n"
                f"DOCUMENT:\n{chunk_content}"
            )
        }]
    )

    return response.content[0].text.strip()


# ─── 3. Map-Reduce Compression (for very long documents) ─────────────────────

def map_reduce_compress(
    long_document:  str,
    query:          str,
    chunk_size:     int = 1_000,  # tokens per map chunk
    target_tokens:  int = 400,
) -> str:
    """
    Compress a document that is too long to fit in the context window at all.

    Map-Reduce approach:
      MAP:    Split document into chunks of chunk_size tokens.
              For each chunk: extract relevant sentences (parallel in production).
      REDUCE: Combine all extracted sentences → final LLM summary.

    WHY this beats just truncating:
      Truncation loses the second half of the document entirely.
      Map-reduce processes the WHOLE document and extracts relevant info
      from every section, then distills it.

    WHY two stages (map then reduce):
      Map chunks independently fit in context window.
      Reduce merges the maps into a coherent answer.
      This pattern scales to arbitrarily long documents.

    Args:
        long_document: Full text of the document.
        query:         The user's question.
        chunk_size:    Approximate tokens per map chunk.
        target_tokens: Final compressed output size.

    Returns:
        Compressed document text covering query-relevant information.
    """

    # ── MAP phase ─────────────────────────────────────────────────────────────

    # Split document into word-level chunks (approximate token split)
    # WHY words not characters: more stable split size
    words = long_document.split()
    words_per_chunk = int(chunk_size * 0.75)  # tokens × 0.75 ≈ words

    chunks      = []
    for i in range(0, len(words), words_per_chunk):
        chunk_text = " ".join(words[i : i + words_per_chunk])
        chunks.append(chunk_text)

    print(f"    Map phase: {len(chunks)} chunks from {count_tokens_local(long_document)} tokens")

    # Extract relevant sentences from each chunk
    # WHY small target per chunk: each chunk contributes one key insight
    per_chunk_target = max(50, target_tokens // len(chunks))

    extracts = []
    for i, chunk in enumerate(chunks):
        extract = extract_relevant_sentences(chunk, query, max_sentences=2)
        if "[IRRELEVANT]" not in extract and len(extract) > 20:
            extracts.append(extract)
        print(f"    Chunk {i+1}/{len(chunks)}: {count_tokens_local(extract)} tokens extracted")

    # ── REDUCE phase ──────────────────────────────────────────────────────────

    if not extracts:
        return "[NO RELEVANT INFORMATION FOUND IN DOCUMENT]"

    combined = "\n".join(extracts)

    # Final LLM compression of all extracts into a coherent answer
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=target_tokens,
        temperature=0,
        messages=[{
            "role":    "user",
            "content": (
                f"Synthesize these extracted document passages into a coherent {target_tokens}-token "
                f"summary that answers the query. Keep only facts directly relevant to the query.\n\n"
                f"QUERY: {query}\n\n"
                f"EXTRACTS:\n{combined}"
            )
        }]
    )

    return response.content[0].text.strip()


# ─── Demo: Token Savings Analysis ────────────────────────────────────────────

def token_savings_demo():
    """
    Compare token counts before and after compression for realistic chunks.
    """

    print("=" * 65)
    print("CONTEXT COMPRESSION: Token Savings Analysis")
    print("=" * 65)

    query  = "How does ReadyOps validate changes before deploying to production?"

    chunks = [
        {
            "id":      "Doc 1",
            "content": (
                "Criterion Networks was founded in 2020. The company specializes in "
                "enterprise infrastructure validation and operations services. "
                "Criterion is a Cisco Premier Advisor and MINT Partner. "
                "ReadyOps is the company's continuous validation platform. "
                "It operates AI agent classes across two deliberately isolated environments: "
                "Live Operations and Production-Representative. "
                "The Production-Representative environment can be a digital twin, "
                "physical lab, or hybrid of both. Operational changes execute in "
                "Live Operations ONLY after validation and formal promotion from the "
                "Production-Representative environment. "
                "This separation ensures that no untested change ever reaches production. "
                "The platform uses four agent classes: Health & Posture monitors ongoing "
                "network health and compliance posture continuously. Validation agents run "
                "automated test suites that simulate real traffic and configuration scenarios. "
                "Operational agents execute approved changes with full audit logging. "
                "Stress & Adversarial agents test resilience under failure conditions."
            ),
        },
        {
            "id":      "Doc 2",
            "content": (
                "Cisco ACI (Application Centric Infrastructure) is a software-defined "
                "networking solution. ACI uses a Leaf-Spine topology. "
                "The APIC controller manages fabric policy. EPGs communicate via contracts. "
                "ACI Multi-Pod extends the fabric geographically using IPN and VXLAN. "
                "Cisco Nexus 9000 series switches are commonly deployed as ACI leaf and spine. "
                "ACI supports integration with Kubernetes via ACI CNI plugin. "
                "The APIC REST API allows programmatic policy management."
            ),
        },
    ]

    print(f"\n  Query: {query}\n")

    for chunk in chunks:
        original_toks = count_tokens_local(chunk["content"])

        # Heuristic extraction
        heuristic      = extract_relevant_sentences(chunk["content"], query)
        heuristic_toks = count_tokens_local(heuristic)

        # LLM compression (API call)
        llm_compressed = llm_chunk_compression(chunk["content"], query, target_tokens=80)
        llm_toks        = count_tokens_local(llm_compressed)

        print(f"  [{chunk['id']}]")
        print(f"  Original:    {original_toks:>5} tokens")
        print(f"  Heuristic:   {heuristic_toks:>5} tokens  ({original_toks - heuristic_toks:+} saved)")
        print(f"  LLM compressed: {llm_toks:>5} tokens  ({original_toks - llm_toks:+} saved)")
        print(f"  LLM output: {llm_compressed[:150]}...")
        print()


def conversation_compression_demo():
    """
    Show sliding window and LLM summary on a simulated conversation.
    """

    print("\n" + "=" * 65)
    print("CONVERSATION HISTORY COMPRESSION")
    print("=" * 65)

    # Simulate 8 turns of conversation (16 messages)
    history = [
        ConversationTurn("user",      "What is Cisco ACI and how does it differ from traditional networking?"),
        ConversationTurn("assistant", "Cisco ACI is a software-defined networking solution using a policy-driven model. Unlike traditional networking with per-device CLI configuration, ACI uses a centralized APIC controller that translates business intent into network policy. The Leaf-Spine topology provides predictable latency and linear scalability."),
        ConversationTurn("user",      "How does ReadyOps integrate with ACI deployments?"),
        ConversationTurn("assistant", "ReadyOps validates ACI deployments through its Validation agent class, which runs automated test suites against a Production-Representative environment (digital twin of your ACI fabric) before promoting changes to Live Operations. This ensures zero untested changes reach production."),
        ConversationTurn("user",      "Can you explain the APIC REST API structure?"),
        ConversationTurn("assistant", "The APIC REST API is organized around the ACI object model. Resources are addressed as JSON objects in a tree hierarchy: uni/ is the root, under which tenants, applications, EPGs, and contracts exist as nested objects. Authentication uses a session cookie obtained via POST /api/aaaLogin.json."),
        ConversationTurn("user",      "What about multi-site ACI setups — how does the Nexus Dashboard Orchestrator work?"),
        ConversationTurn("assistant", "Nexus Dashboard Orchestrator (NDO) provides a single management plane across multiple ACI sites. It stretches Layer 2 and Layer 3 domains across sites using VXLAN tunnels. NDO templates allow deploying consistent policy across all sites simultaneously, with site-specific overrides supported."),
        ConversationTurn("user",      "How does Hypershield complement ACI for microsegmentation?"),
        ConversationTurn("assistant", "Cisco Hypershield uses eBPF to enforce microsegmentation at the kernel level on every workload endpoint. While ACI enforces policy at the network fabric layer, Hypershield adds host-based enforcement. Together they provide Defense-in-Depth: network fabric + endpoint kernel enforcement."),
        ConversationTurn("user",      "Given everything we discussed, what validation strategy do you recommend for an ACI + Hypershield deployment?"),
        ConversationTurn("assistant", "For ACI + Hypershield, I recommend a ReadyOps validation workflow: 1) Use the digital twin to validate ACI fabric policy changes. 2) Run Stress & Adversarial agents to test eBPF policy enforcement under load. 3) Validate Hypershield SGT label propagation through ISE integration. 4) Only promote after all three validation gates pass."),
    ]

    total_toks = sum(t.token_count for t in history)
    print(f"\n  Full history: {len(history)} turns, {total_toks} tokens")

    # Sliding window
    window = sliding_window_history(history, max_history_toks=300)
    window_toks = sum(t.token_count for t in window)
    print(f"\n  After sliding window (budget=300 tokens):")
    print(f"  Kept: {len(window)} turns, {window_toks} tokens")
    print(f"  Dropped: {len(history) - len(window)} turns")

    # Hybrid
    print(f"\n  Hybrid compression (sliding window + LLM summary of dropped turns):")
    summary, recent = hybrid_history_compression(history, max_history_toks=400)

    print(f"  Summary ({count_tokens_local(summary)} tokens):")
    if summary:
        for line in summary[:300].split("\n"):
            print(f"    {line}")

    print(f"\n  Recent turns kept: {len(recent)} turns, {sum(t.token_count for t in recent)} tokens")
    recent_toks  = sum(t.token_count for t in recent)
    summary_toks = count_tokens_local(summary)
    total_after  = recent_toks + summary_toks
    print(f"  Total after hybrid: {total_after} tokens  (vs {total_toks} before — saved {total_toks - total_after})")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    token_savings_demo()
    conversation_compression_demo()
