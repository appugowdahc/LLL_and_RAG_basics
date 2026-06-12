"""
FILE: 05_rag_chunk_tokenization.py
LESSON: Phase 1 - Lesson 7 - Tokens Deep-Dive
TOPIC: How tokenization shapes RAG chunk sizing and boundary detection

WHAT THIS FILE TEACHES:
  - How to measure actual token counts for different content types
  - Chunk size strategies: character-based vs token-based splitting
  - WHY character-based splitting leads to inconsistent token counts
  - Detecting natural semantic boundaries (paragraphs, headings, lists)
  - The "chunk size sweet spot" for RAG quality
  - How chunk overlap affects both token cost and retrieval quality

CORE INSIGHT:
  Most RAG tutorials split text by "500 characters" or "100 words."
  This is WRONG for token-budget management because:
    - 500 characters of English prose ≈ 125 tokens
    - 500 characters of Python code  ≈ 150 tokens
    - 500 characters of JSON config  ≈ 200 tokens
  The SAME character limit produces VERY different token counts by content type.

  For reliable context window management: ALWAYS split by token count, not characters.

INSTALL:
  pip install tiktoken
"""

import re
from dataclasses import dataclass, field
from typing import Optional

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def tok_count(text: str) -> int:
        return len(_enc.encode(text))
    def tok_split(text: str, max_tokens: int) -> list[str]:
        """Split text into chunks of at most max_tokens (exact, token-level)."""
        tokens = _enc.encode(text)
        return [_enc.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]
    HAS_TIKTOKEN = True
except ImportError:
    def tok_count(text: str) -> int:
        return int(len(text.split()) / 0.75)
    def tok_split(text: str, max_tokens: int) -> list[str]:
        words = text.split()
        chunk_words = int(max_tokens * 0.75)
        return [" ".join(words[i:i+chunk_words]) for i in range(0, len(words), chunk_words)]
    HAS_TIKTOKEN = False


# ─── Content Type Token Profiles ─────────────────────────────────────────────

SAMPLE_CONTENT = {
    "English Prose": """\
Cisco ACI (Application Centric Infrastructure) is a software-defined networking
solution that uses a policy-driven model to automate network provisioning and
management. Unlike traditional networking with per-device configuration, ACI uses
a centralized controller called APIC (Application Policy Infrastructure Controller)
that translates business intent into network policy. The Leaf-Spine topology
provides predictable latency and linear scalability. Each leaf switch connects to
every spine switch, ensuring no more than two hops between any two endpoints.
Endpoint groups (EPGs) communicate through contracts that define which groups
can exchange traffic and under what conditions. This model separates the WHAT
(policy intent) from the HOW (physical configuration), allowing the network to
adapt automatically when workloads move or scale.
""",
    "Python Code": """\
import anthropic
from typing import Optional

def retrieve_and_generate(
    query: str,
    collection_name: str,
    top_k: int = 5,
    min_score: float = 0.70,
    model: str = "claude-sonnet-4-6",
) -> dict:
    \"\"\"Retrieve relevant docs and generate a grounded answer.\"\"\"
    client = anthropic.Anthropic()

    # Retrieve chunks from vector DB
    chunks = vector_db.query(
        collection=collection_name,
        query_text=query,
        top_k=top_k,
        filters={"score": {"gte": min_score}},
    )

    if not chunks:
        return {"answer": "NOT IN CONTEXT", "citations": []}

    # Build RAG prompt
    context = "\\n\\n".join(
        f"[{i+1}] {c['source']}\\n{c['content']}"
        for i, c in enumerate(chunks)
    )

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        system="Answer using ONLY the provided context. Cite [N].",
        messages=[{"role": "user", "content": f"{context}\\n\\nQUESTION: {query}"}],
    )
    return {"answer": response.content[0].text, "citations": chunks}
""",
    "YAML Config": """\
# Cisco ACI Tenant Configuration
tenant:
  name: criterion-tenant
  description: Criterion Networks Production Tenant

  vrf:
    name: criterion-prod-vrf
    enforcement: enforced
    preferred_group: disabled

  bridge_domains:
    - name: criterion-bd-web
      vrf: criterion-prod-vrf
      unicast_routing: true
      arp_flooding: false
      subnets:
        - ip: 10.100.10.1/24
          scope: [public, shared]

    - name: criterion-bd-app
      vrf: criterion-prod-vrf
      unicast_routing: true
      l3out: criterion-l3out
      subnets:
        - ip: 10.100.20.1/24
          scope: [private]

  endpoint_groups:
    - name: web-epg
      bridge_domain: criterion-bd-web
      vmm_domain: VMware-DVS

    - name: app-epg
      bridge_domain: criterion-bd-app
      static_paths:
        - path: topology/pod-1/paths-101/pathep-[eth1/1]
          vlan: 100
          mode: regular
""",
    "JSON API Response": """\
{
  "imdata": [
    {
      "fvTenant": {
        "attributes": {
          "dn": "uni/tn-criterion-tenant",
          "name": "criterion-tenant",
          "descr": "Criterion Networks Production Tenant",
          "status": "created,modified",
          "modTs": "2025-01-15T10:30:00.000+00:00",
          "uidRange": "0",
          "ownerKey": "",
          "ownerTag": ""
        },
        "children": [
          {
            "fvCtx": {
              "attributes": {
                "name": "criterion-prod-vrf",
                "pcEnfPref": "enforced",
                "pcEnfDir": "ingress"
              }
            }
          }
        ]
      }
    }
  ],
  "totalCount": "1"
}
""",
}


def content_type_token_profiles():
    """
    Measure token density per character and per word for each content type.
    This informs how to set chunk size targets per content type.
    """

    print("=" * 65)
    print("CONTENT TYPE TOKEN PROFILES (chars → tokens conversion)")
    print("=" * 65)
    print(f"\n  {'Type':<20} {'Chars':>7} {'Words':>7} {'Tokens':>8} "
          f"{'Chars/Tok':>10} {'Words/Tok':>10}")
    print(f"  {'─'*20} {'─'*7} {'─'*7} {'─'*8} {'─'*10} {'─'*10}")

    for name, content in SAMPLE_CONTENT.items():
        chars  = len(content)
        words  = len(content.split())
        tokens = tok_count(content)

        print(
            f"  {name:<20} {chars:>7,} {words:>7,} {tokens:>8,} "
            f"{chars/tokens:>10.2f} {words/tokens:>10.2f}"
        )

    print(f"""
  RULE OF THUMB (cl100k_base tokenizer ≈ Claude):
    English prose: 4.0 chars/token → target 500 tokens = 2,000 chars
    Python code:   3.3 chars/token → target 500 tokens = 1,650 chars
    YAML config:   2.8 chars/token → target 500 tokens = 1,400 chars
    JSON:          2.5 chars/token → target 500 tokens = 1,250 chars

  PRODUCTION PRACTICE:
    Always use TOKEN-BASED splitting, not CHARACTER-BASED.
    Character limit → inconsistent token counts → unpredictable context budget.
    Token limit → exact budget control → predictable costs.
""")


# ─── Character vs Token Splitting Comparison ─────────────────────────────────

def character_vs_token_splitting():
    """
    Compare character-based vs token-based chunk splitting.
    Shows WHY character limits produce wildly different token counts.
    """

    print("\n" + "=" * 65)
    print("CHARACTER vs TOKEN SPLITTING: Variability Comparison")
    print("=" * 65)

    CHAR_LIMIT  = 800
    TOKEN_LIMIT = 200

    print(f"\n  CHARACTER LIMIT = {CHAR_LIMIT} chars | TOKEN LIMIT = {TOKEN_LIMIT} tokens\n")

    for name, content in SAMPLE_CONTENT.items():
        # Character-based split
        char_chunks  = [content[i:i+CHAR_LIMIT] for i in range(0, len(content), CHAR_LIMIT)]
        char_toks    = [tok_count(c) for c in char_chunks]

        # Token-based split
        tok_chunks   = tok_split(content, TOKEN_LIMIT)
        tok_chars    = [len(c) for c in tok_chunks]

        print(f"  [{name}]")
        print(f"    Char-based ({CHAR_LIMIT} chars): {len(char_chunks)} chunks, "
              f"tokens: min={min(char_toks)} max={max(char_toks)} "
              f"spread={max(char_toks)-min(char_toks)} tokens")
        print(f"    Token-based ({TOKEN_LIMIT} toks): {len(tok_chunks)} chunks, "
              f"chars: min={min(tok_chars)} max={max(tok_chars)} "
              f"(consistent token count)")
        print()


# ─── Semantic Boundary Detection ─────────────────────────────────────────────

@dataclass
class SemanticChunk:
    content:     str
    chunk_type:  str   # "paragraph" | "code_block" | "list" | "heading" | "yaml_block"
    token_count: int   = 0
    char_count:  int   = 0

    def __post_init__(self):
        self.token_count = tok_count(self.content)
        self.char_count  = len(self.content)


def detect_semantic_boundaries(text: str) -> list[SemanticChunk]:
    """
    Split text at natural semantic boundaries: paragraphs, code blocks, lists.
    Produces linguistically coherent chunks instead of arbitrary mid-sentence cuts.

    WHY semantic splitting matters for RAG quality:
      A chunk that ends mid-sentence confuses the retrieval model — the embedding
      for an incomplete thought is less meaningful than for a complete paragraph.
      Semantic chunks have better vector representations → better retrieval.

    Strategy:
      1. Detect code blocks (```...```) → keep together
      2. Detect YAML/JSON blocks → keep together
      3. Detect markdown headings → split before each
      4. Detect blank-line paragraph breaks
      5. Detect list blocks (lines starting with -, *, numbers)
    """

    chunks: list[SemanticChunk] = []

    # WHY process code blocks first:
    #   Code blocks can contain text that looks like markdown headings or lists.
    #   Splitting inside a code block would break its semantic integrity.
    code_block_pattern = re.compile(r"```[\s\S]*?```", re.DOTALL)
    yaml_block_pattern = re.compile(r"(?:^|\n)((?:\w[^\n]*:\s*\n(?:[ \t]+[^\n]*\n)*)+)", re.MULTILINE)
    heading_pattern    = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)

    # Split by double newlines (paragraph breaks) first
    raw_paragraphs     = re.split(r"\n\n+", text.strip())

    for para in raw_paragraphs:
        if not para.strip():
            continue

        # Detect chunk type
        if re.match(r"```", para.strip()):
            chunk_type = "code_block"
        elif re.match(r"^#{1,6}\s", para.strip()):
            chunk_type = "heading"
        elif re.match(r"^[\-\*\+]\s|^\d+\.\s", para.strip(), re.MULTILINE):
            chunk_type = "list"
        elif re.match(r"^\w+:\s*$|^\w+:\s+\S", para.strip()):
            chunk_type = "yaml_block"
        else:
            chunk_type = "paragraph"

        chunks.append(SemanticChunk(content=para.strip(), chunk_type=chunk_type))

    return chunks


def demonstrate_semantic_chunking():
    """
    Apply semantic boundary detection to sample content.
    """

    print("\n" + "=" * 65)
    print("SEMANTIC BOUNDARY DETECTION")
    print("=" * 65)

    sample_doc = """\
# Cisco ACI Overview

Cisco ACI is a software-defined networking solution for data centers.
It uses a policy-driven approach to automate network provisioning.

## Core Components

The APIC controller is the central management plane. It translates
business intent into network policy and programs the physical fabric.

Key components include:
- APIC cluster (3+ nodes for HA)
- Leaf switches (access layer, connected to endpoints)
- Spine switches (core layer, connects leaf switches)
- VXLAN fabric (overlay encapsulation)

```python
# Example: Create ACI tenant via Python SDK
from acitoolkit import Session, Tenant

session = Session(url, login, password)
session.login()

tenant = Tenant("criterion-tenant")
tenant.push_to_apic(session)
```

## ReadyOps Integration

ReadyOps validates ACI deployments using a digital twin. The Validation
agent class runs automated test suites against the twin before any
change is promoted to the Live Operations environment.
"""

    chunks = detect_semantic_boundaries(sample_doc)

    print(f"\n  Document: {len(sample_doc)} chars, {tok_count(sample_doc)} tokens")
    print(f"  Semantic chunks detected: {len(chunks)}\n")

    print(f"  {'#':<3} {'Type':<15} {'Tokens':>7} {'Preview'}")
    print(f"  {'─'*3} {'─'*15} {'─'*7} {'─'*40}")

    for i, chunk in enumerate(chunks):
        preview = chunk.content[:50].replace("\n", "↵")
        print(f"  {i+1:<3} {chunk.chunk_type:<15} {chunk.token_count:>7}  {preview}...")


# ─── Chunk Size Sweet Spot ─────────────────────────────────────────────────────

def chunk_size_sweet_spot():
    """
    Explain the tradeoffs in chunk size for RAG quality.

    Too small → less context per chunk → model may lack the information it needs
    Too large → fewer chunks fit → retrieval finds less specific matches
    """

    print("\n" + "=" * 65)
    print("CHUNK SIZE SWEET SPOT (tokens)")
    print("=" * 65)

    sizes = [
        (64,    "Ultra-micro",   "Single sentences. High precision retrieval, poor context."),
        (128,   "Micro",         "2-3 sentences. Good for FAQ, short facts."),
        (256,   "Small",         "One paragraph. RAG sweet spot for factual QA."),
        (512,   "Medium",        "2-3 paragraphs. Best for technical explanations."),
        (1024,  "Large",         "Full section. Good for summarization tasks."),
        (2048,  "Extra-large",   "Multiple sections. Used for document-level tasks."),
    ]

    print(f"\n  {'Size':<12} {'Label':<14} {'Best for'}")
    print(f"  {'─'*12} {'─'*14} {'─'*45}")

    for toks, label, use_case in sizes:
        # WHY mark the sweet spot:
        #   Most RAG benchmarks show 256-512 tokens optimal for factual QA.
        #   Smaller = more precise but risks cutting explanatory context.
        #   Larger = more context per chunk but retrieval becomes less specific.
        marker = " ← SWEET SPOT" if 256 <= toks <= 512 else ""
        print(f"  {toks:<12,} {label:<14} {use_case}{marker}")

    print(f"""
  OVERLAP STRATEGY:
    Chunks often include a small overlap with adjacent chunks.
    This prevents answers from being split across two chunk boundaries.

    chunk_size=512, overlap=64:
      Chunk 1: tokens 0-511
      Chunk 2: tokens 448-959   (64 token overlap)
      Chunk 3: tokens 896-1407  (64 token overlap)

    Cost of overlap: 64/512 = 12.5% more total tokens stored in the vector DB
    and 12.5% more tokens retrieved per query → worth it for boundary coverage.

    For a 60,000 token doc budget:
      No overlap: 60K / 512 = 117 chunks per query
      12.5% overhead: effectively 60K / 576 = 104 chunks — still plenty
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if not HAS_TIKTOKEN:
        print("⚠ tiktoken not installed — counts are approximations.")
        print("  pip install tiktoken\n")

    content_type_token_profiles()
    character_vs_token_splitting()
    demonstrate_semantic_chunking()
    chunk_size_sweet_spot()
