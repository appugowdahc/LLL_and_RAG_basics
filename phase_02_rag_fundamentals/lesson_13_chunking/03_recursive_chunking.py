"""
FILE: 03_recursive_chunking.py
LESSON: Phase 2 - Lesson 13 - Document Processing and Chunking
TOPIC: Recursive chunking — structure-preserving fallback splitting

WHAT THIS FILE TEACHES:
  - The recursive splitter algorithm: try preferred separator, fall back to smaller
  - Separator priority hierarchy: \n\n > \n > . > (space) > character
  - Content-type-specific separator lists (Markdown, code, YAML, plain text)
  - WHY recursive chunking outperforms fixed-size for mixed-structure documents
  - Handling code blocks, headers, and tables within a document
  - The LangChain RecursiveCharacterTextSplitter — what it does under the hood

INSTALL: no external dependencies
"""

import re
from dataclasses import dataclass
from typing import Optional


# ─── Token Approximation ─────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ─── Separator Configurations by Content Type ─────────────────────────────────

# WHY content-type-specific separators:
#   A Markdown document has headings (##), code fences (```), and tables.
#   A Python file has class/def boundaries.
#   A YAML file has key-value block structure.
#   One-size-fits-all separators miss these structural signals.

SEPARATORS = {
    "markdown": [
        "\n## ",     # WHY h2 first: major section boundary
        "\n### ",    # sub-section
        "\n#### ",   # sub-sub-section
        "\n```",     # code block boundary
        "\n\n",      # paragraph
        "\n",        # line
        ". ",        # sentence
        " ",         # word
    ],
    "python": [
        "\nclass ",  # WHY class: top-level semantic unit
        "\ndef ",    # function
        "\n\n",      # blank line (between logical blocks)
        "\n",        # line
        " ",         # word
    ],
    "yaml": [
        "\n---",     # YAML document boundary
        "\n\n",      # blank line between blocks
        "\n  ",      # indented sub-key
        "\n",        # line
        ": ",        # key-value separator (last resort)
    ],
    "plain": [
        "\n\n",      # paragraph
        "\n",        # line
        ". ",        # sentence
        " ",         # word
    ],
}


# ─── Recursive Chunker ────────────────────────────────────────────────────────

def recursive_chunk(
    text:          str,
    separators:    list[str],
    chunk_tokens:  int = 300,
    overlap_tokens: int = 30,
    source:        str = "doc",
    _depth:        int = 0,      # WHY _depth: prevent infinite recursion; for debugging
) -> list[dict]:
    """
    Recursively split text using a priority-ordered list of separators.

    Algorithm:
      1. Try splitting on separators[0] (highest priority).
      2. If all resulting parts fit within chunk_tokens, use them directly.
      3. If a part is too large, recurse on it using separators[1:].
      4. If we've exhausted all separators, fall back to hard character split.
      5. Merge small adjacent parts to stay near (not below) the token target.

    WHY this is better than a single-separator split:
      A paragraph split (\n\n) produces some very large chunks for long paragraphs.
      Falling back to sentence split (.) within those large chunks preserves as much
      structure as possible while still meeting the token budget.

    Args:
        text:          Text to chunk.
        separators:    Ordered list of separators to try (highest priority first).
        chunk_tokens:  Target maximum tokens per chunk.
        overlap_tokens: Tokens of overlap to inject between chunks.
        source:        Source document ID.
        _depth:        Recursion depth (internal use).

    Returns:
        List of chunk dicts.
    """

    # Base case: text fits in one chunk
    if approx_tokens(text) <= chunk_tokens:
        return [{
            "chunk_id":    f"{source}:r{_depth}:0",
            "content":     text.strip(),
            "token_count": approx_tokens(text),
            "separator":   "(whole)",
            "source":      source,
        }]

    # No more separators — hard character split as last resort
    if not separators:
        return _hard_split(text, chunk_tokens, overlap_tokens, source, _depth)

    sep        = separators[0]
    rest_seps  = separators[1:]

    # WHY re.split with keep: preserve the separator in the following piece
    # so structure (e.g., "## heading") stays with its content
    parts = re.split(f"(?={re.escape(sep)})", text) if sep.startswith("\n") else text.split(sep)
    parts = [p for p in parts if p.strip()]

    if len(parts) <= 1:
        # Separator not found in text — try next separator
        return recursive_chunk(text, rest_seps, chunk_tokens, overlap_tokens, source, _depth)

    chunks = []
    idx    = 0

    for part in parts:
        if approx_tokens(part) <= chunk_tokens:
            chunks.append({
                "chunk_id":    f"{source}:r{_depth}:{idx}",
                "content":     part.strip(),
                "token_count": approx_tokens(part),
                "separator":   repr(sep),
                "source":      source,
            })
            idx += 1
        else:
            # Part too large — recurse with next separators
            sub = recursive_chunk(part, rest_seps, chunk_tokens, overlap_tokens, source, _depth + 1)
            chunks.extend(sub)
            idx += len(sub)

    # Merge small adjacent chunks to avoid many tiny fragments
    return _merge_small_chunks(chunks, chunk_tokens, overlap_tokens, source)


def _hard_split(text: str, chunk_tokens: int, overlap_tokens: int, source: str, depth: int) -> list[dict]:
    """
    Last-resort: fixed character split.
    WHY separate function: makes it obvious in code when we fall back to hard split.
    """
    char_size    = chunk_tokens * 4
    char_overlap = overlap_tokens * 4
    chunks       = []
    start        = 0
    idx          = 0

    while start < len(text):
        end     = min(start + char_size, len(text))
        content = text[start:end].strip()
        if content:
            chunks.append({
                "chunk_id":    f"{source}:hard:{depth}:{idx}",
                "content":     content,
                "token_count": approx_tokens(content),
                "separator":   "(hard split)",
                "source":      source,
            })
            idx += 1
        start += char_size - char_overlap

    return chunks


def _merge_small_chunks(
    chunks:        list[dict],
    chunk_tokens:  int,
    overlap_tokens: int,
    source:        str,
) -> list[dict]:
    """
    Merge adjacent small chunks until they reach the target size.
    WHY merge: recursive splitting can produce many tiny fragments
    (e.g., one-line YAML keys) that are too small to embed meaningfully.
    """
    if not chunks:
        return []

    merged = [chunks[0].copy()]

    for chunk in chunks[1:]:
        last      = merged[-1]
        combined  = last["content"] + "\n" + chunk["content"]
        combined_toks = approx_tokens(combined)

        # Merge if the result still fits within budget
        if combined_toks <= chunk_tokens:
            last["content"]     = combined
            last["token_count"] = combined_toks
            last["separator"]   = last["separator"] + "+" + chunk.get("separator", "?")
        else:
            merged.append(chunk.copy())

    return merged


# ─── Content-Type Detector ────────────────────────────────────────────────────

def detect_content_type(text: str) -> str:
    """
    Heuristic content-type detection.
    WHY heuristic: you can't always rely on file extension in a RAG pipeline.
    Documents may be extracted from PDFs, databases, or web pages.
    """
    # WHY count signals:
    #   One indicator could be coincidental; multiple indicators give confidence.
    signals: dict[str, int] = {"markdown": 0, "python": 0, "yaml": 0, "plain": 0}

    if re.search(r"^#{1,6}\s", text, re.MULTILINE):     signals["markdown"] += 3
    if "```" in text:                                     signals["markdown"] += 2
    if re.search(r"^\s*[-*]\s", text, re.MULTILINE):     signals["markdown"] += 1

    if re.search(r"^(def |class )", text, re.MULTILINE): signals["python"] += 3
    if re.search(r"^\s{4}\w", text, re.MULTILINE):       signals["python"] += 1
    if "#" in text and "def " in text:                    signals["python"] += 1

    if re.search(r"^\w[\w_-]+:\s*$", text, re.MULTILINE): signals["yaml"] += 2
    if re.search(r"^  \w[\w_-]+:", text, re.MULTILINE):   signals["yaml"] += 2
    if text.strip().startswith("---"):                     signals["yaml"] += 2

    return max(signals, key=lambda k: signals[k])


# ─── Demo ─────────────────────────────────────────────────────────────────────

MARKDOWN_DOC = """
## ACI Fabric Architecture

Cisco ACI uses a Leaf-Spine topology. Every leaf switch connects to every spine switch,
providing a non-blocking, loop-free fabric. The overlay uses VXLAN for tenant isolation.

### APIC Controller

The APIC cluster manages all fabric policy. Minimum 3 nodes for HA.

```python
import requests

def get_apic_token(apic_url, username, password):
    payload = {"aaaUser": {"attributes": {"name": username, "pwd": password}}}
    resp = requests.post(f"{apic_url}/api/aaaLogin.json", json=payload, verify=False)
    return resp.json()["imdata"][0]["aaaLogin"]["attributes"]["token"]
```

### EPGs and Contracts

EPGs define groups of endpoints with shared policy.
Contracts permit traffic between EPGs. Without a contract, traffic is denied.

## ReadyOps Platform

ReadyOps validates ACI changes in a Production-Representative environment.
All validation tests must pass before promotion to Live Operations.

### Agent Classes

- Health and Posture agents monitor baseline drift.
- Validation agents run pre-change tests.
- Operational agents execute runbooks.
- Stress and Adversarial agents test resilience.
""".strip()

PYTHON_DOC = """
class APICClient:
    def __init__(self, url, username, password):
        self.url = url
        self.token = self._login(username, password)

    def _login(self, username, password):
        payload = {"aaaUser": {"attributes": {"name": username, "pwd": password}}}
        resp = requests.post(f"{self.url}/api/aaaLogin.json", json=payload, verify=False)
        return resp.json()["imdata"][0]["aaaLogin"]["attributes"]["token"]

    def get_epgs(self, tenant):
        headers = {"Cookie": f"APIC-cookie={self.token}"}
        resp = requests.get(f"{self.url}/api/node/mo/uni/tn-{tenant}.json?query-target=subtree&target-subtree-class=fvAEPg", headers=headers)
        return resp.json()["imdata"]

    def create_contract(self, tenant, contract_name, subject_name):
        dn = f"uni/tn-{tenant}/brc-{contract_name}"
        payload = {"vzBrCP": {"attributes": {"dn": dn, "name": contract_name}}}
        headers = {"Cookie": f"APIC-cookie={self.token}"}
        return requests.post(f"{self.url}/api/node/mo/{dn}.json", json=payload, headers=headers)
""".strip()


def run_recursive_demo():
    """Show recursive chunking on Markdown and Python documents."""

    print("=" * 70)
    print("RECURSIVE CHUNKING: Structure-Preserving Splitting")
    print("=" * 70)

    for doc_name, doc_text in [("Markdown", MARKDOWN_DOC), ("Python", PYTHON_DOC)]:
        detected = detect_content_type(doc_text)
        seps     = SEPARATORS[detected]
        chunks   = recursive_chunk(doc_text, seps, chunk_tokens=120, overlap_tokens=15, source=doc_name.lower())

        print(f"\n  [{doc_name}] detected as: '{detected}'")
        print(f"  Total chunks: {len(chunks)} | approx tokens: {approx_tokens(doc_text)}")

        for i, c in enumerate(chunks, 1):
            print(f"\n    Chunk {i} [{c['token_count']} tok] [split on: {c.get('separator','?')}]")
            print(f"      '{c['content'][:100].strip()}'")


def compare_fixed_vs_recursive():
    """
    Show the quality difference between fixed-size and recursive splitting
    on a mixed-structure Markdown document.
    """

    print("\n" + "=" * 70)
    print("FIXED-SIZE vs RECURSIVE: Boundary Quality on Markdown")
    print("=" * 70)

    doc = MARKDOWN_DOC
    target = 150   # tokens

    # Fixed-size character split
    char_size = target * 4
    fixed_chunks = []
    start = 0
    while start < len(doc):
        end = min(start + char_size, len(doc))
        fixed_chunks.append(doc[start:end].strip())
        start += char_size - 30

    # Recursive split
    seps      = SEPARATORS["markdown"]
    rec_chunks = recursive_chunk(doc, seps, chunk_tokens=target, source="demo")

    print(f"\n  Document: {approx_tokens(doc)} approx tokens")
    print(f"\n  Fixed-size ({len(fixed_chunks)} chunks):")
    for i, c in enumerate(fixed_chunks[:3], 1):
        ends_ok = "✓" if re.search(r"[.!?\n]$", c.strip()) else "✗ mid-sentence"
        print(f"    [{i}] {ends_ok}  '{c[:80].strip()}'")

    print(f"\n  Recursive ({len(rec_chunks)} chunks):")
    for i, c in enumerate(rec_chunks[:3], 1):
        ends_ok = "✓" if re.search(r"[.!?\n]$", c["content"].strip()) else "✗ mid-sentence"
        print(f"    [{i}] {ends_ok} [{c['token_count']} tok]  '{c['content'][:80].strip()}'")

    print(f"""
  KEY INSIGHT:
    Fixed-size cuts the Markdown document at arbitrary character positions,
    splitting headings from their content, breaking code blocks mid-function.
    Recursive splitting uses ## headings as natural boundaries first, then
    falls back to paragraphs, then sentences — always preserving structure.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_recursive_demo()
    compare_fixed_vs_recursive()
