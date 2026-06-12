"""
FILE: 02_tokenization_edge_cases.py
LESSON: Phase 1 - Lesson 7 - Tokens Deep-Dive
TOPIC: Tokenization edge cases that trip up RAG developers

WHAT THIS FILE TEACHES:
  - Whitespace rules: leading spaces, tabs, newlines
  - Capitalization: how case changes token IDs and counts
  - Numbers: why long numbers split into multiple tokens
  - Punctuation: how sentence boundaries affect tokenization
  - Code: why code costs more tokens per character than prose
  - Unicode and emoji: byte-level fallback tokenization
  - RAG-specific edge cases: chunk joins, metadata headers

WHY THESE EDGE CASES MATTER FOR RAG:
  When you split documents into chunks and join them with a separator,
  the separator characters directly affect your token count.
  A "\n\n" separator costs 1 token.
  A "\n---\n" separator costs 3+ tokens.
  At 100 chunks × 3 extra tokens = 300 unexpected tokens per query.

INSTALL:
  pip install tiktoken
"""

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
    def tok(text: str) -> list[str]:
        """Tokenize text and return list of decoded token strings."""
        ids = enc.encode(text)
        return [enc.decode([i]) for i in ids]
    def tok_count(text: str) -> int:
        return len(enc.encode(text))
except ImportError:
    HAS_TIKTOKEN = False
    def tok(text: str) -> list[str]:
        return text.split()
    def tok_count(text: str) -> int:
        return int(len(text.split()) / 0.75)


def section(title: str):
    print(f"\n  {'─'*60}")
    print(f"  {title}")
    print(f"  {'─'*60}")


# ─── 1. Whitespace Edge Cases ─────────────────────────────────────────────────

def whitespace_edge_cases():
    section("WHITESPACE EDGE CASES")

    cases = [
        ("No space",          "hello"),
        ("Leading space",     " hello"),         # DIFFERENT token than "hello"
        ("Two leading spaces"," hello"),          # 2 spaces → 2 separate tokens
        ("Tab",              "\thello"),          # Tab is a distinct token
        ("Newline",          "\nhello"),          # Newline creates a break token
        ("Double newline",   "\n\nhello"),        # Paragraph break = 2 tokens
        ("Windows newline",  "\r\nhello"),        # CR+LF = extra tokens
    ]

    print(f"  {'Description':<25} {'Token count':>12}  {'Tokens (repr)'}")
    print(f"  {'─'*25} {'─'*12}  {'─'*30}")

    for desc, text in cases:
        tokens = tok(text)
        print(f"  {desc:<25} {len(tokens):>12}  {[repr(t) for t in tokens]}")

    print(f"""
  RAG CHUNK JOIN IMPACT:
    If you join chunks with "\\n\\n", that costs 1 token per join.
    If you join with "\\n---\\n", that costs 3+ tokens per join.
    For 50 chunks: 1-token join = 50 tokens vs 3-token join = 150 tokens.
    Over 10,000 queries/day: difference of ~1M tokens/day = $3/day.
""")


# ─── 2. Capitalization ────────────────────────────────────────────────────────

def capitalization_edge_cases():
    section("CAPITALIZATION EDGE CASES")

    cases = [
        ("lowercase",  "cisco"),
        ("Titlecase",  "Cisco"),
        ("UPPERCASE",  "CISCO"),
        ("Mixed",      "CiScO"),
        ("Acronym",    "ACI"),
        ("Acronym dot","A.C.I."),
    ]

    for desc, text in cases:
        tokens = tok(text)
        print(f"  {desc:<15} → {len(tokens)} token(s): {[repr(t) for t in tokens]}")

    print(f"""
  RAG IMPLICATIONS:
    When you search for "ACI" in your docs, the model sees different tokens
    than when it sees "aci" or "Aci". This does NOT matter for meaning
    (the transformer handles semantics) but it DOES matter for:
      - Exact keyword matching in prompts ("search for 'ACI' in the doc")
      - Citation detection (checking if model cited "[ACI]" vs "[aci]")
      - Token budget estimation (UPPERCASE can cost more tokens)
""")


# ─── 3. Number Tokenization ───────────────────────────────────────────────────

def number_edge_cases():
    section("NUMBER TOKENIZATION (critical for config/IP/financial data)")

    cases = [
        ("1",               "1"),
        ("12",              "12"),
        ("123",             "123"),
        ("1234",            "1234"),
        ("12345",           "12345"),
        ("123456",          "123456"),
        ("1,234,567",       "1,234,567"),       # commas split each segment
        ("IP: 192.168.1.1", "192.168.1.1"),
        ("Subnet: /24",     "/24"),
        ("Version: v1.2.3", "v1.2.3"),
        ("Port: 8443",      "8443"),
        ("VLAN 100",        "VLAN 100"),
        ("VLAN 4094",       "VLAN 4094"),
    ]

    print(f"  {'Description':<25} {'Count':>6}  {'Tokens'}")
    print(f"  {'─'*25} {'─'*6}  {'─'*40}")

    for desc, text in cases:
        tokens = tok(text)
        print(f"  {desc:<25} {len(tokens):>6}  {[repr(t) for t in tokens]}")

    print(f"""
  WHY NUMBERS SPLIT UNPREDICTABLY:
    BPE learns from training data frequency. "100" appears millions of times → 1 token.
    "1234567" appears rarely as a unit → splits into subwords.
    IP addresses: "192" might be 1 token, ".168" might be 2 tokens (".","168").

  PRODUCTION IMPACT:
    A document with 1,000 IP addresses may cost 3,000-5,000 more tokens
    than you'd estimate at the word level.
    Monitor token counts on your actual corpus — don't rely on rules of thumb.
""")


# ─── 4. Punctuation and Special Characters ────────────────────────────────────

def punctuation_edge_cases():
    section("PUNCTUATION AND SPECIAL CHARACTERS")

    cases = [
        ("Period",          "."),
        ("Comma",           ","),
        ("Colon",           ":"),
        ("Double colon",    "::"),          # common in C++, Rust, namespaces
        ("Arrow",           "->"),
        ("Fat arrow",       "=>"),
        ("Backtick",        "`"),
        ("Triple backtick", "```"),         # code block marker
        ("Pipe",            "|"),
        ("Double pipe",     "||"),
        ("Equals",          "="),
        ("Triple equals",   "==="),
        ("Hash/pound",      "#"),
        ("Double hash",     "##"),          # markdown heading
        ("Ellipsis",        "..."),
        ("Em dash",         "—"),           # Unicode — different from ASCII --
        ("Smart quotes",    "“hello”"),  # " " curly quotes
    ]

    print(f"  {'Description':<20} {'Count':>6}  {'Tokens'}")
    print(f"  {'─'*20} {'─'*6}  {'─'*40}")

    for desc, text in cases:
        tokens = tok(text)
        print(f"  {desc:<20} {len(tokens):>6}  {[repr(t) for t in tokens]}")

    print(f"""
  RAG IMPACT:
    Markdown separators "---" and "===" cost 2-3 tokens each.
    If you use these as chunk separators in your context:
      "---" separator: 3 tokens × 50 chunks = 150 extra tokens per query.
    Use "\\n\\n" (1 token) or nothing (0 tokens) as separator for efficiency.
""")


# ─── 5. Code vs Prose ─────────────────────────────────────────────────────────

def code_vs_prose_comparison():
    section("CODE vs PROSE TOKEN EFFICIENCY")

    prose = (
        "Cisco ACI uses a Leaf-Spine topology where all endpoint groups "
        "communicate through contracts. The APIC controller manages fabric policy "
        "centrally. Multi-tenancy is achieved through VRFs and Bridge Domains."
    )

    python_code = '''\
def configure_aci_tenant(apic_client, tenant_name: str, vrf_name: str):
    """Configure a new ACI tenant with VRF and Bridge Domain."""
    tenant = apic_client.create_tenant(name=tenant_name)
    vrf    = tenant.create_vrf(name=vrf_name, enforcement="enforced")
    bd     = tenant.create_bridge_domain(name=f"{tenant_name}-bd", vrf=vrf)
    return tenant, vrf, bd
'''

    yaml_config = '''\
tenant:
  name: criterion-tenant
  vrf:
    name: criterion-vrf
    enforcement: enforced
  bridge_domain:
    name: criterion-bd
    unicast_routing: true
    arp_flooding: false
'''

    json_config = """\
{
  "tenant": "criterion-tenant",
  "vrf": {"name": "criterion-vrf", "enforcement": "enforced"},
  "bridge_domain": {"name": "criterion-bd", "unicast_routing": true}
}
"""

    for name, text in [
        ("English Prose",  prose),
        ("Python Code",    python_code),
        ("YAML Config",    yaml_config),
        ("JSON Config",    json_config),
    ]:
        chars  = len(text)
        tokens = tok_count(text)
        words  = len(text.split())

        chars_per_tok = chars / max(tokens, 1)
        words_per_tok = words / max(tokens, 1)

        print(
            f"  {name:<18}: {chars:>5} chars  {tokens:>5} tokens  "
            f"{chars_per_tok:.2f} chars/tok  {words_per_tok:.2f} words/tok"
        )

    print(f"""
  KEY FINDING:
    English prose:  ~4.0 chars/token  (best efficiency)
    Python code:    ~3.3 chars/token  (+21% more tokens than prose)
    YAML config:    ~2.8 chars/token  (+43% more tokens than prose)
    JSON config:    ~2.5 chars/token  (+60% more tokens than prose)

  IF YOUR RAG CORPUS IS CODE-HEAVY:
    Estimate 30-60% more tokens than your word count suggests.
    A "500-word" code chunk may cost 700+ tokens.
    Adjust your chunk token budget accordingly.
""")


# ─── 6. Unicode and Emoji ─────────────────────────────────────────────────────

def unicode_edge_cases():
    section("UNICODE AND EMOJI TOKENIZATION")

    cases = [
        ("Latin-1 accented",  "café"),
        ("Chinese (Mandarin)", "网络"),         # "network" in Chinese
        ("Japanese (Kanji)",   "データ"),        # "data" in Japanese
        ("Arabic",            "شبكة"),          # "network" in Arabic
        ("Korean",            "네트워크"),       # "network" in Korean
        ("Russian (Cyrillic)", "сеть"),          # "network" in Russian
        ("Emoji face",        "😀"),
        ("Emoji flag",        "🇺🇸"),            # flag = 2 regional indicator chars
        ("Emoji complex",     "👨‍💻"),            # ZWJ sequence = multiple code points
        ("Mixed",             "Hello 世界!"),     # mixing scripts
    ]

    print(f"  {'Description':<25} {'Text':<20} {'Tokens':>7}  {'Token repr (truncated)'}")
    print(f"  {'─'*25} {'─'*20} {'─'*7}  {'─'*35}")

    for desc, text in cases:
        tokens = tok(text)
        tok_repr = str([repr(t) for t in tokens[:4]])
        if len(tokens) > 4:
            tok_repr = tok_repr[:-1] + ", ...]"
        print(f"  {desc:<25} {text:<20} {len(tokens):>7}  {tok_repr}")

    print(f"""
  CRITICAL FOR MULTILINGUAL RAG:
    If your knowledge base contains Chinese, Japanese, or Arabic documents:
      English text:  ~0.25 tokens/char
      Chinese text:  ~1.0  tokens/char  (4× more expensive!)
    A Chinese document with 5,000 characters ≈ 5,000 tokens.
    The same content in English ≈ 1,250 tokens.

    IMPLICATION: If you mix languages in a RAG corpus, non-English chunks
    will fill your context window faster. Consider:
      - Separate token budgets per language
      - Translating to English before chunking (loses nuance)
      - Using a model with better multilingual tokenization (Gemini, GPT-4o)
""")


# ─── 7. RAG-Specific: Chunk Header Impact ─────────────────────────────────────

def chunk_header_cost():
    section("RAG CHUNK HEADER TOKEN COST")

    chunk_content = (
        "Cisco ACI supports Multi-Pod deployments that extend the fabric "
        "across geographically distributed data centers while maintaining "
        "a single policy domain. The Inter-Pod Network (IPN) connects pods "
        "using VXLAN encapsulation."
    )
    content_toks = tok_count(chunk_content)

    # Different header formats and their costs
    headers = [
        ("No header (baseline)",      ""),
        ("Minimal numeric",           "[Doc 3]\n"),
        ("Source + score",            "[Doc 3] Source: aci_guide.pdf | Score: 0.89\n"),
        ("Full metadata",             "[Doc 3] Source: aci_guide.pdf, p.45 | Score: 0.89 | Tier: core | Date: 2025-01-15\n"),
        ("Verbose header",            "DOCUMENT 3 OF 10\nSource File: /docs/aci_guide.pdf\nPage: 45\nRelevance Score: 0.8900\nDocument Tier: CORE\n---\n"),
    ]

    print(f"  Content tokens (fixed): {content_toks}")
    print()
    print(f"  {'Header Style':<40} {'Header Toks':>12} {'Total Toks':>11} {'Overhead%':>10}")
    print(f"  {'─'*40} {'─'*12} {'─'*11} {'─'*10}")

    for desc, header in headers:
        header_toks = tok_count(header) if header else 0
        total       = content_toks + header_toks
        overhead    = header_toks / max(content_toks, 1) * 100
        print(
            f"  {desc:<40} {header_toks:>12} {total:>11} {overhead:>9.1f}%"
        )

    print(f"""
  RECOMMENDATION:
    Use minimal headers: "[Doc N] Source: filename | Score: X.XX\\n"
    This adds ~8-12 tokens per chunk.
    At 20 chunks per query: 160-240 extra tokens — acceptable overhead.

    AVOID verbose headers with redundant info.
    The model doesn't need "DOCUMENT 3 OF 10" — it can count.
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("TOKENIZATION EDGE CASES: What trips up RAG developers")
    print("=" * 65)

    if not HAS_TIKTOKEN:
        print("\n  ⚠ tiktoken not installed. Install with: pip install tiktoken")
        print("    Running with word-count approximation.\n")

    whitespace_edge_cases()
    capitalization_edge_cases()
    number_edge_cases()
    punctuation_edge_cases()
    code_vs_prose_comparison()
    unicode_edge_cases()
    chunk_header_cost()
