"""
FILE: 04_multilingual_tokens.py
LESSON: Phase 1 - Lesson 7 - Tokens Deep-Dive
TOPIC: Multilingual tokenization — how different languages cost different tokens

WHAT THIS FILE TEACHES:
  - Why non-English text costs more tokens per unit of meaning
  - Token efficiency by language (tokens per word equivalent)
  - How this affects context window capacity for multilingual RAG
  - Design patterns for multilingual RAG at Criterion Networks scale
  - When to translate vs retrieve-in-original-language

INSTALL:
  pip install tiktoken
"""

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def tok_count(text: str) -> int:
        return len(enc.encode(text))
    HAS_TIKTOKEN = True
except ImportError:
    def tok_count(text: str) -> int:
        return int(len(text.split()) / 0.75)
    HAS_TIKTOKEN = False


# ─── Parallel Sentences for Cross-Language Comparison ────────────────────────

# WHY use the same sentence in all languages:
#   We want to measure token cost for the SAME information content.
#   Different sentences would confound language with content length.
#   "network" is a good technical RAG-relevant term to compare.

PARALLEL_SENTENCES = {
    "English":    "The network infrastructure requires continuous validation before deployment.",
    "Spanish":    "La infraestructura de red requiere validación continua antes del despliegue.",
    "French":     "L'infrastructure réseau nécessite une validation continue avant le déploiement.",
    "German":     "Die Netzwerkinfrastruktur erfordert eine kontinuierliche Validierung vor der Bereitstellung.",
    "Portuguese": "A infraestrutura de rede requer validação contínua antes da implantação.",
    "Italian":    "L'infrastruttura di rete richiede una validazione continua prima della distribuzione.",
    "Russian":    "Сетевая инфраструктура требует непрерывной проверки перед развертыванием.",
    "Arabic":     "تتطلب البنية التحتية للشبكة التحقق المستمر قبل النشر.",
    "Chinese":    "网络基础设施在部署前需要持续验证。",
    "Japanese":   "ネットワークインフラは展開前に継続的な検証が必要です。",
    "Korean":     "네트워크 인프라는 배포 전에 지속적인 검증이 필요합니다.",
    "Hindi":      "नेटवर्क अवसंरचना को तैनाती से पहले निरंतर सत्यापन की आवश्यकता है।",
}


def analyze_multilingual_efficiency():
    """
    Compare token counts for the same meaning across languages.
    Shows the 'token tax' for each non-English language.
    """

    print("=" * 70)
    print("MULTILINGUAL TOKEN EFFICIENCY")
    print("=" * 70)

    # English as baseline
    en_tokens = tok_count(PARALLEL_SENTENCES["English"])
    en_chars  = len(PARALLEL_SENTENCES["English"])

    print(f"\n  {'Language':<15} {'Tokens':>7} {'Chars':>7} "
          f"{'vs English':>12} {'Chars/Tok':>10} {'Script'}")
    print(f"  {'─'*15} {'─'*7} {'─'*7} {'─'*12} {'─'*10} {'─'*15}")

    script_map = {
        "English": "Latin", "Spanish": "Latin", "French": "Latin",
        "German": "Latin", "Portuguese": "Latin", "Italian": "Latin",
        "Russian": "Cyrillic", "Arabic": "Arabic", "Chinese": "CJK",
        "Japanese": "CJK/Kana", "Korean": "Hangul", "Hindi": "Devanagari",
    }

    for lang, text in PARALLEL_SENTENCES.items():
        toks        = tok_count(text)
        chars       = len(text)
        ratio       = toks / en_tokens   # WHY ratio: easy to see the multiplier vs English
        chars_per_t = chars / max(toks, 1)
        overhead    = f"{ratio:.2f}×"

        script      = script_map.get(lang, "")

        print(
            f"  {lang:<15} {toks:>7} {chars:>7} "
            f"{overhead:>12} {chars_per_t:>10.2f} {script}"
        )

    print(f"""
  INTERPRETATION:
    1.0× = same token count as English
    2.0× = twice as many tokens for same content (50% context window capacity)

    Latin-script languages (Spanish, French, etc.): ~1.0-1.2×  (BPE handles well)
    Cyrillic (Russian): ~1.5-2.0×  (BPE merges some, but Cyrillic less frequent)
    Arabic: ~2.0-3.0×  (right-to-left, complex morphology, less training data)
    CJK (Chinese, Japanese): ~2.0-4.0×  (each character often its own token)
""")


# ─── Context Window Capacity by Language ─────────────────────────────────────

def context_capacity_by_language():
    """
    Show how many 'sentences of information' fit in the context window
    for each language.
    """

    print("\n" + "=" * 70)
    print("CONTEXT WINDOW CAPACITY BY LANGUAGE")
    print("(Claude Sonnet: 200K context window, 60K doc budget)")
    print("=" * 70)

    DOC_BUDGET = 60_000   # tokens reserved for retrieved documents
    en_tokens  = tok_count(PARALLEL_SENTENCES["English"])

    print(f"\n  {'Language':<15} {'Toks/sentence':>14} {'Max sentences':>14} "
          f"{'vs English':>12}")
    print(f"  {'─'*15} {'─'*14} {'─'*14} {'─'*12}")

    en_sentences = DOC_BUDGET // en_tokens

    for lang, text in PARALLEL_SENTENCES.items():
        toks          = tok_count(text)
        max_sentences = DOC_BUDGET // toks
        vs_english    = max_sentences / en_sentences

        print(
            f"  {lang:<15} {toks:>14} {max_sentences:>14,} "
            f"{vs_english:>11.1%}"
        )

    print(f"""
  IMPLICATION:
    A multilingual RAG system retrieving Chinese documents can fit
    only ~40-50% as many sentences as the same system with English docs.

    Options:
      1. Increase doc token budget for non-English corpora
      2. Retrieve fewer, more precise chunks (raise similarity threshold)
      3. Compress non-English chunks before injecting (translate excerpts)
      4. Use a model with stronger multilingual tokenization
""")


# ─── Multilingual RAG Design Patterns ────────────────────────────────────────

def multilingual_rag_patterns():
    """
    Explain the three main patterns for multilingual RAG and their tradeoffs.
    """

    print("\n" + "=" * 70)
    print("MULTILINGUAL RAG DESIGN PATTERNS")
    print("=" * 70)

    patterns = [
        {
            "name":     "Pattern 1: Retrieve-in-Original, Generate-in-English",
            "flow":     "Query (EN) → Embed → Retrieve (any lang) → Inject original → Generate (EN)",
            "pros":     [
                "Original meaning preserved (no translation errors in source)",
                "Claude understands and can synthesize across languages",
                "Simple pipeline — no translation step",
            ],
            "cons":     [
                "Non-English chunks cost 2-4× more tokens → smaller context budget",
                "Model may produce 'translation artifacts' in the answer",
                "Harder to verify citations (source vs answer in different languages)",
            ],
            "when":     "Knowledge base in multiple languages, answers always in English",
        },
        {
            "name":     "Pattern 2: Translate-Before-Index",
            "flow":     "Ingest → Translate → Embed translated text → Retrieve (EN) → Generate (EN)",
            "pros":     [
                "All chunks in one language → consistent token counts",
                "Full 60K doc budget available (no non-English overhead)",
                "Citation verification easy (source and answer in English)",
            ],
            "cons":     [
                "Translation pipeline required at ingest time (compute + cost)",
                "Translation loses nuance in technical/legal/regulatory text",
                "Must retranslate if source documents update",
            ],
            "when":     "Consistent query/answer language required; accuracy > cost",
        },
        {
            "name":     "Pattern 3: Language-Isolated Index Shards",
            "flow":     "Detect query language → Route to language-specific index → Generate in same language",
            "pros":     [
                "Each language index optimized separately",
                "No translation needed at any stage",
                "Best semantic search quality within each language",
            ],
            "cons":     [
                "Cannot cross-query across languages (Chinese docs for English query)",
                "Multiple indexes = higher infra cost",
                "Language detection adds latency",
            ],
            "when":     "Users query in their own language and expect answers in same language",
        },
    ]

    for pat in patterns:
        print(f"\n  [{pat['name']}]")
        print(f"  Flow: {pat['flow']}")
        print(f"  When to use: {pat['when']}")
        print(f"  Pros:")
        for pro in pat["pros"]:
            print(f"    + {pro}")
        print(f"  Cons:")
        for con in pat["cons"]:
            print(f"    - {con}")


# ─── Token Cost Impact for Criterion Networks Context ─────────────────────────

def enterprise_multilingual_cost_impact():
    """
    Practical cost analysis for a hypothetical enterprise RAG corpus
    with documents in multiple languages.
    """

    print("\n" + "=" * 70)
    print("ENTERPRISE CORPUS: Cost Impact of Language Distribution")
    print("=" * 70)

    QUERIES_PER_DAY     = 2_000
    CHUNKS_PER_QUERY    = 5
    BASE_CHUNK_TOKENS   = 300    # English chunk
    OUTPUT_TOKENS       = 300
    PRICE_INPUT_PER_M   = 3.00   # Sonnet input price

    # Corpus language distribution (enterprise with global ops)
    language_distributions = [
        ("English-only corpus",     {"English": 1.00}),
        ("Mixed US/EU corpus",      {"English": 0.70, "German": 0.15, "French": 0.15}),
        ("Global enterprise corpus",{"English": 0.50, "Chinese": 0.20, "Japanese": 0.15, "Spanish": 0.15}),
    ]

    # Token multipliers per language (vs English baseline)
    # WHY these specific values: derived from the parallel sentence analysis above
    token_multipliers = {
        "English":    1.0,
        "Spanish":    1.1,
        "French":     1.15,
        "German":     1.15,
        "Russian":    1.8,
        "Arabic":     2.5,
        "Chinese":    2.2,
        "Japanese":   2.5,
        "Korean":     2.3,
        "Hindi":      2.0,
    }

    for corpus_name, distribution in language_distributions:
        # Weighted average tokens per chunk for this corpus
        avg_multiplier = sum(
            frac * token_multipliers.get(lang, 1.5)
            for lang, frac in distribution.items()
        )
        avg_chunk_toks = BASE_CHUNK_TOKENS * avg_multiplier
        input_per_q    = avg_chunk_toks * CHUNKS_PER_QUERY + 500  # + system/query overhead
        cost_per_q     = input_per_q / 1_000_000 * PRICE_INPUT_PER_M
        monthly_cost   = cost_per_q * QUERIES_PER_DAY * 30

        print(f"\n  [{corpus_name}]")
        print(f"  Language mix: {', '.join(f'{l}: {f:.0%}' for l, f in distribution.items())}")
        print(f"  Avg tokens/chunk: {avg_chunk_toks:.0f}  (vs {BASE_CHUNK_TOKENS} for English-only)")
        print(f"  Input tokens/query: {input_per_q:.0f}")
        print(f"  Monthly cost (input only): ${monthly_cost:,.2f}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not HAS_TIKTOKEN:
        print("⚠ tiktoken not installed — token counts are approximations.")
        print("  Install with: pip install tiktoken\n")

    analyze_multilingual_efficiency()
    context_capacity_by_language()
    multilingual_rag_patterns()
    enterprise_multilingual_cost_impact()
