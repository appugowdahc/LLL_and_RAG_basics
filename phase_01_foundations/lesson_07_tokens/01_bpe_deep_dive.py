"""
FILE: 01_bpe_deep_dive.py
LESSON: Phase 1 - Lesson 7 - Tokens Deep-Dive
TOPIC: Byte-Pair Encoding (BPE) — the algorithm that creates the token vocabulary

WHAT THIS FILE TEACHES:
  - What BPE is and WHY it was invented
  - The BPE training algorithm: character pairs → merges → vocabulary
  - WHY BPE beats word-level and character-level tokenization
  - Using tiktoken to inspect real Claude/GPT tokenization
  - How vocabulary size affects model capability and cost

WHY BPE WAS INVENTED:
  Word-level tokenizers: "unbelievable" is one token. "unbelievably" is unknown (OOV).
  Any new word → [UNK]. Training corpus must contain EVERY possible word form.
  Result: huge vocabulary OR lots of unknown tokens.

  Character-level tokenizers: every character is a token.
  "hello" = 5 tokens. Sequences are very long. Model must learn to compose words
  from characters — difficult for attention over long sequences.

  BPE SOLUTION: Learn the most common subword units.
  Common words become single tokens. Rare words split into known pieces.
  "unbelievably" → ["un", "believ", "ably"] — all known, no [UNK].

INSTALL:
  pip install tiktoken
"""

import re
import collections
from typing import Optional

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Install tiktoken for real tokenizer: pip install tiktoken")


# ─── Step 1: BPE Training Algorithm ──────────────────────────────────────────

def get_vocab_from_corpus(corpus: str) -> dict[str, int]:
    """
    Convert raw text into a character-level vocabulary with word frequency counts.
    Each word is represented as a space-separated string of characters.

    WHY space-separated characters:
      BPE needs to track which characters are adjacent so it can merge them.
      Representing "hello" as "h e l l o" lets us count ("h","e"), ("e","l"), etc.
      The </w> marker denotes end-of-word — critical for knowing where merges stop.

    Args:
        corpus: Raw text string.

    Returns:
        Dict mapping "space-separated chars + </w>" → frequency count.
    """

    vocab = collections.defaultdict(int)

    # WHY split on whitespace only (not punctuation):
    #   BPE in production handles punctuation as separate tokens.
    #   This simplified version focuses on word-level BPE for clarity.
    for word in corpus.split():
        # Add </w> end-of-word marker → distinguishes "he" in "he" vs "he" in "hello"
        char_repr = " ".join(list(word)) + " </w>"
        vocab[char_repr] += 1

    return dict(vocab)


def get_stats(vocab: dict[str, int]) -> dict[tuple, int]:
    """
    Count how often each adjacent character pair appears across all words.

    WHY pairs not triples:
      BPE merges ONE pair at a time (the most frequent). Greedy, iterative.
      Triples would create too many candidates and slow down training.

    Returns:
        Dict mapping (char_a, char_b) → frequency count.
    """

    pairs = collections.defaultdict(int)

    for word, freq in vocab.items():
        symbols = word.split()

        # WHY zip(symbols, symbols[1:]):
        #   Creates overlapping pairs: "h e l l o" → ("h","e"), ("e","l"), ("l","l"), ("l","o")
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq

    return dict(pairs)


def merge_vocab(best_pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    """
    Merge the most frequent pair throughout the entire vocabulary.

    WHY replace with regex:
      After merging ("l","l") → "ll", ALL occurrences of "l l" must become "ll"
      across every word in the vocab. A simple string replace handles this.

    Args:
        best_pair: The (A, B) pair to merge into "AB".
        vocab:     Current vocabulary.

    Returns:
        New vocabulary with all instances of the pair merged.
    """

    new_vocab = {}

    # WHY re.escape:
    #   The pair chars may contain regex-special characters (e.g., ".").
    #   re.escape ensures they are matched literally.
    bigram  = re.escape(" ".join(best_pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

    for word in vocab:
        merged_word = pattern.sub("".join(best_pair), word)
        new_vocab[merged_word] = vocab[word]

    return new_vocab


def train_bpe(corpus: str, num_merges: int = 20) -> tuple[dict, list]:
    """
    Full BPE training loop.

    Algorithm:
      1. Build initial character-level vocabulary
      2. Count all adjacent pairs
      3. Merge the most frequent pair
      4. Repeat for num_merges iterations

    Args:
        corpus:     Training text.
        num_merges: Number of merge operations (= vocab size growth above base chars).

    Returns:
        (final_vocab, merge_rules_list)
        merge_rules_list: The learned merge operations in priority order.
    """

    vocab       = get_vocab_from_corpus(corpus)
    merge_rules = []

    print(f"\n  Initial character vocabulary ({len(vocab)} unique words):")
    for word, freq in list(vocab.items())[:5]:
        print(f"    '{word}' × {freq}")
    print(f"    ...")

    for i in range(num_merges):
        pairs = get_stats(vocab)

        if not pairs:
            print(f"  Stopped at merge {i} — no more pairs to merge.")
            break

        # The most frequent pair is always the next merge
        # WHY greedy choice:
        #   Merging the most frequent pair gives the most compression per step.
        #   This is the defining characteristic of BPE — always greedy.
        best_pair = max(pairs, key=pairs.get)
        best_freq = pairs[best_pair]

        merge_rules.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)

        merged_token = "".join(best_pair)
        print(f"  Merge {i+1:>3}: {best_pair[0]!r} + {best_pair[1]!r} → {merged_token!r}  "
              f"(appeared {best_freq}× in corpus)")

    return vocab, merge_rules


def apply_bpe(text: str, merge_rules: list[tuple]) -> list[str]:
    """
    Tokenize new text using the learned BPE merge rules.

    WHY apply in SAME ORDER as training:
      BPE merges are ordered by frequency during training.
      Applying them in order produces the same segmentation as training.
      If you apply in wrong order → different tokens → model can't understand input.

    Args:
        text:        Text to tokenize.
        merge_rules: Ordered list of (A, B) merge pairs from training.

    Returns:
        List of BPE tokens.
    """

    # Start with character-level representation
    tokens = list(text) + ["</w>"]

    for pair in merge_rules:
        merged_token = "".join(pair)
        i = 0
        new_tokens = []

        while i < len(tokens):
            # Check if this position starts the target pair
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged_token)
                i += 2   # WHY skip 2: consumed both elements of the pair
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens

    # Remove </w> end marker from final token representation
    return [t.replace("</w>", "") for t in tokens if t != "</w>"]


# ─── Step 2: Real Tokenizer (tiktoken) ───────────────────────────────────────

def explore_real_tokenizer():
    """
    Use tiktoken to inspect real BPE tokenization as used by Claude and GPT-4.
    Claude uses its own proprietary tokenizer but it is BPE-based and cl100k_base
    gives very similar results for English text (within 5-10% token count).

    WHY cl100k_base:
      This is the tiktoken encoding used by GPT-4 and text-embedding-ada-002.
      Claude's tokenizer is not publicly released, but cl100k_base is the closest
      publicly available approximation for English and code.
    """

    if not HAS_TIKTOKEN:
        print("  tiktoken not installed — skipping real tokenizer demo")
        return

    enc = tiktoken.get_encoding("cl100k_base")

    # WHY get_encoding not get_encoding_for_model:
    #   get_encoding gives us the raw BPE encoder by name.
    #   get_encoding_for_model would need a specific OpenAI model name.
    #   For Claude approximation, we use the encoding directly.

    test_cases = [
        ("Single common word",  "hello"),
        ("With leading space",  " hello"),           # Leading space → different token!
        ("Capitalization diff",  "Cisco"),
        ("All caps",             "CISCO"),
        ("Compound word",        "tokenization"),
        ("Rare technical term",  "microsegmentation"),
        ("Number short",         "100"),
        ("Number long",          "10000"),
        ("IP address",           "192.168.1.1"),
        ("Python code",          "def foo(x: int) -> str:"),
        ("JSON snippet",         '{"key": "value", "num": 42}'),
        ("URL",                  "https://api.anthropic.com/v1/messages"),
    ]

    print("\n  REAL TOKENIZER (tiktoken cl100k_base ≈ Claude):")
    print(f"  {'Description':<30} {'Text':<40} {'Tokens':<8} {'IDs'}")
    print(f"  {'─'*30} {'─'*40} {'─'*8} {'─'*30}")

    for desc, text in test_cases:
        token_ids     = enc.encode(text)
        decoded_toks  = [enc.decode([tid]) for tid in token_ids]
        ids_str       = str(token_ids[:5]) + ("..." if len(token_ids) > 5 else "")

        print(
            f"  {desc:<30} {repr(text):<40} "
            f"{len(token_ids):<8} {decoded_toks}"
        )


def vocab_size_analysis():
    """
    Explain WHY vocabulary size matters for a RAG system.
    """

    print("\n\n  VOCABULARY SIZE TRADEOFFS:")
    print(f"  {'Vocab Size':<15} {'Typical Model':<30} {'Implication'}")
    print(f"  {'─'*15} {'─'*30} {'─'*40}")

    vocab_data = [
        (512,       "Character-level (rare)",       "Every char is a token. Long sequences, poor compression."),
        (8_000,     "Early BERT variants",           "Short sequences. Many [UNK] tokens for rare words."),
        (32_000,    "Llama-1, many open models",     "Good balance. Some rare words split badly."),
        (50_257,    "GPT-2, GPT-3",                  "Standard GPT vocabulary. Solid English coverage."),
        (100_000,   "cl100k_base (GPT-4, Claude)",   "Better code, multilingual, and technical coverage."),
        (128_000,   "Llama-3, Mistral-Large",         "Strong multilingual support. Fewer splits for non-English."),
        (256_000,   "Gemini",                         "Broadest multilingual coverage. ~1 token/Chinese char."),
    ]

    for vocab_size, model, implication in vocab_data:
        print(f"  {vocab_size:<15,} {model:<30} {implication}")

    print(f"""
  WHY LARGER VOCAB HELPS RAG:
    1. Fewer tokens per chunk → more content fits in context window budget
    2. Technical terms (Cisco ACI, EVPN, TrustSec) may be single tokens
       instead of splitting into meaningless subwords
    3. Non-English documents cost fewer tokens per equivalent content
    4. BUT larger vocab = larger embedding table = more GPU VRAM for the model
""")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("BPE DEEP-DIVE: Learning the Token Vocabulary")
    print("=" * 65)

    # ── BPE Training Demo ─────────────────────────────────────────────────────
    # Small corpus to show the algorithm clearly

    training_corpus = (
        "the cat sat on the mat "
        "the cat ate the rat "
        "the cat is fat "
        "the rat ran away "
        "the mat is flat "
        "cats and rats and bats "
    )

    print(f"\n  TRAINING CORPUS:\n  '{training_corpus[:80]}...'\n")

    print("  BPE TRAINING — watching the vocabulary build:")
    print("  " + "─" * 60)

    final_vocab, merge_rules = train_bpe(training_corpus, num_merges=15)

    print(f"\n  FINAL VOCABULARY (top entries):")
    for token_str, freq in sorted(final_vocab.items(), key=lambda x: -x[1])[:8]:
        tokens_in_word = token_str.split()
        print(f"    '{token_str}' (freq={freq}) → tokens: {tokens_in_word}")

    # ── Apply learned BPE to new words ───────────────────────────────────────
    print(f"\n  APPLYING LEARNED MERGE RULES TO NEW WORDS:")

    test_words = ["cats", "bats", "mat", "flat", "catfish"]
    for word in test_words:
        tokens = apply_bpe(word, merge_rules)
        print(f"    '{word}' → {tokens}")

    # ── Real tokenizer ────────────────────────────────────────────────────────
    explore_real_tokenizer()
    vocab_size_analysis()
