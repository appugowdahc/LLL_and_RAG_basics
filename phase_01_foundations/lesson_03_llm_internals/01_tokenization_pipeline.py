"""
FILE: 01_tokenization_pipeline.py
LESSON: Phase 1 - Lesson 3 - How LLMs Work Internally
TOPIC: Tokenization — How raw text becomes numbers the model processes

WHAT THIS FILE TEACHES:
  - What tokenization does and why it exists
  - BPE (Byte-Pair Encoding) algorithm intuition
  - How to see real token splits using the `tiktoken` library
  - Why some words cost more tokens than others
  - Special tokens ([BOS], [EOS], [PAD]) and what they do
  - Why token count != word count (pricing impact)

CONCEPT: Why Not Use Characters or Words?
──────────────────────────────────────────
CHARACTERS:
  Problem: "Hello" = 5 units. Long sequences = many steps = slow.
  "A 1000-word document" = ~5000 character steps. Too slow.

WORDS:
  Problem: Vocabulary explodes ("run", "running", "runs", "ran" = 4 entries).
  Typos and rare words become unknown tokens (OOV problem).
  "Antidisestablishmentarianism" → unknown word.

SUBWORDS (BPE — what LLMs actually use):
  Solution: Merge frequent character pairs into single tokens.
  "running" → ["run", "ning"]   (2 tokens, not 7 chars or 1 word)
  "ChatGPT" → ["Chat", "G", "PT"]  (handles novel words)
  Fixed vocabulary of ~50k-100k tokens covers virtually all text.

BPE Algorithm (simplified):
  1. Start with every character as its own token
  2. Count all adjacent token pairs in corpus
  3. Merge the most frequent pair into one new token
  4. Repeat until vocabulary size is reached

INSTALL:
  pip install tiktoken anthropic python-dotenv
"""

import os
from dotenv import load_dotenv
import anthropic

# WHY tiktoken:
#   tiktoken is OpenAI's tokenizer library — it implements the cl100k_base
#   and other tokenizer encodings used by GPT models.
#   Claude uses a DIFFERENT tokenizer internally (not publicly released),
#   but tiktoken is close enough for learning purposes and widely used in RAG.
#   For exact Claude token counts, use client.messages.count_tokens() (Lesson 1).
import tiktoken

load_dotenv()

client = anthropic.Anthropic()


# ─── Load Tokenizer ───────────────────────────────────────────────────────────

# WHY tiktoken.get_encoding("cl100k_base"):
#   cl100k_base is the encoding used by GPT-4 and text-embedding-ada-002.
#   It has a vocabulary of 100,277 tokens.
#   "o200k_base" is the newer encoding used by GPT-4o (200k vocab).
#   We use cl100k_base as a widely-understood reference implementation.
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Vocabulary size of this tokenizer
VOCAB_SIZE = TOKENIZER.n_vocab  # 100,277


# ─── Core Tokenization Functions ──────────────────────────────────────────────

def tokenize(text: str) -> dict:
    """
    Tokenize text and return full analysis: token IDs, decoded tokens, count.

    Args:
        text: Any string to tokenize.

    Returns:
        dict with token_ids, token_strings, count, chars_per_token
    """

    # WHY encode():
    #   Converts a string into a list of integer token IDs.
    #   Each integer is an index into the model's vocabulary table.
    #   The model ONLY sees these integers — never raw text.
    token_ids = TOKENIZER.encode(text)

    # WHY decode each token individually:
    #   By decoding [id] → string for each id separately, we can SEE
    #   exactly how the text was split at subword boundaries.
    #   This is not how the model uses tokens — we do this for visualization.
    token_strings = [
        TOKENIZER.decode([tid]) for tid in token_ids
    ]

    return {
        "original_text":   text,
        "token_ids":       token_ids,
        "token_strings":   token_strings,
        "token_count":     len(token_ids),
        "char_count":      len(text),
        "word_count":      len(text.split()),
        "chars_per_token": round(len(text) / len(token_ids), 2) if token_ids else 0,
    }


def visualize_tokenization(text: str):
    """
    Print a color-coded visualization of how text splits into tokens.

    WHY visualize:
      Seeing "ChatGPT" → ["Chat", "G", "PT"] builds intuition for why
      certain prompts cost more tokens than expected.
    """

    result = tokenize(text)

    print(f"\n  Text:   \"{text}\"")
    print(f"  Tokens: {result['token_count']}  |  Words: {result['word_count']}  |  Chars: {result['char_count']}")
    print(f"  Chars/token: {result['chars_per_token']}")

    # Show each token with its ID
    # WHY format with [brackets]:
    #   Makes token boundaries visible — "run" and "ning" are separate tokens.
    token_display = " | ".join(
        f"[{t!r}:{tid}]" for t, tid in
        zip(result["token_strings"], result["token_ids"])
    )
    print(f"  Split: {token_display}")


# ─── Key Tokenization Insights ────────────────────────────────────────────────

def demonstrate_tokenization_insights():
    """
    Show the non-obvious behaviours of tokenization that affect RAG.

    WHY these examples matter:
      Each insight maps directly to a RAG design decision:
        - Expensive tokens → chunk size decisions
        - Code tokenization → code-specific chunking strategies
        - Special characters → cleaning/normalization pipeline (Phase 4)
        - Non-English → foreign language RAG cost differences
    """

    examples = [
        # Basic words
        ("Simple English",         "Hello world"),
        ("Common word",            "the"),

        # Subword splitting
        ("Technical compound word", "tokenization"),
        ("LLM-specific term",       "Retrieval-Augmented Generation"),
        ("Rare/long word",          "antidisestablishmentarianism"),

        # Numbers and code
        ("Number",                 "12345"),
        ("Phone number format",    "+1 (555) 123-4567"),
        ("Python code",            "def calculate_cosine_similarity(vec_a, vec_b):"),
        ("SQL query",              "SELECT * FROM documents WHERE embedding IS NOT NULL;"),

        # Why spacing matters
        ("Word with leading space", " hello"),  # Different token than "hello"!
        ("Capitalized",             "Hello"),   # Different token than "hello"!

        # Non-English text (more tokens per word)
        ("Japanese",               "私はAIエンジニアです"),
        ("Arabic",                 "مرحبا بالعالم"),
        ("Emoji",                  "Hello 👋 World 🌍"),

        # RAG-specific: typical chunk sizes
        ("~50 word chunk",
         "Retrieval-Augmented Generation combines the strengths of retrieval systems "
         "and generative models. By retrieving relevant documents before generating a "
         "response, RAG systems can provide accurate, up-to-date answers grounded in "
         "real source material rather than relying solely on trained knowledge."),
    ]

    print("\n" + "="*65)
    print("TOKENIZATION INSIGHTS")
    print("="*65)

    for label, text in examples:
        print(f"\n[{label}]")
        visualize_tokenization(text)


# ─── Special Tokens ───────────────────────────────────────────────────────────

def explain_special_tokens():
    """
    Explain special tokens and their role in LLM training and inference.

    WHY special tokens matter for RAG:
      When you build prompts for RAG, you use special tokens implicitly.
      Understanding them prevents accidental injection of token sequences
      that disrupt the model's expected prompt structure.
    """

    print("\n" + "="*65)
    print("SPECIAL TOKENS")
    print("="*65)

    special_tokens_explained = {
        "<|endoftext|>": {
            "id":    100257,
            "role":  "End of document marker",
            "why":   "Tells the model a document ends here. "
                     "During pre-training, multiple documents are packed into one sequence. "
                     "This token separates them so the model doesn't confuse "
                     "the end of doc A with the start of doc B.",
            "rag_impact": "When concatenating retrieved chunks, use clear separators "
                          "to prevent the model from conflating chunks.",
        },
        "[BOS] (Beginning of Sequence)": {
            "id":    "model-specific",
            "role":  "Marks the start of a sequence",
            "why":   "Gives the model a clean starting state. "
                     "Some models require it; Claude handles this internally.",
            "rag_impact": "Handled automatically by the API. No action needed.",
        },
        "[PAD] (Padding)": {
            "id":    "model-specific",
            "role":  "Fills sequences to equal length in batches",
            "why":   "GPU training requires fixed-length tensors. "
                     "Shorter sequences are padded to match the longest in the batch. "
                     "Attention masks prevent the model from attending to padding.",
            "rag_impact": "Irrelevant for API calls (handled server-side), "
                          "but critical when running local models (Llama, Mistral).",
        },
        "[INST] / <|user|> (Instruction tokens)": {
            "id":    "model-specific",
            "role":  "Marks the start of a user turn in chat format",
            "why":   "Instruction-tuned models use special tokens to separate "
                     "user turns from assistant turns. "
                     "Llama-3 uses: <|start_header_id|>user<|end_header_id|>",
            "rag_impact": "When using local models (not API), you MUST format "
                          "prompts with these tokens exactly, or the model ignores instructions.",
        },
    }

    for token, info in special_tokens_explained.items():
        print(f"\n  Token: {token}")
        print(f"  Role:       {info['role']}")
        print(f"  Why exists: {info['why']}")
        print(f"  RAG impact: {info['rag_impact']}")


# ─── Token Cost Calculator ────────────────────────────────────────────────────

def rag_chunk_token_analysis(documents: list[str]):
    """
    Analyze token counts across a list of document chunks.
    Shows distribution of chunk sizes — critical for chunking strategy (Phase 5).

    WHY analyze chunk token distribution:
      In RAG, chunks that are too long eat your context window budget.
      Chunks that are too short may not contain enough context for the LLM.
      Optimal chunk size is typically 256-512 tokens for most use cases.
    """

    print("\n" + "="*65)
    print("RAG CHUNK TOKEN ANALYSIS")
    print("="*65)

    token_counts = []
    for i, doc in enumerate(documents):
        result = tokenize(doc)
        token_counts.append(result["token_count"])
        print(f"\n  Chunk {i+1}: {result['token_count']} tokens | "
              f"{result['word_count']} words | "
              f"First 60 chars: {doc[:60]}...")

    if token_counts:
        avg_tokens  = sum(token_counts) / len(token_counts)
        max_tokens  = max(token_counts)
        min_tokens  = min(token_counts)
        total       = sum(token_counts)

        print(f"\n  Summary:")
        print(f"  Chunks:       {len(token_counts)}")
        print(f"  Total tokens: {total:,}")
        print(f"  Avg/chunk:    {avg_tokens:.0f} tokens")
        print(f"  Min/Max:      {min_tokens} / {max_tokens} tokens")

        # Context window check: how many chunks fit in Claude's 200k window?
        # Assuming 60% of context window is for retrieved docs (from Lesson 2 budget)
        budget = int(200_000 * 0.60)
        chunks_that_fit = budget // avg_tokens if avg_tokens > 0 else 0
        print(f"\n  At avg size: {chunks_that_fit:.0f} chunks fit in RAG doc budget (120k tokens)")
        print(f"  → Phase 5 (Chunking) will show how to optimize this.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"Tokenizer: cl100k_base | Vocabulary size: {VOCAB_SIZE:,}")

    # Demo 1: Tokenization insights
    demonstrate_tokenization_insights()

    # Demo 2: Special tokens
    explain_special_tokens()

    # Demo 3: RAG chunk analysis
    sample_chunks = [
        "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses "
        "by retrieving relevant documents from a knowledge base before generating an answer. "
        "The retrieved documents are injected into the prompt as context.",

        "Vector databases store high-dimensional embedding vectors and support approximate "
        "nearest neighbor (ANN) search. Popular options include Pinecone, Weaviate, Milvus, "
        "Qdrant, and ChromaDB. Each has different trade-offs in performance, cost, and features.",

        "Chunking is the process of splitting documents into smaller pieces before embedding. "
        "Common strategies include fixed-size chunking (split every N tokens), "
        "sliding window chunking (overlapping chunks), and semantic chunking "
        "(split at natural semantic boundaries like paragraphs or sentences).",

        "BM25 is a bag-of-words retrieval function that ranks documents based on term frequency "
        "and inverse document frequency. It is the foundation of keyword search in systems "
        "like Elasticsearch. In hybrid RAG, BM25 is combined with dense vector search "
        "to get the best of both keyword and semantic retrieval.",
    ]

    rag_chunk_token_analysis(sample_chunks)
