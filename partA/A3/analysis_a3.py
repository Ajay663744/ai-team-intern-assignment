#!/usr/bin/env python3

"""
analysis_a3.py -- Corrected multilingual tokenizer comparison

A3 requirements:
- At least two tokenizers
- At least three denominator definitions
- Same multilingual corpus for every tokenizer
- Corpus-level aggregation

Tokenizers:
1. GPT-2
2. XLM-RoBERTa

Denominators:
1. Whitespace words
2. Unicode grapheme clusters
3. UTF-8 bytes
"""

import argparse
import unicodedata

import regex
import tiktoken
from transformers import AutoTokenizer


def load_lines(path):
    lines = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            if not line:
                continue

            line = unicodedata.normalize("NFC", line)
            lines.append(line.lower())

    return lines


def load_tokenizer(spec):
    if spec == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        return enc.encode

    if spec == "xlm-roberta-base":
        tok = AutoTokenizer.from_pretrained(spec)

        return lambda text: tok.encode(
            text,
            add_special_tokens=False
        )

    raise ValueError(f"Unknown tokenizer: {spec}")


def analyze(lines, encode):
    total_tokens = 0
    total_words = 0
    total_graphemes = 0
    total_bytes = 0

    for line in lines:
        tokens = encode(line)

        words = line.split()
        graphemes = regex.findall(r"\X", line)
        utf8_bytes = len(line.encode("utf-8"))

        total_tokens += len(tokens)
        total_words += len(words)
        total_graphemes += len(graphemes)
        total_bytes += utf8_bytes

    tokens_per_word = total_tokens / total_words
    tokens_per_grapheme = total_tokens / total_graphemes
    tokens_per_byte = total_tokens / total_bytes

    return (
        tokens_per_word,
        tokens_per_grapheme,
        tokens_per_byte,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LANG=PATH",
        help="Language and corpus path",
    )

    parser.add_argument(
        "--tokenizer",
        action="append",
        required=True,
        choices=["gpt2", "xlm-roberta-base"],
        help="Tokenizer to evaluate",
    )

    args = parser.parse_args()

    corpora = {}

    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        corpora[lang] = load_lines(path)

    for tokenizer_name in args.tokenizer:

        encode = load_tokenizer(tokenizer_name)

        print()
        print(f"tokenizer: {tokenizer_name}")
        print(
            f"{'lang':<8}"
            f"{'tok/word':>12}"
            f"{'tok/grapheme':>16}"
            f"{'tok/byte':>12}"
        )
        print("-" * 48)

        results = {}

        for lang, lines in corpora.items():

            (
                tokens_per_word,
                tokens_per_grapheme,
                tokens_per_byte,
            ) = analyze(lines, encode)

            results[lang] = (
                tokens_per_word,
                tokens_per_grapheme,
                tokens_per_byte,
            )

            print(
                f"{lang:<8}"
                f"{tokens_per_word:>12.3f}"
                f"{tokens_per_grapheme:>16.3f}"
                f"{tokens_per_byte:>12.3f}"
            )

        print()

        base = "eng"

        for lang in corpora:

            if lang == base:
                continue

            word_ratio = (
                results[lang][0]
                / results[base][0]
            )

            grapheme_ratio = (
                results[lang][1]
                / results[base][1]
            )

            byte_ratio = (
                results[lang][2]
                / results[base][2]
            )

            print(
                f"{lang}: "
                f"{word_ratio:.2f}x English "
                f"(tok/word), "
                f"{grapheme_ratio:.2f}x English "
                f"(tok/grapheme), "
                f"{byte_ratio:.2f}x English "
                f"(tok/byte)"
            )


if __name__ == "__main__":
    main()