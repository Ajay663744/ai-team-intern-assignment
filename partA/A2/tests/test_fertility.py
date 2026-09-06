import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fertility_fixed import analyze
import math

from fertility_fixed import analyze


def simple_encoder(text):
    # One token per Unicode code point.
    return list(text)


def test_whitespace_splitting():
    lines = ["hello  world"]

    fertility, _ = analyze(lines, simple_encoder)

    # "hello  world" has 2 words.
    # The simple encoder produces 12 tokens:
    # 5 + 2 spaces + 5 = 12
    # fertility = 12 / 2 = 6.0
    assert fertility == 6.0


def test_corpus_level_aggregation():
    lines = ["a b c d", "e"]

    fertility, _ = analyze(lines, simple_encoder)

    # "a b c d" = 7 codepoints/tokens
    # "e" = 1 codepoint/token
    #
    # Total tokens = 8
    # Total words = 5
    #
    # Corpus-level fertility = 8 / 5 = 1.6
    #
    # Buggy mean-of-line-ratios would be:
    # (7/4 + 1/1) / 2 = 1.375
    assert fertility == 1.6


def test_grapheme_cluster_counting():
    # "कि" contains multiple Unicode code points
    # but represents one grapheme cluster.
    lines = ["कि"]

    fertility, tok_per_char = analyze(lines, simple_encoder)

    # 2 Unicode code-point tokens / 1 grapheme cluster
    assert fertility == 2.0
    assert tok_per_char == 2.0


def test_results_are_finite():
    lines = ["hello world"]

    fertility, tok_per_char = analyze(lines, simple_encoder)

    assert math.isfinite(fertility)
    assert math.isfinite(tok_per_char)