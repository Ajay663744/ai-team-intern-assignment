#!/usr/bin/env python3
"""
build_corpus.py  --  Part A1: Build FLORES-200 multilingual eval corpus
========================================================================

Source (primary): Direct download of the official FLORES-200 dataset archive
  URL:     https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
  Archive: flores200_dataset.tar.gz  (~25.6 MB)
  License: CC-BY-SA 4.0
  Repo:    https://github.com/facebookresearch/flores  (archived)
           Newer OLDI fork: https://github.com/openlanguagedata/flores

  Note on HuggingFace alternatives: as of 2026-09, both
  openlanguagedata/flores_plus and facebook/flores on the HF Hub are GATED
  (require account login + access approval); Muennighoff/flores200 is blocked
  by the HF datasets library because it uses a legacy dataset script.
  The direct CDN download above is publicly accessible with no authentication.

Split used: devtest  (1012 sentences per language, line-aligned)

LANGUAGES:
  eng_Latn  - English  (Latin script)
  hin_Deva  - Hindi    (Devanagari script)
  tam_Taml  - Tamil    (Tamil script)
  kan_Knda  - Kannada  (Kannada script)

PREPROCESSING (minimal, deliberately):
  1. Strip leading/trailing whitespace per line (incl. Windows CRLF).
  2. Unicode NFC normalisation applied consistently to every language.
  3. NO lowercasing: Hindi/Tamil/Kannada have no case distinction. Lowercasing
     only English would introduce a systematic cross-language asymmetry --
     the exact inconsistency Part A2 audits. Omitting is the correct choice.
  4. No other transformations (internal whitespace, punctuation unchanged).
     Zero-width spaces (U+200B) found in Hindi/Kannada text are preserved as-is;
     silently removing them would be an undocumented transformation.

OUTPUT (UTF-8, LF line endings, one sentence per line):
  partA/corpus/eng_Latn.txt   (1012 lines)
  partA/corpus/hin_Deva.txt   (1012 lines)
  partA/corpus/tam_Taml.txt   (1012 lines)
  partA/corpus/kan_Knda.txt   (1012 lines)

USAGE:
  # Step 1: download and extract the dataset (one-time)
  # curl -O https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
  # tar -xzf flores200_dataset.tar.gz

  # Step 2: run this script
  python build_corpus.py --flores-dir flores200_dataset --output-dir partA/corpus

  # Dry run (stats only, no files written):
  python build_corpus.py --flores-dir flores200_dataset --output-dir partA/corpus --dry-run
"""

import argparse
import sys
import unicodedata
from pathlib import Path

LANGUAGES = ["eng_Latn", "hin_Deva", "tam_Taml", "kan_Knda"]
SPLIT = "devtest"
SOURCE_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
LICENSE = "CC-BY-SA 4.0"


def preprocess_line(line):
    """
    Minimal preprocessing pipeline for a single raw sentence line.

    Steps applied:
      1. rstrip('\\r\\n')  -- remove the newline character (not strip() yet)
      2. .strip()         -- remove any remaining leading/trailing whitespace
      3. unicodedata.normalize('NFC', ...)  -- Unicode NFC normalisation

    Deliberately NOT done:
      - Lowercasing: Hindi, Tamil, and Kannada scripts have no case
        distinction. Lowercasing only English would introduce a cross-language
        asymmetry. Part A2 audits exactly this kind of inconsistency.
      - Internal whitespace collapse: language-specific spacing conventions
        (e.g. around Devanagari punctuation) must not be silently altered.
      - Punctuation normalisation: scripts use distinct punctuation marks
        (Devanagari danda, Tamil-specific characters, ASCII period) --
        normalising across scripts would erase linguistically valid variation.
      - Removing zero-width spaces (U+200B): these appear in the source
        files in Hindi and Kannada text and are preserved as-is. Silently
        removing them would be an undocumented transformation.
    """
    line = line.rstrip('\r\n').strip()
    line = unicodedata.normalize('NFC', line)
    return line


def load_split(flores_dir, split=SPLIT):
    """
    Load all 4 target languages from the FLORES-200 devtest directory.
    Returns dict: lang_code -> list[str] of preprocessed sentences.
    """
    split_dir = Path(flores_dir) / split
    if not split_dir.is_dir():
        # Maybe the user passed the devtest dir directly
        split_dir = Path(flores_dir)

    corpus = {}
    for lang in LANGUAGES:
        fpath = split_dir / (lang + '.' + split)
        if not fpath.exists():
            print(f"ERROR: File not found: {fpath}", file=sys.stderr)
            print(f"  Expected: <flores_dir>/{split}/{lang}.{split}", file=sys.stderr)
            sys.exit(1)
        with fpath.open(encoding='utf-8') as f:
            sentences = [preprocess_line(ln) for ln in f]
        # Drop trailing empty lines (artifact of some editors)
        while sentences and not sentences[-1]:
            sentences.pop()
        corpus[lang] = sentences
        print(f"  Loaded {lang}: {len(sentences)} sentences  ({fpath})")
    return corpus


def verify_alignment(corpus):
    """Confirm all languages have the same sentence count. Exit if not."""
    counts = {lang: len(sents) for lang, sents in corpus.items()}
    unique = set(counts.values())
    if len(unique) != 1:
        print("ALIGNMENT ERROR: Languages have different sentence counts!", file=sys.stderr)
        for lang, n in counts.items():
            print(f"  {lang}: {n}", file=sys.stderr)
        sys.exit(1)
    n = unique.pop()
    print(f"\nAlignment verified: all {len(corpus)} languages have {n} sentences. OK")
    return n


def compute_stats(corpus):
    stats = {}
    for lang, sents in corpus.items():
        text = '\n'.join(sents)
        stats[lang] = {
            'sentences': len(sents),
            'words': sum(len(s.split()) for s in sents),
            'chars': sum(len(s) for s in sents),
            'bytes_utf8': len(text.encode('utf-8')),
        }
    return stats


def print_stats_table(stats):
    hdr = f"{'Language':<12} {'Sentences':>10} {'Words':>10} {'Chars':>10} {'Bytes(UTF-8)':>14}"
    print('\n' + hdr)
    print('-' * len(hdr))
    for lang, s in stats.items():
        print(f"{lang:<12} {s['sentences']:>10,} {s['words']:>10,} {s['chars']:>10,} {s['bytes_utf8']:>14,}")


def write_corpus(corpus, output_dir, dry_run):
    if dry_run:
        print("\n[dry-run] Would write to:", output_dir)
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for lang, sents in corpus.items():
        out_path = output_dir / (lang + '.txt')
        with out_path.open('w', encoding='utf-8', newline='\n') as fh:
            for sent in sents:
                fh.write(sent + '\n')
        print(f"  Wrote {out_path}  ({len(sents)} lines)")


def main():
    parser = argparse.ArgumentParser(
        description="Build FLORES-200 multilingual eval corpus for Part A1."
    )
    parser.add_argument(
        '--flores-dir', required=True,
        help='Path to the extracted flores200_dataset/ directory.'
    )
    parser.add_argument(
        '--output-dir', default='partA/corpus',
        help='Directory to write output .txt files (default: partA/corpus).'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print stats without writing any files.'
    )
    args = parser.parse_args()

    print('=' * 60)
    print('Part A1 -- FLORES-200 corpus builder')
    print('=' * 60)
    print(f'Source    : {SOURCE_URL}')
    print(f'Split     : {SPLIT}')
    print(f'Languages : {", ".join(LANGUAGES)}')
    print(f'Input dir : {args.flores_dir}')
    print(f'Output    : {args.output_dir}')
    print()

    corpus = load_split(args.flores_dir, split=SPLIT)
    verify_alignment(corpus)
    stats = compute_stats(corpus)
    print_stats_table(stats)
    print()
    write_corpus(corpus, args.output_dir, dry_run=args.dry_run)

    if not args.dry_run:
        print('\nDone. All corpus files written successfully.')


if __name__ == '__main__':
    main()
