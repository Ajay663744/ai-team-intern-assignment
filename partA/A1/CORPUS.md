# CORPUS.md — Part A1: FLORES-200 Multilingual Eval Corpus

## Source

| Field | Value |
|---|---|
| **Dataset name** | FLORES-200 (Evaluation Benchmark for Low-Resource and Multilingual Machine Translation) |
| **Split used** | devtest (the standard ~1012-sentence evaluation split) |
| **Download URL** | https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz |
| **Archive size** | ~25.6 MB (compressed) |
| **Original repo** | https://github.com/facebookresearch/flores (archived; no longer updated) |
| **Current OLDI fork** | https://github.com/openlanguagedata/flores |
| **License** | **CC-BY-SA 4.0** (confirmed from LICENSE_CC-BY-SA file present in the flores-main GitHub upload and from the root README.md which states: "FLORES-200: CC-BY-SA 4.0") |
| **Citation** | NLLB Team et al., "No Language Left Behind: Scaling Human-Centered Machine Translation," 2022. |

**Version/revision note:** The downloaded archive (lores200_dataset.tar.gz) does not embed a Git commit SHA or a dated version string. The README file inside the archive states only: "FLORES 200 dataset" with no version number. The archive was downloaded directly from Meta's CDN on 2026-09-05. The canonical release described in the NLLB 2022 paper is the version linked from the official GitHub README; this is that release. For exact reproducibility, record the SHA-256 of the downloaded archive alongside this file.

**HuggingFace access note:** As of 2026-09-05, both openlanguagedata/flores_plus and acebook/flores on the HuggingFace Hub are gated datasets (require account login and access approval). The Muennighoff/flores200 mirror is blocked by the datasets library because it uses a deprecated dataset script format. The direct CDN download above (dl.fbaipublicfiles.com) is publicly accessible with no authentication and is the primary access path used in uild_corpus.py.

---

## Size

**1012 sentence pairs** used (full FLORES-200 devtest split — no sub-sampling).

### Per-language statistics

| Language | Code | Script | Sentences | Whitespace-words | Characters | UTF-8 bytes |
|---|---|---|---:|---:|---:|---:|
| English | eng_Latn | Latin | 1,012 | 21,901 | 131,966 | 133,107 |
| Hindi | hin_Deva | Devanagari | 1,012 | 25,643 | 131,180 | 338,450 |
| Tamil | 	am_Taml | Tamil | 1,012 | 16,775 | 154,131 | 422,646 |
| Kannada | kan_Knda | Kannada | 1,012 | 16,100 | 138,027 | 376,352 |

**Alignment check:** all 4 files have exactly 1012 lines (verified both before and after preprocessing — see Step 6 below).

**Byte-count interpretation:** The large byte/character ratios for Hindi (338,450 bytes / 131,180 chars ≈ 2.58 bytes/char), Tamil (422,646 / 154,131 ≈ 2.74 bytes/char), and Kannada (376,352 / 138,027 ≈ 2.73 bytes/char) are expected: Devanagari, Tamil, and Kannada scripts encode mostly in the 3-byte range of UTF-8, while Latin/ASCII encodes at 1 byte/char. This is a known property of Unicode block assignments, not an encoding error.

---

## Domain

FLORES-200 sentences are **professionally translated** from source articles drawn from four Wikimedia properties:

- **Wikipedia** — encyclopedic articles across diverse topics (biology, geography, history, science, culture, technology)
- **Wikinews** — short news articles on recent events
- **Wikijunior** — simplified encyclopedic content for younger audiences
- **Wikivoyage** — travel guides and destination descriptions

The 1012 devtest sentences come from 842 distinct source articles (per the FLORES-200 README), averaging approximately 21 words per sentence. The topic mix is deliberately broad to avoid single-domain bias. The register is **formal and encyclopedic** throughout: third-person narrative, full sentences, written prose style. The corpus contains **no conversational dialogue, no social-media text, no code-mixed text, and no colloquial register** — this is a deliberate property of the benchmark's Wikimedia sourcing.

Translations were produced by professional translators under Meta's NLLB programme. Some languages in FLORES-200 were translated not from English but from Spanish, French, Russian, or Modern Standard Arabic (per the official README); for the four languages used here (English, Hindi, Tamil, Kannada), English is the source language.

---

## Preprocessing

The following steps were applied **consistently and identically** to all 4 language files. No step was applied selectively to a subset of languages.

### Steps applied

1. **Strip leading/trailing whitespace per line** (Python .strip() after removing the newline character). This removes accidental indentation, trailing spaces, and Windows CRLF line endings. It does **not** alter any character within the sentence body.

2. **Unicode NFC normalisation** (unicodedata.normalize('NFC', line) in Python 3). This converts any NFD or NFKD-decomposed characters to their precomposed equivalents. In practice all 4 source files were already in NFC form (verified: character-level diff between raw and NFC output was zero for all languages), so this step was a no-op on this specific dataset — but it is applied unconditionally as a defensive measure to guarantee output form.

3. **Remove trailing empty lines** at the end of each file only. The .devtest files end with a trailing newline; processing produces one empty string at the end of the list, which is dropped. This does not affect any sentence content or internal alignment.

### Steps NOT applied (and reasoning)

- **Lowercasing (omitted deliberately):** Hindi (Devanagari), Tamil, and Kannada scripts have no uppercase/lowercase distinction — these scripts are unicameral. Lowercasing only English while leaving the other three languages unchanged would introduce a **systematic cross-language inconsistency** in the representation of the same token across languages. Part A2 of this audit examines exactly this class of inconsistency in tokenizer behaviour. Applying lowercasing here would confound that analysis by introducing a preprocessing asymmetry that mimics or masks the tokenizer-level asymmetry being measured. Therefore: **no language was lowercased**.

- **Internal whitespace collapse (omitted):** Different scripts use different whitespace conventions around punctuation and within sentences. Collapsing multiple spaces would silently alter content in language-specific ways.

- **Punctuation normalisation (omitted):** Devanagari uses the danda (।), Tamil uses its own punctuation conventions, and Kannada similarly. Normalising punctuation across scripts would erase linguistically valid variation.

- **Zero-width space removal (not applied):** Zero-width spaces (U+200B) appear in a small number of Hindi and Kannada sentences in the source files (visible as \u200b in Python repr output). These are preserved exactly as found in the source data. Silently removing them would be an undocumented transformation; they are flagged in the Caveats section instead.

---

## Caveats

**What this corpus cannot tell you.** FLORES-200 devtest sentences are professionally translated, formal, encyclopedic/news-register text drawn from Wikimedia sources (Wikipedia, Wikinews, Wikijunior, Wikivoyage). This register is *not* representative of the casual conversational language, code-mixed text (e.g., Hindi-English or Kannada-English mixing common in Indian social media and messaging), domain-specific vocabulary (medical, legal, technical), or colloquial/slang expressions that a production multilingual NLP system would actually encounter in real deployment traffic. Any tokenizer performance metric derived here should be understood as reflecting behaviour on formal written prose, not on the distribution of text that real users produce. Second, the sample of 1,012 sentences is small relative to the diversity of real production traffic; a single-decimal-place "fertility ratio" computed from this corpus is a directional estimate with meaningful statistical uncertainty, not a precise production measurement — it should be treated as an order-of-magnitude signal rather than a precise number. Third, because all four languages are translations of the same English-rooted source text (with English as the source language for all four in this corpus), the translations may carry "translationese" artifacts: professional translations tend to preserve source-language sentence structure more than natively-authored text in the target language would, which could cause cross-language tokenizer comparisons to understate the structural divergence that would appear in a corpus of naturally authored text per language. Fourth, zero-width spaces (U+200B) were found in a number of Hindi and Kannada sentences in the source .devtest files; these were preserved rather than silently removed, but their presence means that some whitespace-tokenised word counts may be inflated or split differently than expected. Fifth, the FLORES-200 devtest split does not contain any transliterated or Romanised text for Hindi, Tamil, or Kannada — the corpus is entirely in the native scripts — which means that tokenizer behaviour on transliterated or Romanised Indian-language text (common in practice) is not captured here.

---

## Step 6 — Validation (Sanity Table)

The following statistics were computed from the **final output files** in partA/corpus/ after all preprocessing:

| Language | File | Sentences | Whitespace-words | Characters | UTF-8 bytes |
|---|---|---:|---:|---:|---:|
| English | eng_Latn.txt | 1,012 | 21,901 | 131,966 | 133,107 |
| Hindi | hin_Deva.txt | 1,012 | 25,643 | 131,180 | 338,450 |
| Tamil | 	am_Taml.txt | 1,012 | 16,775 | 154,131 | 422,646 |
| Kannada | kan_Knda.txt | 1,012 | 16,100 | 138,027 | 376,352 |

**Alignment re-confirmed:** all 4 output files have exactly **1,012 non-empty lines**. Line N across all 4 files is a translation of the same source sentence (verified from the parallel structure of the FLORES-200 devtest split and by the sentence-level content match on lines 1-3 across all languages).

**NFC check:** character-level diff between raw source bytes and NFC-normalised output was **zero** for all 4 languages — the source files were already in NFC form.

---

## Reproducibility

To reproduce the corpus from scratch:

```bash
# 1. Download the official FLORES-200 dataset archive (publicly accessible, no auth required)
curl -O https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz

# 2. Extract
tar -xzf flores200_dataset.tar.gz

# 3. Run the build script (Python 3.8+, no external dependencies beyond stdlib)
python partA/build_corpus.py \
    --flores-dir flores200_dataset \
    --output-dir partA/corpus
```

The build script (`partA/build_corpus.py`) uses only Python standard library modules (`unicodedata`, `pathlib`, `argparse`, `sys`) and has no third-party dependencies. It reads the 4 target `.devtest` files, applies the documented preprocessing, verifies alignment, and writes the output files.
