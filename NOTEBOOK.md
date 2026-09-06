# Audit Notebook

## Part B — 2026-09-04

### B1: KV-cache sizing

**Hypothesis**: The per-token KV-cache cost follows `layers × 2 × kv_heads × head_dim × bytes_per_element`, where GQA means using 8 KV heads (not 24 Q heads), head_dim=128, fp16=2 bytes.

**Calculation**:
```
bytes_per_tok = 28 × 2 × 8 × 128 × 2 = 114 688 bytes/token
kv_budget     = (24 GB × 0.92) − (4.2B × 2 B) − 1.6 GB = 12.08 GB
kv_per_seq    = 114 688 × 4096 = 469.76 MB
max_seqs      = floor(12 080 MB / 469.76 MB) = 25
```

**Check against log**: At prompt_len=3584, bs=24 sits at kv_cache_util=0.93 with 0 preemptions; bs=32 hits kv_cache_util=0.97 with 7 preemptions. This is exactly what the ceiling of 25 predicts — no issues on first try.

**Conclusion**: Predicted ceiling of 25 sequences confirmed by the batch-size grid in the log.

---

### B2: Throughput anomaly

**Hypothesis**: Throughput should scale linearly with batch size; a break indicates resource saturation.

**Calculation**: Computed scaling ratios across all six long-prompt rows. Found throughput peaks at bs=24 (1607.4 tok/s, ratio 1.226 vs expected 1.500) then *drops* at bs=32 (ratio 0.861 — below 1). Linear extrapolation from bs=24:
```
bs=32: expected 2143.2 tok/s, actual 1384.0 => −35.4% / wall +16.1%
bs=48: expected 3214.8 tok/s, actual 1298.5 => −59.6% / wall +23.8%
```

**Check against log**: The kv_cache_util, preempted_seqs, ttft_ms columns all agree — preemption begins exactly where the KV ceiling is crossed. No first-try issues; all columns corroborate the mechanism.

**Conclusion**: KV-cache exhaustion at bs≥25 causes preemption; `max_num_seqs=24` is the concrete fix.

---

### B3: Auditing REPORT_v0 Section 2

**Hypothesis (formula identification)**: Tested two candidates — `gen_len×nr/wall` vs `(pl+gen_len)×nr/wall`. Formula B matched all tested rows to <0.02%; Formula A was off by −66.7% (short prompts) and −87.5% (long prompts) systematically, confirming the metric counts prompt tokens.

**Honest goodput (bs=24, pl=3584)**:
- Method 1: `512×24/61.16 = 200.92 tok/s`
- Method 2: `512×24/(61.16−0.5005) = 202.57 tok/s`
- Difference: 0.83% — agreed on first try.

**First-try issue with Method 2b (ITL-based)**: `24/0.09607 = 249.8 tok/s` diverged 23% from Methods 1 & 2. Root cause: p50 ITL only captures mid-flight decode steps; it ignores prefill scheduling overhead and tail latency. Methods 1 & 2 use total wall clock and are the reliable derivation. Method 2b is noted but flagged as an overestimate.

**Conclusion**: REPORT_v0 inflates throughput 8× for long-prompt rows and 3× for short-prompt rows by using `(pl+gl)×nr/wall` as its "throughput" metric. Both Section 2 conclusions are wrong: honest goodput is 201 tok/s at bs=24, not ~1600; and long prompts have *worse* honest goodput than short.

---

### B4: Production metric

**Conclusion**: `vllm:num_preemptions_total` (Prometheus counter, vLLM `/metrics` endpoint) is the single counter that directly confirms the preemption/KV-pressure mechanism. Expected to be near zero below concurrency=25 and to rise sharply above it.



---

## Part A -- A1: Build multilingual eval corpus -- 2026-09-05

### What the uploaded folder structure actually looked like

The workspace contained flores-main/ -- the GitHub repository clone from facebookresearch/flores (archived). This is the documentation and tooling repo only; it contains READMEs, the flores_move.py script used internally at Meta to stage data files, license files (LICENSE_CC-BY-SA, LICENSE_CC-BY-NC4.0), and subdirectories for older releases (previous_releases/flores101/, floresv1/) and auxiliary datasets (nllb_md/, nllb_seed/, ocr/, toxicity/).

CRITICAL FINDING: flores-main/ does NOT contain the actual parallel text data files. The data (flores200_dataset/ with dev/ and devtest/ subfolders) is distributed separately from the repo, as a .tar.gz archive. This mismatch between the uploaded repo clone and the expected data layout was the main friction point for this task.

### Friction log -- access issues and how they were resolved

Attempt 1 -- Search the workspace for .devtest / .dev files:
Get-ChildItem -Recurse -Include returned no results. Confirmed: the uploaded flores-main/ is the repo documentation only, not the data.

Attempt 2 -- HuggingFace datasets library (openlanguagedata/flores_plus):
Installed datasets + pandas into the venv. Attempted load_dataset('openlanguagedata/flores_plus', name='eng_Latn', split='devtest').
Error: DatasetNotFoundError: Dataset 'openlanguagedata/flores_plus' is a gated dataset on the Hub. You must be authenticated to access it.
This is the current legitimate OLDI-maintained successor, but it requires a HuggingFace account and approved access request.

Attempt 3 -- HuggingFace mirror (Muennighoff/flores200):
Attempted load_dataset('Muennighoff/flores200', ...).
Error: RuntimeError: Dataset scripts are no longer supported, but found flores200.py
The datasets library (v5.x) dropped support for custom dataset scripts; this mirror is incompatible.

Attempt 4 -- HuggingFace (facebook/flores):
Attempted load_dataset('facebook/flores', ...).
Error: DatasetNotFoundError: Dataset 'facebook/flores' is a gated dataset on the Hub.
Same authentication barrier.

Resolution -- Direct CDN download:
The official FLORES-200 GitHub README (flores-main/flores200/README.md) references a direct download via tinyurl.com/flores200dataset which redirects to https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz. This is Meta's own CDN; it is publicly accessible with no authentication. Downloaded the archive (~25.6 MB), extracted to flores200_dataset/.

### Actual structure of flores200_dataset/ (post-extraction)

  flores200_dataset/
    README                     -- 8 lines; states sentences are in same order per language
    dev/                       -- 997 sentences per language, 204 language files
    devtest/                   -- 1012 sentences per language, 204 language files
    metadata_dev.tsv           -- sentence-level metadata for dev split
    metadata_devtest.tsv       -- sentence-level metadata for devtest split

File naming: <lang_code>.<split> (e.g., eng_Latn.devtest). This exactly matches the expected FLORES-200 layout. No deviations.

All 4 required language files confirmed present in devtest/:
  eng_Latn.devtest (133,108 bytes)
  hin_Deva.devtest (338,106 bytes)
  tam_Taml.devtest (422,653 bytes)
  kan_Knda.devtest (376,492 bytes)

### Alignment verification

All 4 files: 1012 lines each. Confirmed by Python line-count before preprocessing. No truncation or padding needed.

### Encoding observations

- All 4 files decoded cleanly as UTF-8 (no BOM, no encoding errors).
- NFC diff was zero for all 4 languages -- the source files were already in NFC form. The NFC normalisation step was a no-op on this data, but is applied unconditionally as a defensive measure.
- Zero-width spaces (U+200B) detected in Hindi and Kannada text via Python repr() output. These are preserved in the output rather than silently removed -- their presence is documented in CORPUS.md.

### Output

All 4 corpus files written to partA/corpus/, 1012 lines each, verified aligned on read-back.

  Language      Sentences      Words      Chars   Bytes(UTF-8)
  ------------------------------------------------------------
  eng_Latn          1,012     21,901    131,966        133,107
  hin_Deva          1,012     25,643    131,180        338,450
  tam_Taml          1,012     16,775    154,131        422,646
  kan_Knda          1,012     16,100    138,027        376,352

## A2 —## A2 — Baseline: Original v0 implementation

### Method

Ran the original `fertility.py` without modifications on the four
languages in `partA/corpus/` using the GPT-2 tokenizer.

### Results

| Language | Fertility (tokens/word) | Tokens/character |
|---|---:|---:|
| English | 1.29 | 0.214 |
| Hindi | 7.87 | 1.529 |
| Kannada | 22.57 | 2.660 |
| Tamil | 25.13 | 2.724 |

Using English as the baseline, the reported fertility ratios were:

- Hindi: 6.11x
- Kannada: 17.53x
- Tamil: 19.52x

These values are used as the reference for the subsequent A2
experiments.


 Experiment 1: Effect of lowercasing

### Hypothesis
The `line.lower()` operation may alter the fertility measurement,
especially for languages with case distinctions.

### Method
Ran the original v0 implementation on the four-language corpus,
then repeated the experiment with only `line.lower()` disabled.

Tokenizer: GPT-2

### Results

| Language | With lower() | Without lower() |
|---|---:|---:|
| English | 1.29 | 1.24 |
| Hindi | 7.87 | 7.86 |
| Kannada | 22.57 | 22.57 |
| Tamil | 25.13 | 25.13 |

### Observation
Lowercasing changed English fertility from 1.29 to 1.24.
Hindi changed only slightly, while Kannada and Tamil were unchanged.

### Conclusion
`lower()` has a measurable effect on the English result, but it is
not by itself evidence of an implementation bug. It is a preprocessing
choice whose effect should be considered when interpreting the benchmark.

Experiment 2: Word splitting

### Hypothesis
`line.split(" ")` may incorrectly count words when the corpus contains
multiple or non-standard whitespace characters.

### Method
Compared the original implementation using:

    words = line.split(" ")

against a version using:

    words = line.split()

Only this line was changed. The same four-language corpus and GPT-2
tokenizer were used.

### Results

| Language | Original `split(" ")` | `split()` |
|---|---:|---:|
| English | 1.29 | 1.29 |
| Hindi | 7.87 | 7.87 |
| Kannada | 22.57 | 23.02 |
| Tamil | 25.13 | 25.25 |

Tokens/character remained unchanged because only word counting was
modified.

### Conclusion
The original literal-space split produces different fertility values
for Kannada and Tamil on the corpus. Therefore, the word-counting
implementation is sensitive to whitespace and is a confirmed issue.
Using Python's general `split()` avoids empty fields from repeated
whitespace and handles other whitespace separators.

Experiment 3: Character counting

### Hypothesis
`len(line)` counts Unicode code points rather than user-perceived
characters. This may affect the tokens/character metric for Indic scripts.

### Method
Compared the original:

    chars = len(line)

with a grapheme-cluster count using the `regex` package:

    chars = len(regex.findall(r"\X", line))

Only the character-counting operation was changed. The same corpus,
GPT-2 tokenizer, lowercasing, and word-counting logic were retained.

### Results

| Language | Code-point count | Grapheme-cluster count |
|---|---:|---:|
| English | 0.214 | 0.214 |
| Hindi | 1.529 | 2.341 |
| Kannada | 2.660 | 4.066 |
| Tamil | 2.724 | 4.215 |

### Observation
The fertility values were unchanged because word counting was not
modified. However, tokens/character changed substantially for the
Indic languages.

### Conclusion
The v0 implementation defines "character" as a Unicode code point.
This produces substantially different tokens/character values from
grapheme-cluster counting for Hindi, Kannada, and Tamil. The benchmark
therefore needs an explicit definition of the character unit being
measured.

 Experiment 4: Per-line averaging vs corpus-level aggregation

### Hypothesis
The v0 implementation averages fertility ratios calculated separately
for each line. This may produce a different result from calculating
fertility from the total number of tokens and total number of words in
the corpus.

### Method
The original implementation was compared with a version that accumulates
total tokens, total words, and total characters across all lines and
then calculates the ratios from those totals.

Only the aggregation method was changed. The same four-language corpus,
GPT-2 tokenizer, lowercasing, and word splitting were retained.

### Results

| Language | Per-line average | Corpus-level ratio |
|---|---:|---:|
| English | 1.29 | 1.28 |
| Hindi | 7.87 | 7.82 |
| Kannada | 22.57 | 22.30 |
| Tamil | 25.13 | 24.90 |

### Conclusion
The two aggregation methods produce different results on the corpus.
The v0 implementation therefore gives each line equal weight rather
than weighting lines according to their number of words. This is a
statistical/aggregation issue that can affect the reported fertility.

Experiment 5: Random seed

### Hypothesis

The `random.seed(1337)` statement may be unnecessary because the
benchmark does not appear to use the `random` module elsewhere.

### Method

Compared the original implementation containing:

    random.seed(1337)

with a version where only this statement was disabled.

The same four-language corpus and GPT-2 tokenizer were used.

### Results

| Language | Original | Without random seed |
|---|---:|---:|
| English | 1.29 | 1.29 |
| Hindi | 7.87 | 7.87 |
| Kannada | 22.57 | 22.57 |
| Tamil | 25.13 | 25.13 |

The cross-language fertility ratios were also identical.

### Conclusion

Removing the random seed produced exactly the same results. The
`random` module is not otherwise used by the current benchmark, so
`random.seed(1337)` has no effect on the reported measurements. It is
therefore redundant/dead code rather than a functional bug.


this
multilingual comparison.

The original `fertility.py` was preserved unchanged so that all
corrections remain traceable to the original v0 implementation.