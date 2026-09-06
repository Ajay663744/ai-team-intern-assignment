# AI Usage

I used AI assistance during this assignment as a coding, debugging, and review aid. I remained responsible for checking the results, running the experiments locally, and deciding what evidence to include.

## Where AI helped

- Helped inspect and reason about the original `fertility.py` implementation and identify candidate issues to test.
- Helped design controlled experiments for:
  - whitespace splitting,
  - character/grapheme counting,
  - corpus-level aggregation,
  - lowercasing,
  - and the unused random seed.
- Helped implement the corrected fertility analysis and regression tests.
- Helped identify the need to count Unicode grapheme clusters with `regex` rather than treating Unicode code points as user-perceived characters.
- Helped structure the A3 comparison between GPT-2 and XLM-RoBERTa using multiple denominator definitions.
- Helped organize the chronological `NOTEBOOK.md` and format the A1–A4 sections.
- Helped reason through the Part B calculations, including KV-cache sizing, throughput scaling, and reverse-engineering the `reported_tok_s` column.
- Helped draft and revise the Part A4 and Part C recommendation memos.

## Where AI was misleading or needed correction

The AI-generated reasoning was not treated as evidence by itself. I repeatedly ran the proposed commands and tests locally and corrected conclusions when the measured results differed from assumptions.

One concrete example was the fertility audit. The initial focus was on the whitespace-splitting and aggregation issues, but the important conceptual problem with `chars = len(line)` became clear only after comparing the actual `tok/char` results. Unicode code-point counting understated the number of user-perceived characters for Indic scripts. Replacing it with Unicode grapheme-cluster counting materially changed the Hindi, Kannada, and Tamil `tok/grapheme` values.

Another example was the test for corpus-level aggregation. An initial test used lines with equal word counts, which could not distinguish mean-of-line-ratios from corpus-level `sum(tokens) / sum(words)`. I corrected the test to use unequal word counts (`"a b c d"` and `"e"`), where the two aggregation methods produce different results.

I also encountered an import/path issue after moving the A2 files into `partA/A2/`. Running the tests from the new directory initially produced:

```text
ModuleNotFoundError: No module named 'fertility_fixed'
```

I fixed the repository/test layout and then reran the suite successfully.

For Part B, I verified the calculations by executing `partB/_calc.py` locally rather than relying on AI arithmetic. The resulting calculations showed that the benchmark's `reported_tok_s` includes prompt plus generation tokens, which is why it differs substantially from generated-token goodput.

## Verification principle

For this assignment, AI suggestions were treated as hypotheses, not as evidence. Final claims were retained only after checking them against the provided files, controlled experiments, tests, or locally executed calculations.

The final repository therefore contains the code, tests, calculations, and written analysis needed to reproduce the main conclusions.
