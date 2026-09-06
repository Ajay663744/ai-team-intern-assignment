# A4 — Tokenizer Audit Recommendation Memo

## Corrected headline

The original GPT-2 benchmark substantially overstates the apparent
Indic-language tokenizer penalty when compared with a multilingual
tokenizer.

On the A1 corpus, GPT-2 produces 6.12x, 17.85x and 19.60x the English
tokens/word for Hindi, Kannada and Tamil respectively. With XLM-RoBERTa,
the corresponding ratios fall to 1.05x, 1.81x and 1.73x.

The choice of tokenizer therefore materially changes the estimated
multilingual serving cost. For routing and capacity planning, the
relevant operational quantity is the number of tokenizer input tokens
processed per request.

## Routing recommendation

Do not route or capacity-plan multilingual traffic using the original
GPT-2 fertility ratios unless GPT-2 is actually the production
tokenizer.

For a multilingual deployment, use measurements from the tokenizer
actually deployed. Tokens per whitespace word should be the primary
offline cost-comparison metric because it provides an intuitive
language-level comparison of tokenization efficiency. Tokens per
grapheme
should be retained as a cross-script diagnostic because whitespace
words are not necessarily comparable across writing systems.

For actual production capacity and routing decisions, monitor the
number of tokenizer input tokens per request, since model context and
inference workload are determined by tokenizer tokens rather than
characters or bytes.

Based on this audit, an Indic/multilingual tokenizer such as XLM-R shows
far more balanced tokenization across the tested languages than GPT-2.
However, this experiment alone does not establish that XLM-R is the
best production tokenizer.

## Biggest caveat

The A1 corpus contains only English, Hindi, Kannada and Tamil, and its
sample size and domain limit generalization. The results do not
establish tokenization behavior for Telugu, Malayalam, Bengali,
Marathi or other languages, nor do they establish production traffic
distributions.

The corpus also does not represent every possible domain or text
style. Routing decisions should therefore be validated against
representative production-like traffic before deployment.

## Production metric

Monitor **input tokens per request by language and tokenizer** in
production.

This directly connects the audit to inference cost and capacity.
Track the distribution, especially p50 and p95, rather than only the
mean. Compare production observations against the offline estimates
and investigate if observed token counts materially exceed the expected
range.

The production tokenizer and actual token count per request should be
treated as the source of truth for capacity planning rather than the
original GPT-2 fertility numbers.