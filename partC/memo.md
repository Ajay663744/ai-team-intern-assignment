# Part C — Decision Memo: Casual Multilingual Responses

## Recommendation

Choose **(a) SFT on synthetic "casualized" response pairs** over an
inference-time rewriter. SFT puts the desired behavior directly into the
production model and avoids adding a second model invocation and its
associated latency at serving time.

However, because the same fine-tuned checkpoint affects all six languages,
the launch decision must be language-gated. A single full-model SFT
checkpoint cannot cleanly "un-apply" the fine-tune to four languages at
inference time. If independent language holdback is required, use separate
checkpoints or language/cluster-specific LoRA adapters rather than one
jointly modified checkpoint.

## Assumptions

- One A100-80GB is available for two weeks and can support the SFT workload.
- Synthetic formal-to-casual pairs can be generated without an external API.
- The reviewer provides 10 h/week for two weeks = 20 reviewer-hours.
- Hindi and Kannada receive native-speaker review; the other four languages
  require automated pre-launch checks.
- The objective is style improvement without changing meaning or task
  correctness.

## Back-of-envelope arithmetic

### Data volume

Create **12,000 synthetic pairs**:

- 2,000 examples × 6 languages = 12,000 pairs.
- Hold out approximately **1,200 examples** (200 per language) for evaluation.

### Reviewer throughput

20 reviewer-hours = 20 × 60 = 1,200 minutes.

At approximately 2 minutes/example:

- 1,200 / 2 = **600 manually reviewed examples**.

Use this capacity primarily for Hindi and Kannada and difficult cases rather
than attempting to review the full training set.

### Compute / serving cost

SFT uses the A100 primarily during training and does not add a second model
invocation at serving time. A rewriter would add an additional inference
stage to every request, increasing serving latency and compute.

## Success metric

On the held-out evaluation set:

- **≥70%** of reviewed responses must be judged casual/conversational.
- **≥95%** must preserve the original meaning/task correctness.

The 70% threshold measures style improvement; the 95% threshold prevents
casualization from causing unacceptable semantic regressions.

## Kill criterion

Kill the SFT checkpoint if **any** of these conditions occur by the end of
Week 1:

- **Hindi or Kannada:** native-speaker review shows <60% casual acceptance
  or <95% meaning preservation.
- **Tamil, Telugu, Bengali, or Marathi:** because native review is
  unavailable, run multilingual NLI entailment between the original formal
  response and the casualized response. If **either-direction entailment is
  below 0.90 for more than 5% of examples in any language**, treat that
  language as a failure and do not ship the joint checkpoint. This NLI check
  is a weaker substitute for native review, not equivalent to it.
- If the checkpoint passes semantic checks but fails to reach the **70%
  casualness target**, do not spend the remaining GPU budget scaling the
  same approach; use the Day-1 evidence to revise the data or fall back to
  prompt engineering.

## Day-1 experiment

Create **200 evaluation prompts**:

- 50 Hindi
- 50 Kannada
- 25 Tamil
- 25 Telugu
- 25 Bengali
- 25 Marathi

Compare three blinded systems:

1. Current model.
2. Prompt-engineered casual response.
3. Small SFT pilot trained on a few hundred synthetic pairs.

Measure casual acceptance, meaning preservation, and task correctness. The
native reviewer evaluates Hindi/Kannada; automated NLI checks provide the
initial safety screen for the other four languages.

Continue with full SFT only if the pilot provides evidence that casualness
can improve without unacceptable semantic drift.

## Final decision

**At week 3, Hindi and Kannada ship the casual SFT behavior only if they pass
native-speaker validation; Tamil, Telugu, Bengali, and Marathi remain on the
original formal behavior unless they pass the automated semantic-safety
check and have a language-specific adapter/checkpoint that allows them to be
gated independently.**