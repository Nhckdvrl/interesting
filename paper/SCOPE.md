# Scope: what the mother paper does, and what we need to match it

Written after reading NoRA (arXiv 2608.31036) end to end, because the question
"is this enough for the venue" should be answered from the actual bar and not
from intuition.

## What NoRA actually runs

| regime | model | data | benchmarks |
|---|---|---|---|
| pretraining | custom MLA / MHA stacks, L24-D1024 | SlimPajama | LAMBADA, WikiText, ARC-e/c, HellaSwag, PIQA, OpenBookQA, WinoGrande |
| SFT | **Llama-3.2-3B** | MetaMath, CodeFeedback | GSM8K, MATH500, HumanEval, MBPP, plus catastrophic forgetting (PEFT-Arena) |
| RL | **DeepSeek-R1-Distill-Qwen-1.5B** | DAPO-Math-17k | AIME24/25, MATH500, Minerva, AMC, HMMT |

Plus ablations on the normalisation dimension and the initialisation
distribution.

**The mother paper's largest model is 3B.** Parameter count is therefore not
where we are short -- our 8B panel already exceeds it. What NoRA has and we do
not is **breadth**: three regimes, three data sources, roughly fifteen
benchmarks, all reported as accuracy.

## Where we actually stand

| axis | NoRA | us, before this expansion |
|---|---|---|
| largest model | 3B | **8B** |
| model families | Llama, Qwen (distill), custom | Qwen3 only |
| SFT datasets | MetaMath, CodeFeedback | GSM8K only (Dolly in flight) |
| regimes | pretrain + SFT + RL | SFT only |
| metric | accuracy on ~15 benchmarks | eval loss (accuracy in flight) |
| runs | not stated; several tables | ~1700 |

So the gap is one model family, two datasets, and accuracy -- not scale.

## What is now queued to close it

* **Llama-3.2-3B** (`results/llama3b`, 19 runs) -- NoRA's own SFT model, and a
  genuinely different architecture family, tokenizer and gradient geometry.
  Tests whether the gauge result is about LoRA or about Qwen3. `meta-llama` is
  gated so this uses the ungated mirror of the same weights.
* **MetaMath and CodeFeedback** (`results/task_*`, 24 runs) -- NoRA's own two
  SFT datasets. Code is the interesting one: its gradient structure differs
  enough from maths that a coordinate defined through the gradient metric could
  behave differently, and if it does not, that is the result.
* **GSM8K exact-match** (`results/acc`, `results/long`) -- the metric anyone
  would act on. A 0.002 nat gap need not move it at all, and we report either
  way.
* **Dolly** (`results/dolly_frame`) -- a non-mathematical instruction task.

## What we are deliberately not doing, and why

**The RL regime.** NoRA runs GRPO-style RL on a 1.5B distilled model. That is a
real regime and its absence is a limitation we state rather than hide. The
reason is the standing compute constraint against large RL sweeps, and the fact
that the claim under test -- which reparameterisation symmetries an optimizer
respects -- is derived, not fitted, and does not depend on the regime. An RL
arm would be evidence that it transfers, not evidence that it holds.

**Pretraining from scratch.** Same reasoning, plus the pretraining arm of NoRA
is on custom small architectures, which would test a different question from
the one we are asking about adapters on pretrained models.
