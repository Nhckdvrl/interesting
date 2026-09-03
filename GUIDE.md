# Repository guide

Start here. `README.md` is the original topic registration and is deliberately
left unchanged; this file is the map of what has actually been built and where
the current main line is.

## The main line, in one paragraph

Three topics were registered from the Normalized LoRA (NoRA) mother paper.
Topics **01** and **02** converged and are now one line of work; topic **03** is
a scoped non-reproduction. The line is:

> LoRA's factorisation `ΔW = sBA` carries a `GL(r)` redundancy -- `(SA, BS⁻¹)`
> is the same function for every invertible `S`. **Optimizers do not all see the
> same part of it**, and which part they see is decided by one structural
> property of the update rule: whether the preconditioner is diagonal in the
> coordinates the group acts on. We give the map, a training-free label for the
> part that separates optimizers, and the consequence for how LoRA
> initialisations are compared.

Nine optimizers, each normalised by its own noise floor
(`01-.../src/hierarchy.py`):

```
   signed permutations  ⊂  O(r)  ⊂  GL(r)

   sees O(r) only:            SGD, SGD+momentum, Muon, matrix-preconditioned Adam
   sees O(r) AND the frame:   AdamW, Lion, RMSprop, Adagrad, Adadelta
```

Twelve orders of magnitude separate the two groups. The frame carries **zero
function-space information** -- `(A,B)` and `(QA,BQᵀ)` have the same function,
loss and gradient -- yet five of the nine respond to it, including every
optimizer in common use.

## Where to read, in order

| file | what it is |
|---|---|
| **`paper/OUTLINE.md`** | **the main line.** Eight sections, each naming the experiment that carries it, plus a requirements table to re-read before every step. **Start here.** |
| `paper/POSITION.md` | related work (four papers treat this gauge as a defect to remove) and scale calibration against NoRA and the nearest comparable paper |
| `paper/RESULTS.md` | every headline number, regenerated from the run files by `paper/make_results.py` |
| `paper/hierarchy.txt` | the optimizer→class map as the script prints it |
| `01-hidden-preconditioner/results/README.md` | index of ~30 result directories |
| `0*/STATUS.md` | per-topic records: what was run, what it showed, what was falsified |
| `01-.../PREDICTIONS_*.md`, `paper/8b_predictions.md` | four pre-registrations, predictions unedited, outcomes appended -- two were falsified |
| `paper/superseded/` | two earlier framings and why each was replaced |
| `README.md`, `0*/README.md` | the original registrations |

## Reproducing the numbers

```bash
.venv/bin/python paper/make_results.py > paper/RESULTS.md
```

Everything it prints is read from the committed `results/` directories; no
training is re-run.

## Code layout

```
common/            shared library — the only place with reusable machinery
  intrinsic.py       THE scientific object: M_x, its coordinates, exact constructions
  pinit.py           A-initialisers and the Schur–Horn (spectrum, diagonal) constructions
  initializers.py    faithful re-implementations of the published initializers
  gauge.py           exact function-preserving Transformer gauges (V/O and residual)
  lora.py            minimal LoRA with full control over A0, B0 and the base weight
  train.py           the training loop and its optimisation diagnostics
  pstats.py          statistics of P = A^T A
  data.py            SFT corpora, completion masking, fixed batch order
  evaluate.py        GSM8K exact-match scoring
  cluster.py         multi-host GPU scheduler (arch-pinned)

0*/src/            per-topic runners, panels and analyses — see each src/README.md
0*/results/        raw run files, one JSON per run, grouped by panel tag
0*/plots/          figures
paper/             the write-up and the scripts that generate its numbers
failed_topics/     the pre-registered rejection registry
```

## Compute

`common/cluster.py` schedules across the fleet and **pins each panel to one GPU
architecture**: the A100 and RTX PRO 6000 nodes differ by ≈4e-4 nats on an
identical configuration, which is above the 2.7e-4 measurement null, so a panel
whose conclusions live at that resolution must not be split across them.

```bash
.venv/bin/python common/cluster.py        # list free slots
```

Two virtualenvs, because the driver versions differ: `.venv` (cu130, RTX PRO
6000) and `.venv-a100` (cu124, A100).
