# Paper outline (ICLR form) — rewritten to build rather than defend

The previous version had two consecutive defensive sections ("Limits, stated"
and "What we are not claiming", the latter three consecutive negations). It got
that way by narrowing the claim every time an experiment failed: the
prescription weakened, so it was demoted; accuracy did not move, so it was cut;
the mechanism was collinear, so it was dropped. The result apologised for
itself.

The root cause is the framing. *"LoRA initialisation comparisons are
optimizer-specific"* is a **critique**, and a critique's value is defined by
someone else's error, so it is defensive by construction. The same data
supports a **constructive** claim, and that version does not narrow when an
experiment fails, because the classification is the contribution and a failed
prescription simply is not part of it.

---

## Title

**What Is a LoRA Initialization?**
*An equivalence-class answer, and why it depends on your optimizer*

## The claim, positively

An initialization is not the pair `(A, B)`. It is an **equivalence class**, and
*which* class depends on the optimizer that will consume it. We give the
classes, the invariants that label them, a training-free label for the part that
distinguishes them, and the map from optimizer to class.

Everything else in the paper is a consequence of that answer, not a caveat to it.

---

## 1. The question

LoRA writes `ΔW = s B A`, and the factorisation is not unique: `(SA, B S^{-1})`
is the same function for every invertible `S`. So the object a practitioner
chooses — "an initialisation" — is not a pair of matrices. It is whatever
survives the redundancy.

*What survives depends on who is looking.* That is the paper.

Four recent works (LoRA-RITE ICLR'25, Balanced LoRA, FedRot-LoRA, Riemannion
ICLR'26) treat the redundancy as a defect to remove. We ask instead what it
**is**, and answer with a classification.

## 2. The classes

`GL(r)` splits by polar decomposition into a rotation part `O(r)` and a scaling
part. That gives a chain of candidate equivalence relations:

```
signed permutations  ⊂  O(r)  ⊂  GL(r)
```

An initialisation's identity is its orbit under whichever of these the optimizer
respects. Under a fully invariant method it is the `GL(r)` class; under SGD or
Muon the `O(r)` class, i.e. the orbit `P₀ = AᵀA`; under AdamW the orbit **plus
the frame**, `r(r−1)/2` further coordinates.

## 3. The map from optimizer to class

Nine optimizers, float64, each at its own tuned learning rate, each normalised
by its own signed-permutation noise floor (`src/hierarchy.py`):

| | `O(r)`/floor | class it sees |
|---|---|---|
| SGD, SGD+momentum, Muon, matrix-preconditioned Adam | 0.3 – 2.3 | `O(r)` |
| AdamW, Lion, RMSprop, Adagrad, Adadelta | 8e11 – 2e13 | `O(r)` **plus the frame** |

**Twelve orders of magnitude**, and the line is neither adaptivity nor the norm:
it is whether the preconditioner is *diagonal in the coordinates the group acts
on*.

The map **predicts**. Lion's class was named before it was measured, from its
update rule alone, and it landed there — analytically (1.3e-3 against Muon's
4.4e-9) and in training (0.00068 against Muon's 0.00026).

It also settles a live disagreement: Riemannion (ICLR'26) motivates its
construction by asserting per-factor Muon is not invariant to "scalings **or**
rotations". Muon is exactly `O(r)`-invariant and not scaling-invariant — the two
halves differ by six orders of magnitude, and the classification says which is
which.

## 4. Labelling the classes without training

The `O(r)` class is labelled by the invariants of the metric triple
`(AAᵀ, AΣAᵀ, AC_gAᵀ)`; classical invariant theory gives nine at order ≤ 2, and
we verify they are `O(r)`-invariant to 4e-16.

The frame — the part only diagonal methods see — is labelled by
`Λ₁ = ‖GAᵀ‖₁² / (d_out·r·‖GAᵀ‖_F²)`, the exact ratio of AdamW's first-order
descent rate to SGD's. **One probe pass, no training.** Schur–Horn bounds its
reachable range given the invariants, so the two labels are not independent.

## 5. Where the literature sits

Applying the labels to 20 published configurations across **seven model
families**, with no training (`src/analyze_audit.py`):

* the zoo spans the frame coordinate 3–4.3×;
* the separation is perfect on every model — every frame-based method above
  every data-aware one;
* the ordering is stable across architectures, so the frame is a property of the
  **method**, not the backbone: one number per method, computable in a minute.

This is what the classification buys: a published initialiser now has a label,
and the labels are informative — they recover the data-aware/frame-based
distinction that the methods' own descriptions imply.

## 6. The classes are real, not bookkeeping

Moving *within* a class changes nothing; moving *between* them does.

One gauge orbit of the vanilla draw, every logged preconditioner statistic equal
to 1.5e-8, only the frame moving (`results/frame`, three model families):

| optimizer | spread over the orbit |
|---|---|
| SGD | 0.00001 |
| matrix-preconditioned Adam | 0.00022 |
| Muon | 0.00026 |
| Lion | 0.00068 |
| **AdamW** | **0.00222** |

The `O(r)`-blind methods sit at their reproducibility floor; the frame-sighted
ones do not. And the effect is **exactly zero at `r = 1`**, where `O(1)` *is*
the signed-permutation group and the class collapses — growing through
`r = 4, 16, 64, 128` as the quotient opens up.

Rotating published initialisers within their own orbit (`results/rot`, three
families) helps or is neutral 6/6 one way and hurts or is neutral 6/6 the other,
with `BA`, `P` and all nine invariants preserved to 1e-15.

## 7. The consequence for practice

Because the class depends on the optimizer, so does the comparison. Running the
same six published initialisers under a frame-sighted and a frame-blind
optimizer: the **ordering is preserved** (Kendall τ = +0.87), while the
**margins are not** — Kaiming's advantage over gradsub, EVA and PiSSA falls 31%,
29% and 37%.

So roughly a third of the margin between published LoRA initialisers is a
property of the optimizer, not of the method. The label in §4 says how much,
before any training.

## 8. Scope

Effects are 0.002–0.006 nats: large against the 2e-4 reproducibility floor and
against the gaps between published methods, and they do not move GSM8K
exact-match at n = 200. The classification is exact; the training-level
consequences are measured under SFT on three model families and four tasks.
Two pre-registered predictions of ours were falsified and one reading
over-called; the prediction files are in the repository unedited with outcomes
appended.

---

## Narrative shape

Each section is a step in one construction, not a defence of the previous one:

> **ask** what an initialisation is → **classify** (the group splits) →
> **map** optimizers to classes → **label** the classes without training →
> **locate** the literature in them → **verify** the classes are real →
> **derive** the consequence for comparison

Nothing in it depends on the prescription working, on accuracy moving, or on the
mechanism being resolved. Those were experiments; this is the answer to a
question.

## Evidence inventory

Only runs that carry a main-line claim. The project has 2348 trained runs in
total, but most are exploration and abandoned lines the paper does not cite;
quoting that as "experiment volume" would be misleading.

| § | claim | source | runs | coverage |
|---|---|---|---|---|
| 3 | optimizer → class map | `src/hierarchy.py` | analytic | 9 optimizers |
| 4 | the labels | `common/intrinsic.py` | verified to 4e-16 | — |
| 5 | where the literature sits | `second_order_*.json` | **0**, training-free | 7 families |
| 6 | the classes are real | `results/frame` + `llama3b` + `olmo` | ~215 | 3 families, 5 optimizers |
| 6 | class collapse at r = 1 | `results/rank` | 50 | r = 1…128 |
| 6 | rotation within orbit | `results/rot` + `llama_zoo` + `olmo_zoo` | ~180 | 3 families, 6 methods |
| 7 | margin transfer | Muon arms of the above | ~90 | 3 families |
| 8 | scope | `q8b`, `q8bprobe`, `frame_bf16`, `acc`, `long`, `task_*` | ~215 | 0.6B–8B, 4 tasks |

Main-line total once the third family lands: **~750**, plus the Stage-1 audit
(582) cited in §5.
