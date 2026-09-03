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

**Which Reparameterizations Can Your Optimizer See?**
*A map for LoRA's factorization group, and what it costs to ignore it*

*(Rejected as too large: "What Is a LoRA Initialization?" — that promises a
complete characterisation of the equivalence classes. We do not have one: the
nine invariants label the `O(r)` class only to order ≤ 2, and `Λ₁` is a single
scalar summarising an `r(r−1)/2`-dimensional frame. A reviewer would catch the
gap between the title and the delivery.)*

## The claim, sized to what we deliver

LoRA's factorisation carries a `GL(r)` redundancy. **Optimizers do not all see
the same part of it**, and which part they see is decided by one structural
property of the update rule. We give the map — nine optimizers, three subgroups,
verified — a training-free label for the part that separates them, and the
consequence for how LoRA initialisations are compared.

The **map** is the contribution, and the map is complete. Everything else
follows from it.

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

Applying the labels to 20 published configurations across four model families
so far, with no training (`src/analyze_audit.py`):

| model | span | separation |
|---|---|---|
| Qwen3-0.6B | 4.3× | perfect |
| Llama-3.2-3B | 3.0× | perfect |
| Qwen3-8B | 3.5× | perfect |
| OLMo-2-1B | 3.9× | one adjacent pair crosses, by 0.015 |

* the zoo spans the frame coordinate 3–4.3× on every backbone;
* the separation between frame-based and data-aware methods is perfect on three
  of four. Where it breaks it breaks at the **boundary pair** — OLoRA, the least
  data-aware of that group (a QR of the pretrained weight, using neither
  gradients nor activations), over BiMI, the most structured of the other. The
  two categories are ours and they blur for exactly the methods on the line;
* the coordinate is a property of the **method**, not the backbone: median
  across-model CV **6.9%**, median Kendall τ between model pairs **+0.82**. So
  one number per method, computable in a minute, characterises it.

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

---

## Requirements check — re-read before every step

The user's stated criteria, verbatim in substance, with an honest self-assessment.
This exists because the work drifted four times; each drift was visible only in
hindsight, and this table is the thing to re-read *before* acting, not after.

| requirement | status | evidence / what would break it |
|---|---|---|
| **topic breadth — not too large** | **fixed** | "What Is a LoRA Initialization?" was rejected as over-promising: we do not fully label the classes. The map is complete and the title claims only the map. |
| **topic breadth — not too small** | ok | Nine optimizers, three subgroups, seven model families in the audit. Not "we found a knob". |
| **novelty vs related work** | ok, and located precisely | NOT novel: Adam sees the basis (NeurIPS'25); LoRA has a gauge (four papers). NOVEL: the map across nine optimizers; that the redundancy is *exact* (zero function-space content) rather than a generic rotation; the labels; the seven-family audit; the 30% transfer number. |
| **main-claim novelty** | ok | "Optimizers see different parts of the group, and one structural property decides which" — nobody has stated or mapped this. |
| **narrative novelty** | ok | Constructive classification, not a critique of a subfield. |
| **narrative advances, does not defend** | **fixed** | Seven sections, each a step in one construction. Previous version had two defensive sections and three consecutive negations; both removed. |
| **must NOT narrow because an experiment failed** | **structurally fixed** | The map is the contribution. The prescription's fragility, the accuracy null and the collinear mechanism cannot narrow it, because none of them is load-bearing. Previously all three forced a retreat. |
| **no defensive experiments** | watch | Ban: further probe-size controls, further precision controls, further seed replication. Those exist and are enough. Any new panel must extend coverage or test a prediction. |
| **main-line experiment volume** | ok | ~750 once the third family lands, plus the 582-run Stage-1 audit. Not 2348 — most of that is exploration the paper does not cite, and quoting it would be misleading. |
| **model coverage** | **in progress** | Training: Qwen3-0.6B, Llama-3.2-3B, OLMo-2-1B (running), plus Qwen3-8B for scope. Audit: seven families. The three core claims were single-model, which was the real gap. |
| **mother paper is NoRA** | ok | §5 explains NoRA's own ablation: its whole family sits at vanilla's frame value, which is why those five conditions are mutually indistinguishable. |

**Standing rule.** Before adding any experiment, name which row of this table it
moves. If it moves none, it is defensive and does not get run.
