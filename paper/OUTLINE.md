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

Applying the labels to 20 published configurations across **seven model
families**, with no training (`src/analyze_audit.py`):

| model | span | separation |
|---|---|---|
| Qwen3-0.6B | 4.3× | perfect |
| Qwen3-1.7B | 4.5× | one pair crosses, by 0.0014 |
| Gemma-2-2B | 5.1× | perfect |
| Llama-3.2-3B | 3.0× | perfect |
| OLMo-2-1B | 3.9× | one pair crosses, by 0.0147 |
| Mistral-7B | 2.9× | perfect |
| Qwen3-8B | 3.5× | perfect |

* the zoo spans the frame coordinate **2.9–5.1×** on every backbone;
* the separation between frame-based and data-aware methods is **perfect on five
  of seven**, and in both exceptions the violation is the *same single adjacent
  pair* — OLoRA over BiMI, by 0.0014 and 0.0147. That is the boundary pair by
  construction: OLoRA is the least data-aware of its group (a QR of the
  pretrained weight, using neither gradients nor activations) and BiMI the most
  structured of the other. The two categories are ours and they blur for exactly
  the methods on the line between them;
* the coordinate is a property of the **method**, not the backbone: median
  across-model CV **10.3%**, median Kendall τ over all **21 model pairs**
  **+0.82**. One number per method, computable in a minute, characterises it —
  across four vendors, four tokenizers and a 13× range in width.

## 5b. The map holds on a second family, with one honest wrinkle

The optimizer-to-class map (§3) was demonstrated in training only on Qwen3-0.6B.
Repeated on OLMo-2-1B, each optimizer judged against **its own** floor measured
at **its own** tuned learning rate (signed permutations, exactly invariant for
every optimizer here):

| optimizer | tuned lr | frame spread | own floor | ratio | class |
|---|---|---|---|---|---|
| SGD | 0.1 | 0.00010 | — (spread *is* the floor) | 1× | `O(r)` (blind) |
| Muon | 3e-4 | 0.000058 | 0.000027 | 2.2× | `O(r)` (blind) |
| AdamW | 3e-4 | 0.00165 | 0.00016 (n=2) | **10.3×** | frame (sees it) |
| Lion | 1e-4 | 0.00388 | 0.000039 | 99× | frame (sees it) |

The split is unambiguous: the blind optimizers sit at **1–2.2×** their own floor,
the sighted ones at **10–99×**. AdamW's floor is its own signed-permutation
spread (0.00016), *not* SGD's (2.9e-8) — the two differ by ~5500× here, and
dividing by SGD's would have reported a spurious 63000× where the honest number
is 10.3×. This is the second standing rule paying for itself.

matprec is the wrinkle, and it resolves in our favour. At its tuned lr (0.01) it
reads 6.2× its floor — apparently frame-sensitive, contradicting the analytic
2.2e-16 covariance. But the per-lr spread is:

```
lr = 0.001   0.000000      lr = 0.01   0.002163  (tuned optimum)
lr = 0.003   0.000002      lr = 0.03   0.005037  (diverging)
```

matprec is exactly covariant at low lr (five to six decimals) and its spread
grows *monotonically* with lr — the signature of stability-edge float chaos, not
of frame sensitivity. Its optimum happens to sit where that chaos begins. So the
map holds; the naive ratio at the tuned point is a measurement artefact of an
exactly-covariant method run near its stability limit, and we report it that way.

## 6. The classes are real, not bookkeeping

Moving *within* a class changes nothing; moving *between* them does.

One gauge orbit of the vanilla draw, every logged preconditioner statistic equal
to 1.5e-8, only the frame moving. **Three model families, all in fp32, each
carrying its own floor** — SGD is exactly gauge-covariant, so wherever it appears
its spread *is* that panel's reproducibility floor, measured rather than
imported:

| family | kaiming | frame0 | frame1 | AdamW spread | AdamW own floor | ratio | SGD covariance |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.44551 | **0.44329** | 0.44506 | 0.00222 | 0.00021 | 11× | 1.0e-05 |
| Llama-3.2-3B | 0.51783 | **0.51615** | 0.51853 | 0.00239 | 0.00040 | 6× | 2.6e-04 |
| OLMo-2-1B | 0.54168 | **0.54004** | 0.54164 | 0.00165 | 0.00016 | 10× | 2.6e-08 |

`frame0` is best on all three, with the same ordering. The ratio is against
AdamW's **own** reduction-order floor (its signed-permutation spread), which is
the like-for-like comparison and lands at a consistent **6–11×** on all three
families. The last column is a separate check: under SGD, which is exactly
gauge-covariant, the four gauge-related draws agree to between five and eight
decimal places — on OLMo to **eight** (2.6e-08) — confirming they carry
identical preconditioner content and that the AdamW spread is a pure frame
effect. (Reading the ratio against that covariance floor instead, as an earlier
draft did, gave a spurious 200×/9×/63000× — the SGD floor swings four orders
across families and is not AdamW's floor. Qwen's and Llama's own floors come from
`gradsub@frame0`, an exact identity map under AdamW, and from the sister
`llama_map` panel; OLMo's from its AdamW signed-permutation seeds.)

> **Flag, under verification — see `01-hidden-preconditioner/FINDING_scale_attenuation.md`.**
> The Llama row's 6× is not robust. The clean replication (`llama_map`, a larger
> probe batch so the frames are constructed more accurately) gives 1.4× at the
> same lr, not 6× — part of `llama_fp32`'s spread was probe noise, not frame
> signal. Ordered by size the own-floor ratios read 0.6B 11×, 1B 10×, 3B 1.4–6×,
> 8B marginal: the margin over the floor **attenuates with scale** (AdamW's
> reduction-order floor grows, the frame signal does not). This does not touch the
> §3 map, which is the *relative* gap AdamW ≫ blind; the decisive check is the
> `llama_map` blind arm (running). Framing is the user's call; the number is
> flagged, not yet rewritten.

Five optimizers on the Qwen3-0.6B orbit, ordered by what the map in §3 predicts:

| optimizer | spread | vs floor |
|---|---|---|
| SGD | 0.00001 | 1× |
| matrix-preconditioned Adam | 0.00022 | 1.1× |
| Muon | 0.00026 | 1.3× |
| Lion | 0.00068 | 3.4× |
| **AdamW** | **0.00222** | **11×** |

**Precision is not neutral here, and we say so.** The same Llama ladder run with
bf16 matmuls gives an AdamW spread of 0.00104 against an SGD floor of 0.00090 —
a ratio of 1.2, i.e. nothing. bf16 raises the floor 3.5× *and* compresses the
conditions, burying the effect from both ends. Any panel carrying a frame claim
must be fp32 or carry its own floor.

**Structural control:** the effect is exactly zero at `r = 1`, where `O(1)` *is*
AdamW's own symmetry group and the class collapses, and grows through
`r = 4, 16, 64, 128`.

Rotating published initialisers within their own orbit, on **two families**,
with `BA`, `P` and all nine invariants preserved to 1e-15:

| | `@frame0` helps or neutral | `@frame1` hurts or neutral |
|---|---|---|
| Qwen3-0.6B | 5/6 | 6/6 |
| OLMo-2-1B | 6/6 | 4/6 |

**11 of 12** rotations toward the gradient-metric eigenframe help or do nothing.

Two floors are in play here and they measure different things, which matters for
reading the near-boundary cases:

* **SGD's spread** (1.1e-05 on Qwen, 2.6e-08 on OLMo) is how far two
  gauge-related runs drift under an *exactly covariant* optimizer;
* **an identity rotation under AdamW** (~2e-04) is how far two
  signed-permutation-equivalent runs drift under AdamW itself, which adds the
  reduction-order chaos its nonlinearity amplifies.

Judging an AdamW rotation needs the second. We have it for free: gradsub is
*already* at the `M_g` eigenframe, so `gradsub@frame0` is the identity map, and
its measured +0.00021 on Qwen **is** that floor rather than a result. That is
also the one cell keeping Qwen at 5/6 instead of 6/6.

## 7. The consequence: which initialiser is best depends on the optimizer

Because the class depends on the optimizer, so does the comparison. Running the
same six published initialisers under a frame-sighted optimizer (AdamW) and a
frame-blind one (Muon), on three model families, with every cell at an interior
optimum and each panel carrying its own SGD-measured floor:

| family | floor | pairs that reverse | reversals above 2× the floor on **both** |
|---|---|---|---|
| Qwen3-0.6B | 1.1e-05 | 1/15 | **0** |
| Llama-3.2-3B | 2.6e-04 | 6/15 | **4** |
| OLMo-2-1B | 2.6e-08 | 3/15 | **3** |

On Llama and OLMo — two independent families, different vendors, different
tokenizers, different pretraining corpora — **the same three pairs reverse**,
and together they are a complete reversal of one triple:

```
                    AdamW                          Muon
Llama-3.2-3B   gradsub < eva < pissa   ->   pissa < eva < gradsub
OLMo-2-1B      gradsub < eva < pissa   ->   pissa < eva < gradsub
```

Gradient-subspace initialisation is the best of the three under AdamW and the
**worst** under Muon, on both models, with every gap more than twice the panel's
own floor. So "which LoRA initialiser is better" is not a property of the
initialiser alone.

**It is model-dependent, and we say so.** Qwen3-0.6B shows zero reversals above
its floor; there the ordering is stable (τ = +0.87). What travels is not "the
ranking always reverses" but "the ranking is not safe to quote without naming
the optimizer" — on two of three families it does reverse, reproducibly and in
the same place.

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
| **model coverage** | **in progress** | Every main line now on three families {Qwen3-0.6B, OLMo-2-1B, Llama-3.2-3B}. The central map (§3) was the last on two; its third family (Llama-3.2-3B, five optimizers, true fp32) is running on the A100 pool — AdamW arm done at an interior optimum (lr 2e-4), blind/sighted optimizers filling in. Plus Qwen3-8B for scope: fp32 ladder + its own AdamW floor (three signed-permutation seeds) complete — the effect sits **at or below** that floor at 8B (ratio 0.2–1.2×), cleanly resolved only at 0.6B/1B; the frame signal is flat with scale and the floor grows toward the stability edge (see `01-hidden-preconditioner/FINDING_scale_attenuation.md`). Audit: seven families. |
| **mother paper is NoRA** | ok | §5 explains NoRA's own ablation: its whole family sits at vanilla's frame value, which is why those five conditions are mutually indistinguishable. |

**Standing rule.** Before adding any experiment, name which row of this table it
moves. If it moves none, it is defensive and does not get run.

**Second standing rule, learned the hard way.** A reproducibility floor is a
property of *a panel*, not of the project. It must be measured inside the panel
it is used on — by SGD, which is exactly gauge-covariant, or by a
signed-permutation control — never imported from another model or another
precision. Importing Qwen3-0.6B's fp32 floor of 2e-4 onto a bf16 Llama panel
turned a null (AdamW 0.00104 against that panel's own SGD floor of 0.00090, a
ratio of 1.2) into a reported "5.2x". Every panel that carries a claim must
contain its own floor.
