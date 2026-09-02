# Position: what is ours, and what is already known

Written after a literature check, before the claims harden. Two halves of the
argument are individually well established and must be cited, not claimed.

## Already known (cite, do not claim)

**Adam is not rotation-equivariant.** SGD's update is the gradient, which is
equivariant under an orthogonal change of basis; Adam's elementwise second
moment is not, so orthogonally-related parameterisations give genuinely
different trajectories. Established generally -- e.g. *Understanding Adam
Requires Better Rotation Dependent Assumptions* (NeurIPS 2025), *The Loss Does
Not See the Basis, but Adam Does*, and the norm-geometry view in *Old
Optimizer, New Norm* (Bernstein & Newhouse), which identifies Adam with
steepest descent under the elementwise max norm.

**LoRA has a gauge freedom.** `(A, B) -> (QA, BQ^T)` for orthogonal Q -- and
more generally `(RA, BR^{-1})` for invertible R -- leaves `BA` unchanged. This
is standard, and three recent lines act on it:

| work | what it does with the gauge |
|---|---|
| **LoRA-RITE** (ICLR 2025) | makes the *optimizer* invariant to it, with a transformation-invariant preconditioner. Gemma-7B GSM8K 48.37 -> 55.50. |
| **Balanced LoRA** | projects onto the balanced manifold `A^T A = B B^T` after each step, to improve conditioning. Standard initialisation; no frame analysis. |
| **FedRot-LoRA** | treats rotational misalignment as aggregation *noise* to be removed by Procrustes alignment. |

Every one of them treats the gauge as a **problem to be eliminated** -- from the
optimizer, from the iterate, or from the aggregation.

## Ours

We invert the framing: the gauge frame is not a defect to remove but a **free,
prescribable coordinate of the initialisation**, and the initialisation
literature is already setting it -- by accident.

1. **The frame is the part of an initialisation that only an adaptive optimizer
   can see.** SGD's *entire trajectory* is exactly gauge-covariant (with
   momentum, decay and global-norm clipping), so under SGD the frame is not
   underdetermined -- it is *vacuous*. Under AdamW it is a real degree of
   freedom. Which part of an initialisation is visible is decided by the
   optimizer's norm geometry.

2. **A coordinate that says how to set it.**
   `Lambda_1 = ||G A^T||_1^2 / (d_out r ||G A^T||_F^2)` in `(0, 1]` is the exact
   ratio of AdamW's first-order descent rate to SGD's -- the dual norm of the
   `l_inf` geometry over the dual norm of the `l2` geometry. Schur-Horn pins its
   reachable range at fixed invariants, so the frame's value is bounded by the
   invariants but not determined by them.

3. **The zoo varies it by 4.3x without knowing.** Measured with no training on
   23 published initializer configurations: data-aware methods sit low
   (LoRA-One 0.086, gradsub 0.152, PiSSA 0.273, EVA 0.285), frame-based ones
   high (BiMI 0.328, flat-diagonal 0.356, ETF 0.357, NoRA 0.366, Kaiming
   0.366). The whole NoRA
   family sits at vanilla's value -- which is why we measure those conditions as
   mutually indistinguishable to within 1.2x the measurement null. Published
   comparisons between these methods are partly comparisons of an unreported
   coordinate.

4. **A cheap win, and a cheap one specifically.** Rotating an existing
   initialiser preserves `B A` to 1e-15, `P = s^2 A^T A` to 1e-15, and every
   invariant of the triple exactly: it is the same initialiser in different
   coordinates by every definition those papers use. It adds no parameters and
   no optimizer state. It does need one probe gradient -- but that is strictly
   less than EVA, PiSSA, gradient-subspace or LoRA-One already spend, and it
   buys `r(r-1)/2` numbers rather than an `r x d_in` subspace.

## The relationship to LoRA-RITE is synthesis, not competition

LoRA-RITE's 7-point GSM8K gain from making the optimizer gauge-invariant is
independent evidence that the frame carries real signal. Our account says
*why*, and asks whether most of it is available for free by choosing the frame
once at initialisation instead of changing the optimizer.
