"""Gate 0 for topic 01.

Numerically establishes the structural facts that the matched-control design
rests on, and verifies theorem T1 (first-step P-equivalence).

Run:  python 01-hidden-preconditioner/src/gate0_feasible_region.py
"""
import json, os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from common.pinit import (kaiming_A, A_from_spectrum, A_with_spectrum_and_diagonal,
                          set_gram_diagonal_, normalize_columns, spectrum_shape,
                          make_A, _rand_orth)
from common.pstats import p_stats

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gate0_feasible_region.json")
torch.set_default_dtype(torch.float64)
report = {}


def hdr(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# ---------------------------------------------------------------- F1
hdr("F1  Schur-Horn construction is exact "
    "(prescribed spectrum AND prescribed diagonal)")
g = torch.Generator().manual_seed(0)
res = []
for (r, d_in) in [(8, 64), (16, 896), (32, 512), (64, 2048)]:
    for shape, kw in [("flat", {}), ("linear", {}), ("geometric", dict(decay=0.7))]:
        lam = spectrum_shape(shape, r, trace=1.0, device="cpu", **kw)
        A = A_with_spectrum_and_diagonal(r, d_in, lam, 1.0 / d_in, g, "cpu")
        got_spec = torch.linalg.svdvals(A) ** 2
        spec_err = float((got_spec.sort().values - lam.sort().values).abs().max())
        diag_err = float((A.pow(2).sum(0) - 1.0 / d_in).abs().max()) * d_in  # relative
        res.append(dict(r=r, d_in=d_in, shape=shape,
                        spec_err=spec_err, diag_err=diag_err))
        print(f"  r={r:3d} d_in={d_in:5d} {shape:10s} "
              f"max|spec err|={spec_err:.3e}  max rel|diag err|={diag_err:.3e}")
report["F1_schur_horn_exactness"] = res
assert max(x["spec_err"] for x in res) < 1e-12
assert max(x["diag_err"] for x in res) < 1e-8   # relative to the mean diagonal
print("  -> spectrum and diagonal are BOTH prescribable to machine precision.")


# ---------------------------------------------------------------- F2
hdr("F2  Crosstalk MAGNITUDE is a deterministic function of (spectrum, diagonal);\n"
    "    only the crosstalk PATTERN is free.")
r, d_in = 16, 896
lam = spectrum_shape("linear", r, trace=1.0, device="cpu")
rows = []
Ps = []
for seed in range(6):
    g = torch.Generator().manual_seed(100 + seed)
    A = A_with_spectrum_and_diagonal(r, d_in, lam, 1.0 / d_in, g, "cpu")
    st = p_stats(A)
    Ps.append(A)
    rows.append(st["crosstalk"])
    print(f"  draw {seed}: crosstalk c(P)={st['crosstalk']:.12f}  "
          f"eff_rank={st['eff_rank']:.4f}  diag_var={st['diag_var']:.3e}")
pred = math.sqrt(1.0 - (d_in * (1.0 / d_in) ** 2) / float((lam ** 2).sum()))
print(f"  closed form  sqrt(1 - sum(d^2)/sum(lam^2)) = {pred:.12f}")
# but the *patterns* differ: compare P matrices of two draws
P0 = Ps[0].T @ Ps[0]; P1 = Ps[1].T @ Ps[1]
rel = float((P0 - P1).norm() / P0.norm())
print(f"  yet ||P_0 - P_1||_F / ||P_0||_F = {rel:.4f}  (patterns are unrelated)")
report["F2_crosstalk_determined"] = dict(
    crosstalks=rows, closed_form=pred,
    max_dev=max(abs(x - pred) for x in rows), pattern_rel_diff=rel)
assert max(abs(x - pred) for x in rows) < 1e-10
band_lo = math.sqrt(1.0 - r / d_in)
print(f"\n  Corollary: with a flat diagonal, c(P)^2 = 1 - sum(d^2)/sum(lam^2) and")
print(f"  sum(lam^2) >= (tr P)^2 / r, so for ANY rank-r P with r << d_in")
print(f"      c(P) in [ sqrt(1 - r/d_in), 1 ] = [{band_lo:.6f}, 1]  (r={r}, d_in={d_in}).")
print(f"  The pre-registered crosstalk statistic is therefore essentially PINNED in")
print(f"  the realistic regime and cannot serve as a causal knob.  Reported as a")
print(f"  Gate-0 kill of statistic 3.3.")
report["F2_crosstalk_band"] = dict(lo=band_lo, hi=1.0, r=r, d_in=d_in)
print("  -> matched (trace, diagonal, spectrum) forces matched crosstalk magnitude.")
print("     Experiment family B must therefore be a PATTERN experiment.")


# ---------------------------------------------------------------- F3
hdr("F3  What NoRA's column normalisation actually changes")
r, d_in = 16, 896
g = torch.Generator().manual_seed(7)
A0 = kaiming_A(r, d_in, g, "cpu")
variants = {
    "kaiming (vanilla LoRA)": A0,
    "nora  (trace-matched)": make_A("nora", r, d_in, g, "cpu", ref_A=A0),
    "nora_unit (literal)": make_A("nora_unit", r, d_in, g, "cpu", ref_A=A0),
    "kaimingspec_flatdiag": make_A("kaimingspec_flatdiag", r, d_in, g, "cpu", ref_A=A0),
    "flatspec_flatdiag": make_A("flatspec_flatdiag", r, d_in, g, "cpu", ref_A=A0),
}
tab = {}
print(f"  {'variant':24s} {'tr P':>10s} {'diag_imb':>10s} {'crosstalk':>10s} "
      f"{'eff_rank':>9s} {'spec_max/min':>13s}")
for k, A in variants.items():
    st = p_stats(A)
    tab[k] = st
    print(f"  {k:24s} {st['tr_P']:10.5f} {st['diag_imbalance']:10.3e} "
          f"{st['crosstalk']:10.6f} {st['eff_rank']:9.4f} "
          f"{st['spec_max']/max(st['spec_min'],1e-30):13.2f}")
report["F3_variant_pstats"] = {k: {kk: vv for kk, vv in v.items()
                                   if kk != 'spec_top4'} for k, v in tab.items()}
print("\n  NOTE: 'nora' (trace-matched) changes the SPECTRUM as well as the diagonal;")
print("        it is therefore NOT a pure diagonal intervention.  The control")
print("        'kaimingspec_flatdiag' is: same trace, same spectrum, flat diagonal.")


# ---------------------------------------------------------------- F4
hdr("F4  Theorem T1.  The P0-equivalence class is EXACTLY the left-O(r) orbit.")
print("""  If A1, A2 in R^{r x d_in} have rank r and A1^T A1 = A2^T A2, then the polar
  decomposition A_i = S_i V_i^T (S_i = (A_i A_i^T)^{1/2}) is unique, so
  A2 = Q A1 with Q = S_2 S_1^{-1} ... orthogonal.  I.e. 'same hidden
  preconditioner' <=> 'same adapter up to an O(r) factor gauge'.""")
d_out, d_in, r = 32, 128, 8
g = torch.Generator().manual_seed(3)
A = kaiming_A(r, d_in, g, "cpu")
Q = _rand_orth(r, r, g, "cpu")
A2 = Q @ A
# recover Q numerically from the Grams alone, to show the claim is constructive
Qhat = A2 @ torch.linalg.pinv(A)      # unique since rank(A) = r
print(f"  ||A2^T A2 - A^T A||_F     = {float((A2.T@A2 - A.T@A).norm()):.3e}")
print(f"  ||A2 - A||_F              = {float((A2-A).norm()):.3e}   (different A)")
print(f"  ||Qhat^T Qhat - I||_F     = {float((Qhat.T@Qhat - torch.eye(r)).norm()):.3e}")
print(f"  ||A2 - Qhat A||_F         = {float((A2 - Qhat@A).norm()):.3e}")
report["F4_orbit"] = dict(
    gram_diff=float((A2.T@A2 - A.T@A).norm()),
    A_diff=float((A2-A).norm()),
    Q_orth_err=float((Qhat.T@Qhat - torch.eye(r)).norm()),
    reconstruct_err=float((A2 - Qhat@A).norm()))

# first-step formula
G = torch.randn(d_out, d_in, generator=g)
eta, s = 0.1, 2.0
B = torch.zeros(d_out, r)
gB = s * G @ A.T
gA = s * B.T @ G
dW1 = s * (B - eta * gB) @ (A - eta * gA)
pred = -eta * (s ** 2) * (G @ (A.T @ A))
print(f"  first step: ||dW - (-eta G P)||_F/||dW||_F = "
      f"{float((dW1-pred).norm()/dW1.norm()):.3e};  max|dL/dA|_{{B=0}} = {float(gA.abs().max()):.1e}")
report["F4_first_step"] = dict(dW_vs_formula=float((dW1-pred).norm()/dW1.norm()))


# ---------------------------------------------------------------- F5
hdr("F5  Under plain SGD the left-O(r) gauge is preserved EXACTLY, for all time.\n"
    "    => P0 is a COMPLETE descriptor of the initialisation, not just a\n"
    "       first-order one.  Method identity is unidentifiable under SGD.")

def sgd_traj(A0, steps, eta, s, Gs, wd=0.0):
    Ai = A0.clone(); B = torch.zeros(Gs[0].shape[0], A0.shape[0], dtype=A0.dtype)
    out = []
    for t in range(steps):
        G = Gs[t]
        gB = s * G @ Ai.T
        gA = s * B.T @ G
        B = B - eta * gB
        Ai = Ai - eta * gA
        out.append(s * B @ Ai)
    return out

def adam_traj(A0, steps, lr, s, Gs, b1=0.9, b2=0.999, eps=1e-8):
    Ai = A0.clone(); B = torch.zeros(Gs[0].shape[0], A0.shape[0], dtype=A0.dtype)
    mA = torch.zeros_like(Ai); vA = torch.zeros_like(Ai)
    mB = torch.zeros_like(B);  vB = torch.zeros_like(B)
    out = []
    for t in range(1, steps + 1):
        G = Gs[t - 1]
        gB = s * G @ Ai.T
        gA = s * B.T @ G
        for (p, gp, m, v) in ((B, gB, mB, vB), (Ai, gA, mA, vA)):
            m.mul_(b1).add_(gp, alpha=1 - b1)
            v.mul_(b2).addcmul_(gp, gp, value=1 - b2)
            p -= lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
        out.append(s * B @ Ai)
    return out

Gs = [torch.randn(d_out, d_in, generator=g) for _ in range(30)]
# a genuinely different P0 (independent draw) as the scale reference
A3 = kaiming_A(r, d_in, torch.Generator().manual_seed(99), "cpu")

def reldiff(t1, t2):
    return [float((a - b).norm() / (a.norm() + 1e-30)) for a, b in zip(t1, t2)]

sgd_rows = {}
for eta in [0.3, 0.1, 0.03]:
    same_P = reldiff(sgd_traj(A, 30, eta, s, Gs), sgd_traj(A2, 30, eta, s, Gs))
    diff_P = reldiff(sgd_traj(A, 30, eta, s, Gs), sgd_traj(A3, 30, eta, s, Gs))
    sgd_rows[str(eta)] = dict(same_P=same_P, diff_P=diff_P)
    print(f"  SGD  eta={eta:<5}  same-P0 pair rel.diff @steps 1/5/15/30: "
          f"{same_P[0]:.1e} {same_P[4]:.1e} {same_P[14]:.1e} {same_P[29]:.1e}"
          f"   |  different-P0 pair: {diff_P[29]:.2f}")
report["F5_sgd"] = sgd_rows
assert max(sgd_rows["0.3"]["same_P"]) < 1e-12
print("  -> identical to machine precision for 30 steps at every LR.")


# ---------------------------------------------------------------- F6
hdr("F6  AdamW BREAKS the gauge.  Two initialisations with *identical* P0 give\n"
    "    different trajectories under Adam, because the coordinatewise\n"
    "    normalisation m/sqrt(v) is not equivariant under A -> QA, B -> BQ^T.")
adam_rows = {}
for lr in [3e-3, 1e-3, 3e-4]:
    same_P = reldiff(adam_traj(A, 30, lr, s, Gs), adam_traj(A2, 30, lr, s, Gs))
    diff_P = reldiff(adam_traj(A, 30, lr, s, Gs), adam_traj(A3, 30, lr, s, Gs))
    adam_rows[str(lr)] = dict(same_P=same_P, diff_P=diff_P)
    print(f"  Adam lr={lr:<7} same-P0 pair rel.diff @steps 1/5/15/30: "
          f"{same_P[0]:.1e} {same_P[4]:.2f} {same_P[14]:.2f} {same_P[29]:.2f}"
          f"   |  different-P0 pair @30: {diff_P[29]:.2f}")
report["F6_adam"] = adam_rows
print("""
  INTERPRETATION.  Under SGD, 'which initialisation method' is *unidentifiable*
  given P0: the entire trajectory of the merged update depends on A0 only
  through P0 = s^2 A0^T A0.  Under Adam it is identifiable, and the size of the
  same-P0 divergence is a direct measurement of how much of the widely reported
  'LoRA initialisation matters' effect is really 'Adam is not covariant under
  the adapter-factor gauge' rather than a property of the preconditioner.
  This gives A1 a clean two-way decomposition of any initialisation effect:
        total  =  P0 channel  (visible under SGD)
                + gauge-representative channel  (Adam-only, P0-invisible).""")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(report, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
