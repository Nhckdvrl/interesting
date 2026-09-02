"""Stage 2, wave 1 -- a star design through the vanilla reference point.

Every point is an exact, independently-controlled intervention in the intrinsic
coordinates (S, D, rho).  The design is a star rather than a full grid so that
the first wave buys main effects and the reference reconstruction cheaply; the
interaction fill-in is wave 2, sited where wave 1 says it is needed.

The reference reconstruction is the sharpest control in the whole project:
an A placed at the vanilla draw's OWN intrinsic coordinates has ~30x the
parameter-space trace of the vanilla draw.  If it nevertheless trains
identically, tr(A^T A) is demonstrably not the causal coordinate.

Training configuration is byte-for-byte the one used by the `lit` audit
(0.6B, GSM8K, r=16, alpha=32, fp32, 300 steps, constant LR, warmup 10,
clip 1.0), so the 13 published initializers already measured there become a
zero-cost out-of-distribution test set for whatever law the atlas produces.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/01-hidden-preconditioner/src/run_atlas.py"
LRS = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]

# (S, D, rho);  D=None means "the vanilla draw's own D", rho='ref' likewise
POINTS = [("ref", 1.0, None, "ref")]                       # reconstruction
POINTS += [(f"S{s:g}", s, None, "ref") for s in (0.1, 0.3, 3.0, 10.0, 30.0)]
POINTS += [(f"D{d:g}", 1.0, d, "ref") for d in (2.0, 3.0, 4.0, 8.0, 12.0, 16.0)]
POINTS += [(f"R{r}", 1.0, None, r) for r in ("0.1", "0.3", "1.0", "3.0", "30.0")]
POINTS += [("Rrand", 1.0, None, "rand")]
POINTS = [(n, S, D, r, 0.5) for n, S, D, r in POINTS]

# Wave 2.  The out-of-distribution test on the published initializers found a
# coordinate wave 1 does not span: W = tr(A A^T) / tr(A Sigma A^T), the ratio of
# the parameter metric that Adam's per-coordinate step sees to the data metric
# that the function sees.  Wave 1 sits at W/W_vanilla in [27, 42]; every
# published initializer sits in [0.018, 1.0], and the wave-1 residual correlates
# with log W at r = -0.79.  The whitening exponent q in A = Atil Sigma^-q sweeps
# it: q = 0.5 is wave 1, q = 0 puts the row space in the parameter metric, and
# q < 0 concentrates it on the highest-variance directions.
WAVE2 = [(f"W{q:g}", 1.0, None, "ref", q) for q in (-0.25, 0.0, 0.15, 0.3)]
WAVE2 += [(f"W0_S{s:g}", s, None, "ref", 0.0) for s in (0.1, 10.0)]
WAVE2 += [(f"W0_D{d:g}", 1.0, d, "ref", 0.0) for d in (2.0, 12.0)]

# Wave 3.  The exact causal ladder for W.  Unlike wave 2, S and D are fixed by
# construction (M_x = Lambda identically) and the *spectrum-weighted* alignment
# R_g = tr(A C_g A^T)/tr(A Sigma A^T) -- the quantity that actually enters
# <G, GP>, unlike the unweighted statistic wave 1 swept -- is driven to the
# vanilla draw's value.  Only W moves.  The W/W0 = 1 rung reproduces the vanilla
# draw in ALL FOUR coordinates and is the sufficiency test.
WAVE3 = [(f"MW{w:g}", 1.0, None, "ref", w) for w in
         (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="atlas")
    ap.add_argument("--arch", default="blackwell")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lrs", default=None)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--wave", default="1", choices=["1", "2", "3", "all"])
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    lrs = [float(x) for x in a.lrs.split(",")] if a.lrs else LRS
    pts = {"1": POINTS, "2": WAVE2, "3": WAVE3,
           "all": POINTS + WAVE2 + WAVE3}[a.wave]
    jobs = []
    for _, S, D, rho, q in pts:
        for lr in lrs:
            for sd in [int(x) for x in a.seeds.split(",")]:
                d = f"--D {D:g} " if D is not None else ""
                w = (f"--matchW {q:g} " if a.wave == "3"
                     else (f"--wexp {q:g} " if q != 0.5 else ""))
                jobs.append(f"{{PY}} {RUN} --tag {a.tag} --S {S:g} {d}"
                            f"--rho {rho} {w}--lr {lr:g} --seed {sd} "
                            f"--steps {a.steps}")
    if a.reverse:
        jobs = jobs[::-1]
    print(f"{len(pts)} atlas points, {len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/01-hidden-preconditioner/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
