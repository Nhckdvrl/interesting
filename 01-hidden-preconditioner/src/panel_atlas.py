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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="atlas")
    ap.add_argument("--arch", default="blackwell")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lrs", default=None)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    lrs = [float(x) for x in a.lrs.split(",")] if a.lrs else LRS
    jobs = []
    for _, S, D, rho in POINTS:
        for lr in lrs:
            for sd in [int(x) for x in a.seeds.split(",")]:
                d = f"--D {D:g} " if D is not None else ""
                jobs.append(f"{{PY}} {RUN} --tag {a.tag} --S {S:g} {d}"
                            f"--rho {rho} --lr {lr:g} --seed {sd} "
                            f"--steps {a.steps}")
    if a.reverse:
        jobs = jobs[::-1]
    print(f"{len(POINTS)} atlas points, {len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/01-hidden-preconditioner/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
