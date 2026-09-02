"""The held-out set for the atlas: published initializers, re-run on the same
architecture and configuration as the atlas so that the out-of-distribution
prediction is not confounded by a hardware offset (measured at ~4e-4 nats
between A100 and RTX PRO 6000, which is inside the seed spread of 2.3e-3 but
above the 2.7e-4 gauge null).

These runs are NEVER used to fit the law; they are located in the intrinsic
coordinates afterwards and predicted.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/01-hidden-preconditioner/src/run_lit.py"
LRS = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
CONDS = [("kaiming", "trace"), ("nora", "trace"), ("nora_unit", "none"),
         ("bimi", "none"), ("etf", "none"), ("flatspec_flatdiag", "trace"),
         ("geomspec_flatdiag0.5", "trace"), ("eva", "none"),
         ("gradsub", "none"), ("pissa", "none"), ("pissa_minor", "none"),
         ("olora", "none"), ("lora_one", "none")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ood")
    ap.add_argument("--arch", default="blackwell")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = [f"{{PY}} {RUN} --tag {a.tag} --cond {c} --lr {lr:g} --match {m} "
            f"--steps {a.steps}" for c, m in CONDS for lr in LRS]
    if a.reverse:
        jobs = jobs[::-1]
    print(f"{len(CONDS)} held-out initializers, {len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/01-hidden-preconditioner/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
