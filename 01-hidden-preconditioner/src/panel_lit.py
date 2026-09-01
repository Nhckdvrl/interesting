"""Topic 01 -- the literature audit.

Every published LoRA initializer is run as a sample in P-space, against two
references that bound the measurement:

  * `kaiming`     -- vanilla LoRA, 3 seeds
  * `left_gauge`  -- A_0 -> Q A_0, Q in O(r): IDENTICAL P_0, hence provably zero
                     preconditioner content.  Any method effect smaller than the
                     spread of this condition is not a preconditioner effect.

and under three matching conventions:

  none       each method at its published scale
  trace      tr P matched to the vanilla draw  (the parameter-space norm --
             what LoRAM/NoRA/the magnitude literature control)
  trace_act  tr(P Sigma_x) matched            (the DATA-space norm: the size of
             the function change the adapter can make)

The pilot measurement that motivates the third arm: at matched tr P, EVA has
55x the activation-weighted trace and 72x the gradient-weighted trace of a
vanilla draw; the gradient-subspace init has 43x/97x; PiSSA 4.6x/4.5x.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/01-hidden-preconditioner/src/run_lit.py"

LRS = [3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
OTHERS = ["nora", "nora_unit", "etf", "eva", "gradsub", "flatspec_flatdiag",
          "geomspec_flatdiag0.5", "pissa", "pissa_minor", "olora", "lora_one"]
PUBLISHED = ["nora_unit", "etf", "eva", "gradsub", "pissa", "pissa_minor",
             "olora", "lora_one"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lit")
    ap.add_argument("--arch", default="a100")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lrs", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    lrs = [float(x) for x in a.lrs.split(",")] if a.lrs else LRS
    jobs = []

    def add(cond, lr, match, extra=""):
        jobs.append(f"{{PY}} {RUN} --tag {a.tag} --cond {cond} --lr {lr:g} "
                    f"--match {match} --steps {a.steps} {extra}")

    for lr in lrs:
        # --- reference + null, 3 replicates each, in the primary arm
        for sd in (0, 1, 2):
            add("kaiming", lr, "trace", f"--seed {sd}")
        for gs in (1, 2, 3):
            add("left_gauge", lr, "trace", f"--gauge_seed {gs}")
        for c in OTHERS:
            add(c, lr, "trace")
        # --- data-metric-matched arm
        add("kaiming", lr, "trace_act")
        add("left_gauge", lr, "trace_act", "--gauge_seed 1")
        for c in OTHERS:
            add(c, lr, "trace_act")
        # --- published scale
        for c in PUBLISHED:
            add(c, lr, "none")
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/01-hidden-preconditioner/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
