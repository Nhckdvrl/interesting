"""Topic 02 -- gauge dose-response panel.

Every rung is an EXACT function-preserving reparameterisation of the same
pretrained model (fp32-verified to 7 s.f. by E0).  The ladder increases only
how many residual-stream coordinates a rotation mixes together:

    none  ->  perm (1)  ->  block4  ->  block16  ->  block64  ->  block256
          ->  rand (1024) / hadamard (1024)

Pre-registered predictions:
  * SGD (with momentum, decoupled WD, global-norm clipping -- all covariant):
    FLAT.  Every rung must give the same loss up to fp32 roundoff.  This is the
    control that says any AdamW effect is not numerical.
  * AdamW: covariant under permutations/sign flips only, so `perm` must match
    `none`, and the penalty should grow with block size.
  * If LoRA shows a larger penalty than FullFT, PEFT carries *excess* gauge
    sensitivity beyond the optimizer's own.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/02-representation-gauge/src/run_dose.py"

GAUGES = ["none", "perm", "block4", "block16", "block64", "block256",
          "rand", "hadamard"]
LRS = {("lora", "adamw"): [1e-4, 3e-4, 1e-3],
       ("lora", "sgd"):   [3e-2, 1e-1, 3e-1],
       ("full", "adamw"): [1e-6, 3e-6, 1e-5],
       ("full", "sgd"):   [1e-3, 3e-3, 1e-2]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dose")
    ap.add_argument("--arch", default="blackwell")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--gauge_seeds", default="0")
    ap.add_argument("--methods", default="lora,full")
    ap.add_argument("--opts", default="adamw,sgd")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for meth in a.methods.split(","):
        for opt in a.opts.split(","):
            for lr in LRS[(meth, opt)]:
                for gauge in GAUGES:
                    seeds = ([0] if gauge in ("none", "hadamard")
                             else [int(x) for x in a.gauge_seeds.split(",")])
                    for gs in seeds:
                        jobs.append(
                            f"{{PY}} {RUN} --tag {a.tag} --method {meth} "
                            f"--optimizer {opt} --lr {lr:g} --gauge {gauge} "
                            f"--gauge_seed {gs} --steps {a.steps}")
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/02-representation-gauge/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
