"""Topic 02 -- "is Adam's advantage over SGD a property of the coordinates?"

The dose panel suggests it is, in part: Adam's edge over SGD on FullFT falls
from +0.0058 to +0.0040 nats (-31%) between the pretrained basis and an exactly
function-preserving Hadamard rotation of the residual stream, while SGD itself
is flat across the whole ladder.

That claim rests on both optima being resolved, so this panel replaces the
3-point LR grids with dense ones, and adds gauge seeds and replicate seeds.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/02-representation-gauge/src/run_dose.py"
LRS = {("full", "adamw"): [3e-7, 1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5],
       ("full", "sgd"):   [3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2],
       ("lora", "adamw"): [3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3],
       ("lora", "sgd"):   [1e-2, 3e-2, 6e-2, 1e-1, 2e-1, 3e-1, 5e-1]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="edge")
    ap.add_argument("--arch", default="blackwell")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--gauges", default="none,block64,hadamard,rand")
    ap.add_argument("--methods", default="full,lora")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--reverse", action="store_true",
                    help="walk the job list backwards, so a second scheduler "
                         "can drain the same queue from the other end without "
                         "colliding with the first")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for meth in a.methods.split(","):
        for opt in ("adamw", "sgd"):
            for lr in LRS[(meth, opt)]:
                for gauge in a.gauges.split(","):
                    for sd in [int(x) for x in a.seeds.split(",")]:
                        jobs.append(
                            f"{{PY}} {RUN} --tag {a.tag} --method {meth} "
                            f"--optimizer {opt} --lr {lr:g} --gauge {gauge} "
                            f"--gauge_seed 0 --seed {sd} --steps {a.steps}")
    if a.reverse:
        jobs = jobs[::-1]
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/02-representation-gauge/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
