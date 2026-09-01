"""Topic 02 -- does the gauge effect grow with model size?

The single most important open question: at 0.6B the best-tuned penalty of an
exactly function-preserving Hadamard rotation is 1.7 millinats.  If that number
grows with scale the phenomenon is a practical concern; if it shrinks it is a
curiosity.  Three rungs of the ladder (pretrained basis, block64, Hadamard) on
a 1.7B backbone, both optimizers, both methods.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/02-representation-gauge/src/run_dose.py"
LRS = {("full", "adamw"): [1e-6, 3e-6, 1e-5],
       ("full", "sgd"):   [1e-3, 3e-3, 1e-2],
       ("lora", "adamw"): [1e-4, 3e-4, 1e-3],
       ("lora", "sgd"):   [3e-2, 1e-1, 3e-1]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="scale17")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--arch", default="blackwell")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--micro_bs", type=int, default=8)
    ap.add_argument("--gauges", default="none,block64,hadamard")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for meth in ("full", "lora"):
        for opt in ("adamw", "sgd"):
            for lr in LRS[(meth, opt)]:
                for gauge in a.gauges.split(","):
                    jobs.append(
                        f"{{PY}} {RUN} --tag {a.tag} --model {a.model} "
                        f"--method {meth} --optimizer {opt} --lr {lr:g} "
                        f"--gauge {gauge} --steps {a.steps} "
                        f"--micro_bs {a.micro_bs}")
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/02-representation-gauge/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
