"""Topic 02 at 7B -- the coordinate-mixing ladder on Mistral-7B-v0.3."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/02-representation-gauge/src/run_dose.py"
LRS = {("lora", "adamw"): [1e-4, 2e-4, 4e-4],
       ("lora", "sgd"):   [3e-2, 1e-1, 3e-1],
       ("full", "adamw"): [1e-6, 3e-6, 1e-5],
       ("full", "sgd"):   [1e-3, 3e-3, 1e-2]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dose7b")
    ap.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    ap.add_argument("--arch", default="blackwell")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--methods", default="lora")
    ap.add_argument("--gauges", default="none,perm,block64,rand,hadamard")
    ap.add_argument("--micro_bs", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for meth in a.methods.split(","):
        for opt in ("adamw", "sgd"):
            for lr in LRS[(meth, opt)]:
                for g in a.gauges.split(","):
                    jobs.append(
                        f"{{PY}} {RUN} --tag {a.tag} --model {a.model} "
                        f"--method {meth} --optimizer {opt} --lr {lr:g} "
                        f"--gauge {g} --steps {a.steps} --bs 16 "
                        f"--micro_bs {a.micro_bs} --eval_batches 16 "
                        f"--task gsm8k --n_train 6000 --n_eval 256")
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/02-representation-gauge/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
