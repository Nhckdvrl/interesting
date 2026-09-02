"""Topic 02 at 7B scale.

At 0.6B the best-tuned penalty of an exactly function-preserving rotation of the
residual stream is 1.7 millinats under AdamW and 0 under SGD.  The decisive
question for whether this is a curiosity or a practical concern is whether the
penalty grows with scale.  Mistral-7B-v0.3 (hidden 4096, a different
architecture family from the 0.6B Qwen3 runs) is verified exact in fp32 to
2.7e-7 nats by `e0_7b`.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import cluster

REPO = cluster.REPO
RUN = f"{REPO}/02-representation-gauge/src/run_dose.py"
LRS = {"adamw": [1e-4, 3e-4, 1e-3], "sgd": [3e-2, 1e-1, 3e-1]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dose7b")
    ap.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    ap.add_argument("--task", default="metamath")
    ap.add_argument("--arch", default="blackwell")
    ap.add_argument("--hosts", default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--gauges", default="none,perm,block64,hadamard")
    ap.add_argument("--method", default="lora")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for opt in ("adamw", "sgd"):
        for lr in LRS[opt]:
            for gauge in a.gauges.split(","):
                jobs.append(
                    f"{{PY}} {RUN} --tag {a.tag} --model {a.model} "
                    f"--task {a.task} --method {a.method} --optimizer {opt} "
                    f"--lr {lr:g} --gauge {gauge} --gauge_seed 0 "
                    f"--steps {a.steps} --bs 16 --micro_bs 2 --max_len 512 "
                    f"--r {a.r} --alpha {a.alpha:g} --n_train 60000 "
                    f"--n_eval 512 --eval_batches 24")
    print(f"{len(jobs)} jobs")
    cluster.main(jobs, a.tag,
                 f"{REPO}/02-representation-gauge/results/{a.tag}/logs",
                 arch=a.arch,
                 hosts=a.hosts.split(",") if a.hosts else None, dry=a.dry)
