"""Topic 02 -- gauge dose-response summary.

Reports, per (method, optimizer), the best-tuned loss at each rung of the
coordinate-mixing ladder.  `perm` is the zero-dose control (AdamW is exactly
covariant under permutations and sign flips), and every SGD row is a full
positive control (SGD is covariant under ALL of them, so any SGD spread is
numerical).
"""
import glob, json, os, sys, statistics as st
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ORDER = ["none", "perm", "block4", "block16", "block64", "block256", "rand",
         "hadamard"]
MIX = {"none": 1, "perm": 1, "block4": 4, "block16": 16, "block64": 64,
       "block256": 256, "rand": 1024, "hadamard": 1024}


def main(tag="dose"):
    d = os.path.join(REPO, "02-representation-gauge", "results", tag)
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        rows.append(dict(m=a["method"], o=a["optimizer"], g=a["gauge"],
                         gs=a["gauge_seed"], lr=a["lr"],
                         loss=r["log"]["final_eval_loss"]))
    print(f"{len(rows)} runs\n")
    for m in sorted({r["m"] for r in rows}):
        for o in sorted({r["o"] for r in rows}):
            sub = [r for r in rows if r["m"] == m and r["o"] == o]
            if not sub:
                continue
            lrs = sorted({r["lr"] for r in sub})
            print(f"=== {m} + {o} " + "=" * 52)
            print(f"  {'gauge':10s} {'mix':>5s} "
                  + "".join(f"{('lr=%.0e' % l):>11s}" for l in lrs)
                  + f"{'best':>11s}{'vs none':>10s}")
            base = None
            for g in ORDER:
                gr = [r for r in sub if r["g"] == g]
                if not gr:
                    continue
                cells = []
                for l in lrs:
                    v = [x["loss"] for x in gr if x["lr"] == l]
                    cells.append(f"{st.mean(v):11.5f}" if v else f"{'-':>11s}")
                best = min(st.mean([x["loss"] for x in gr if x["lr"] == l])
                           for l in lrs
                           if [x for x in gr if x["lr"] == l])
                if g == "none":
                    base = best
                print(f"  {g:10s} {MIX[g]:5d}" + "".join(cells)
                      + f"{best:11.5f}{best-base:+10.5f}")
            print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dose")
