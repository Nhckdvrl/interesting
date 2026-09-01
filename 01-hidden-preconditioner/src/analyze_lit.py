"""Topic 01 -- the literature audit.

Central question: after conditioning on a small set of P-statistics, does
knowing WHICH published initializer produced A_0 add anything?

The measurement floor is `left_gauge`: A_0 -> Q A_0 with Q in O(r) leaves P_0
bit-identical, so its run-to-run spread is a pure lower bound on what any
method effect could mean.
"""
import glob, json, math, os, sys, statistics as st
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")

ZEROB = ["kaiming", "left_gauge", "nora", "nora_unit", "etf", "eva", "gradsub",
         "flatspec_flatdiag", "geomspec_flatdiag0.5"]
NONZEROB = ["pissa", "pissa_minor", "olora", "lora_one"]


def load(tag):
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        ps = list(r["init_pstats"].values())
        rows.append(dict(
            cond=a["cond"], lr=a["lr"], seed=a["seed"],
            gs=a.get("gauge_seed", 0), match=a.get("match", "trace"),
            r_eff=st.mean(p["eff_rank"] for p in ps),
            rel_trP=st.mean(p["rel_tr_P"] for p in ps),
            rel_act=st.mean(p["rel_tr_act"] for p in ps),
            rel_grad=st.mean(p["rel_tr_grad"] for p in ps),
            B0=st.mean(p["B0_norm"] for p in ps),
            dW=r["log"]["diag"][-1]["dW_norm"],
            loss=r["log"]["final_eval_loss"]))
    return rows


def main(tag="lit"):
    rows = load(tag)
    print(f"{len(rows)} runs in {tag}\n")
    for match in ["trace", "trace_act", "none"]:
        sub = [r for r in rows if r["match"] == match]
        if not sub:
            continue
        lrs = sorted({r["lr"] for r in sub})
        print(f"{'='*100}\nMATCHING: {match}\n{'='*100}")
        print(f"  {'condition':22s} {'r_eff':>6s} {'trP':>8s} {'trPS':>8s} "
              f"{'trPCg':>8s} {'B0':>7s} "
              + "".join(f"{('%.0e' % l):>9s}" for l in lrs) + f"{'best':>9s}")
        best = {}
        for c in ZEROB + NONZEROB:
            cr = [r for r in sub if r["cond"] == c]
            if not cr:
                continue
            cells = []
            for l in lrs:
                v = [x["loss"] for x in cr if x["lr"] == l]
                cells.append(f"{st.mean(v):9.5f}" if v else f"{'-':>9s}")
            b = min(st.mean([x["loss"] for x in cr if x["lr"] == l])
                    for l in lrs if [x for x in cr if x["lr"] == l])
            best[c] = b
            e = cr[0]
            tag2 = "" if c in ZEROB else "  (B0!=0)"
            print(f"  {c:22s} {e['r_eff']:6.2f} {e['rel_trP']:8.2f} "
                  f"{e['rel_act']:8.2f} {e['rel_grad']:8.2f} "
                  f"{e['B0']:7.2f}" + "".join(cells) + f"{b:9.5f}{tag2}")
        # null scale
        g = [r for r in sub if r["cond"] == "left_gauge"]
        k = [r for r in sub if r["cond"] == "kaiming"]
        if g and k:
            per_lr = []
            for l in lrs:
                vals = [x["loss"] for x in g + k if x["lr"] == l]
                if len(vals) > 2:
                    per_lr.append(st.pstdev(vals))
            null = max(per_lr) if per_lr else float("nan")
            print(f"\n  NULL (kaiming seeds + left_gauge, identical P0): "
                  f"max per-LR sd = {null:.5f}")
            if "kaiming" in best:
                print(f"  {'condition':22s} {'effect vs kaiming':>18s} "
                      f"{'in null sds':>12s}")
                for c, b in sorted(best.items(), key=lambda kv: kv[1]):
                    if c == "kaiming":
                        continue
                    d = b - best["kaiming"]
                    print(f"  {c:22s} {d:+18.5f} {d/max(null,1e-9):12.1f}"
                          + ("" if c in ZEROB else "   (B0!=0)"))
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lit")
