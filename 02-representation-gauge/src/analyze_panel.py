"""Summarise the topic-02 gauge panel."""
import glob, json, os, sys, statistics as st
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main(tag="g1"):
    d = os.path.join(REPO, "02-representation-gauge", "results", tag)
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")))]
    print(f"{len(recs)} runs in {tag}\n")
    by = {}
    for r in recs:
        a = r["args"]
        by.setdefault((a["optimizer"], a["lr"]), {})[
            (a["init"], a["mode"], a["gauge_seed"])] = r

    for key in sorted(by, key=lambda k: (k[0], k[1])):
        opt, lr = key
        cells = by[key]
        print(f"--- {opt}  lr={lr:g}  " + "-" * 44)
        def fe(init, mode, gs=0):
            r = cells.get((init, mode, gs))
            return r["log"]["final_eval_loss"] if r else None
        ref_k = fe("kaiming", "orig")
        ref_n = fe("nora", "orig")
        print(f"  kaiming orig                  {ref_k}")
        print(f"  nora    orig                  {ref_n}")
        gseeds = sorted({g for (_, _, g) in cells})
        rows = {}
        for gs in gseeds:
            for init, mode in [("kaiming", "coupled_oracle"),
                               ("nora", "coupled_oracle"),
                               ("nora", "coupled_algo")]:
                v = fe(init, mode, gs)
                if v is None:
                    continue
                ref = ref_k if init == "kaiming" else ref_n
                rows.setdefault((init, mode), []).append(v - ref)
                print(f"  {init:7s} {mode:15s} gauge{gs}  {v:.6f}   "
                      f"delta vs orig = {v-ref:+.6f}")
        print()
        floor = rows.get(("kaiming", "coupled_oracle"), [])
        eff = rows.get(("nora", "coupled_algo"), [])
        orc = rows.get(("nora", "coupled_oracle"), [])
        def summ(name, xs):
            if not xs: return
            print(f"    {name:34s} mean|delta|={st.mean(abs(x) for x in xs):.6f}"
                  f"  rms={ (sum(x*x for x in xs)/len(xs))**0.5:.6f}  n={len(xs)}")
        summ("FLOOR  kaiming (optimizer only)", floor)
        summ("ORACLE nora N(A) then rotate", orc)
        summ("EFFECT nora N(AR) (algorithmic)", eff)
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "g1")
