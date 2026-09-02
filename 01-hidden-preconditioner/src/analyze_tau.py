"""Wave 4 -- is omega a dynamical timescale, or a static geometric preference?

The prediction.  The initial down-projection is rewritten by Adam at a relative
rate ||dA||_F/||A||_F ~ eta_A sqrt(r d)/sqrt(S W), so the scaffold's persistence
timescale is tau_A ~ sqrt(S W)/eta_A.  If the tuned optimum corresponds to a
roughly fixed tau_A, then at fixed S

        omega*  proportional to  eta_A^2 ,

i.e. the minimum of the omega ladder should move by 4x per doubling of
eta_A/eta_B.  Frozen A (eta_A = 0) has no remodelling at all, so if the omega
response is a timescale effect it should flatten there.
"""
import glob, json, math, os, statistics as st, sys
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results", "atlas")


def load():
    d = {}
    for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        if a.get("matchW") is None:
            continue
        al = a.get("a_lr_ratio", 1.0)
        m = list(r["init_stats"].values())
        k = (al, a["matchW"])
        e = d.setdefault(k, dict(w=st.mean(x["W"] / x["W_ref"] for x in m),
                                 curve={}))
        e["curve"][a["lr"]] = r["log"]["final_eval_loss"]
    return d


def opt(curve, lrs_common):
    c = {l: curve[l] for l in lrs_common if l in curve}
    if not c:
        return None, None
    b = min(c, key=c.get)
    return c[b], b


def main():
    D = load()
    als = sorted({k[0] for k in D})
    ws = sorted({k[1] for k in D})
    # wave 3 swept 7 LRs; wave 4 swept 3.  Compare on the shared grid.
    common = [1e-4, 2e-4, 3e-4]
    print("Best loss over the shared LR grid {1e-4, 2e-4, 3e-4}\n")
    print(f"  {'omega':>8s} " + "".join(f"{('eta_A/eta_B=%g'%a):>18s}" for a in als))
    tab = {}
    for w in ws:
        row, wv = [], None
        for a in als:
            e = D.get((a, w))
            if e is not None and wv is None:
                wv = e["w"]
            L, lr = opt(e["curve"], common) if e else (None, None)
            tab[(a, w)] = L
            row.append(f"{L:11.5f} @{lr:.0e}" if L else f"{'-':>18s}")
        print(f"  {wv:8.3f} " + "".join(row))
    print()
    for a in als:
        v = [(w, tab[(a, w)]) for w in ws if tab.get((a, w)) is not None]
        if len(v) < 3:
            continue
        best = min(v, key=lambda t: t[1])
        # continuous minimum of a quadratic in log omega
        x = np.log([D[(a, w)]["w"] for w, _ in v])
        y = np.array([t[1] for t in v])
        c = np.polyfit(x, y, 2)
        wstar = math.exp(-c[1] / (2 * c[0])) if c[0] > 0 else float("nan")
        lab = "frozen A" if a == 0 else f"eta_A/eta_B = {a:g}"
        print(f"  {lab:22s} grid argmin omega = {D[(a,best[0])]['w']:6.3f}, "
              f"quadratic omega* = {wstar:7.3f}, "
              f"range over ladder = {max(y)-min(y):.5f} nats")
    print("\n  prediction: omega* should scale as (eta_A/eta_B)^2, i.e. 0.25 : 1 : 4")
    fr = [(D[(0.0, w)]["w"], tab[(0.0, w)]) for w in ws
          if tab.get((0.0, w)) is not None]
    on = [(D[(1.0, w)]["w"], tab[(1.0, w)]) for w in ws
          if tab.get((1.0, w)) is not None and (0.0, w) in tab
          and tab[(0.0, w)] is not None]
    if len(fr) >= 3:
        print(f"\n  FROZEN-A CONTROL")
        print(f"    range of the omega response with A frozen   : "
              f"{max(y for _, y in fr) - min(y for _, y in fr):.5f} nats")
        print(f"    range of the omega response with A trainable: "
              f"{max(y for _, y in on) - min(y for _, y in on):.5f} nats")
        print(f"    -> if omega acted through A-remodelling the frozen response "
              f"would flatten.")


if __name__ == "__main__":
    main()
