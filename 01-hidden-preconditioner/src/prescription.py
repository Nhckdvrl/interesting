"""Topic 01 -- the prescription test.

If a B0=0 initializer's only effect were the data-metric scale it induces, then
its whole loss-vs-LR curve should be the vanilla curve shifted along the LR axis
by exactly

    lr*_method / lr*_vanilla  =  ( tr(P Sigma)_vanilla / tr(P Sigma)_method )^{1/2}

because the size of the function change scales as lr * sqrt(tr(P Sigma)).
We test the prediction two ways:
  (i) does the predicted shift match the measured argmin ratio?
  (ii) after shifting, does the method's entire curve lie on the vanilla curve,
       or only touch it at the optimum?
Residual curvature after the shift is the signature of the second channel.
"""
import glob, json, math, os, statistics as st, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")


def curves(tag, match):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        if a.get("match", "trace") != match:
            continue
        out.setdefault(a["cond"], {}).setdefault(a["lr"], []).append(
            r["log"]["final_eval_loss"])
    return {c: {lr: st.mean(v) for lr, v in d.items()} for c, d in out.items()}


def interp(curve, x):
    xs = sorted(curve)
    if x <= xs[0] or x >= xs[-1]:
        return None
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (math.log(x) - math.log(xs[i])) / (math.log(xs[i + 1]) - math.log(xs[i]))
            return curve[xs[i]] * (1 - t) + curve[xs[i + 1]] * t


def main(tag="lit"):
    P = json.load(open(os.path.join(RES, "pstat_table_trace.json")))
    C = curves(tag, "trace")
    if "kaiming" not in C:
        print("no data"); return
    van = C["kaiming"]
    van_star = min(van, key=van.get)
    print(f"vanilla optimum: lr* = {van_star:.1e}, loss = {van[van_star]:.5f}\n")
    print(f"  {'method':22s} {'trPS':>8s} {'pred shift':>11s} {'meas shift':>11s} "
          f"{'shifted rms':>12s} {'shifted max':>12s} {'n':>3s}")
    for c in sorted(C):
        if c == "kaiming" or c not in P:
            continue
        cur = C[c]
        star = min(cur, key=cur.get)
        pred = 1.0 / math.sqrt(P[c]["rel_act"])
        devs = []
        for lr, l in cur.items():
            v = interp(van, lr / pred)       # vanilla at the equivalent LR
            if v is not None:
                devs.append(l - v)
        if len(devs) < 3:
            print(f"  {c:22s} {P[c]['rel_act']:8.2f} {pred:11.3f} "
                  f"{star/van_star:11.3f} {'-':>12s} {'-':>12s} {len(devs):3d}")
            continue
        rms = (sum(d * d for d in devs) / len(devs)) ** .5
        print(f"  {c:22s} {P[c]['rel_act']:8.2f} {pred:11.3f} "
              f"{star/van_star:11.3f} {rms:12.5f} "
              f"{max(abs(d) for d in devs):12.5f} {len(devs):3d}")
    print("\n  'pred shift' = (tr(P Sigma)_vanilla / tr(P Sigma)_method)^(1/2)")
    print("  'meas shift' = argmin_lr(method) / argmin_lr(vanilla)")
    print("  'shifted rms' = rms residual of the method's curve against the")
    print("                  vanilla curve after applying the PREDICTED shift.")
    print("  The AdamW gauge null on this setup is 2.7e-4 nats.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lit")
