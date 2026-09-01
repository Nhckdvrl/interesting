"""Topic 01, Experiment family D -- does method identity survive conditioning
on P-statistics?

We treat the published initializers as *samples in P-space* and ask whether a
small set of pre-registered statistics predicts the best-tuned loss, and whether
the method label adds anything on top.

Statistics (from `pstat_table.py`), all measured at initialisation with 4
minibatches and no training:
    tr P          parameter-space scale
    tr(P Sigma)   data-space scale       (function-change size)
    r_eff(P)      parameter-metric effective rank
    r_eff^Sigma   DATA-metric effective rank, (tr M)^2/||M||_F^2, M = A Sigma A^T
    cos_adam      first-step descent efficiency under the optimizer actually used
    ||B_0||       whether the method leaves the P_0 equivalence class at all
"""
import glob, json, math, os, statistics as st, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")

DATA_AGNOSTIC_ZEROB = ["kaiming", "left_gauge", "nora", "nora_unit", "etf",
                       "flatspec_flatdiag", "geomspec_flatdiag0.5"]
DATA_AWARE_ZEROB = ["eva", "gradsub"]
NONZEROB = ["pissa", "pissa_minor", "olora", "lora_one"]


def best_by_cond(tag, match):
    rows = {}
    for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        if a.get("match", "trace") != match:
            continue
        rows.setdefault(a["cond"], {}).setdefault(a["lr"], []).append(
            r["log"]["final_eval_loss"])
    out = {}
    for c, d in rows.items():
        means = {lr: st.mean(v) for lr, v in d.items()}
        blr = min(means, key=means.get)
        interior = min(means) < means[min(means)] or (
            blr != min(means.keys()) and blr != max(means.keys()))
        out[c] = dict(loss=means[blr], lr=blr, n_lr=len(means),
                      bracketed=(blr != min(means.keys()) and
                                 blr != max(means.keys())))
    return out


def pearson(a, b):
    n = len(a); ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** .5
    return num / max(den, 1e-30)


def main(tag="lit"):
    P = json.load(open(os.path.join(RES, "pstat_table_trace.json")))
    for match in ("trace", "trace_act", "none"):
        B = best_by_cond(tag, match)
        if not B:
            continue
        print(f"\n{'='*104}\nMATCHING = {match}\n{'='*104}")
        print(f"  {'condition':22s} {'group':10s} {'r_eff':>6s} {'rS':>6s} "
              f"{'trPS':>8s} {'cosAdam':>8s} {'||B0||':>7s} "
              f"{'best loss':>10s} {'at lr':>8s} {'bracketed':>10s}")
        for c in DATA_AGNOSTIC_ZEROB + DATA_AWARE_ZEROB + NONZEROB:
            if c not in B or c not in P:
                continue
            grp = ("agnostic" if c in DATA_AGNOSTIC_ZEROB else
                   "data-aware" if c in DATA_AWARE_ZEROB else "B0!=0")
            p, b = P[c], B[c]
            print(f"  {c:22s} {grp:10s} {p['r_eff']:6.2f} {p['r_eff_act']:6.2f} "
                  f"{p['rel_act']:8.2f} {p['cos_adam']:8.4f} {p['B0']:7.2f} "
                  f"{b['loss']:10.5f} {b['lr']:8.1e} "
                  f"{'yes' if b['bracketed'] else 'EDGE':>10s}")
        # within-group predictiveness
        for grp, names in (("data-agnostic B0=0", DATA_AGNOSTIC_ZEROB),
                           ("all B0=0", DATA_AGNOSTIC_ZEROB + DATA_AWARE_ZEROB),
                           ("all methods",
                            DATA_AGNOSTIC_ZEROB + DATA_AWARE_ZEROB + NONZEROB)):
            xs = [(P[c], B[c]["loss"]) for c in names if c in B and c in P]
            if len(xs) < 4:
                continue
            print(f"\n  [{grp}]  n={len(xs)}")
            for lab, f in (("1/sqrt(r_eff)", lambda p: 1 / math.sqrt(p["r_eff"])),
                           ("1/sqrt(r_eff^Sigma)",
                            lambda p: 1 / math.sqrt(p["r_eff_act"])),
                           ("log tr(P Sigma)", lambda p: math.log(p["rel_act"])),
                           ("1/cos_adam", lambda p: 1 / p["cos_adam"]),
                           ("||B0||>0", lambda p: 1.0 if p["B0"] > 1e-6 else 0.0)):
                r = pearson([f(p) for p, _ in xs], [l for _, l in xs])
                print(f"    loss vs {lab:22s} pearson r = {r:+.3f}")
        # spread of the agnostic cluster vs the null
        cl = [B[c]["loss"] for c in DATA_AGNOSTIC_ZEROB[:6] if c in B]
        if len(cl) > 3:
            print(f"\n  data-agnostic cluster: mean={st.mean(cl):.5f} "
                  f"sd={st.pstdev(cl):.5f} range={max(cl)-min(cl):.5f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lit")
