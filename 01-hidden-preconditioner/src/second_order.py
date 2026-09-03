"""The nine second-order invariants, and the one coordinate Adam can see.

S, D, W and R_g use only four of the nine order-<=2 invariants of the triple
(M_0, M_x, M_g) = (A A^T, A Sigma A^T, A C_g A^T).  That alone is a reason the
(S, D, omega) law leaves a systematic out-of-distribution residual.  But there
is a second, sharper reason, and it is a statement about symmetry groups.

SGD's first-order descent rate is sum_j ||G a_j||_2^2 = tr M_g, invariant under
the full gauge A -> QA.  AdamW's is sum_j ||G a_j||_1, invariant only under
signed permutations of the rows of A.  So AdamW's trajectory depends on a
functional of A that is NOT a function of the conjugacy class -- and the
measurement null already showed this experimentally, since the left gauge
A -> QA costs 1.5e-6 nats under SGD and 2.7e-4 under AdamW.

The natural such functional is the equipartition of the gradient energy over
the r rows of A,

    E_g = (sum_j sqrt((M_g)_jj))^2 / (r tr M_g)   in (1/r, 1],

which is 1 when the r rows carry equal gradient energy and 1/r when one row
carries it all.  E_g is exactly signed-permutation invariant and exactly not
O(r) invariant, which is the fingerprint of an Adam-only coordinate.

This script measures all nine invariants plus E_0, E_x, E_g, without training,
for every cached atlas construction and for every published initializer, so the
law can be re-selected on the full set.
"""
import argparse, glob, hashlib, json, os, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from common.pinit import make_A, kaiming_A
from common import initializers as IN
from common.intrinsic import (whiten_ops, sym_pow, second_order, triple,
                              l1_flatness, offdiag_mass)
from common.data import build_sft, FixedOrderLoader
from common.train import load_model
from run_lit import collect_grads, collect_act_cov, ACT_GROUP, cached_make_A
from run_atlas import ACACHE
from intrinsic_table import CONDS

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KEYS = ("D_0", "D_x", "D_g", "E_0", "E_x", "E_g",
        "Psi_0x", "Psi_0g", "Psi_xg", "Lam1", "Dout",
        "Off_x", "Off_g", "Off_0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every_layer", type=int, default=4)
    ap.add_argument("--probe_batches", type=int, default=4)
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16"],
                    help="backbone dtype for the probe.  The audit measures "
                         "EIGENVECTOR DIRECTIONS of r x r metrics, and every "
                         "linear-algebra step is promoted to float64 anyway, "
                         "so bf16 weights are adequate and halve both the "
                         "weights and the per-module gradients -- which is "
                         "what an 8B probe cannot otherwise fit.")
    ap.add_argument("--extra", default="",
                    help="comma-separated extra conditions, e.g. "
                         "frame0,frame1,gradsub@frame1 -- each may carry a "
                         "'|match' suffix (default |trace)")
    ap.add_argument("--no_atlas", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        REPO, "01-hidden-preconditioner", "results", "second_order.json"))
    a = ap.parse_args()

    model, tok = load_model(a.model, dtype=getattr(torch, a.dtype))
    tr, _ = build_sft(tok, a.task, 6000, 256, 384, seed=0)
    ld = FixedOrderLoader(tr, 16, tok.pad_token_id, seed=0)
    T = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
         "down_proj")
    allm = {n: m for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in T}
    G = collect_grads(model, allm, ld, a.probe_batches)
    ACT = collect_act_cov(model, allm, ld, a.probe_batches)
    mods = {n: m for n, m in allm.items()
            if int(n.split("layers.")[1].split(".")[0]) % a.every_layer == 0}
    print(f"{len(mods)} sampled modules of {len(allm)}", flush=True)
    s = a.alpha / a.r

    # Move the backbone to CPU before accumulating per-module probes: it is
    # only needed afterwards for the weight-derived initialisers, which read one
    # module at a time.  Holding 33 GB of fp32 weights alongside the covariances
    # is what OOMed the 8B audit twice.
    model.cpu()
    torch.cuda.empty_cache()
    P = {}
    for name in mods:
        key = name.rsplit(".", 1)[0] + "." + ACT_GROUP[name.split(".")[-1]]
        Sig = ACT[key].cuda().double()
        Gc = G[name].cuda().double()
        P[name] = dict(Sig=Sig, Cg=Gc.T @ Gc, G=Gc, Sh=sym_pow(Sig, 0.5))
    print("probes done", flush=True)

    def row(A, name):
        p = P[name]
        Ad = A.double().cuda()
        out = second_order(*triple(Ad, p["Sig"], p["Cg"]))
        lam, _, dout = l1_flatness(Ad, p["G"])
        out["Lam1"], out["Dout"] = lam, dout
        M0, Mx, Mg = triple(Ad, p["Sig"], p["Cg"])
        out["Off_0"] = offdiag_mass(M0)
        out["Off_x"] = offdiag_mass(Mx)
        out["Off_g"] = offdiag_mass(Mg)
        return out

    rows = {}

    CONDS_ALL = list(CONDS)
    for e in filter(None, a.extra.split(",")):
        CONDS_ALL.append((e.split("|")[0],
                          e.split("|")[1] if "|" in e else "trace"))

    # ---- the synthetic atlas, straight from the construction cache ---------
    n_at = 0
    if a.no_atlas:
        print("skipping the atlas cache")
    for f in ([] if a.no_atlas else sorted(glob.glob(os.path.join(ACACHE, "*.pt")))):
        try:
            c = torch.load(f, map_location="cpu")
        except Exception:
            continue
        if not set(mods) <= set(c["A"]):
            continue
        acc = {}
        for name in mods:
            for k, v in row(c["A"][name], name).items():
                acc.setdefault(k, []).append(v)
        rows["atlas:" + os.path.basename(f)[:-3]] = {
            k: st.mean(acc[k]) for k in KEYS}
        n_at += 1
    print(f"{n_at} cached atlas constructions", flush=True)

    # ---- the published initializers, rebuilt exactly as run_lit does ------
    for cond, match in CONDS_ALL:
        acc = {}
        gauge_t = None
        if "@frame" in cond:
            cond, gauge_t = cond.split("@frame")[0], float(cond.split("@frame")[1])
        for name, mod in mods.items():
            d_in = mod.weight.shape[1]
            h = int(hashlib.md5(f"{a.seed}:{name}".encode()).hexdigest()[:12], 16)
            g = torch.Generator().manual_seed(h)
            base = kaiming_A(a.r, d_in, g, "cpu")
            ref_tr = float(base.pow(2).sum())
            Wt = mod.weight.detach().float().cuda()
            A, B = None, None
            if cond == "kaiming":
                A = base
            elif cond == "left_gauge":
                g2 = torch.Generator().manual_seed(int(hashlib.md5(
                    f"gauge1:{name}".encode()).hexdigest()[:12], 16))
                A = make_A("left_gauge", a.r, d_in, g2, "cpu", ref_A=base)
            elif cond == "etf":
                A = IN.init_etf(a.r, d_in, g, ref_tr)
            elif cond == "eva":
                key = name.rsplit(".", 1)[0] + "." + ACT_GROUP[name.split(".")[-1]]
                A = IN.init_eva(ACT[key].cuda(), a.r, d_in, ref_tr).cpu().double()
            elif cond.startswith("framex"):
                _k = name.rsplit(".", 1)[0] + "." + ACT_GROUP[name.split(".")[-1]]
                A = IN.init_frame(base.cuda(), G[name].cuda(), float(cond[6:]),
                                  Sigma=ACT[_k].cuda()).cpu().double()
            elif cond.startswith("frame"):
                _t = cond[5:]
                A = IN.init_frame(base.cuda(), G[name].cuda(),
                                  _t if _t in ("opt", "min") else
                                  float(_t)).cpu().double()
            elif cond == "gradsub":
                A = IN.init_gradsubspace(G[name].cuda(), a.r, ref_tr).cpu().double()
            elif cond in ("pissa", "pissa_minor"):
                A, B = IN.init_pissa(Wt, a.r, s, minor=(cond == "pissa_minor"))
                A = A.cpu().double()
            elif cond == "olora":
                A, B = IN.init_olora(Wt, a.r, s); A = A.cpu().double()
            elif cond == "lora_one":
                A, B = IN.init_lora_one(G[name].cuda(), a.r, s, W=Wt)
                A = A.cpu().double()
            else:
                A = cached_make_A(cond, a.r, d_in, f"{a.seed}:{name}", base)
            if gauge_t is not None:
                from common.intrinsic import frame_ladder
                Ad2 = A.double().cuda(); Gd2 = G[name].cuda().double()
                GA2 = Gd2 @ Ad2.T
                Qg = frame_ladder(GA2.T @ GA2, [gauge_t])[0]
                A = (Qg @ Ad2).cpu().double()
            if match != "none":
                p = P[name]
                Ad, bd = A.double().cuda(), base.double().cuda()
                def q(M):
                    return (float(M.pow(2).sum()),
                            float(((M @ p["Sig"]) * M).sum()),
                            float((G[name].cuda().double() @ M.T).pow(2).sum()))
                idx = {"trace": 0, "trace_act": 1, "trace_grad": 2}[match]
                A = A * (q(bd)[idx] / max(q(Ad)[idx], 1e-30)) ** 0.5
            for k, v in row(A, name).items():
                acc.setdefault(k, []).append(v)
        tagn = cond + (f"@frame{gauge_t:g}" if gauge_t is not None else "")
        rows[f"lit:{tagn}|{match}"] = {k: st.mean(acc[k]) for k in KEYS}
        r_ = rows[f"lit:{tagn}|{match}"]
        print(f"  {tagn:26s} {match:10s} Lam1={r_['Lam1']:.4f} "
              f"E_g={r_['E_g']:.4f} Off_x={r_['Off_x']:.4f} "
              f"D_g={r_['D_g']:.4f} Psi_0x={r_['Psi_0x']:.4f}", flush=True)

    if os.path.exists(a.out):          # merge, never clobber earlier rows
        old = json.load(open(a.out)); old.update(rows); rows = old
    json.dump(rows, open(a.out, "w"), indent=2)
    print("wrote", a.out, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
