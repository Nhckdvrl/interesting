"""Locate the published initializers in the atlas coordinates.

For every (initializer, matching convention) we rebuild A exactly as
`run_lit.py` does — same per-module seed, same probes, same rescaling — and
then measure where it sits in the intrinsic state space:

    S    = tr(A Sigma A^T), relative to the vanilla draw in the same layer
    D    = r_eff(A Sigma A^T)
    rho  = captured whitened-gradient energy, relative to a random row space

These coordinates are computed WITHOUT training and WITHOUT ever looking at the
atlas, so using them to predict the held-out runs is a genuine out-of-sample
test rather than a fit.
"""
import argparse, hashlib, json, os, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from common.pinit import make_A, kaiming_A
from common.pstats import p_stats
from common import initializers as IN
from common.intrinsic import (intrinsic_state, whiten_ops, captured_of,
                              sym_pow)
from common.data import build_sft, FixedOrderLoader
from common.train import load_model
from run_lit import collect_grads, collect_act_cov, ACT_GROUP, cached_make_A

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONDS = [("kaiming", "trace"), ("left_gauge", "trace"), ("nora", "trace"),
         ("nora_unit", "none"), ("nora_unit", "trace"), ("bimi", "none"),
         ("bimi", "trace"), ("etf", "none"), ("etf", "trace"),
         ("flatspec_flatdiag", "trace"), ("geomspec_flatdiag0.5", "trace"),
         ("eva", "none"), ("eva", "trace"), ("eva", "trace_act"),
         ("gradsub", "none"), ("gradsub", "trace"), ("gradsub", "trace_act"),
         ("pissa", "none"), ("pissa", "trace"), ("pissa_minor", "none"),
         ("olora", "none"), ("lora_one", "none"), ("lora_one", "trace")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every_layer", type=int, default=4)
    ap.add_argument("--probe_batches", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(
        REPO, "01-hidden-preconditioner", "results", "intrinsic_table.json"))
    a = ap.parse_args()

    model, tok = load_model(a.model, dtype=torch.float32)
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

    # whitening operators, computed once per sampled module
    W = {}
    for name in mods:
        key = name.rsplit(".", 1)[0] + "." + ACT_GROUP[name.split(".")[-1]]
        Sig = ACT[key].cuda().double()
        Cg = (G[name].cuda().double().T @ G[name].cuda().double())
        Sh = sym_pow(Sig, 0.5)
        _, _, tau, U = whiten_ops(Sig, Cg)
        W[name] = dict(Sig=Sig, Sh=Sh, tau=tau, U=U)
    print("whitening done", flush=True)

    rows = {}
    for cond, match in CONDS:
        acc = {}
        for name, mod in mods.items():
            d_in = mod.weight.shape[1]
            h = int(hashlib.md5(f"{a.seed}:{name}".encode()).hexdigest()[:12], 16)
            g = torch.Generator().manual_seed(h)
            base = kaiming_A(a.r, d_in, g, "cpu")
            ref_tr = float(base.pow(2).sum())
            Wt = mod.weight.detach().float()
            A, B = None, None
            if cond == "kaiming":
                A = base
            elif cond == "left_gauge":
                g2 = torch.Generator().manual_seed(
                    int(hashlib.md5(f"gauge1:{name}".encode()).hexdigest()[:12], 16))
                A = make_A("left_gauge", a.r, d_in, g2, "cpu", ref_A=base)
            elif cond == "etf":
                A = IN.init_etf(a.r, d_in, g, ref_tr)
            elif cond == "eva":
                key = name.rsplit(".", 1)[0] + "." + ACT_GROUP[name.split(".")[-1]]
                A = IN.init_eva(ACT[key].cuda(), a.r, d_in, ref_tr).cpu().double()
            elif cond == "gradsub":
                A = IN.init_gradsubspace(G[name].cuda(), a.r, ref_tr).cpu().double()
            elif cond in ("pissa", "pissa_minor"):
                A, B = IN.init_pissa(Wt, a.r, s, minor=(cond == "pissa_minor"))
                A, B = A.cpu().double(), B.cpu().double()
            elif cond == "olora":
                A, B = IN.init_olora(Wt, a.r, s)
                A, B = A.cpu().double(), B.cpu().double()
            elif cond == "lora_one":
                A, B = IN.init_lora_one(G[name].cuda(), a.r, s, W=Wt)
                A, B = A.cpu().double(), B.cpu().double()
            else:
                A = cached_make_A(cond, a.r, d_in, f"{a.seed}:{name}", base)

            w = W[name]
            def wt(M):
                Md = M.double().cuda()
                key2 = name.rsplit(".", 1)[0] + "." + ACT_GROUP[name.split(".")[-1]]
                return (float(Md.pow(2).sum()),
                        float(((Md @ w["Sig"]) * Md).sum()),
                        float((G[name].cuda().double() @ Md.T).pow(2).sum()))
            if match != "none":
                idx = {"trace": 0, "trace_act": 1, "trace_grad": 2}[match]
                k = (wt(base)[idx] / max(wt(A)[idx], 1e-30)) ** 0.5
                A = A * k
                if B is not None:
                    B = B / k
            Ad = A.double().cuda()
            S_abs, D_ = intrinsic_state(Ad, w["Sig"])
            S_ref, D_ref = intrinsic_state(
                base.double().cuda(), w["Sig"])
            V = torch.linalg.qr((Ad @ w["Sh"]).T)[0]
            rho = captured_of(V, w["tau"], w["U"])
            stt = p_stats(A.cpu(), s=1.0)
            for k2, v in (("S_rel", S_abs / S_ref), ("D", D_), ("rho", rho),
                          ("tr_P", stt["tr_P"]), ("r_eff_param", stt["eff_rank"]),
                          ("A_fro", float(Ad.norm())),
                          ("B0", float(B.norm()) if B is not None else 0.0)):
                acc.setdefault(k2, []).append(v)
        rows[f"{cond}|{match}"] = {k2: st.mean(v) for k2, v in acc.items()}
        r = rows[f"{cond}|{match}"]
        print(f"  {cond:22s} {match:10s} S={r['S_rel']:9.2f} D={r['D']:6.2f} "
              f"rho={r['rho']:8.2f} trP={r['tr_P']:9.2f} B0={r['B0']:6.2f}",
              flush=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
