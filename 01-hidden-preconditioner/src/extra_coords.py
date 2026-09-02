"""Is (S, D, rho) sufficient?

The atlas contains two points with the same S, the same D and the same captured
gradient energy rho, built differently: one from a contiguous window of the
whitened-gradient eigenbasis, one from a Haar-random row space.  They differ by
4.2e-3 nats -- more than any single-axis effect in the atlas.  So the row space
carries information beyond its captured energy.

This script recomputes, for every cached atlas point and with no training, a set
of candidate fourth coordinates:

    R_g   = tr(A C_g A^T)/tr(A Sigma A^T)
                                  the LAMBDA-WEIGHTED task alignment, i.e. the
                                  first-order descent rate per unit data-space
                                  scale.  This -- not the unweighted captured
                                  energy used to build wave 1 -- is the quantity
                                  that enters <G, GP>.
    D_g   = r_eff(A C_g A^T)      spectral dimension in the GRADIENT metric
                                  (the analogue of D, which is measured in the
                                  activation metric)
    W     = tr(A A^T)/tr(A Sigma A^T)
                                  parameter metric over data metric
    Cdis  = r_eff of the captured-energy distribution of the row space over the
            eigenbasis of T = Sigma^{-1/2} C_g Sigma^{-1/2}, i.e. how many
            whitened-gradient directions the adapter actually spreads over
    cos1  = cos(G, GP), the measured first-order descent efficiency
"""
import argparse, glob, hashlib, json, os, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from common.intrinsic import whiten_ops, sym_pow, gradient_alignment
from common.data import build_sft, FixedOrderLoader
from common.train import load_model
from run_atlas import collect_probes, ACT_GROUP, ACACHE

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def erank(M):
    t = float(torch.diagonal(M).sum())
    return t * t / (float((M * M).sum()) + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--every_layer", type=int, default=4)
    ap.add_argument("--probe_batches", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(
        REPO, "01-hidden-preconditioner", "results", "extra_coords.json"))
    a = ap.parse_args()

    model, tok = load_model(a.model, dtype=torch.float32)
    tr, _ = build_sft(tok, a.task, 6000, 256, 384, seed=0)
    ld = FixedOrderLoader(tr, 16, tok.pad_token_id, seed=0)
    T = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
         "down_proj")
    allm = {n: m for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in T}
    SIG, G = collect_probes(model, allm, ld, a.probe_batches)
    mods = [n for n in allm
            if int(n.split("layers.")[1].split(".")[0]) % a.every_layer == 0]
    print(f"{len(mods)} sampled modules", flush=True)
    W = {}
    for n in mods:
        key = n.rsplit(".", 1)[0] + "." + ACT_GROUP[n.split(".")[-1]]
        Sig = SIG[key].cuda().double()
        Gc = G[n].cuda().double()
        Cg = Gc.T @ Gc
        _, _, tau, U = whiten_ops(Sig, Cg)
        W[n] = dict(Sig=Sig, Cg=Cg, G=Gc, Sh=sym_pow(Sig, 0.5), tau=tau, U=U)
    del model
    torch.cuda.empty_cache()
    print("whitening done", flush=True)

    rows = {}
    for f in sorted(glob.glob(os.path.join(ACACHE, "*.pt"))):
        try:
            c = torch.load(f, map_location="cpu")
        except Exception:
            continue
        if not set(mods) <= set(c["A"]):
            continue
        acc = {}
        for n in mods:
            A = c["A"][n].double().cuda()
            w = W[n]
            Dg = erank(A @ w["Cg"] @ A.T)
            Rg = gradient_alignment(A, w["Sig"], w["Cg"])
            Wm = float(A.pow(2).sum()) / (float(((A @ w["Sig"]) * A).sum())
                                          + 1e-30)
            V = torch.linalg.qr((A @ w["Sh"]).T)[0]
            e = (w["U"].T @ V).pow(2).sum(1) * w["tau"]     # captured per T-mode
            Cdis = float(e.sum()) ** 2 / float(e.pow(2).sum() + 1e-30)
            GP = (w["G"] @ A.T) @ A
            cos1 = float((w["G"] * GP).sum() /
                         (w["G"].norm() * GP.norm() + 1e-30))
            for k, v in (("D_g", Dg), ("Cdis", Cdis), ("cos1", cos1),
                         ("R_g", Rg), ("W", Wm)):
                acc.setdefault(k, []).append(v)
        s = c["stats"][mods[0]]
        rows[os.path.basename(f)[:-3]] = dict(
            S=st.mean(c["stats"][n]["S_rel"] for n in mods),
            D=st.mean(c["stats"][n]["D"] for n in mods),
            rho=st.mean(c["stats"][n]["rho_rel"] for n in mods),
            **{k: st.mean(v) for k, v in acc.items()})
        r = rows[os.path.basename(f)[:-3]]
        print(f"  S={r['S']:8.3f} D={r['D']:7.3f} rho={r['rho']:8.3f} | "
              f"D_g={r['D_g']:7.3f} Cdis={r['Cdis']:9.2f} cos1={r['cos1']:.5f}",
              flush=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
