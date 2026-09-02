"""Stage 2 -- map the intrinsic state space of LoRA initialisation.

A point in this atlas is not a method.  It is a *controlled intervention*: an
initialisation constructed to sit at an exactly specified location

    S    data-space scale        tr(A Sigma A^T), relative to a vanilla draw
    D    spectral dimension      r_eff(A Sigma A^T), in [1, r]
    rho  task alignment          captured whitened-gradient energy, relative to
                                 a random row space (so random = 1)

The three are exactly independent by construction (see `common/intrinsic.py`),
so the atlas is a designed experiment rather than a correlational sweep.

Published initializers are deliberately NOT used to build the atlas.  They are
held out, located in the same coordinates afterwards, and used as an
out-of-distribution test of whatever law the atlas produces.

Trajectory logging: S_t and D_t are recorded through training in the *fixed*
initial data metric, so that different runs are directly comparable and we can
ask whether trajectories collapse in intrinsic coordinates.
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from common.lora import apply_lora, lora_parameters, DEFAULT_TARGETS
from common.pinit import kaiming_A
from common.pstats import p_stats
from common.intrinsic import (build_A, build_A_matched, intrinsic_state,
                              whiten_ops, captured_of, output_state, sym_pow,
                              metric_ratio, gradient_alignment)


def sym_half(M):
    return sym_pow(M, 0.5)
from common.data import build_sft, FixedOrderLoader
from common.train import load_model, train, eval_loss, set_amp
from common.evaluate import gsm8k_accuracy, gsm8k_eval_set

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ACACHE = os.path.expanduser("~/.cache/nora_repo_atlasA")
ACT_GROUP = {"q_proj": "q_proj", "k_proj": "q_proj", "v_proj": "q_proj",
             "o_proj": "o_proj", "gate_proj": "gate_proj",
             "up_proj": "gate_proj", "down_proj": "down_proj"}


def collect_probes(model, mods, loader, n_batches, device="cuda"):
    """One pass gives both the input second moment Sigma (shared within a
    group) and the input-side gradient covariance C_g = G^T G per module."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in mods.values():
        m.weight.requires_grad_(True)
    reps = {n: m for n, m in mods.items()
            if ACT_GROUP[n.split(".")[-1]] == n.split(".")[-1]}
    cov, hooks = {}, []

    def mk(n):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            cov[n] = cov.get(n, 0) + x.T @ x
        return hook
    for n, m in reps.items():
        hooks.append(m.register_forward_hook(mk(n)))
    ntok = 0
    for i in range(n_batches):
        b = {k: v.to(device) for k, v in loader.get(i).items()}
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        lg = out.logits[:, :-1].float(); lb = b["labels"][:, 1:]
        msk = lb != -100
        ntok += int(msk.sum())
        (nn.functional.cross_entropy(lg[msk], lb[msk]) / n_batches).backward()
    for h in hooks:
        h.remove()
    G = {n: m.weight.grad.detach().float() for n, m in mods.items()}
    for m in mods.values():
        m.weight.grad = None
        m.weight.requires_grad_(False)
    return {n: c / max(ntok, 1) for n, c in cov.items()}, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--S", type=float, default=1.0,
                    help="data-space scale, RELATIVE to the vanilla draw")
    ap.add_argument("--D", type=float, default=None,
                    help="target r_eff(A Sigma A^T); default = the vanilla draw's")
    ap.add_argument("--a_lr_ratio", type=float, default=1.0,
                    help="eta_A / eta_B.  The timescale on which Adam rewrites "
                         "the initial A is tau_A ~ sqrt(S W)/eta_A, so at fixed "
                         "S the optimal W should scale as eta_A^2.  0 freezes A.")
    ap.add_argument("--matchW", type=float, default=None,
                    help="wave 3: hold S, D and the LAMBDA-WEIGHTED alignment "
                         "R_g at the vanilla draw's values and drive "
                         "W/W_vanilla to this target.  Unlike --wexp this is an "
                         "exactly matched intervention: M_x = Lambda by "
                         "construction, so S and D cannot drift.")
    ap.add_argument("--matchW_iters", type=int, default=400)
    ap.add_argument("--wexp", type=float, default=0.5,
                    help="whitening exponent q in A = Atil Sigma^-q; sweeps the "
                         "parameter-vs-data metric ratio W")
    ap.add_argument("--rho", default="1.0",
                    help="captured-energy ratio; 'rand' for a Haar row space; "
                         "'ref' to copy the vanilla draw's own alignment")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--micro_bs", type=int, default=0)
    ap.add_argument("--probe_bs", type=int, default=0)
    ap.add_argument("--probe_batches", type=int, default=4)
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sched", default="constant")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--amp", default="none")
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_eval", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--traj_every", type=int, default=25)
    ap.add_argument("--acc_n", type=int, default=0)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    args = ap.parse_args()

    rho_s = args.rho
    cell = (f"S{args.S:g}_D{args.D if args.D is not None else 'ref'}"
            + (f"_MW{args.matchW:g}" if args.matchW is not None
               else f"_R{rho_s}" + (f"_W{args.wexp:g}" if args.wexp != 0.5
                                    else ""))
            + (f"_a{args.a_lr_ratio:g}" if args.a_lr_ratio != 1.0 else "")
            + f"_lr{args.lr:g}_s{args.seed}")
    outdir = os.path.join(REPO, "01-hidden-preconditioner", "results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, cell + ".json")
    if os.path.exists(outfile):
        print("skip (exists)", outfile); return

    set_amp(args.amp)
    torch.manual_seed(args.seed)
    model, tok = load_model(args.model, dtype=dict(
        float32=torch.float32, bfloat16=torch.bfloat16)[args.dtype])
    tr, te = build_sft(tok, args.task, args.n_train, args.n_eval, args.max_len,
                       seed=0)
    micro = args.micro_bs or args.bs
    accum = max(args.bs // micro, 1)
    trl = FixedOrderLoader(tr, micro, tok.pad_token_id, seed=0)
    tel = FixedOrderLoader(te, args.bs, tok.pad_token_id, seed=999)
    targets = tuple(args.targets.split(","))
    mods = {n: m for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in targets}

    probe_bs = args.probe_bs or micro
    probe_ld = (trl if probe_bs == micro
                else FixedOrderLoader(tr, probe_bs, tok.pad_token_id, seed=0))
    SIG, G = {}, {}
    torch.cuda.empty_cache()

    rho = None if rho_s in ("rand", "random", "none") else \
        ("ref" if rho_s == "ref" else float(rho_s))
    stats, SIGMA_CPU = {}, {}

    # The construction is deterministic given (model, task, seed, r, S, D, rho)
    # and is shared by every learning rate at this atlas point, so building it
    # once and caching removes 6/7 of the cost of a 7-LR sweep.  The dominant
    # term is one eigendecomposition of the whitened gradient operator per
    # adapted module.
    os.makedirs(ACACHE, exist_ok=True)
    ckey = hashlib.md5(
        f"{args.model}|{args.task}|{args.seed}|{args.r}|{args.S}|{args.D}|"
        f"{rho_s}|{args.wexp}|{args.matchW}|{args.matchW_iters}|"
        f"{args.probe_batches}|{probe_bs}|{args.n_train}|"
        f"{args.max_len}".encode()).hexdigest()
    cfile = os.path.join(ACACHE, ckey + ".pt")
    CACHED = None
    if os.path.exists(cfile):
        try:
            CACHED = torch.load(cfile, map_location="cpu")
            print(f"  reusing cached atlas point ({len(CACHED['A'])} modules)",
                  flush=True)
        except Exception:
            CACHED = None
    if CACHED is None:
        SIG, G = collect_probes(model, mods, probe_ld, args.probe_batches)
        torch.cuda.empty_cache()

    def factory(name, r, d_in, d_out):
        if CACHED is not None:
            stats[name] = CACHED["stats"][name]
            SIGMA_CPU[name] = CACHED["sigma"][name]
            return CACHED["A"][name]
        h = int(hashlib.md5(f"{args.seed}:{name}".encode()).hexdigest()[:12], 16)
        g = torch.Generator(device="cuda").manual_seed(h)
        key = name.rsplit(".", 1)[0] + "." + ACT_GROUP[name.split(".")[-1]]
        Sig = SIG[key].double()
        Cg = (G[name].T @ G[name]).double()
        # the vanilla draw fixes the reference point of the S axis, and the
        # default D, so that every atlas coordinate is expressed relative to
        # what plain LoRA would have done in this same layer
        gk = torch.Generator().manual_seed(h)
        A_ref = kaiming_A(r, d_in, gk, "cpu").double().cuda()
        S_ref, D_ref = intrinsic_state(A_ref, Sig)
        cache = {}
        # the vanilla draw's own alignment, so that a "reconstruction" point can
        # be placed at exactly its intrinsic coordinates.  This is the sharpest
        # internal control in the atlas: if (S, D, rho) is sufficient, an A with
        # a completely different parameter-space norm must train identically.
        _, S_ih, tau, U = whiten_ops(Sig, Cg)
        cache.update(S_half=sym_half(Sig), S_ihalf=S_ih, tau=tau, U=U)
        V_ref = torch.linalg.qr((A_ref @ cache["S_half"]).T)[0]
        rho_ref = captured_of(V_ref, tau, U)
        rr = rho_ref if rho == "ref" else rho
        if args.matchW is not None:
            W_ref = metric_ratio(A_ref, Sig)
            Rg_ref = gradient_alignment(A_ref, Sig, Cg)
            A, _, _ = build_A_matched(
                args.S * S_ref, args.D if args.D is not None else D_ref,
                args.matchW * W_ref, Rg_ref, r, Sig, Cg, cache=cache,
                generator=torch.Generator(device=Sig.device).manual_seed(h),
                iters=args.matchW_iters)
        else:
            A = build_A(args.S * S_ref,
                        args.D if args.D is not None else D_ref,
                        rr, r, Sig, Cg, generator=g, cache=cache,
                        wexp=args.wexp)
        s_got, d_got = intrinsic_state(A, Sig)
        V = torch.linalg.qr((A @ cache["S_half"]).T)[0]
        st = p_stats(A.cpu(), s=1.0); st.pop("spec_top4", None)
        st.update(S_abs=s_got, S_rel=s_got / S_ref, D=d_got, D_ref=D_ref,
                  rho_rel=captured_of(V, cache["tau"], cache["U"]),
                  rho_ref=rho_ref, trP_ref=float((A_ref * A_ref).sum()),
                  W=metric_ratio(A, Sig),
                  W_ref=metric_ratio(A_ref, Sig),
                  R_g=gradient_alignment(A, Sig, Cg),
                  R_g_ref=gradient_alignment(A_ref, Sig, Cg),
                  A_fro=float(A.norm()))
        stats[name] = st
        SIGMA_CPU[name] = Sig.float().cpu()
        return A.float().cpu()

    adapters = apply_lora(model, args.r, args.alpha, factory, targets=targets)
    params = lora_parameters(adapters)
    if CACHED is None:
        tmp = cfile + f".tmp{os.getpid()}"
        torch.save(dict(A={n: a.lora_A.detach().float().cpu()
                           for n, a in adapters.items()},
                        stats=stats, sigma=SIGMA_CPU), tmp)
        os.replace(tmp, cfile)
    del G
    torch.cuda.empty_cache()

    # trajectory in the FIXED initial data metric
    traj = []
    sample = [n for n in adapters
              if n.endswith(("layers.0.self_attn.q_proj",
                             "layers.13.mlp.down_proj",
                             "layers.27.self_attn.o_proj"))]

    @torch.no_grad()
    def cb(t, model_, adapters_):
        if t % args.traj_every and t != 0:
            return
        rec = {"step": t, "per_layer": {}}
        Ss, Ds, Bn = [], [], []
        for n in sample:
            ad = adapters_[n]
            Sg = SIGMA_CPU[n].double().cuda()
            A = ad.lora_A.detach().double()
            s_, d_ = intrinsic_state(A, Sg)
            Ss.append(s_); Ds.append(d_)
            Bn.append(float(ad.lora_B.norm()))
            rec["per_layer"][n] = dict(S=s_, D=d_, B=float(ad.lora_B.norm()),
                                       dW=float(ad.delta_w().norm()))
            del Sg
        rec.update(S_mean=sum(Ss) / len(Ss), D_mean=sum(Ds) / len(Ds),
                   B_mean=sum(Bn) / len(Bn))
        traj.append(rec)

    base_eval = eval_loss(model, tel, args.eval_batches, "cuda")
    acc_set = gsm8k_eval_set(args.acc_n) if args.acc_n else None
    cfg = dict(steps=args.steps, accum=accum, optimizer=args.optimizer,
               lr=args.lr, wd=0.0, warmup=args.warmup, sched=args.sched,
               grad_clip=args.grad_clip, a_lr_ratio=args.a_lr_ratio,
               momentum=args.momentum)
    log = train(model, adapters, params, trl, tel, cfg, log_every=5,
                eval_every=args.eval_every, eval_batches=args.eval_batches,
                sample_layers=sample, callback=cb)
    acc = None
    if acc_set:
        try:
            acc, _ = gsm8k_accuracy(model, tok, acc_set, bs=32)
        except Exception as e:                                    # noqa: BLE001
            print("acc failed:", e)
    json.dump(dict(cell=cell, args=vars(args), base_eval_loss=base_eval,
                   final_acc=acc, init_stats=stats, traj=traj, log=log),
              open(outfile, "w"))
    m = list(stats.values())
    import statistics as st
    print(f"[{cell}] S_rel={st.mean(x['S_rel'] for x in m):.3f} "
          f"D={st.mean(x['D'] for x in m):.2f} "
          f"W/W0={st.mean(x['W']/x['W_ref'] for x in m):.2f} "
          f"Rg/Rg0={st.mean(x['R_g']/x['R_g_ref'] for x in m):.2f} "
          f"trP={st.mean(x['tr_P'] for x in m):.3f} "
          f"base={base_eval:.5f} final={log['final_eval_loss']:.5f} "
          f"({log['wall_time']:.0f}s)")


if __name__ == "__main__":
    main()
