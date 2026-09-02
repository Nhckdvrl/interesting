"""Topic 01 -- the literature audit inside the matched framework.

Runs published LoRA initializers as *samples in P-space* rather than as
competing methods, alongside:
  * `left_gauge`, the provably content-free null (identical P_0), and
  * the exact matched constructions from `common/pinit.py`.

`--match_trace 1` rescales A_0 (compensating B_0 so the initial function is
untouched) to the trace of the vanilla kaiming draw, removing the update
magnitude confound; `--match_trace 0` runs each method at its published scale.
The difference between the two arms is the magnitude channel.
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from common.lora import apply_lora, lora_parameters, DEFAULT_TARGETS
from common.pinit import make_A, kaiming_A
from common.pstats import p_stats
from common import initializers as IN
from common.data import build_sft, FixedOrderLoader
from common.train import load_model, train, eval_loss, set_amp
from common.evaluate import gsm8k_accuracy, gsm8k_eval_set

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ACACHE = os.path.expanduser("~/.cache/nora_repo_A0")


def cached_make_A(kind, r, d_in, seed_key, ref_A):
    """Disk-cache the Schur-Horn constructions: they are O(d_in) python-level
    Givens rotations and are identical across every job that shares
    (kind, r, d_in, seed)."""
    if "flatdiag" not in kind:
        g = torch.Generator().manual_seed(0)
        return make_A(kind, r, d_in, g, "cpu", ref_A=ref_A)
    os.makedirs(ACACHE, exist_ok=True)
    key = hashlib.md5(f"{kind}|{r}|{d_in}|{seed_key}".encode()).hexdigest()
    f = os.path.join(ACACHE, key + ".pt")
    if os.path.exists(f):
        try:
            return torch.load(f)
        except Exception:
            pass
    g = torch.Generator().manual_seed(0)
    A = make_A(kind, r, d_in, g, "cpu", ref_A=ref_A)
    tmp = f + f".tmp{os.getpid()}"
    torch.save(A, tmp); os.replace(tmp, f)
    return A

# which module's input covariance each target reuses (q/k/v share an input,
# gate/up share an input)
ACT_GROUP = {"q_proj": "q_proj", "k_proj": "q_proj", "v_proj": "q_proj",
             "o_proj": "o_proj", "gate_proj": "gate_proj",
             "up_proj": "gate_proj", "down_proj": "down_proj"}


def collect_grads(model, mods, loader, n_batches, device="cuda"):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in mods.values():
        m.weight.requires_grad_(True)
    for i in range(n_batches):
        b = {k: v.to(device) for k, v in loader.get(i).items()}
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        lg = out.logits[:, :-1].float(); lb = b["labels"][:, 1:]
        msk = lb != -100
        loss = nn.functional.cross_entropy(lg[msk], lb[msk])
        (loss / n_batches).backward()
    G = {n: m.weight.grad.detach().float().cpu() for n, m in mods.items()}
    for m in mods.values():
        m.weight.grad = None
        m.weight.requires_grad_(False)
    return G


@torch.no_grad()
def collect_act_cov(model, mods, loader, n_batches, device="cuda", on_cpu=False):
    reps = {n: m for n, m in mods.items()
            if ACT_GROUP[n.split(".")[-1]] == n.split(".")[-1]}
    cov, hooks = {}, []
    def mk(n):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            c = x.T @ x
            if on_cpu:
                c = c.cpu()
            cov[n] = cov.get(n, 0) + c
        return hook
    for n, m in reps.items():
        hooks.append(m.register_forward_hook(mk(n)))
    for i in range(n_batches):
        b = {k: v.to(device) for k, v in loader.get(i).items()}
        model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    for h in hooks:
        h.remove()
    return {n: (c if c.device.type == "cpu" else c.cpu()) for n, c in cov.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gauge_seed", type=int, default=0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--match", default="trace",
                    choices=["none", "trace", "trace_act", "trace_grad"],
                    help="which scale invariant to match to the vanilla kaiming "
                         "draw.  tr P is the parameter-space norm; but the "
                         "preconditioner acts on DATA, so the size of the "
                         "function change is set by tr(P Sigma_x) and the "
                         "first-order descent by tr(P G^T G).  Matching tr P "
                         "alone leaves those free -- which is exactly the "
                         "loophole every data-aware initializer exploits.")
    ap.add_argument("--subtract", type=int, default=1)
    ap.add_argument("--b0_rel", type=float, default=0.01)
    ap.add_argument("--dtype", default="float32",
                    help="bf16 rounding of the base-weight subtraction used by "
                         "PiSSA/OLoRA/LoRA-One is a 4e-3 nat confound; fp32 "
                         "keeps every condition function-identical at init")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--micro_bs", type=int, default=0,
                    help="0 = no accumulation (micro_bs = bs).  Needed at 7B, "
                         "where a fp32 forward at bs=16 does not fit.")
    ap.add_argument("--probe_bs", type=int, default=0,
                    help="batch size for the one-shot gradient/activation "
                         "probes; 0 = same as micro_bs")
    ap.add_argument("--act_cov_device", default="auto",
                    help="'cpu' keeps the d_in x d_in activation covariances "
                         "off the GPU; at 7B the down_proj covariance alone is "
                         "822 MB per layer")
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_eval", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--probe_batches", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--scaling", default="standard",
                    choices=["standard", "rsqrt"],
                    help="rsqrt = rsLoRA rank-stabilised scaling alpha/sqrt(r)")
    ap.add_argument("--b_lr_ratio", type=float, default=1.0,
                    help=">1 reproduces LoRA+ (larger LR on the up-projection)")
    ap.add_argument("--amp", default="none", choices=["none", "bf16"])
    ap.add_argument("--sched", default="constant")
    ap.add_argument("--acc_n", type=int, default=0,
                    help="if >0, score GSM8K exact-match accuracy on this many "
                         "held-out problems before and after training")
    ap.add_argument("--acc_bs", type=int, default=16)
    ap.add_argument("--acc_max_new", type=int, default=320)
    args = ap.parse_args()

    cell = (f"{args.cond}_lr{args.lr:g}_s{args.seed}"
            + (f"_g{args.gauge_seed}" if args.cond == "left_gauge" else "")
            + ("" if args.match == "trace" else f"_m{args.match}")
            + ("" if args.subtract else "_nosub")
            + (f"_{args.optimizer}" if args.optimizer != "adamw" else "")
            + ("" if args.dtype == "float32" else f"_{args.dtype}")
            + ("" if args.scaling == "standard" else f"_{args.scaling}")
            + ("" if args.b_lr_ratio == 1.0 else f"_bl{args.b_lr_ratio:g}")
            + (f"_r{args.r}" if args.r != 16 else "")
            + (f"_a{args.alpha:g}" if args.alpha != 32.0 else ""))
    outdir = os.path.join(REPO, "01-hidden-preconditioner", "results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, cell + ".json")
    if os.path.exists(outfile):
        print("skip (exists)", outfile); return

    torch.manual_seed(args.seed)
    set_amp(args.amp)
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

    # both probes are always collected: they are needed for the weighted-trace
    # statistics that every condition is scored on, not only for the
    # initializers that use them.
    probe_bs = args.probe_bs or micro
    probe_ld = (trl if probe_bs == micro
                else FixedOrderLoader(tr, probe_bs, tok.pad_token_id, seed=0))
    on_cpu = (args.act_cov_device == "cpu" or
              (args.act_cov_device == "auto" and
               model.config.hidden_size >= 2048))
    G = collect_grads(model, mods, probe_ld, args.probe_batches)
    ACT = collect_act_cov(model, mods, probe_ld, args.probe_batches,
                          on_cpu=on_cpu)
    torch.cuda.empty_cache()
    # the weighted-trace statistics stream Sigma / C_g back to the GPU one
    # module at a time, so the probes never need to be resident together
    s = args.alpha / args.r
    pstats = {}

    def weighted_traces(A, name, d_in):
        """tr(P), tr(P Sigma_x), tr(P C_g) up to the common factor s^2."""
        Ad = A.float().cuda()
        key = name.rsplit(".", 1)[0] + "." + ACT_GROUP[name.split(".")[-1]]
        Sig = ACT[key].cuda()
        Gc = G[name].cuda()
        t_p = float(Ad.pow(2).sum())
        t_act = float(((Ad @ Sig) * Ad).sum())
        GA = Gc @ Ad.T
        t_grad = float(GA.pow(2).sum())      # tr(A C_g A^T) = ||G A^T||_F^2
        return t_p, t_act, t_grad

    def factory(name, r, d_in, d_out):
        h = int(hashlib.md5(f"{args.seed}:{name}".encode()).hexdigest()[:12], 16)
        g = torch.Generator().manual_seed(h)
        base = kaiming_A(r, d_in, g, "cpu")
        ref_tr = float(base.pow(2).sum())
        W = mods[name].weight.detach().float()   # stays on GPU for SVD/QR
        A, B, sub = None, None, False
        c = args.cond
        if c in ("kaiming",):
            A = base
        elif c == "left_gauge":
            g2 = torch.Generator().manual_seed(
                int(hashlib.md5(f"gauge{args.gauge_seed}:{name}".encode())
                    .hexdigest()[:12], 16))
            A = make_A("left_gauge", r, d_in, g2, "cpu", ref_A=base)
        elif c == "etf":
            A = IN.init_etf(r, d_in, g, ref_tr)
        elif c == "eva":
            key = name.rsplit(".", 1)[0] + "." + ACT_GROUP[name.split(".")[-1]]
            A = IN.init_eva(ACT[key].cuda(), r, d_in, ref_tr).cpu().double()
        elif c == "gradsub":
            A = IN.init_gradsubspace(G[name].cuda(), r, ref_tr).cpu().double()
        elif c in ("pissa", "pissa_minor"):
            A, B = IN.init_pissa(W, r, s, minor=(c == "pissa_minor"))
            A, B = A.cpu().double(), B.cpu().double()
            sub = bool(args.subtract)
        elif c == "olora":
            A, B = IN.init_olora(W, r, s)
            A, B = A.cpu().double(), B.cpu().double()
            sub = bool(args.subtract)
        elif c == "lora_one":
            A, B = IN.init_lora_one(G[name].cuda(), r, s, W=W,
                                    b0_rel=args.b0_rel)
            A, B = A.cpu().double(), B.cpu().double()
            sub = bool(args.subtract)
        else:
            A = cached_make_A(c, r, d_in, f"{args.seed}:{name}", base)
        if args.match != "none":
            idx = {"trace": 0, "trace_act": 1, "trace_grad": 2}[args.match]
            tgt = weighted_traces(base, name, d_in)[idx]
            cur = weighted_traces(A, name, d_in)[idx]
            k = (tgt / max(cur, 1e-30)) ** 0.5
            A = A * k
            if B is not None:
                B = B / k            # product (hence initial function) preserved
        st = p_stats(A, s=1.0); st.pop("spec_top4", None)
        tp, ta, tg = weighted_traces(A, name, d_in)
        bp, ba, bg = weighted_traces(base, name, d_in)
        st.update(tr_P_act=ta, tr_P_grad=tg,
                  rel_tr_P=tp / bp, rel_tr_act=ta / ba, rel_tr_grad=tg / bg,
                  B0_norm=float(B.norm()) if B is not None else 0.0)
        pstats[name] = st
        if B is None:
            return A.float()
        return dict(A=A.float(), B=B.float(), subtract=sub)

    adapters = apply_lora(model, args.r, args.alpha, factory, targets=targets,
                          scaling=args.scaling)
    params = lora_parameters(adapters)
    base_eval = eval_loss(model, tel, args.eval_batches, "cuda")
    cfg = dict(steps=args.steps, accum=accum, optimizer=args.optimizer, lr=args.lr,
               wd=0.0, warmup=args.warmup, sched=args.sched,
               grad_clip=args.grad_clip, momentum=args.momentum,
               b_lr_ratio=args.b_lr_ratio)
    acc_set = gsm8k_eval_set(args.acc_n) if args.acc_n else None
    base_acc = None
    if acc_set:
        base_acc, _ = gsm8k_accuracy(model, tok, acc_set, bs=args.acc_bs,
                                     max_new=args.acc_max_new)
        print(f"  base GSM8K acc = {base_acc:.4f}", flush=True)
    log = train(model, adapters, params, trl, tel, cfg, log_every=5,
                eval_every=args.eval_every, eval_batches=args.eval_batches,
                sample_layers=[n for n in adapters
                               if n.endswith("layers.13.mlp.down_proj")])
    acc, samples = (gsm8k_accuracy(model, tok, acc_set, bs=args.acc_bs,
                                   max_new=args.acc_max_new)
                    if acc_set else (None, None))
    json.dump(dict(cell=cell, args=vars(args), base_eval_loss=base_eval,
                   base_acc=base_acc, final_acc=acc,
                   acc_samples=samples,
                   init_pstats=pstats, log=log), open(outfile, "w"))
    print(f"[{cell}] base={base_eval:.5f} final={log['final_eval_loss']:.5f}"
          + (f" acc={acc:.4f} (base {base_acc:.4f})" if acc is not None else "")
          + f" ({log['wall_time']:.0f}s)")


if __name__ == "__main__":
    main()
