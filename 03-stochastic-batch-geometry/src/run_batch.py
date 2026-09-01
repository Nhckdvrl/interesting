"""Topic 03, E1/E2/E6 -- batch-size reproduction.

Fixed *example budget*, varying logical batch size, so that every cell sees the
same number of training examples and only the number of optimizer steps and the
gradient noise level change.  This is the axis along which LoRA Without Regret
reports the LoRA-vs-FullFT gap widening.

methods:
  full        full fine-tuning of all Linear weights in the transformer blocks
  lora        vanilla LoRA
  nora        NoRA-init, trace-matched (removes the pure magnitude confound)
  nora_unit   literal unit-norm columns
  lora_fa     LoRA with A frozen (LoRA-FA): P is constant for the whole run,
              which is the clean test of whether the pathology needs A to move
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from common.lora import apply_lora, lora_parameters, DEFAULT_TARGETS
from common.pinit import make_A, kaiming_A
from common.pstats import p_stats
from common.data import build_sft, FixedOrderLoader
from common.train import load_model, train, eval_loss, set_amp

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def full_ft_params(model, targets):
    ps = []
    for n, p in model.named_parameters():
        p.requires_grad_(False)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in targets:
            mod.weight.requires_grad_(True)
            ps.append(mod.weight)
    return ps


def build_factory(kind, seed):
    store = {}

    def factory(name, r, d_in, d_out):
        h = int(hashlib.md5(f"{seed}:{name}".encode()).hexdigest()[:12], 16)
        g = torch.Generator().manual_seed(h)
        base = kaiming_A(r, d_in, g, "cpu")
        A = base if kind in ("lora", "lora_fa") else make_A(kind, r, d_in, g,
                                                            "cpu", ref_A=base)
        st = p_stats(A, s=1.0); st.pop("spec_top4", None)
        store[name] = st
        return A.float()
    factory.stats = store
    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", required=True,
                    choices=["full", "lora", "nora", "nora_unit", "lora_fa"])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--bs", type=int, required=True, help="logical batch size")
    ap.add_argument("--micro_bs", type=int, default=16)
    ap.add_argument("--budget", type=int, default=16384,
                    help="number of training examples consumed (fixed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--sched", default="cosine")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--n_train", type=int, default=7000)
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--eval_batches", type=int, default=32)
    ap.add_argument("--n_evals", type=int, default=10)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    args = ap.parse_args()

    steps = max(args.budget // args.bs, 1)
    micro = min(args.micro_bs, args.bs)
    accum = args.bs // micro
    cell = f"{args.method}_bs{args.bs}_lr{args.lr:g}_s{args.seed}"
    outdir = os.path.join(REPO, "03-stochastic-batch-geometry", "results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, cell + ".json")
    if os.path.exists(outfile):
        print("skip (exists)", outfile); return

    torch.manual_seed(args.seed)
    # FullFT and LoRA must be treated identically: fp32 master weights,
    # bf16 matmuls for both.  (bf16 master weights would make FullFT and LoRA
    # differ in optimizer precision, which is a confound for a study about
    # gradient noise.)
    set_amp("bf16")
    model, tok = load_model(args.model, dtype=torch.float32)
    tr, te = build_sft(tok, args.task, args.n_train, args.n_eval, args.max_len,
                       seed=0)
    trl = FixedOrderLoader(tr, micro, tok.pad_token_id, seed=args.seed)
    tel = FixedOrderLoader(te, 16, tok.pad_token_id, seed=999)
    targets = tuple(args.targets.split(","))

    adapters, fac = {}, None
    if args.method == "full":
        params = full_ft_params(model, targets)
    else:
        fac = build_factory(args.method, args.seed)
        adapters = apply_lora(model, args.r, args.alpha, fac, targets=targets,
                              train_A=(args.method != "lora_fa"))
        params = lora_parameters(adapters)
    ntr = sum(p.numel() for p in params)
    base_eval = eval_loss(model, tel, args.eval_batches, "cuda")
    cfg = dict(steps=steps, accum=accum, optimizer="adamw", lr=args.lr, wd=0.0,
               warmup=max(int(args.warmup_frac * steps), 1), sched=args.sched,
               grad_clip=args.grad_clip)
    ev = max(steps // args.n_evals, 1)
    log = train(model, adapters, params, trl, tel, cfg,
                log_every=max(steps // 60, 1), eval_every=ev,
                eval_batches=args.eval_batches,
                sample_layers=[n for n in adapters
                               if n.endswith("layers.13.mlp.down_proj")])
    json.dump(dict(cell=cell, args=vars(args), steps=steps, accum=accum,
                   micro_bs=micro, n_trainable=ntr, base_eval_loss=base_eval,
                   init_pstats=(fac.stats if fac else {}), log=log),
              open(outfile, "w"))
    print(f"[{cell}] steps={steps} accum={accum} base={base_eval:.4f} "
          f"final={log['final_eval_loss']:.5f} ({log['wall_time']:.0f}s)")


if __name__ == "__main__":
    main()
