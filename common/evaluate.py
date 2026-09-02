"""Downstream accuracy evaluation.

The mother paper (NoRA) reports benchmark accuracy — GSM8K exact match,
HumanEval/MBPP pass@1 — not held-out loss, and its headline is a +5.44 point
average SFT gain.  Loss deltas of 1e-3 nats are not comparable to that, so every
decisive comparison here is also scored on accuracy.
"""
import math
import re

import torch

GSM_GOLD = re.compile(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
ANSWER_IS = re.compile(r"[Tt]he answer is:?\s*(-?[0-9][0-9,]*\.?[0-9]*)")


def _norm(x):
    if x is None:
        return None
    x = x.replace(",", "").rstrip(".")
    if len(x) > 40:           # a runaway digit string overflows float()
        return x
    try:
        v = float(x)
        if not math.isfinite(v):
            return x
        return str(int(v)) if v == int(v) else str(v)
    except (ValueError, OverflowError):
        return x


def extract_pred(text):
    """Accept both the GSM8K '#### x' convention and MetaMath's
    'The answer is: x'; otherwise fall back to the last number."""
    m = GSM_GOLD.search(text)
    if m:
        return _norm(m.group(1))
    m = ANSWER_IS.search(text)
    if m:
        return _norm(m.group(1))
    nums = NUM.findall(text)
    return _norm(nums[-1]) if nums else None


@torch.no_grad()
def gsm8k_accuracy(model, tok, examples, device="cuda", max_new=320, bs=16,
                   amp=None, stop_str="\nQuestion:"):
    """examples: list of (prompt, gold_answer_string)."""
    model.eval()
    was_cache = model.config.use_cache
    model.config.use_cache = True
    tok.padding_side = "left"
    n_ok, n = 0, 0
    outs = []
    ctx = amp if amp is not None else torch.autocast("cuda", enabled=False)
    for i in range(0, len(examples), bs):
        chunk = examples[i:i + bs]
        enc = tok([p for p, _ in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with ctx:
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id)
        for j, (p, gold) in enumerate(chunk):
            txt = tok.decode(gen[j][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            if stop_str and stop_str in txt:
                txt = txt.split(stop_str)[0]
            try:
                pred = extract_pred(txt)
                g = (_norm(GSM_GOLD.search(gold).group(1))
                     if GSM_GOLD.search(gold) else _norm(gold))
            except Exception:
                pred, g = None, None
            n_ok += int(pred is not None and pred == g)
            n += 1
            if len(outs) < 4:
                outs.append((pred, g, txt[:200]))
    model.config.use_cache = was_cache
    tok.padding_side = "right"
    model.train()
    return n_ok / max(n, 1), outs


def gsm8k_eval_set(n=250, seed=0):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")["test"].shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    return [(f"Question: {e['question'].strip()}\nAnswer:", e["answer"])
            for e in ds]
