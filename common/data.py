"""Cheap deterministic SFT data.  Completion-only loss masking."""
import os, re, torch
from datasets import load_dataset

GSM8K_DIR = None  # resolved from the HF cache automatically


def _fmt_gsm8k(ex):
    q = ex["question"].strip()
    a = ex["answer"].strip()
    prompt = f"Question: {q}\nAnswer:"
    completion = " " + a
    return prompt, completion


def _fmt_numina(ex):
    return (f"Question: {ex['problem'].strip()}\nAnswer:",
            " " + ex["solution"].strip())


FORMATTERS = {"gsm8k": _fmt_gsm8k, "numina": _fmt_numina}


CACHE_DIR = os.path.expanduser("~/.cache/nora_repo_sft")


def build_sft(tokenizer, task="gsm8k", n_train=4000, n_eval=400, max_len=512,
              seed=0):
    """Tokenised, completion-masked SFT pairs.  Cached on disk because the same
    (task, n, max_len, seed, tokenizer) tuple is rebuilt by every panel job."""
    import pickle, hashlib
    key = hashlib.md5(
        f"{task}|{n_train}|{n_eval}|{max_len}|{seed}|"
        f"{tokenizer.name_or_path}|{len(tokenizer)}".encode()).hexdigest()
    os.makedirs(CACHE_DIR, exist_ok=True)
    cf = os.path.join(CACHE_DIR, key + ".pkl")
    if os.path.exists(cf):
        try:
            with open(cf, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    out = _build_sft(tokenizer, task, n_train, n_eval, max_len, seed)
    tmp = cf + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, cf)
    return out


def _build_sft(tokenizer, task="gsm8k", n_train=4000, n_eval=400, max_len=512,
               seed=0):
    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")
        tr, te = ds["train"], ds["test"]
    elif task == "numina":
        # single large SFT pool; a disjoint held-out slice serves as eval.
        d = load_dataset("AI-MO/NuminaMath-CoT", split="train")
        d = d.select(range(min(len(d), n_train + n_eval + 20000)))
        d = d.shuffle(seed=1234)
        te = d.select(range(n_eval))
        tr = d.select(range(n_eval, len(d)))
    else:
        raise ValueError(task)
    fmt = FORMATTERS[task]

    def encode(split, n):
        split = split.shuffle(seed=seed).select(range(min(n, len(split))))
        out = []
        for ex in split:
            prompt, comp = fmt(ex)
            pi = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            ci = tokenizer(comp, add_special_tokens=False)["input_ids"]
            ci = ci + [tokenizer.eos_token_id]
            ids = (pi + ci)[:max_len]
            labels = ([-100] * len(pi) + ci)[:max_len]
            if sum(x != -100 for x in labels) < 4:
                continue
            out.append((ids, labels))
        return out

    return encode(tr, n_train), encode(te, n_eval)


def collate(batch, pad_id):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a)
        lab[i, :len(b)] = torch.tensor(b)
        att[i, :len(a)] = 1
    return dict(input_ids=ids, labels=lab, attention_mask=att)


class FixedOrderLoader:
    """Deterministic, reproducible stream of *token-budget-balanced* batches.

    Every condition in a matched comparison sees the SAME ordered sequence of
    examples, so trajectories are directly comparable.  Batches are formed by a
    fixed permutation, then sorted-by-length within a bucket to limit padding.
    """
    def __init__(self, data, batch_size, pad_id, seed=0, bucket=8):
        self.data, self.bs, self.pad_id = data, batch_size, pad_id
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(data), generator=g).tolist()
        # length-bucketing inside blocks of bucket*batch_size to cut padding
        blk = bucket * batch_size
        order = []
        for i in range(0, len(perm), blk):
            chunk = perm[i:i + blk]
            chunk.sort(key=lambda j: len(data[j][0]))
            order.extend(chunk)
        self.batches = [order[i:i + batch_size]
                        for i in range(0, len(order) - batch_size + 1, batch_size)]

    def __len__(self):
        return len(self.batches)

    def get(self, i):
        idx = self.batches[i % len(self.batches)]
        return collate([self.data[j] for j in idx], self.pad_id)


ANS_RE = re.compile(r"####\s*([\-0-9\.,]+)")


def gsm8k_gold(ans):
    m = ANS_RE.search(ans)
    return m.group(1).replace(",", "").strip() if m else None
