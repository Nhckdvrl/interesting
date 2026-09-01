"""A minimal, fully-controllable LoRA implementation.

We deliberately do NOT use peft: the experiments need
  * arbitrary user-supplied A0 (Schur-Horn constructions, gauge rotations, ...),
  * optional frozen A (LoRA-FA control),
  * per-adapter diagnostics of P = s^2 A^T A and of the merged update,
  * exact control of the scaling convention.

Convention (matches 01/README):
    A in R^{r x d_in},  B in R^{d_out x r},  B_0 = 0
    y = W x + s * B (A x),      dW = s * B A,      P = s^2 A^T A
    s = alpha / r          ("standard")
      = alpha / sqrt(r)    ("rsqrt", the rank-stabilised convention)
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, A0: torch.Tensor,
                 scaling: str = "standard", train_A: bool = True,
                 B0: torch.Tensor = None, subtract_from_base: bool = False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        if scaling == "standard":
            self.s = alpha / r
        elif scaling == "rsqrt":
            self.s = alpha / math.sqrt(r)
        else:
            raise ValueError(scaling)
        d_out, d_in = base.weight.shape
        assert A0.shape == (r, d_in), (A0.shape, (r, d_in))
        dt = base.weight.dtype
        self.lora_A = nn.Parameter(A0.to(device=base.weight.device, dtype=torch.float32),
                                   requires_grad=train_A)
        if B0 is None:
            B0 = torch.zeros(d_out, r)
        self.lora_B = nn.Parameter(B0.to(device=base.weight.device,
                                         dtype=torch.float32))
        if subtract_from_base:
            # PiSSA / OLoRA / LoRA-One style: the adapter starts at a NONZERO
            # dW and the same amount is removed from the frozen base weight, so
            # the initial *function* is unchanged while B_0 != 0.  This is the
            # only way an initializer can leave the P_0 equivalence class.
            with torch.no_grad():
                dw = self.s * (self.lora_B.float() @ self.lora_A.float())
                base.weight.data -= dw.to(base.weight.dtype)
        self.register_buffer("A0", self.lora_A.detach().clone(), persistent=False)
        self._compute_dtype = dt

    def forward(self, x):
        out = self.base(x)
        xa = torch.nn.functional.linear(x.to(self.lora_A.dtype), self.lora_A)
        delta = torch.nn.functional.linear(xa, self.lora_B) * self.s
        return out + delta.to(out.dtype)

    @torch.no_grad()
    def delta_w(self):
        return self.s * (self.lora_B @ self.lora_A)


def _get_parent(model, name):
    parts = name.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")


def apply_lora(model, r, alpha, a_factory, targets=DEFAULT_TARGETS,
               scaling="standard", train_A=True, layer_filter=None):
    """a_factory(name, r, d_in, d_out) -> A0 (r, d_in), or a dict
    {A: ..., B: ..., subtract: bool} for nonzero-B initializers."""
    to_patch = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in targets:
            if layer_filter is not None and not layer_filter(name):
                continue
            to_patch.append((name, mod))
    adapters = {}
    for name, mod in to_patch:
        d_out, d_in = mod.weight.shape
        out = a_factory(name, r, d_in, d_out)
        if isinstance(out, dict):
            A0, B0, sub = out["A"], out.get("B"), out.get("subtract", False)
        else:
            A0, B0, sub = out, None, False
        new = LoRALinear(mod, r, alpha, A0, scaling=scaling, train_A=train_A,
                         B0=B0, subtract_from_base=sub)
        parent, attr = _get_parent(model, name)
        setattr(parent, attr, new)
        adapters[name] = new
    for n, p in model.named_parameters():
        p.requires_grad_("lora_A" in n or "lora_B" in n)
    for name, ad in adapters.items():
        ad.lora_A.requires_grad_(train_A)
    return adapters


def lora_parameters(adapters):
    ps = []
    for ad in adapters.values():
        if ad.lora_A.requires_grad:
            ps.append(ad.lora_A)
        ps.append(ad.lora_B)
    return ps
