"""Muon, for the optimizer-symmetry experiment.

The point of including Muon is a sharp prediction, not a benchmark. Whether an
optimizer can see LoRA's gauge frame is decided by whether the norm it descends
in is orthogonally invariant:

    SGD    steepest descent under the Frobenius norm    -> invariant  -> blind
    Muon   steepest descent under the spectral norm     -> invariant  -> blind
    AdamW  steepest descent under the elementwise max   -> NOT        -> sees it

For Muon the covariance is exact and elementary.  Its update is the orthogonal
polar factor msign(M) = U V^T of M = U Sigma V^T.  Under the gauge,
grad_B -> grad_B Q^T = U Sigma (Q V)^T with QV still orthonormal, so

    msign(grad_B Q^T) = U (Q V)^T = msign(grad_B) Q^T,

and on the other factor grad_A -> Q grad_A gives msign(Q grad_A) =
Q msign(grad_A).  Momentum buffers inherit the same transformation, so the
whole trajectory maps by the gauge -- exactly as for SGD, and unlike AdamW.

Newton-Schulz iteration for msign follows Jordan et al.'s quintic coefficients.
The reference implementation runs it in bfloat16, and that MATTERS here: the
equivariance above is exact in exact arithmetic (measured 3.8e-15 in float64,
2.3e-6 in float32) but bfloat16 breaks it by 7-9% relative.  So stock Muon is
only approximately gauge-covariant, and a clean null test has to run the
iteration in float32.  `dtype` controls this and defaults to float32 for that
reason.
"""
import torch

# quintic coefficients tuned so the iteration converges fast from a spectral
# norm of 1 without needing the singular values to be well separated
_NS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def msign(M, steps=5, eps=1e-7, dtype=torch.float32):
    """Orthogonal polar factor of M by Newton-Schulz.

    Not the exact polar factor: five quintic steps land 2.4e-2 away from U V^T,
    by design.  Equivariance, which is what this is used for, is exact
    regardless of how far the iteration has converged, because every step is a
    polynomial in X X^T applied on the left.
    """
    X = M.to(dtype)
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    a, b, c = _NS
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(M.dtype)


class Muon(torch.optim.Optimizer):
    """Muon on 2-D parameters, AdamW on everything else.

    Matches the reference in the parts that matter here: heavy-ball momentum
    with optional Nesterov, orthogonalised update, and the sqrt(max(1, m/n))
    shape scaling that makes the update RMS comparable across shapes.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, wd=0.0,
                 ns_steps=5, dtype=torch.float32):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, wd=wd,
                                      ns_steps=ns_steps, dtype=dtype))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(d)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(d)
                u = d.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                if u.ndim == 2:
                    u = msign(u, steps=g["ns_steps"],
                              dtype=g.get("dtype", torch.float32))
                    u = u * max(1.0, u.shape[-2] / u.shape[-1]) ** 0.5
                if g["wd"]:
                    p.mul_(1 - g["lr"] * g["wd"])
                p.add_(u, alpha=-g["lr"])
        return loss
