"""Adam whose second moment is shared across blocks of the rank index.

This is the paper's map turned from a table into an axis.  The map says AdamW
sees the frame and Muon does not; that is a binary fact about two optimizers,
and a reviewer can always answer "those two differ in many ways besides
symmetry".  `GroupAdam_b` removes that answer, because b moves the symmetry
group and nothing else.

Construction.  The gauge acts on the rank index only: (A, B) -> (QA, BQ^T).
Partition the r rank coordinates into r/b contiguous blocks.  For A (r x d_in),
keep ONE second-moment scalar per (block J, input coordinate j):

    v^A_{J,j} <- beta2 v^A_{J,j} + (1 - beta2) * (1/b) ||G^A_{J,j}||_2^2

and step  dA_{J,j} = -eta m^A_{J,j} / (sqrt(v^A_{J,j}) + eps).  For B
(d_out x r) the same with (output row i, block J).

Theorem.  Let Q = diag(Q_1, ..., Q_{r/b}) with each Q_k in O(b), aligned to the
blocks.  Under the gauge the gradients transform as G^A -> Q G^A and
G^B -> G^B Q^T (chain rule on B'A' = BA).  Then per block

    ||Q_J G^A_{J,j}||^2 = ||G^A_{J,j}||^2      so v is INVARIANT,
    m^A_{J,j} -> Q_J m^A_{J,j}                 so m is COVARIANT,

hence dA_{J,j} -> Q_J dA_{J,j}, i.e. dA -> Q dA: the update commutes with the
gauge action exactly, at every step, with momentum.  So

    GroupAdam_b  is exactly equivariant to  H_b = O(b)^{r/b}.

The two ends are the optimizers we already have:

    b = 1   H_1 = O(1)^r = signed permutations' continuous part -> AdamW
            (with the 1/b mean over a single element, this IS Adam elementwise)
    b = r   H_r = O(r)                                 -> fully gauge-blind

and in between the invisible group has dimension (r/b)*b(b-1)/2 = r(b-1)/2, so
the frame dimension the optimizer can still resolve is

    r(r-1)/2 - r(b-1)/2 = r(r-b)/2.

For r = 16 that is 120, 112, 96, 64, 0 visible dimensions at b = 1, 2, 4, 8, 16.

Why this is not the one-parameter family of Singh et al. (2608.05136).  Theirs
interpolates coordinate-wise -> shared-scalar preconditioning, an ANISOTROPY
axis whose interior points are equivariant to no exact group; it yields a
monotone trend.  Ours is a subgroup CHAIN, O(1)^r < O(2)^{r/2} < ... < O(r),
exact at every rung, so it predicts a discrete staircase: a rotation confined to
k-blocks is invisible iff k <= b.  That is falsifiable in a way a trend is not.
It also blocks the rank index ONLY, leaving per-input-coordinate adaptivity
untouched, so b is not a proxy for "how adaptive the optimizer is".
"""
import torch


def _rank_axis(shape):
    """The gauge acts on the rank index, which is the short side of a LoRA
    factor (r = 16 against d ~ 2048).  Returns None for non-2-D parameters."""
    if len(shape) != 2:
        return None
    return 0 if shape[0] <= shape[1] else 1


class GroupAdam(torch.optim.Optimizer):
    """Adam with the second moment averaged over blocks of b rank coordinates.

    b = 1 reproduces AdamW exactly (the mean over a one-element block is the
    element itself); b = r shares one scalar across the whole rank index and is
    exactly O(r)-equivariant.  Non-2-D parameters fall back to elementwise Adam,
    which is moot here because only the LoRA factors train.
    """

    def __init__(self, params, lr=1e-3, b=1, betas=(0.9, 0.999), eps=1e-8,
                 wd=0.0):
        if b < 1:
            raise ValueError(f"block size b must be >= 1, got {b}")
        super().__init__(params, dict(lr=lr, b=b, betas=betas, eps=eps, wd=wd))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            b1, b2 = g["betas"]
            blk = g["b"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                st = self.state[p]
                ax = _rank_axis(d.shape)
                if ax is not None:
                    r = d.shape[ax]
                    if r % blk:
                        raise ValueError(
                            f"block size b={blk} does not divide rank r={r}")
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(d)
                    st["v"] = torch.zeros_like(
                        _blocked(d, ax, blk).mean(dim=_MEAN_DIM[ax])
                        if ax is not None else d)
                st["t"] += 1
                t = st["t"]
                st["m"].mul_(b1).add_(d, alpha=1 - b1)
                mh = st["m"] / (1 - b1 ** t)
                if ax is None:
                    st["v"].mul_(b2).addcmul_(d, d, value=1 - b2)
                    upd = mh / ((st["v"] / (1 - b2 ** t)).sqrt() + g["eps"])
                else:
                    # mean square over the b rank coordinates of each block:
                    # the 1/b keeps the step scale comparable across b, and
                    # makes b = 1 identical to elementwise Adam.
                    ms = _blocked(d * d, ax, blk).mean(dim=_MEAN_DIM[ax])
                    st["v"].mul_(b2).add_(ms, alpha=1 - b2)
                    vh = st["v"] / (1 - b2 ** t)
                    den = (vh.sqrt() + g["eps"]).unsqueeze(_MEAN_DIM[ax])
                    upd = (_blocked(mh, ax, blk) / den).reshape(d.shape)
                if g["wd"]:
                    p.mul_(1 - g["lr"] * g["wd"])
                p.add_(upd, alpha=-g["lr"])
        return loss


# where the b-axis lands after _blocked(), per rank axis
_MEAN_DIM = {0: 1, 1: 2}


def _blocked(x, ax, blk):
    """Split the rank axis into (r/b, b).

    A (r x d_in), ax = 0  ->  (r/b, b, d_in),   b-axis = 1
    B (d_out x r), ax = 1 ->  (d_out, r/b, b),  b-axis = 2
    """
    if ax == 0:
        return x.reshape(x.shape[0] // blk, blk, x.shape[1])
    return x.reshape(x.shape[0], x.shape[1] // blk, blk)


def block_rotation(r, k, device="cpu", dtype=torch.float64, generator=None):
    """Q = diag(Q_1, ..., Q_{r/k}) with each Q_i uniform in O(k).

    k = 1 gives random signs, which is O(1)^r -- the group AdamW already
    respects, so it is the null cell of the staircase.  Blocks are contiguous,
    so for powers of two the k-blocks nest inside the b-blocks and a rotation
    is inside H_b exactly when k <= b.
    """
    if r % k:
        raise ValueError(f"k={k} must divide r={r}")
    Q = torch.zeros(r, r, device=device, dtype=dtype)
    for i in range(r // k):
        if k == 1:
            blk = torch.randint(0, 2, (1, 1), device=device,
                                generator=generator, dtype=torch.int64)
            blk = (blk * 2 - 1).to(dtype)
        else:
            M = torch.randn(k, k, device=device, dtype=dtype,
                            generator=generator)
            blk, R = torch.linalg.qr(M)
            # fix the sign convention so Q is Haar-distributed on O(k)
            blk = blk * torch.sign(torch.diagonal(R)).unsqueeze(0)
        Q[i * k:(i + 1) * k, i * k:(i + 1) * k] = blk
    return Q
