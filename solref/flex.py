"""Sol-Attn executed through torch flex_attention, with the approximate
correction kept.

Why this exists
---------------
NVIDIA's Triton reference (`fwd.py`) walks the selected KV blocks **serially**
inside the kernel::

    for _ in range(tl.sum(exact.to(tl.int32))):

so its cost grows with the number of exact blocks. `flex_attention` lowers to
fused FlashAttention-style kernels and has no such loop. Credit for that
observation goes to KingGore's ComfyUI_sol-attn_Blackwell -- including the
detail that the BlockMask must be built directly via
``BlockMask.from_kv_blocks`` rather than ``create_block_mask``, which vmaps the
mask function over the whole [B,H,T,T] index space and explodes.

What this does differently
--------------------------
KingGore's version masks unselected KV blocks away entirely. Sol-Attn does not:
skipped blocks still contribute through their block summaries. This module keeps
that correction.

`kc` is a block **mean** of k; `vc` is a block **sum** of v. NVIDIA accumulates
``numerator += p @ vc`` and ``denominator += sum(p * block_len)``. Since
``exp(s + log L) == L * exp(s)``, feeding ``vc/len`` as the values and adding
``log(len)`` to the scores reproduces both terms exactly. So the approximate
branch is plain attention over ~T/128 summary keys, and the two branches merge
by log-sum-exp -- algebraically the same as NVIDIA's fused online softmax.

Routing runs at 128-token granularity (flex's mask block size) rather than
NVIDIA's 64, so slightly more work is done exactly. That moves output toward
dense attention, never away.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

FLEX_BLOCK = 128

_compiled_flex = None


def _flex():
    global _compiled_flex
    if _compiled_flex is None:
        from torch.nn.attention.flex_attention import flex_attention

        _compiled_flex = torch.compile(flex_attention, dynamic=False)
    return _compiled_flex


def _summaries(k_h, v_h, tokens, n_blocks):
    """Block mean of k, block sum of v, and true block lengths."""
    b, h, _, d = k_h.shape
    kb = k_h.view(b, h, n_blocks, FLEX_BLOCK, d).float()
    vb = v_h.view(b, h, n_blocks, FLEX_BLOCK, d).float()

    lens = torch.full((n_blocks,), float(FLEX_BLOCK), device=k_h.device)
    tail = tokens - (n_blocks - 1) * FLEX_BLOCK
    lens[-1] = float(tail)
    L = lens.view(1, 1, n_blocks, 1)

    # padded rows are zero, so a plain sum is already the real sum
    kc = kb.sum(dim=3) / L
    vc = vb.sum(dim=3)
    return kc, vc, lens


def _routing(q_h, kc, tokens, n_blocks, tau):
    """fwd.py's diag threshold. The `scale` factor cancels on both sides."""
    b, h, _, d = q_h.shape
    qb = q_h.view(b, h, n_blocks, FLEX_BLOCK, d).float()
    lens = torch.full((n_blocks,), float(FLEX_BLOCK), device=q_h.device)
    lens[-1] = float(tokens - (n_blocks - 1) * FLEX_BLOCK)
    q_bar = qb.sum(dim=3) / lens.view(1, 1, n_blocks, 1)

    kc_mean = kc.mean(dim=2, keepdim=True)
    kc_var = kc.var(dim=2, unbiased=False, keepdim=True)
    mean_term = (q_bar @ kc_mean.transpose(-1, -2)).squeeze(-1)
    var_term = ((q_bar * q_bar) @ kc_var.transpose(-1, -2)).squeeze(-1)
    threshold = mean_term + tau * torch.sqrt(var_term.clamp_min(0) + 1e-6)

    score = torch.einsum("bhnd,bhmd->bhnm", q_bar, kc)

    idx = torch.arange(n_blocks, device=q_h.device)
    neighbour = (idx.view(-1, 1) - idx.view(1, -1)).abs() <= 1
    return (score > threshold.unsqueeze(-1)) | neighbour


def _block_mask(selected, q_len, kv_len):
    """Build a BlockMask directly. create_block_mask() must NOT be used here."""
    from torch.nn.attention.flex_attention import BlockMask

    n = selected.shape[-1]
    kv_num_blocks = selected.sum(dim=-1).to(torch.int32)
    # stable sort on the negated flag: selected first, original order preserved
    order = torch.argsort((~selected).to(torch.int8), dim=-1, stable=True)
    kv_indices = order.to(torch.int32)

    def boundary(b, h, q_idx, kv_idx):
        return (q_idx < q_len) & (kv_idx < kv_len)

    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        BLOCK_SIZE=FLEX_BLOCK,
        mask_mod=boundary,
        seq_lengths=(q_len, kv_len),
    )


def _merge(out_a, lse_a, out_b, lse_b):
    m = torch.maximum(lse_a, lse_b)
    wa = torch.exp(lse_a - m).unsqueeze(-1)
    wb = torch.exp(lse_b - m).unsqueeze(-1)
    return (out_a.float() * wa + out_b.float() * wb) / (wa + wb)


def sol_attn_flex(q, k, v, *, scale=None, tau: float = 1.0):
    """Sol-Attn on BTHD bf16 inputs via flex_attention."""
    b, t, h, d = q.shape
    scale = d ** -0.5 if scale is None else float(scale)
    n_blocks = math.ceil(t / FLEX_BLOCK)
    t_pad = n_blocks * FLEX_BLOCK

    to_bhsd = lambda x: x.permute(0, 2, 1, 3).contiguous()
    q_h, k_h, v_h = to_bhsd(q), to_bhsd(k), to_bhsd(v)
    if t_pad != t:
        pad = (0, 0, 0, t_pad - t)
        q_h, k_h, v_h = F.pad(q_h, pad), F.pad(k_h, pad), F.pad(v_h, pad)

    kc, vc, lens = _summaries(k_h, v_h, t, n_blocks)
    selected = _routing(q_h, kc, t, n_blocks, float(tau))

    flex = _flex()

    # exact branch over the real k/v
    out_e, lse_e = flex(
        q_h, k_h, v_h,
        block_mask=_block_mask(selected, t_pad, t_pad),
        scale=scale, return_lse=True,
    )

    # approximate branch over the block summaries
    approx = ~selected
    n_pad = math.ceil(n_blocks / FLEX_BLOCK) * FLEX_BLOCK
    vc_v = (vc / lens.view(1, 1, -1, 1)).to(q.dtype)  # block mean of v
    kc_k = kc.to(q.dtype)
    log_len = torch.log(lens)
    if n_pad != n_blocks:
        kc_k = F.pad(kc_k, (0, 0, 0, n_pad - n_blocks))
        vc_v = F.pad(vc_v, (0, 0, 0, n_pad - n_blocks))
        log_len = F.pad(log_len, (0, n_pad - n_blocks))
        approx = F.pad(approx, (0, n_pad - n_blocks))

    approx_blocks = approx.view(b, h, n_blocks, n_pad // FLEX_BLOCK, FLEX_BLOCK).any(-1)

    def length_bias(score, bi, hi, q_idx, kv_idx):
        return score + log_len[kv_idx]

    out_a, lse_a = flex(
        q_h, kc_k, vc_v,
        block_mask=_block_mask(approx_blocks, t_pad, n_pad),
        score_mod=length_bias,
        scale=scale, return_lse=True,
    )

    out = _merge(out_e, lse_e, out_a, lse_a)
    return out[:, :, :t, :].permute(0, 2, 1, 3).contiguous().to(q.dtype)


def warmup(device=None, heads: int = 8, tokens: int = 1024) -> None:
    """Pay the torch.compile cost at startup instead of mid-render."""
    device = device or torch.device("cuda")
    try:
        z = torch.randn(1, tokens, heads, 128, device=device, dtype=torch.bfloat16)
        sol_attn_flex(z, z.clone(), z.clone())
        logger.info("[Sol-Attn] flex_attention compiled (warmup done)")
    except Exception as exc:  # noqa: BLE001
        logger.info("[Sol-Attn] flex warmup skipped (%s)", exc)


__all__ = ["sol_attn_flex", "warmup"]
