"""Sol-Attn (NVIDIA Sana / Sol-Engine) as an opt-in ComfyUI attention backend.

Uses the Triton reference kernel, which JIT-compiles for any arch Triton
targets -- including SM120 (RTX 50xx). The upstream sm90/sm100 gate guards the
hand-written CuTe kernels only and is not applicable here.

Hard requirements of the kernel (anything else falls back to your normal
backend, e.g. SageAttention):
  - head_dim == 128
  - bfloat16
  - no attention mask
  - 4D q/k/v (skip_reshape=True path)

MiniMax H3 satisfies all four (56 heads x 128, bf16, mask=None).
"""

import logging
import torch

from .solref import sol_attn, sol_attn_flex

log = logging.getLogger(__name__)

BLOCK = 64
MIN_TOKENS = BLOCK * 4  # below this the routing overhead isn't worth it


class _Unsupported(Exception):
    pass


class _Stats:
    """So you can tell whether Sol actually engaged, instead of guessing."""

    def __init__(self):
        self.hits = 0
        self.fallbacks = 0
        self.reason = None
        self.logged = False
        self.sched_logged = False

    def hit(self):
        self.hits += 1
        if not self.logged:
            # root logger: guaranteed to surface in the ComfyUI console
            logging.info("[Sol-Attn] ACTIVE - attention is running on Sol-Attn")
            self.logged = True

    def miss(self, reason):
        self.fallbacks += 1
        if self.reason != reason:
            self.reason = reason
            logging.info("[Sol-Attn] falling back to default backend: %s", reason)

    def scheduled_off(self):
        # Not a fallback -- the user asked for dense attention on these steps.
        if not self.sched_logged:
            logging.info("[Sol-Attn] outside start/end window - dense attention")
            self.sched_logged = True


STATS = _Stats()


def _make_override(tau: float, backend: str = "triton",
                   sigma_start: float = None, sigma_end: float = None):
    kernel = sol_attn_flex if backend == "flex" else sol_attn
    scheduled = sigma_start is not None and sigma_end is not None

    def override(func, q, k, v, heads, *args, **kwargs):
        mask = kwargs.get("mask", None)
        skip_reshape = kwargs.get("skip_reshape", False)
        skip_output_reshape = kwargs.get("skip_output_reshape", False)

        # Sigmas fall as sampling proceeds, so the active window is
        # sigma_end < sigma <= sigma_start. Same convention as core EasyCache.
        if scheduled:
            sigmas = (kwargs.get("transformer_options") or {}).get("sigmas", None)
            if sigmas is not None:
                s = float(sigmas[0])
                if s > sigma_start or s <= sigma_end:
                    STATS.scheduled_off()
                    return func(q, k, v, heads, *args, **kwargs)

        try:
            if mask is not None:
                raise _Unsupported("attention mask present")
            if not skip_reshape or q.dim() != 4:
                raise _Unsupported("not the 4D skip_reshape path")

            b, h, n, d = q.shape
            if d != 128:
                raise _Unsupported(f"head_dim {d} != 128")
            if q.dtype != torch.bfloat16:
                raise _Unsupported(f"dtype {q.dtype} != bfloat16")
            if k.shape != q.shape or v.shape != q.shape:
                raise _Unsupported("q/k/v shape mismatch (cross-attention?)")
            if n < MIN_TOKENS:
                raise _Unsupported(f"only {n} tokens")

            # Comfy hands us BHSD; Sol-Attn wants BTHD, contiguous.
            qt = q.transpose(1, 2).contiguous()
            kt = k.transpose(1, 2).contiguous()
            vt = v.transpose(1, 2).contiguous()

            out = kernel(qt, kt, vt, tau=tau)  # (B, T, H, D)

            STATS.hit()
            if skip_output_reshape:
                return out.transpose(1, 2)  # (B, H, T, D)
            return out.reshape(b, n, h * d)  # (B, T, H*D)

        except _Unsupported as e:
            STATS.miss(str(e))
        except Exception as e:  # kernel/compile failure -- never kill a render
            STATS.miss(f"{type(e).__name__}: {e}")

        return func(q, k, v, heads, *args, **kwargs)

    return override


class SolAttentionPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "tau": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Routing threshold. Higher = more blocks take "
                        "the approximate path = faster, lower fidelity. "
                        "1.0 is upstream default.",
                    },
                ),
                "backend": (
                    ["triton", "flex"],
                    {
                        "default": "triton",
                        "tooltip": "triton = NVIDIA's reference kernel, routing "
                        "at 64 tokens. flex = torch flex_attention, routing at "
                        "128 tokens, correction term kept.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Fraction of sampling before Sol-Attn engages. "
                        "Raise it to leave the early steps (composition, camera) "
                        "on dense attention.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Fraction of sampling after which Sol-Attn "
                        "stops. Lower it to leave the final detail-resolving "
                        "steps on dense attention.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/attention"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Always re-execute. Otherwise ComfyUI caches this node, patch() never
        # re-runs, the stats never reset, and the console goes quiet even
        # though Sol is live -- making the log useless as a signal.
        return float("nan")
    DESCRIPTION = (
        "Route this model's self-attention through Sol-Attn (sparse, "
        "training-free). Only affects the model you wire it to. Falls back to "
        "your normal backend for any shape it can't handle."
    )

    def patch(self, model, enabled, tau, backend="triton",
              start_percent=0.0, end_percent=1.0):
        if not enabled:
            return (model,)
        m = model.clone()

        # percent -> sigma, the same way core's EasyCache does it. If the model
        # can't report its sampling schedule, run unscheduled rather than fail.
        sigma_start = sigma_end = None
        if float(start_percent) > 0.0 or float(end_percent) < 1.0:
            try:
                ms = m.get_model_object("model_sampling")
                sigma_start = float(ms.percent_to_sigma(float(start_percent)))
                sigma_end = float(ms.percent_to_sigma(float(end_percent)))
            except Exception as e:  # noqa: BLE001
                logging.info("[Sol-Attn] schedule disabled (%s); running full range", e)
                sigma_start = sigma_end = None

        opts = dict(m.model_options.get("transformer_options", {}))
        opts["optimized_attention_override"] = _make_override(
            float(tau), backend, sigma_start, sigma_end
        )
        m.model_options["transformer_options"] = opts
        STATS.__init__()  # reset per-patch so the log reflects this run
        logging.info(
            "[Sol-Attn] patch applied to model (tau=%.2f, backend=%s, %.2f-%.2f)",
            float(tau), backend, float(start_percent), float(end_percent),
        )
        return (m,)


NODE_CLASS_MAPPINGS = {"SolAttentionPatch": SolAttentionPatch}
NODE_DISPLAY_NAME_MAPPINGS = {"SolAttentionPatch": "Sol-Attn (sparse attention)"}
