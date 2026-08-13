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
import os
import traceback

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
        self.dense_logged = False
        self.sink_logged = False
        self.sink_warned = False

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

    def dense_block(self):
        if not self.dense_logged:
            logging.info("[Sol-Attn] dense_blocks active - some blocks kept dense")
            self.dense_logged = True

    def sink(self, kv_blocks, rows):
        if not self.sink_logged:
            logging.info(
                "[Sol-Attn] sink_conditioning: %d leading KV block(s) exact%s",
                kv_blocks, ", those query rows dense too" if rows else "",
            )
            self.sink_logged = True

    def sink_missing(self):
        # The span comes from the H3 forward wrapper. If it never arrives the
        # sink silently does nothing, which is exactly the failure you would
        # not notice -- so say it once, loudly.
        if not self.sink_warned:
            logging.warning(
                "[Sol-Attn] sink_conditioning is on but no H3 video span was "
                "found - the sink is INACTIVE (non-H3 model, or the layout "
                "could not be rebuilt)"
            )
            self.sink_warned = True


STATS = _Stats()


# Set while a block chosen to stay dense is executing. The H3 block loop is
# single-threaded and hands every block the same transformer_options, so there
# is no block index inside the attention hook -- this flag is how we get one.
_DENSE_BLOCK = False


def _make_dense_wrapper(previous=None):
    def wrapper(args, extra):
        global _DENSE_BLOCK
        was = _DENSE_BLOCK
        _DENSE_BLOCK = True
        try:
            if previous is not None:          # don't clobber another node's patch
                return previous(args, extra)
            return extra["original_block"](args)
        finally:
            _DENSE_BLOCK = was
    return wrapper


def _make_span_wrapper(original_forward):
    """Publish H3's video-segment span into transformer_options.

    The packed layout is built inside MiniMaxH3Model._forward and never
    reaches the attention hook, so rebuild it here the same way the model does
    and hand the span down. Any failure just leaves the span absent, which
    means the sink stays off -- it must never take a render down with it.
    """

    def forward(x, timestep, context, transformer_options={},
                minimax_payload=None, **kwargs):
        try:
            from comfy.ldm.minimax.model import PackedLayout
        except Exception:
            PackedLayout = None
        if PackedLayout is not None and isinstance(transformer_options, dict):
            payload = minimax_payload or {}
            try:
                video_x, audio_x = x[0], x[1]
                signature = (
                    context.shape[1],
                    video_x.shape[2],
                    -(-video_x.shape[3] // 2) * 2,
                    -(-video_x.shape[4] // 2) * 2,
                    audio_x.shape[-1],
                )
                layout = payload.get("layout")
                if layout is None or layout.signature != signature:
                    layout = PackedLayout(
                        *signature,
                        keyframes=payload.get("keyframes"),
                        refs=payload.get("refs"),
                        frame_count=payload.get("frame_count"),
                    )
                span = next(
                    ((a, b) for a, b, kind in layout.segments if kind == "video"),
                    None,
                )
                if span is not None:
                    transformer_options["sol_h3_video_span"] = span
            except Exception as e:  # noqa: BLE001
                logging.info("[Sol-Attn] H3 span unavailable (%s: %s)",
                             type(e).__name__, e)
        return original_forward(x, timestep, context, transformer_options,
                                minimax_payload=minimax_payload, **kwargs)

    forward._sol_attn_span_fallback = original_forward
    return forward


def _make_override(tau: float, backend: str = "triton",
                   sigma_start: float = None, sigma_end: float = None,
                   tau_end: float = None, sink: str = "off",
                   min_tokens: int = MIN_TOKENS, thresh_type: str = "diag"):
    kernel = sol_attn_flex if backend == "flex" else sol_attn
    scheduled = sigma_start is not None and sigma_end is not None
    ramped = tau_end is not None and abs(tau_end - tau) > 1e-6
    # flex_attention decides sparsity through its own mask, not the block
    # summary loop, so the sink has nowhere to attach on that backend.
    sinking = sink != "off" and kernel is sol_attn
    if sink != "off" and kernel is not sol_attn:
        logging.info("[Sol-Attn] sink_conditioning ignored: not supported on the flex backend")

    def override(func, q, k, v, heads, *args, **kwargs):
        mask = kwargs.get("mask", None)
        skip_reshape = kwargs.get("skip_reshape", False)
        skip_output_reshape = kwargs.get("skip_output_reshape", False)
        tau_now = tau

        if _DENSE_BLOCK:
            STATS.dense_block()
            return func(q, k, v, heads, *args, **kwargs)

        # Sigmas fall as sampling proceeds, so the active window is
        # sigma_end < sigma <= sigma_start. Same convention as core EasyCache.
        if scheduled or ramped:
            sigmas = (kwargs.get("transformer_options") or {}).get("sigmas", None)
            if sigmas is not None:
                s = float(sigmas[0])
                if scheduled and (s > sigma_start or s <= sigma_end):
                    STATS.scheduled_off()
                    return func(q, k, v, heads, *args, **kwargs)
                if ramped:
                    # progress 0->1 across the active window (or the whole run)
                    hi = sigma_start if scheduled else float(sigmas.max())
                    lo = sigma_end if scheduled else 0.0
                    span = hi - lo
                    p = 0.0 if span <= 0 else min(1.0, max(0.0, (hi - s) / span))
                    tau_now = tau + (tau_end - tau) * p

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
            if n < min_tokens:
                raise _Unsupported(f"only {n} tokens (min_tokens={min_tokens})")

            # H3 packs text/conditioning/reference/audio rows ahead of the
            # video rows, so "the conditioning" is exactly the leading blocks.
            # The span is injected by the forward wrapper this node installs.
            sink_kv = sink_q = 0
            if sinking:
                span = (kwargs.get("transformer_options") or {}).get("sol_h3_video_span")
                video_start = int(span[0]) if span else 0
                if 0 < video_start <= n:
                    sink_kv = -(-video_start // BLOCK)
                    if sink == "exact_kv_and_rows":
                        sink_q = sink_kv
                    STATS.sink(sink_kv, bool(sink_q))
                else:
                    STATS.sink_missing()

            # Comfy hands us BHSD; Sol-Attn wants BTHD, contiguous.
            qt = q.transpose(1, 2).contiguous()
            kt = k.transpose(1, 2).contiguous()
            vt = v.transpose(1, 2).contiguous()

            if kernel is sol_attn:
                out = kernel(qt, kt, vt, tau=tau_now, thresh_type=thresh_type,
                             sink_kv_blocks=sink_kv, sink_q_blocks=sink_q)
            else:
                out = kernel(qt, kt, vt, tau=tau_now)  # (B, T, H, D)

            STATS.hit()
            if skip_output_reshape:
                return out.transpose(1, 2)  # (B, H, T, D)
            return out.reshape(b, n, h * d)  # (B, T, H*D)

        except _Unsupported as e:
            STATS.miss(str(e))
        except Exception as e:  # kernel/compile failure -- never kill a render
            # name the frame that actually raised: without it an OOM only tells
            # you the size it asked for, not which allocation asked
            tb = traceback.extract_tb(e.__traceback__)
            where = f"{os.path.basename(tb[-1].filename)}:{tb[-1].lineno}" if tb else "?"
            STATS.miss(f"{type(e).__name__} at {where}: {e}")

        return func(q, k, v, heads, *args, **kwargs)

    return override


def _parse_block_ranges(spec: str, total: int):
    """'33-35, 39-42' -> {33,34,35,39,40,41,42}, clipped to [0, total).

    Commas, semicolons and newlines all separate. Reversed pairs ('42-39') are
    accepted. Anything unparseable is reported and skipped rather than raising:
    a typo in a text box should not kill a render that is already loaded.
    """
    keep, bad, dropped = set(), [], []
    for piece in spec.replace(";", ",").replace("\n", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            if "-" in piece.lstrip("-"):
                lo_s, hi_s = piece.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if lo > hi:
                    lo, hi = hi, lo
                span = range(lo, hi + 1)
            else:
                span = (int(piece),)
        except ValueError:
            bad.append(piece)
            continue
        for i in span:
            if 0 <= i < total:
                keep.add(i)
            else:
                dropped.append(i)
    if bad:
        logging.info("[Sol-Attn] dense_block_ranges: ignored %s", ", ".join(bad))
    if dropped:
        logging.info(
            "[Sol-Attn] dense_block_ranges: %d index(es) outside 0-%d dropped",
            len(dropped), total - 1,
        )
    return keep


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
                "tau_end": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "0 = off (constant tau). Above 0, tau ramps "
                        "linearly from `tau` at the start of the window to this "
                        "value at the end — accurate early, faster late.",
                    },
                ),
                "dense_blocks": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16,
                        "tooltip": "Keep the first N and last N transformer "
                        "blocks on dense attention. Cheap quality insurance; "
                        "lets you push tau higher. 0 = off. Ignored when "
                        "dense_block_ranges is set.",
                    },
                ),
                "dense_block_ranges": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Explicit block indices to keep dense, e.g. "
                        "'33-35, 39-42'. Overrides dense_blocks when non-empty. "
                        "Use this when the blocks worth protecting are not the "
                        "first/last ones -- Kijai's H3 advice is 39-42. "
                        "Out-of-range indices are dropped with a log line.",
                    },
                ),
                "sink_conditioning": (
                    ["exact_kv", "exact_kv_and_rows", "off"],
                    {
                        "default": "exact_kv",
                        "tooltip": "MiniMax H3 only. H3 packs text, conditioning, "
                        "reference and audio rows ahead of the video rows; this "
                        "keeps those leading KV blocks on the exact path so "
                        "prompt adherence and audio sync survive sparsification. "
                        "exact_kv_and_rows also runs those query rows dense "
                        "(costlier). off restores upstream behaviour. Has no "
                        "effect on non-H3 models or the flex backend -- both say "
                        "so in the log.",
                    },
                ),
                "min_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": BLOCK * 4,
                        "max": 131072,
                        "step": BLOCK,
                        "tooltip": "Use the normal backend below this sequence "
                        "length. Under a few thousand tokens the routing "
                        "overhead costs more than the sparsity saves.",
                    },
                ),
                "thresh_type": (
                    ["diag", "exact"],
                    {
                        "default": "diag",
                        "tooltip": "How the routing threshold is estimated. "
                        "diag is the evaluated default; exact uses "
                        "second-moment statistics for more precise routing at "
                        "extra precompute cost. Triton backend only.",
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
              start_percent=0.0, end_percent=1.0, tau_end=0.0, dense_blocks=0,
              dense_block_ranges="", sink_conditioning="exact_kv",
              min_tokens=4096, thresh_type="diag"):
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
            float(tau), backend, sigma_start, sigma_end,
            float(tau_end) if float(tau_end) > 0.0 else None,
            sink_conditioning, int(min_tokens), thresh_type,
        )
        m.model_options["transformer_options"] = opts

        # The sink needs to know where H3's conditioning rows end, and only the
        # model forward can tell it. Installed on demand so nothing wraps the
        # forward when the sink is off.
        if sink_conditioning != "off":
            try:
                original = m.get_model_object("diffusion_model.forward")
                if hasattr(original, "_sol_attn_span_fallback"):
                    original = original._sol_attn_span_fallback  # re-patch cleanly
                m.add_object_patch("diffusion_model.forward",
                                   _make_span_wrapper(original))
            except Exception as e:  # noqa: BLE001
                logging.info("[Sol-Attn] span wrapper not installed (%s: %s)",
                             type(e).__name__, e)

        n_dense = int(dense_blocks)
        spec = (dense_block_ranges or "").strip()
        if spec or n_dense > 0:
            try:
                blocks = m.get_model_object("diffusion_model.blocks")
                total = len(blocks)
                if spec:
                    # explicit indices win: the blocks worth protecting are not
                    # always symmetric around the stack (H3 wants 39-42).
                    keep = _parse_block_ranges(spec, total)
                else:
                    keep = set(range(min(n_dense, total))) | \
                           set(range(max(0, total - n_dense), total))
                if not keep:
                    raise _Unsupported("no valid block indices")
                existing = (opts.get("patches_replace", {}) or {}).get("dit", {})
                for i in sorted(keep):
                    m.set_model_patch_replace(
                        _make_dense_wrapper(existing.get(("double_block", i))),
                        "dit", "double_block", i,
                    )
                logging.info(
                    "[Sol-Attn] %d/%d blocks kept dense: %s",
                    len(keep), total, sorted(keep),
                )
            except Exception as e:  # noqa: BLE001
                logging.info("[Sol-Attn] dense_blocks skipped (%s: %s)", type(e).__name__, e)

        STATS.__init__()  # reset per-patch so the log reflects this run
        logging.info(
            "[Sol-Attn] patch applied (tau=%.2f%s, backend=%s, %.2f-%.2f, "
            "dense=%s, sink=%s, min_tokens=%d, thresh=%s)",
            float(tau),
            "" if float(tau_end) <= 0 else "->%.2f" % float(tau_end),
            backend, float(start_percent), float(end_percent),
            spec if spec else n_dense, sink_conditioning,
            int(min_tokens), thresh_type,
        )
        return (m,)


NODE_CLASS_MAPPINGS = {"SolAttentionPatch": SolAttentionPatch}
NODE_DISPLAY_NAME_MAPPINGS = {"SolAttentionPatch": "Sol-Attn (sparse attention)"}
