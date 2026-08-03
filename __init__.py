"""ComfyUI-SolAttn - Sol-Attn sparse attention as an opt-in per-model patch.

Import is guarded: if Triton or the kernel can't load, this pack registers
nothing and ComfyUI starts normally rather than throwing an IMPORT FAILED.
"""

import logging

log = logging.getLogger(__name__)

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:  # noqa: BLE001
    log.warning("[Sol-Attn] not loaded (%s: %s)", type(e).__name__, e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
