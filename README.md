# ComfyUI-SolAttn

NVIDIA's **Sol-Attn** sparse attention as an opt-in ComfyUI node — including on
**RTX 50-series (SM120)**, which upstream doesn't support.

Sparse attention means the model skips attention work it decides won't matter.
It's an approximation: faster picture, slightly different picture. The kernel is
NVIDIA's. Getting it to run on a 5090 and wiring it into ComfyUI is the only
thing this repo adds.

---

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/sumeetprashant/ComfyUI-SolAttn.git
```

No pip dependencies. Needs Triton 3.4.0+ with TMA (`triton.tools.tensor_descriptor`).

Wire it between your model loader and whatever uses the model:

```
UNETLoader → Sol-Attn → BasicGuider
```

It only touches the model you wire it to.

---

## ⚠️ Two nodes will silently switch this off

This is the first thing to check if the console says `patch applied` and nothing
else.

| node | what it does | result |
|---|---|---|
| **MiniMax H3 Memory Efficient Sage Attention** (KJNodes) | replaces the attention function on every block | Sol-Attn never runs, at any wiring order |
| **ModelAttentionBackend** (core) | writes the same override slot Sol-Attn uses | whichever is applied **last** wins |

Both load without error. Neither warns you. Bypass them if you want Sol-Attn.

**How to know it actually ran** — this line, every render:

```
[Sol-Attn] ACTIVE - attention is running on Sol-Attn
```

No `ACTIVE` line and no `falling back` line means something upstream took the
attention path. That's the bug above.

---

## The knobs

Only two matter to start with: **`tau`** and **`backend`**. Leave the rest alone
until you have a reason.

| knob | plain English | when to touch it |
|---|---|---|
| `tau` | How much work to skip. Higher = faster and rougher. | **The main dial.** Start 1.4. See the table below for where it breaks. |
| `backend` | `triton` or `flex`. | Leave on `triton`. `flex` pays ~80 s of compile per resolution — only worth it at 30+ steps. |
| `enabled` | Off switch. | A/B without unplugging cables. |
| `start_percent` / `end_percent` | Which slice of the render uses sparse attention. `0.0–1.0` = all of it. | Only if you see a specific artifact. Narrowing this costs speed — see below. |
| `tau_end` | Ramps `tau` across the render: careful early, fast late. `0` = off. | When one fixed `tau` is either too slow or too rough. |
| `dense_blocks` | Keeps the first and last N transformer blocks exact. | Cheap insurance if a high `tau` is misbehaving. |
| `dense_block_ranges` | Same, but you name the blocks: `39-42`. | When the blocks worth protecting aren't the first/last ones. |
| `sink_conditioning` | **MiniMax H3 only.** H3 packs the text prompt and audio at the front of the sequence. This keeps those rows exact so prompt adherence and lip-sync survive. | Leave `exact_kv` on H3. No effect on other models — it says so in the log. |
| `min_tokens` | Don't bother below this sequence length. | Rarely. Short clips aren't worth the routing overhead. |
| `thresh_type` | How the skip decision is estimated. `diag` is the default; `exact` is more precise and costs more. | Rarely. |

### Just tell me what to set

- **Want it faster:** `tau 1.8`, everything else default. Measured 24% faster than
  ComfyUI's own attention, picture held.
- **Being careful:** `tau 1.4`. 14% faster, closest to baseline.
- **Something looks wrong:** drop `tau` first. Don't reach for `dense_blocks` —
  it cost 6% and fixed nothing measurable here.

---

## Measured

RTX 5090 · MiniMax H3 · 832×640 · 10 s clip · 6 steps · turbo LoRA · same seed ·
back-to-back, one session.

Baseline is **ComfyUI's own `comfy kitchen attention`**, not Sage — kitchen is
the faster and more honest thing to beat, because it's exact.

| setting | time | vs baseline | texture metric | picture |
|---|---|---|---|---|
| baseline (comfy kitchen attention) | 53.7 s | — | 0.80–0.97 | fine |
| `tau 1.0` | 51.9 s | −3% | 0.79–0.90 | fine |
| `tau 1.4` | 46.1 s | −14% | 0.78–0.85 | fine |
| `tau 1.4` + `dense_block_ranges 39-42` | 48.8 s | −9% | 0.75–0.92 | fine |
| **`tau 1.8`** | **40.6 s** | **−24%** | 0.65–0.76 | fine |
| `tau 2.2` | 37.8 s | −30% | 0.79 → **1.36 rising** | **breaks** |

**`tau 2.2` is past the edge**: duplicated characters, wrong costume. And the
texture metric *climbs through the clip* — that rise is the early warning. If
your number goes up frame over frame, drop `tau` before you trust the render.

Texture metric = median local standard deviation (5×5 window) on the luma frame.
On flat cartoon art it sits near 0.8 when clean; a visibly mottled render reads
3.2+. It's a cheap objective check for "did this quietly damage my output."

A narrow `start/end` window plus `dense_blocks` plus `sink_conditioning` all at
once measured **22% slower than doing nothing**. Sparse attention only pays when
you let it be sparse.

### Caveats, honestly

- One rig, one model, one scene. Not a benchmark.
- "Picture held" is my eye on a side-by-side, not a user study.
- Same seed still gives a *different take*, not a degraded one. You can't diff
  two runs and read the difference as quality loss.
- Sol's cost depends on content, so fix seed **and** input image.
- Triton autotunes per resolution — the first run at a new size pays a JIT sweep
  inside the sampling loop. **Measure the second run.**

---

## Other nodes that do the same job

I haven't benchmarked these head-to-head. Several are more capable than this
one. Pick on what you need, not on who wrote it.

| project | what it is | why you'd pick it |
|---|---|---|
| [Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn) | Sol-Attn with a much bigger node set — int8 QK, a tau curve with a graph, FFN chunking, a dedicated H3 variant | **More features than this repo.** Also covers SM89. Start here if you want options |
| [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) | parallel Sol-Attn port | kijai's nodes tend to land first and get maintained |
| [KingGore/ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell) | Sol-Attn on Blackwell via `flex_attention` | different route to the same place |
| [woct0rdho/ComfyUI-RadialAttn](https://github.com/woct0rdho/ComfyUI-RadialAttn) | Radial Attention — a *different* sparse method | not Sol-Attn; worth trying on its own terms |
| **comfy kitchen attention** (in ComfyUI core) | exact, not sparse | **Try this first.** It's fast, it's exact, and it's already installed. This node only beat it by 24% |
| SageAttention | quantized attention | different tradeoff — approximates the maths, not the sparsity |

Sparse attention is not the biggest lever available. On this rig, moving CUDA
12.6 → 13.0 was worth **2.1× total** with no quality cost at all. Do the exact
speedups before the approximate ones.

---

## Requirements

Enforced at runtime — anything else **falls back to your normal backend** and
logs why. It never raises into your render.

| | |
|---|---|
| `head_dim` | exactly 128 |
| dtype | bfloat16 |
| attention mask | none |
| layout | 4D q/k/v (`skip_reshape=True`) |

Tested only on: RTX 5090 (SM120) · torch 2.11.0+cu130 · Triton 3.4.0 · Python
3.12 · ComfyUI 0.32.0 · Windows 11 · MiniMax H3.

SM89/SM86 are **untested**. The Triton kernel has no architecture dependency in
principle, but TMA descriptors do, and nobody has run it there.

### About the architecture gate

Upstream blocks non-SM90/SM100 cards. That check is a **dispatch guard** — it
picks between NVIDIA's hand-written CuTe kernels, which only exist for those two
architectures. It is not a hardware-safety check, and no userspace CUDA library
can affect thermal or power limits. The Triton kernel has no such dependency and
inherited the guard by sharing `_validate()`. This repo lifts it for the Triton
path only; every other check is kept verbatim.

The risk is **wrong output, not damaged hardware** — and you'd see it in the
picture. The exact diff is in [`NOTICE`](NOTICE).

If you get bad numerics on some other architecture, that's this repo's problem.
Don't file it upstream against NVlabs/Sana.

---

## Credits

Almost all of the engineering here is other people's.

**Sol-Attn** — Haopeng Li, Yitong Li, Junsong Chen, Tian Ye, Haozhe Liu,
Jincheng Yu, Duomin Wang, Ruihua Zhang, Zeke Xie, Enze Xie, Song Han (NVIDIA
Research). The kernel, the method and everything in `solref/` are theirs.
[Project](https://nvlabs.github.io/Sana/Sol-Attn/) ·
[Source](https://github.com/NVlabs/Sana/tree/sol-engine) ·
[Paper](https://arxiv.org/abs/2607.24027)

```bibtex
@misc{li2026solattnacceleratingvideogeneration,
      title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly
      Attention Sparsification},
      author={Haopeng Li and Yitong Li and Junsong Chen and Tian Ye and Haozhe
      Liu and Jincheng Yu and Duomin Wang and Ruihua Zhang and Zeke Xie and
      Enze Xie and Song Han},
      year={2026}, eprint={2607.24027}, archivePrefix={arXiv},
      primaryClass={cs.CV}, url={https://arxiv.org/abs/2607.24027},
}
```

**ComfyUI** (comfyanonymous and contributors) — the
`optimized_attention_override` hook that makes a per-model patch possible.

**Design lineage** — no code from these, but it wouldn't look like this without
them: [ComfyUI-RadialAttn](https://github.com/woct0rdho/ComfyUI-RadialAttn)
(@woct0rdho) established the opt-in `MODEL → MODEL` patch pattern ·
[RadialAttention](https://github.com/mit-han-lab/radial-attention) (MIT Han Lab)
· [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton)
(@kijai), source of the `start_percent`/`end_percent` idea ·
[ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell)
(@KingGore), the `BlockMask.from_kv_blocks` detail the `flex` backend relies on.

**Triton** (OpenAI and contributors) — compiles the kernel.
**FlashAttention** (Tri Dao et al.) — lineage recorded in [`NOTICE`](NOTICE).

Integration, packaging and the SM120 change:
[@sumeetprashant](https://github.com/sumeetprashant).

## License

Apache 2.0 — see [`LICENSE`](LICENSE), inherited from NVlabs/Sana. Modifications
to NVIDIA source are itemised in [`NOTICE`](NOTICE) per Apache-2.0 §4(b).
