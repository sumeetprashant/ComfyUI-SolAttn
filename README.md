# ComfyUI-SolAttn

Run NVIDIA's **Sol-Attn** sparse attention inside ComfyUI, as an opt-in
per-model patch — including on **consumer Blackwell (SM120 / RTX 50-series)**,
which upstream does not support.

> To do that it lifts an architecture guard in NVIDIA's dispatcher — a check
> that picks between their SM90/SM100 kernels, not a hardware-safety check. See
> [FAQ](#faq) for what that does and doesn't mean; the exact diff is in
> [`NOTICE`](NOTICE).

---

## FAQ

**Is that architecture check a safety thing — will this overheat my GPU?**
No. It's a *dispatch guard*. It selects between NVIDIA's hand-written CuTe
kernels, which only exist for SM90 and SM100 — so on any other card it means
"no kernel compiled for you," not "unsafe." Thermal and power limits live in
firmware and the driver; no userspace CUDA library can affect them.

**Then why did it block the Triton kernel too?**
Because `_validate()` is shared between both code paths. The Triton kernel has
no architecture dependency — Triton compiles it for whatever card you have — it
just inherited a guard written for the CuTe path.

**So what's the actual risk?**
Wrong output, not damaged hardware. If a kernel assumed something
architecture-specific that doesn't hold, you'd see it in the picture. Every
other check is kept verbatim (head_dim 128, bf16, contiguity, device, shape),
and any shape it can't handle falls back to your normal attention backend
instead of erroring.

---

## What this is

[Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/) is training-free sparse
attention: each query block is routed to either exact attention or a cheap
block-summary approximation, decided on the fly inside a single online-softmax
pass. No tuning run, no calibration, one knob (`tau`).

NVIDIA ship two implementations — hand-written CuTe kernels for SM90 (H100) and
SM100 (B200), and a Triton reference. The CuTe kernels genuinely do not exist
for other architectures. The **Triton kernel has no such dependency**: it is
JIT-compiled for whatever architecture Triton is handed. Upstream gates both
paths behind the same SM90/SM100 check. This repository removes that check for
the Triton path only.

It works on an RTX 5090. That is the entire contribution here — the kernel is
NVIDIA's.

## Requirements

Sol-Attn's own constraints, enforced at runtime:

| | |
|---|---|
| `head_dim` | **exactly 128** |
| dtype | **bfloat16** |
| attention mask | **none** |
| layout | 4D q/k/v (ComfyUI's `skip_reshape=True` path) |

Anything else **falls back to your normal attention backend** and logs why. It
never raises into your render.

Also needs Triton with `triton.tools.tensor_descriptor` (TMA). Verified on
Triton 3.4.0.

## Tested on

```
RTX 5090 (SM120)  ·  torch 2.11.0+cu130  ·  Triton 3.4.0
Python 3.12.10    ·  ComfyUI 0.30.0      ·  Windows 11
MiniMax H3 (56 heads x 128, bf16, mask=None) -- satisfies all four constraints
```

Only SM120 has been tested. SM89/SM86 are untested — the Triton kernel has no
architecture dependency in principle, but TMA descriptors do, and nobody has
run it there.

## Usage

Add **Sol-Attn (sparse attention)** (`model_patches/attention`) between your
model loader and whatever consumes the model:

```
UNETLoader → Sol-Attn → BasicGuider
```

It patches only the model you wire it to, via ComfyUI's
`transformer_options["optimized_attention_override"]` hook. Every other model in
your graph is untouched.

| widget | default | |
|---|---|---|
| `enabled` | `true` | flip to `false` to A/B without rewiring |
| `tau` | `1.0` | routing threshold. Higher = more blocks take the approximate path = faster, lower fidelity |
| `backend` | `triton` | which kernel runs the routing — see [Backends](#backends) |
| `start_percent` | `0.0` | sampling fraction before Sol engages — raise to keep early steps dense |
| `end_percent` | `1.0` | sampling fraction after which Sol stops — lower to keep final steps dense |

`start_percent` / `end_percent` restrict Sol to part of the sampling run: dense
attention early (composition, camera) and late (fine detail), sparse in the
middle where most of the time goes. Defaults `0.0 / 1.0` = always on. Idea taken
from [kijai's ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton);
percent→sigma conversion follows ComfyUI core's own EasyCache.

The console tells you exactly what happened, every run:

```
[Sol-Attn] patch applied to model (tau=1.00, backend=flex, 0.00-1.00)
[Sol-Attn] ACTIVE - attention is running on Sol-Attn
[Sol-Attn] outside start/end window - dense attention
[Sol-Attn] falling back to default backend: <reason>
```

## Backends

Two implementations of the same method. Both keep Sol-Attn's approximate
correction — skipped blocks still contribute through their block summaries.

| | `triton` | `flex` |
|---|---|---|
| kernel | NVIDIA's Triton reference | `torch.nn.attention.flex_attention` |
| routing granularity | 64 tokens | 128 tokens |
| cos vs dense SDPA | 0.999993 | 0.9993 |
| first run cost | Triton autotune sweep | one `torch.compile` |

`flex` builds its mask with `BlockMask.from_kv_blocks` — **not**
`create_block_mask`, which vmaps over the whole `[B,H,T,T]` index space and OOMs
(64 GiB at 32k tokens). Credit to
[KingGore](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell) for the
flex_attention approach and that detail.

A demo workflow is in [`workflows/`](workflows/).

## Measured

> **Still testing.** These are early numbers from one rig, one model. More runs
> and more models to come — treat this as work in progress, not a benchmark.

MiniMax H3, 15s, 480×864, 20 steps, `res_multistep`, fixed seed, same input
image. One session, back-to-back:

| | s/it | vs Sage |
|---|---|---|
| SageAttention 2.2.0 (Sol bypassed) | 9.37 | baseline |
| Sol-Attn `triton`, `tau=1.0` | 9.70 | +3.5% ⚠️ |
| Sol-Attn `flex`, `tau=1.0` | **8.67** | **−7.5%** |
| Sol-Attn `flex`, `tau=1.12` | **7.93** | **−15.4%** |

⚠️ That `triton` run had ~8.4 GB VRAM free vs ~25 GB for the others — not a
matched comparison, re-run pending. An earlier session, different input image,
measured `triton` `tau=1.0` at **8.92** vs Sage **9.91** (−10.0%).

Fidelity — cosine vs dense SDPA, structured input, 8 heads × 128, bf16:

| `triton` | `flex` |
|---|---|
| **0.999993** | **0.9993** |

Numerical only; a perceptual video A/B is not published yet. On *random Gaussian*
input this reads ≈ 0.67 — an artifact of feeding noise to a structure-exploiting
method, not a fidelity result. Benchmark on structured input.

Two things that will skew your own numbers: Sol's cost is **content-dependent**,
so fix seed *and* input image; and Triton autotunes on `key=["T"]`, so the first
run at any new resolution pays a JIT sweep inside the sampling loop — measure the
second run.

## Scope and caveats

- **Sol-Attn is approximate.** Output will not be bit-identical to dense
  attention. Whether that is visible in your content is your call — A/B it.
- **Fidelity is numerical, not perceptual.** A side-by-side video A/B is still
  outstanding. If you run one, please open an issue with the pair.
- **`triton` timing is unresolved** — two sessions disagree (−10.0% vs +3.5%
  against Sage), the second under VRAM pressure. `flex` has the clean
  matched-conditions measurement.
- **The architecture gate was removed by this repository, not by NVIDIA.**
  NVIDIA has not validated Sol-Attn outside SM90/SM100. If you get bad numerics
  on some other architecture, that is this repo's problem to report — do not
  file it upstream against NVlabs/Sana.
- NVIDIA's published 2.1–2.3× figures are for **Sol-Engine as a whole** — CuTe
  kernels plus NVFP4 quantization plus DiT block fusion, on datacenter GPUs.
  This is the Triton reference kernel alone. Different thing entirely.

## Credits

Essentially all of the engineering here is other people's.

**Sol-Attn** — Haopeng Li, Yitong Li, Junsong Chen, Tian Ye, Haozhe Liu,
Jincheng Yu, Duomin Wang, Ruihua Zhang, Zeke Xie, Enze Xie, and Song Han
(NVIDIA Research, Efficient AI Team & Singapore Lab). The kernel, the method,
and the preprocessing in `solref/` are theirs.

- Project page: https://nvlabs.github.io/Sana/Sol-Attn/
- Source: https://github.com/NVlabs/Sana/tree/sol-engine
- Paper: https://arxiv.org/abs/2607.24027

```bibtex
@misc{li2026solattnacceleratingvideogeneration,
      title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly
      Attention Sparsification},
      author={Haopeng Li and Yitong Li and Junsong Chen and Tian Ye and Haozhe
      Liu and Jincheng Yu and Duomin Wang and Ruihua Zhang and Zeke Xie and
      Enze Xie and Song Han},
      year={2026},
      eprint={2607.24027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.24027},
}
```

**FlashAttention** (Tri Dao et al.) — NVIDIA's own third-party notices record
that parts of the SM90/SM100 scaffold in Sol-Engine derive from FlashAttention
(BSD-3-Clause). Those files are not redistributed here, but the lineage is
recorded in [`NOTICE`](NOTICE).

**ComfyUI** (comfyanonymous and contributors) — the
`optimized_attention_override` hook this integrates against, which is what makes
a clean per-model attention patch possible at all.

### Prior art and design lineage

This repository contains no code from the projects below, but it would not look
the way it does without them.

- **[ComfyUI-RadialAttn](https://github.com/woct0rdho/ComfyUI-RadialAttn)**
  ([@woct0rdho](https://github.com/woct0rdho)) — established the pattern this
  follows: expose sparse attention as an opt-in `MODEL → MODEL` patch node that
  writes into `model_options["transformer_options"]`, rather than flipping a
  global backend for every model in the graph. That design decision is theirs;
  this repo just applies it to a different kernel.
- **[RadialAttention](https://github.com/mit-han-lab/radial-attention)**
  (MIT Han Lab) — the sparse attention method that port wraps.
- **[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)**
  ([@kijai](https://github.com/kijai)) — `WanVideoSetRadialAttention` is
  parallel prior art for the same idea inside the wrapper workflow.
- **[ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton)**
  ([@kijai](https://github.com/kijai)) — a parallel Sol-Attn port, and the source
  of the `start_percent` / `end_percent` idea implemented here.
- **[ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell)**
  ([@KingGore](https://github.com/KingGore)) — independently brought Sol-Attn to
  Blackwell via `flex_attention`, and got right the non-obvious
  `BlockMask.from_kv_blocks` detail. The `flex` backend here follows that
  approach; it differs in keeping Sol-Attn's approximate correction term.

**Triton** (OpenAI and contributors) — compiles the kernel.

**SageAttention** (thu-ml) and **woct0rdho**'s Windows builds — the backend this
falls back to, and the baseline every number above is measured against.

Integration, packaging, and the SM120 change: [@sumeetprashant](https://github.com/sumeetprashant).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE), inherited from NVlabs/Sana.
Modifications to NVIDIA source are itemised in [`NOTICE`](NOTICE) per
Apache-2.0 §4(b).
