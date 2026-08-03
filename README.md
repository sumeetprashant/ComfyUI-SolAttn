# ComfyUI-SolAttn

Run NVIDIA's **Sol-Attn** sparse attention inside ComfyUI, as an opt-in
per-model patch — including on **consumer Blackwell (SM120 / RTX 50-series)**,
which upstream does not support.

> **This repository removes a GPU architecture check that NVIDIA put in their
> code.** That is not a bug fix and it is not sanctioned by NVIDIA. Read
> [Scope and caveats](#scope-and-caveats) before you use it. The full diff and
> its rationale are documented in [`NOTICE`](NOTICE).

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

The console tells you exactly what happened, every run:

```
[Sol-Attn] patch applied to model (tau=1.00)
[Sol-Attn] ACTIVE - attention is running on Sol-Attn
[Sol-Attn] falling back to default backend: <reason>
```

A demo workflow is in [`workflows/`](workflows/).

## Measured results — read the caveats

Everything below is from one machine, one model. **Do not treat it as a
benchmark.**

**MiniMax H3, 15s, 480×864, 20 steps, `res_multistep`, fixed seed, same input
image, SageAttention as the baseline:**

| | s/it | |
|---|---|---|
| SageAttention 2.2.0 | 9.91 | baseline |
| Sol-Attn, `tau=1.0` | **8.92** | **−10.0%** |

That is **one** controlled pair. Two further Sol runs on *different* input
images measured 9.66 and 9.87 s/it, but no matched baseline was captured for
those, so they prove nothing on their own.

Sol-Attn's cost is **content-dependent** in a way dense attention is not — the
fraction of blocks routed to the exact path depends on the actual attention
distribution. Expect run-to-run variation with subject matter. Fix your seed and
your input image before comparing anything.

Synthetic microbenchmark vs SageAttention, random tensors, 8 heads × 128:

| tokens | Sol (ms) | Sage (ms) | speedup |
|---|---|---|---|
| 2,048 | 0.12 | 0.20 | 1.67× |
| 8,192 | 0.56 | 0.77 | 1.38× |
| 16,384 | 1.76 | 2.38 | 1.35× |
| 32,768 | 10.51 | 12.47 | 1.19× |

Random Gaussian q/k produce a near-uniform attention distribution, which is the
worst possible input for a method that exploits structure. Treat these as a
smoke test that the kernel runs and is not catastrophically slow — not as a
performance claim, and not as a fidelity measurement.

Also note the compile tax: Triton autotunes with `key=["T"]`, so the **first run
at any new token count pays a JIT sweep inside the sampling loop.** Change your
resolution or duration and you pay it again. Benchmark the second run.

## Scope and caveats

- **Sol-Attn is approximate.** Output will not be bit-identical to dense
  attention. Whether that is visible in your content is your call — A/B it.
- **No fidelity A/B is published here yet.** The timing above is solid; a
  matched quality comparison has not been completed. If you run one, please open
  an issue with the pair.
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

**Triton** (OpenAI and contributors) — compiles the kernel.

**SageAttention** (thu-ml) and **woct0rdho**'s Windows builds — the backend this
falls back to, and the baseline every number above is measured against.

Integration, packaging, and the SM120 change: [@sumeetprashant](https://github.com/sumeetprashant).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE), inherited from NVlabs/Sana.
Modifications to NVIDIA source are itemised in [`NOTICE`](NOTICE) per
Apache-2.0 §4(b).
