# ComfyUI-SolAttn

NVIDIA's **Sol-Attn** sparse attention in ComfyUI. Works on **RTX 50-series
(SM120)**, which upstream refuses.

## What it does

Model compares every part of the picture to every other part. Most of those
comparisons are empty sky checking on empty sky. You pay for all of them.

This node lets the model skim. Look at chunk, decide "nothing here," move on.
Spend the effort where the stuff is.

Skim a little: save a little. Skim too much: model forgets what it already drew
and draws it again. Two heads. Two people. Fun for nobody.

**One dial. Slow and right, or fast and wrong. You find the middle.**

Kernel is NVIDIA's. This repo just makes it run on a 5090 and gives it a node.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/sumeetprashant/ComfyUI-SolAttn.git
```

No pip installs. Needs Triton 3.4.0+.

## Use it

Open [`workflows/minimax_h3_i2v_solattn_simple.json`](workflows/minimax_h3_i2v_solattn_simple.json).
17 nodes, all ComfyUI core except this one. Image → video with sound, MiniMax H3.

```
UNETLoader → Sol-Attn → SigmaShift → Guider ┐
LoadImage + CLIP + VAE → H3ImageToVideo     ┴→ Sampler → decode → SaveVideo
```

1. Point `UNETLoader`, `CLIPLoader`, `LoadImage` at your files.
2. Set `tau` to **1.4**. Touch nothing else.
3. Render once with `enabled` off, once with it on. Same seed.
4. **Look at both.** Same picture? Keep it. Grew an extra person? Drop `tau`.

Run it twice before you time it. First run at a new size compiles inside the
sampling loop. That number is a lie.

This line means it's working:

```
[Sol-Attn] ACTIVE - attention is running on Sol-Attn
```

No `ACTIVE`, no `falling back` → it isn't running. See [Conflicts](#conflicts).

## Numbers

Demo workflow, 243 frames, 20 steps, same seed, only `tau` changed.

| `tau` | time | vs off | picture |
|---|---|---|---|
| off | 166.8 s | — | fine |
| 1.0 | 173.1 s | **4% slower** | fine |
| 1.4 | 156.4 s | **6% faster** | fine |
| 1.8 | 135.2 s | 19% faster | **broke** |

Three things in that table:

- **Too low is worse than off.** Deciding what to skip isn't free.
- **The good speed and the breakage sit right next to each other.**
- **Nothing warns you.** The render "succeeds." You have to look.

On another graph (6 steps, turbo LoRA) `tau 1.8` was fine and gave 24%. So the
ceiling belongs to your setup, not to this node. Two renders and you know yours.

**Long clips win more.** Same workflow, only length changed:

| clip | off | on | gain |
|---|---|---|---|
| 3 s | 35.1 s | 32.1 s | 9% |
| 10 s | 166.8 s | 135.2 s | 19% |

Cost grows faster than the clip does. Small clip, small pile, small saving.
**Short clips? This is not your problem. Close the tab.**

## What's wrong with it

1. Too far and it draws things twice. No warning.
2. `ACTIVE` means it ran, not that it was right. Only your eyes know.
3. Too low and it's slower than off.
4. Safe setting moves when your model, size, steps or LoRA move.
5. Gain is small. CUDA 12.6 → 13.0 on this machine was worth **2.1×**, free, no
   quality cost. Do that first.
6. First run at a new size is slow. Ignore it.
7. One GPU, one model, one person tested this. Elsewhere, you're the test.

It's a dial you check, not a switch you forget.

## Conflicts

Two nodes switch this off silently. Both load fine. Neither warns you.

| node | what it does |
|---|---|
| **MiniMax H3 Memory Efficient Sage Attention** (KJNodes) | replaces attention on every block — Sol-Attn never runs, any wiring order |
| **ModelAttentionBackend** (core) | writes the same slot — last one applied wins |

Bypass them if you want this node.

## The knobs

Two matter: `tau` and `backend`. Rest can wait.

| knob | what it does | when |
|---|---|---|
| `tau` | How much the model may skip. Low = careful. High = fast. Too high = twins. | **The dial.** Start 1.4 |
| `backend` | `triton` or `flex` | Leave on `triton`. `flex` burns ~80 s compiling per size |
| `enabled` | Off switch | Your A/B button |
| `start_percent` / `end_percent` | Which slice of the render skims | Chasing one specific artifact |
| `tau_end` | Careful at the start, fast at the end | When one fixed `tau` won't do both |
| `dense_blocks` | First and last N blocks stay exact | Insurance. Bought nothing here |
| `dense_block_ranges` | Pick blocks by hand: `39-42` | When the ends aren't the problem |
| `sink_conditioning` | **H3 only.** Stops it skimming your prompt and audio | Leave on `exact_kv` for H3 |
| `min_tokens` | Below this, don't bother | Rarely |
| `thresh_type` | How it decides. `exact` thinks harder, charges more | Rarely |

## Other nodes, same job

Not benchmarked against each other. Some are better. Pick on what you need.

| project | why |
|---|---|
| [Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn) | **More features than this.** int8 QK, tau curve, FFN chunking, H3 variant. Also SM89 |
| [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) | kijai's stuff lands early and stays maintained |
| [KingGore/ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell) | Blackwell via `flex_attention` |
| [woct0rdho/ComfyUI-RadialAttn](https://github.com/woct0rdho/ComfyUI-RadialAttn) | different method entirely, worth a look |
| **comfy kitchen attention** (core) | **Try first.** Exact, fast, already installed |

## Requirements

Anything else falls back to your normal backend and says why. Never breaks a
render.

| | |
|---|---|
| `head_dim` | exactly 128 |
| dtype | bfloat16 |
| mask | none |
| layout | 4D q/k/v |

Tested: RTX 5090 · torch 2.11.0+cu130 · Triton 3.4.0 · ComfyUI 0.32.0 ·
Windows 11 · MiniMax H3. SM89/SM86 untested.

**The architecture gate:** upstream blocks non-SM90/SM100. That check picks
between NVIDIA's hand-written kernels, which only exist for those two chips.
It's not a safety check — no userspace CUDA library can touch power or thermal
limits. The Triton kernel has no such dependency and only inherited the guard by
sharing a validator. Lifted for the Triton path only; every other check kept.
Risk is wrong pixels, not a dead GPU. Diff in [`NOTICE`](NOTICE).

Bad numerics on another chip is this repo's problem. Don't file it upstream
against NVlabs/Sana.

## Credits

Nearly all of this is other people's work.

**Sol-Attn** — Haopeng Li, Yitong Li, Junsong Chen, Tian Ye, Haozhe Liu,
Jincheng Yu, Duomin Wang, Ruihua Zhang, Zeke Xie, Enze Xie, Song Han (NVIDIA
Research). Kernel, method and everything in `solref/` are theirs.
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

**ComfyUI** — the `optimized_attention_override` hook this builds on.

**Lineage** (no code taken):
[ComfyUI-RadialAttn](https://github.com/woct0rdho/ComfyUI-RadialAttn) (@woct0rdho)
set the opt-in `MODEL → MODEL` patch pattern ·
[RadialAttention](https://github.com/mit-han-lab/radial-attention) (MIT Han Lab) ·
[ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) (@kijai),
where `start_percent`/`end_percent` came from ·
[ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell)
(@KingGore), the `BlockMask.from_kv_blocks` detail `flex` needs.

**Triton** — compiles it. **FlashAttention** (Tri Dao et al.) — lineage in
[`NOTICE`](NOTICE).

SM120 change, integration, packaging:
[@sumeetprashant](https://github.com/sumeetprashant).

## License

Apache 2.0 — see [`LICENSE`](LICENSE), inherited from NVlabs/Sana. Changes to
NVIDIA source itemised in [`NOTICE`](NOTICE) per Apache-2.0 §4(b).
