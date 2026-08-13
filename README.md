# ComfyUI-SolAttn

A ComfyUI node that makes MiniMax H3 video generation faster. It runs NVIDIA's
Sol-Attn on RTX 50-series cards, which NVIDIA's own code refuses to do.

## What it does

While generating a video, the model repeatedly compares every part of the
picture with every other part. A lot of that work is wasted — most of those
comparisons are between areas where nothing is happening.

This node lets the model skip the comparisons it judges to be unimportant, and
spend its time on the parts that matter. That makes rendering faster.

The trade-off is that skipping is a judgement call, not a certainty. Skip a
little and you save a little time with no visible change. Skip too much and the
model loses track of what it has already drawn, and can draw the same character
or object twice.

So this is a dial, not a switch. Turned down it is safe and saves little. Turned
up it saves more and eventually damages the picture. The useful part of this
page is helping you find where that line sits on your machine.

The underlying code is NVIDIA's. What this repository adds is making it run on a
5090 and packaging it as a ComfyUI node.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/sumeetprashant/ComfyUI-SolAttn.git
```

Nothing to pip install. You need Triton 3.4.0 or newer.

## How to use it

Open [`workflows/minimax_h3_i2v_solattn_simple.json`](workflows/minimax_h3_i2v_solattn_simple.json).
It is 17 nodes, and everything in it except this node ships with ComfyUI. It
turns one image into a video with sound using MiniMax H3.

```
UNETLoader → Sol-Attn → SigmaShift → Guider ┐
LoadImage + CLIP + VAE → H3ImageToVideo     ┴→ Sampler → decode → SaveVideo
```

1. Point `UNETLoader`, `CLIPLoader` and `LoadImage` at your own files. The
   filenames saved in the workflow are from my machine.
2. Set `tau` to **1.4** and leave every other setting alone.
3. Render the same clip twice with the same seed: once with `enabled` off, once
   with it on.
4. Compare them. If they show the same thing, keep the setting. If the second
   one has gained an extra character, lower `tau` and try again.

Two things worth knowing before you judge the speed:

- **Ignore the first render at any new resolution.** The node compiles itself
  the first time it sees a new frame size, and that happens during rendering, so
  the first result is always slower than the real figure.
- **Check the console.** This line means the node is actually working:

  ```
  [Sol-Attn] ACTIVE - attention is running on Sol-Attn
  ```

  If you see neither that line nor a `falling back` line, the node is not
  running at all. See [Things that stop it working](#things-that-stop-it-working).

## Measured results

All from the demo workflow above: 243 frames, 20 steps, same seed, with only
`tau` changed between runs.

| `tau` | time | compared to off | picture |
|---|---|---|---|
| off | 166.8 s | — | correct |
| 1.0 | 173.1 s | **4% slower** | correct |
| 1.4 | 156.4 s | **6% faster** | correct |
| 1.8 | 135.2 s | 19% faster | **a character was duplicated** |

Three things are worth taking from that table:

- **Setting it too low is worse than not using it.** Working out what to skip
  costs time of its own, so at low settings you pay that cost without saving
  enough to cover it.
- **The best speed sits immediately next to the point where it breaks.**
- **Nothing warns you when it breaks.** The render completes normally and the
  console looks healthy. You have to look at the video.

On a different setup — 6 steps with a turbo LoRA — `tau 1.8` was fine and saved
24%. The safe maximum therefore depends on your models and settings rather than
on this node, so it is worth finding your own. Two renders will tell you.

### Longer clips benefit more

Same workflow and seed, with only the clip length changed:

| clip length | off | on | saving |
|---|---|---|---|
| 3 seconds | 35.1 s | 32.1 s | 9% |
| 10 seconds | 166.8 s | 135.2 s | 19% |

The comparison work grows faster than the clip length does, and skipping that
work is all this node does. A short clip simply does not have much to skip. If
you mostly make short clips, this node will not help you much.

## Known problems

1. Pushed too far, it duplicates characters or objects, with no warning.
2. The `ACTIVE` message confirms the node ran. It does not confirm the picture
   is correct. Only checking the video does that.
3. Set too low, it is slower than not using it at all.
4. The safe setting changes when your model, resolution, step count or LoRA
   changes, so it is worth re-checking after any of those.
5. The saving is modest. For comparison, updating CUDA from 12.6 to 13.0 on this
   machine was worth 2.1× with no loss of quality at all. Exact speed-ups like
   that are worth doing before approximate ones like this.
6. The first render at any new resolution is slow, for the reason above.
7. It has been tested on one GPU, with one model, by one person.

## Things that stop it working

Two other nodes will disable this one without any error message.

| node | what happens |
|---|---|
| **MiniMax H3 Memory Efficient Sage Attention** (KJNodes) | It replaces the attention step on every layer, so this node never gets called, regardless of the order you wire them in |
| **ModelAttentionBackend** (in ComfyUI core) | It uses the same slot as this node, so whichever is applied last wins |

Bypass either one if you want this node to run.

## Settings

Only `tau` and `backend` matter to begin with.

| setting | what it does | when to change it |
|---|---|---|
| `tau` | How much the model is allowed to skip. Lower is safer, higher is faster. | This is the main dial. Start at 1.4 |
| `backend` | Which implementation runs. | Leave on `triton`. `flex` spends about 80 seconds compiling for each new frame size |
| `enabled` | Turns the node off without unplugging it. | For comparing with and without |
| `start_percent` / `end_percent` | Limits skipping to part of the render. | Only when chasing a specific fault |
| `tau_end` | Starts careful and gets faster as the render progresses. | When a single fixed `tau` is either too slow or too rough |
| `dense_blocks` | Forces the first and last few layers to do full work. | As insurance. It made no measurable difference here |
| `dense_block_ranges` | The same, but you choose which layers, e.g. `39-42`. | When the layers you want to protect are not at either end |
| `sink_conditioning` | MiniMax H3 only. Stops the model skipping over your prompt and the audio. | Leave on `exact_kv` for H3. It does nothing on other models and says so in the log |
| `min_tokens` | Below this clip size, don't skip anything. | Rarely |
| `thresh_type` | How the skip decision is calculated. `exact` is more careful and costs more time. | Rarely |

## Alternatives

I have not benchmarked these against each other. Several do more than this one.
Choose on what you need.

| project | why you might prefer it |
|---|---|
| [Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn) | More features than this repository, including a dedicated H3 version. Also supports SM89 |
| [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) | Another version of the same thing, actively maintained |
| [KingGore/ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell) | A different route to running this on Blackwell cards |
| [woct0rdho/ComfyUI-RadialAttn](https://github.com/woct0rdho/ComfyUI-RadialAttn) | A different method for the same goal |
| **comfy kitchen attention**, built into ComfyUI | Worth trying first. It is already installed, it is fast, and it does not approximate anything |

## Requirements

The node checks these at runtime. Anything it cannot handle falls back to your
normal settings and logs the reason, so it will not break a render.

| | |
|---|---|
| head size | exactly 128 |
| precision | bfloat16 |
| attention mask | none |
| layout | 4D q/k/v |

Tested on: RTX 5090, torch 2.11.0+cu130, Triton 3.4.0, ComfyUI 0.32.0, Windows
11, MiniMax H3. Older NVIDIA cards are untested.

**About the card check:** NVIDIA's code blocks anything that isn't an H100 or
B200. That check exists to choose between hand-written versions of the code that
only exist for those two cards. It is not a safety check, and no software of
this kind can affect your GPU's power or temperature limits. The version this
repository uses is compiled for whatever card you have and never needed that
restriction; it only inherited it by sharing the same validation function. This
repository removes it for that version only, and keeps every other check. The
risk is a wrong-looking picture, not damaged hardware. The exact change is
recorded in [`NOTICE`](NOTICE).

If you get bad results on another card, that is this repository's problem to
fix. Please don't report it to NVlabs/Sana.

## Credits

Nearly all of the work here is other people's.

**Sol-Attn** — Haopeng Li, Yitong Li, Junsong Chen, Tian Ye, Haozhe Liu,
Jincheng Yu, Duomin Wang, Ruihua Zhang, Zeke Xie, Enze Xie and Song Han (NVIDIA
Research). The method and everything in `solref/` is theirs.
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

**ComfyUI** — for the hook that makes a per-model patch like this possible.

**Design lineage**, though no code was taken from them:
[ComfyUI-RadialAttn](https://github.com/woct0rdho/ComfyUI-RadialAttn)
(@woct0rdho) established the opt-in patch-node pattern this follows ·
[RadialAttention](https://github.com/mit-han-lab/radial-attention) (MIT Han Lab)
· [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton)
(@kijai), where the `start_percent` / `end_percent` idea came from ·
[ComfyUI_sol-attn_Blackwell](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell)
(@KingGore), for the `BlockMask.from_kv_blocks` detail the `flex` backend needs.

**Triton** — compiles the code. **FlashAttention** (Tri Dao and others) —
lineage recorded in [`NOTICE`](NOTICE).

SM120 change, integration and packaging:
[@sumeetprashant](https://github.com/sumeetprashant).

## License

Apache 2.0 — see [`LICENSE`](LICENSE), inherited from NVlabs/Sana. Changes to
NVIDIA's source are itemised in [`NOTICE`](NOTICE) as required by Apache-2.0
§4(b).
