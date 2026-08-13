# ComfyUI-SolAttn

NVIDIA's **Sol-Attn** sparse attention as an opt-in ComfyUI node — including on
**RTX 50-series (SM120)**, which upstream doesn't support.

## What it does, without the words

Every frame, the model compares every bit of the picture to every other bit.
Most of those comparisons are two bits of empty sky asking each other how it's
going. Pointless, and you pay for all of them.

Sol-Attn lets the model skim. It glances at each chunk, decides "nothing
happening here," and moves on — spending its real effort where things actually
are. Fewer pointless conversations, same film, less time.

The catch is the skimming is a guess. Skim a little and you save a little and
nothing changes. Skim too much and the model loses track of what it already
drew — and cheerfully draws it a second time. In testing it turned one woman
walking down a street into two women walking down a street. It was very pleased
with itself.

**So: it's a dial between "slower and correct" and "faster and hallucinating a
twin."** Your job is finding where that line sits on your machine, with your
model. This page tells you how.

The kernel is NVIDIA's. Getting it to run on a 5090 and wiring it into ComfyUI
is the only thing this repo adds.

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

## How to use it — 5 steps

**1. Open the workflow.**
[`workflows/minimax_h3_i2v_solattn_simple.json`](workflows/minimax_h3_i2v_solattn_simple.json)
— 17 nodes, image → video with audio on MiniMax H3. All ComfyUI core except this
node, so there's nothing else to install.

```
UNETLoader → Sol-Attn → MiniMaxH3SigmaShift → BasicGuider ┐
LoadImage + CLIPLoader + VAELoader → MiniMaxH3ImageToVideo ┴→ Sampler → decode → SaveVideo
```

**2. Repoint three nodes at your own files** — `UNETLoader`, `CLIPLoader`,
`LoadImage`. The filenames in there are from my machine and mean nothing on
yours.

**3. Queue it. Watch the console for this line:**

```
[Sol-Attn] ACTIVE - attention is running on Sol-Attn
```

That line is the whole point. No `ACTIVE` and no `falling back` means it isn't
running — jump to [the two nodes that switch it off](#-two-nodes-will-silently-switch-this-off).

**4. Run it a second time.** The first run at any new resolution pays a
one-time Triton compile *inside* the sampling loop. Judging the node on run one
is like timing a car during the handbrake turn out of the driveway.

**5. Set `enabled` to `false` and run again.** Two numbers, same seed. That's
the entire A/B — no rewiring.

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

| knob | what it actually does | when to touch it |
|---|---|---|
| `tau` | How much work the model is allowed to skip. Low = careful. High = fast. Too high = two Brians. | **This is the dial.** Start at 1.4. |
| `backend` | `triton` or `flex`. | Leave it on `triton`. `flex` spends ~80 s compiling per resolution and only earns that back over 30+ steps. |
| `enabled` | The off switch. | Your A/B button. |
| `start_percent` / `end_percent` | Which slice of the render gets sparse attention. `0.0–1.0` is all of it. | Only to chase a specific artifact. Narrowing it costs speed — that's the point of it. |
| `tau_end` | Slides `tau` from careful at the start to fast at the end. `0` = off. | When one fixed `tau` is either too slow or too rough and you want both. |
| `dense_blocks` | Keeps the first and last N transformer blocks doing full exact work. | Insurance. Measured no benefit here, so try `tau` first. |
| `dense_block_ranges` | Same, but you pick the blocks by hand: `39-42`. | When the blocks worth protecting aren't at the ends. |
| `sink_conditioning` | **MiniMax H3 only.** H3 stacks your prompt and the audio at the front of the sequence. This stops the sparse path from skimming them, so the model keeps listening to what you asked for. | Leave it on `exact_kv` for H3. On anything else it does nothing and says so in the log. |
| `min_tokens` | Below this sequence length, don't bother — the bookkeeping costs more than the skipping saves. | Rarely. |
| `thresh_type` | How it decides what to skip. `diag` is the default; `exact` thinks harder and charges for it. | Rarely. |

### Just tell me what to set

1. **Set `tau` to 1.4.** Leave everything else alone.
2. **Render one clip with `enabled` off, one with it on.** Same seed.
3. **Look at them.** If they show the same thing, keep it. If the second one
   grew an extra person, drop `tau` to 1.2 and look again.

That's the whole method. There is no correct value I can give you, because it
moves with your model, your resolution and your clip length.

### Find your own ceiling

Every number below is the demo workflow, 243 frames, 20 steps, same seed, only
`tau` changed:

| setting | time | vs off | picture |
|---|---|---|---|
| off | 166.8 s | — | one woman |
| `tau 1.0` | 173.1 s | **4% slower** | one woman |
| `tau 1.4` | 156.4 s | **6% faster** | one woman |
| `tau 1.8` | 135.2 s | 19% faster | **two women** ❌ |

Read that table properly, because it's the honest shape of this thing:

- **`tau 1.0` is slower than not using the node at all.** Deciding what to skip
  costs something. Skip too little and you've paid the bill without ordering
  anything.
- **`tau 1.4` bought 6%.** Real, and not going to change your life.
- **`tau 1.8` bought 19% and grew a second woman.** The useful speed and the
  breakage live *right next to each other*, and you cannot tell them apart from
  the console. You have to look at the picture.

On a different graph (6 steps, turbo LoRA) `tau 1.8` held fine and gave 24%. So
the ceiling is not a property of the node — it's a property of your setup. Find
yours. It takes two renders.

### Longer clips are worth more

Same workflow, same seed, only the clip length changed:

| clip | off | on (`tau 1.8`) | gain |
|---|---|---|---|
| 73 frames (3 s) | 35.1 s | 32.1 s | **9%** |
| 243 frames (10 s) | 166.8 s | 135.2 s | **19%** |

The cost of comparing everything to everything grows faster than the clip does,
and skipping those comparisons is the only thing this node does. Short clip,
small pile, small saving.

**If you make three-second clips, this is not your bottleneck. Close the tab.**

---

## Measured

RTX 5090 · MiniMax H3 · 832×640 · 10 s clip · 6 steps · turbo LoRA · same seed ·
back-to-back, one session.

Baseline is **ComfyUI's own `comfy kitchen attention`**, not Sage — kitchen is
the faster and more honest thing to beat, because it's exact.

This table is from a production graph (6 steps, turbo LoRA, chunked feed-forward).
The plain demo workflow above, at 20 steps with no LoRA, measured **19%** at the
same clip length. Same direction, different graph — which is why the exact
percentage you get will be your own.

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

## What's wrong with it

Straight, before you wire it into anything you care about.

1. **It can quietly duplicate things.** Push `tau` too far and characters split
   in two. It happened at `tau 1.8` in the demo workflow on this page. Nothing
   in the log warns you — the render "succeeds."
2. **The console cannot tell you if the picture is fine.** `ACTIVE` means it
   ran, not that it was right. Only your eyes close that loop.
3. **Set too low, it's slower than not using it.** `tau 1.0` measured 4% slower
   than off. There's a floor below which it's pure overhead.
4. **The safe ceiling moves.** `tau 1.8` broke on one graph and was fine on
   another. Change your model, resolution, step count or LoRA and you should
   re-check.
5. **The gain is modest.** 6% at a setting I'd actually ship. Upgrading CUDA
   12.6 → 13.0 on this machine was worth **2.1×** with no quality cost at all.
   Do the free wins first.
6. **First run at any resolution is slow** — a one-time compile happens inside
   the sampling loop. Ignore run one.
7. **Barely tested.** One GPU, one model, one person. If you're on anything
   else you are the test.

None of that makes it useless. It makes it a dial you have to check, not a
switch you flip and forget.

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
