---
layout: post
title: "A VLA from scratch: 29% tokens, 0% grasps, and a GRPO that wouldn't budge"
date: 2026-06-07 09:00:00
description: "Building a vision-language-action model from scratch on LIBERO-Spatial: SigLIP + Qwen2.5-3B, behavior cloning to 29% action-token accuracy but 0% closed-loop success, and a GRPO run that trained cleanly without improving. An honest negative result, plus the six bugs in the way."
tags: vla robotics grpo reinforcement-learning libero
categories: deep-learning
featured: false
chart:
  plotly: true
toc:
  sidebar: left
---

A vision-language-action (VLA) model, built from scratch on a Qwen2.5-3B backbone and pushed through the full robot-learning pipeline: behavior cloning on LIBERO-Spatial demos, closed-loop evaluation in the MuJoCo simulator, and GRPO fine-tuning. One A100, one suite, ~6 GPU-hours.

It did not work, and that is the post. Behavior cloning reached ~29% action-token accuracy but **0% closed-loop success**. GRPO ran cleanly — finite loss, stable ratios, no divergence — and did not move the policy. The interesting content is *why* each stage behaved the way it did, and the six bugs between "scaffold" and "a number," several of which fail silently.

| Stage | Closed-loop success | Open-loop action-token acc |
|---|---|---|
| Behavior cloning | 0 / 15 (0.0%) | ~0.29 (plateau) |
| GRPO (dense reaching reward, 5 iters) | 0.0% | — |

Backbone Qwen2.5-3B + SigLIP-so400m (frozen), LoRA rank 32, **66.4M trainable parameters**.

## The architecture

```
agentview_rgb   ─► SigLIP-so400m (frozen) ─┐
eye_in_hand_rgb ─► SigLIP-so400m (frozen) ─┤
                                            ├─► MLP projector ─► Qwen2.5-3B (LoRA) ─► 7 action tokens ─► robot
"pick up the black bowl ..." ─► tokenizer ─┘
```

Two things join the LLM's sequence very differently, and this asymmetry matters:

- **Vision goes in as continuous vectors.** SigLIP turns each 384px camera frame into ~729 patch embeddings; a 2-layer MLP projects them into Qwen's embedding space, and they are spliced into the input sequence as-is — no discretization, no vocab lookup.
- **Actions come out as discrete tokens.** The 7-DoF action is binned and mapped onto existing Qwen vocabulary token ids. The model predicts them autoregressively, exactly like text.

The transformer attends over `[image vecs][prompt tokens][action tokens]` uniformly and has no idea which vectors came from pixels. Loss is applied only on the 7 action positions. This is the OpenVLA recipe, minus an embedding resize.

## Action tokenization, and a silent out-of-bounds bug

Each action dimension is normalized to its empirical 1st–99th percentile range and split into 256 uniform bins:

$$\text{bin}_d(a) = \left\lfloor \frac{\text{clip}(a,\ell_d,h_d)-\ell_d}{h_d-\ell_d}\cdot 256 \right\rfloor, \qquad \text{token id} = \text{offset} + \text{bin}_d(a)$$

The fitted ranges over the LIBERO-Spatial demos (the gripper dim is binary, hence ±1):

```
low  = [-0.761 -0.656 -0.938 -0.108 -0.205 -0.186 -1.0]
high = [ 0.938  0.873  0.935  0.105  0.176  0.144  1.0]
```

The `offset` is where this got interesting. The first version set `offset = embed_rows` — the *count* of embedding rows (151936) — used as a starting *index*. Valid indices stop at 151935, so every action token id was out of bounds. On CUDA, an out-of-bounds embedding lookup does not raise; it reads adjacent memory. The result was a loss that fell to ~7.7 and then sat there, with token accuracy pinned at exactly 0 — the classic signature of a model training on garbage.

Worse, the obvious fix (`offset = embed_rows − 256`, the genuine "last 256 rows") is *also* wrong: Qwen pads its vocabulary with ~292 near-identical trailing rows (they differ by ~6e-5). A tied, frozen `lm_head` cannot produce different logits for identical target rows, so those bins would be indistinguishable. The fix detects the padding and maps onto the last 256 *distinct* real rows, with an assertion so it can never silently go out of bounds again:

```python
emb_w = self.llm.get_input_embeddings().weight.data
pad = (emb_w - emb_w[-1:]).abs().amax(dim=1) < 1e-3        # trailing identical padding
non_pad = (~pad).nonzero().flatten()
last_real = int(non_pad.max()) + 1
offset = last_real - self.atok.n_bins                       # -> [151388, 151644)
assert 0 <= offset and offset + self.atok.n_bins <= self.embed_rows
self.atok.set_token_offset(offset)
```

## Behavior cloning: a fast climb, then a wall

BC is next-token cross-entropy masked to the 7 action positions. Loss starts near $\ln(151936)\approx 11.9$ (uniform over the full vocab) and has to drop below $\ln(256)\approx 5.5$ before the model is meaningfully concentrating mass on the action region. It did that in ~400 steps — and then stopped improving.

```plotly
{"data":[{"x":[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,740,760,780,800,820,840,860,880,900,920,940,960,980,1000,1020,1040,1060,1080,1100,1120,1140,1160,1180,1200,1220,1240],"y":[13.685,10.728,7.678,6.831,6.418,6.158,6.109,5.914,5.719,5.837,5.758,5.725,5.514,5.549,5.583,5.485,5.543,5.595,5.318,5.401,5.369,5.346,5.373,5.536,5.26,5.371,5.383,5.532,5.424,5.414,5.376,5.34,5.411,5.399,5.316,5.387,5.418,5.384,5.357,5.271,5.374,5.416,5.484,5.299,5.368,5.364,5.129,5.302,5.194,5.241,5.256,5.461,5.171,5.636,5.438,5.255,5.362,5.194,5.237,5.349,5.299,5.234],"name":"BC loss","type":"scatter","mode":"lines","line":{"color":"#636efa","width":2},"hovertemplate":"step %{x}<br>loss %{y:.3f}<extra></extra>"},{"x":[20,1240],"y":[5.545,5.545],"name":"ln(256) = 5.545","type":"scatter","mode":"lines","line":{"color":"#00cc96","width":1.5,"dash":"dash"},"hoverinfo":"skip"},{"x":[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,740,760,780,800,820,840,860,880,900,920,940,960,980,1000,1020,1040,1060,1080,1100,1120,1140,1160,1180,1200,1220,1240],"y":[0.005,0.009,0.075,0.107,0.158,0.176,0.166,0.228,0.232,0.229,0.246,0.232,0.258,0.255,0.243,0.26,0.256,0.254,0.287,0.272,0.269,0.258,0.266,0.238,0.286,0.274,0.265,0.241,0.262,0.249,0.265,0.26,0.264,0.26,0.279,0.258,0.273,0.263,0.283,0.277,0.271,0.263,0.249,0.25,0.26,0.275,0.321,0.296,0.317,0.268,0.291,0.26,0.28,0.18,0.256,0.271,0.276,0.319,0.271,0.29,0.26,0.272],"name":"action-token acc","yaxis":"y2","type":"scatter","mode":"lines","line":{"color":"#EF553B","width":2},"hovertemplate":"step %{x}<br>acc %{y:.3f}<extra></extra>"}],"layout":{"title":{"text":"Behavior cloning: loss crosses ln(256) fast, accuracy plateaus at ~0.29"},"xaxis":{"title":"step"},"yaxis":{"title":"loss","range":[4.5,14]},"yaxis2":{"title":"token acc","overlaying":"y","side":"right","range":[0,0.4]},"height":460,"margin":{"l":60,"r":60,"t":60,"b":50},"hovermode":"x unified","legend":{"x":0.55,"y":0.95}}}
```

The plateau at ~0.29 is the whole story, and it traces back to the action head. Actions are mapped onto existing Qwen vocab rows, and with a **tied, frozen `lm_head`** those rows are arbitrary fixed vectors that were never trained to *be* action representations. A LoRA-only model can only steer its hidden states toward those fixed targets — it cannot reshape the targets. That caps how finely it can separate 256 bins per dimension. OpenVLA full-finetunes the head precisely so the action rows *become* good action representations; we kept it frozen, and 0.29 is roughly what that buys.

## Closed-loop: 0%

Token accuracy is a proxy, and a deceptive one. Closed-loop control runs the policy for up to 150 steps and asks whether the LIBERO goal predicate fires; per-step errors compound. A 29%-per-token policy means roughly $0.29^7 \approx 0.02\%$ of full 7-token actions are exactly right, and even "close" actions drift the arm over a long rollout.

```
[eval] task 0 ...black_bowl_between_the_plate_and_the_ramekin...: 0/5
[eval] task 1 ...black_bowl_next_to_the_ramekin...:             0/5
[eval] task 2 ...black_bowl_from_table_center...:               0/5
```

0/15, terminated early — the GRPO gate is 15% and this cannot meet it. Not a bug; an underfit policy doing exactly what an underfit policy does.

## GRPO, and the cold-start trap

GRPO needs no critic — the group mean is the baseline. For $G$ rollouts from one init state with rewards $r_1\dots r_G$:

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)+\varepsilon}, \qquad \mathcal{L} = -\mathbb{E}\big[\min(\rho A,\ \text{clip}(\rho,1-\epsilon,1+\epsilon)A)\big],\quad \rho = e^{\log\pi_\theta-\log\pi_{\theta_\text{old}}}$$

Here is the trap, sitting right in the advantage formula: if the policy succeeds ~never, every $r_i$ in a group is 0, so $\text{std}(r)=0$, every $A_i=0$, and the update is a silent no-op. That is the same failure as a contrastive method with no contrast. The pipeline gates GRPO on BC clearing 15% for exactly this reason; BC scored 0%, so the gate refused.

To run GRPO at all on a 0% policy, the reward needs variance. LIBERO's task reward is sparse (even `reward_shaping=True` leaves the per-step reward at 0), but the observation exposes per-object `<obj>_to_robot0_eef_pos` vectors. So the reward became a dense **reaching** signal — how close the gripper got to the nearest object over the episode:

```python
def _reach_dist(obs):
    ds = [float(np.linalg.norm(np.asarray(v)))
          for k, v in obs.items() if k.endswith("_to_robot0_eef_pos")]
    return min(ds) if ds else 0.0

# per rollout:  dense = -min_reach_distance_over_episode + 10 * success
# then group-relative advantage:
dense = np.array(dense)
adv = (dense - dense.mean()) / (dense.std() + 1e-6)
```

This gives non-zero variance across rollouts even at 0% task success, so the update has signal. And it did — the run trained cleanly. It just didn't help:

```plotly
{"data":[{"x":[0,1,2,3,4],"y":[-0.260,-0.313,-0.309,-0.285,-0.279],"name":"reach reward","type":"scatter","mode":"lines+markers","line":{"color":"#636efa","width":2.5},"marker":{"size":7},"hovertemplate":"iter %{x}<br>reach_R %{y:.3f}<extra></extra>"},{"x":[0,1,2,3,4],"y":[0.041,0.018,0.017,0.016,0.013],"name":"reward std (group)","yaxis":"y2","type":"scatter","mode":"lines+markers","line":{"color":"#EF553B","width":2,"dash":"dot"},"marker":{"size":6},"hovertemplate":"iter %{x}<br>std %{y:.3f}<extra></extra>"}],"layout":{"title":{"text":"GRPO: reach reward stays flat while group reward variance collapses"},"xaxis":{"title":"iteration","dtick":1},"yaxis":{"title":"mean reach reward","range":[-0.34,-0.24]},"yaxis2":{"title":"group reward std","overlaying":"y","side":"right","range":[0,0.05]},"height":440,"margin":{"l":60,"r":60,"t":60,"b":50},"hovermode":"x unified","legend":{"x":0.5,"y":0.95}}}
```

Two honest reasons it stayed flat:

1. **The curve is confounded.** Each iteration samples *different* random tasks and init states, so the reach reward across iterations mixes policy change with task-sampling noise. A fixed eval set per iteration would have been the correct design; this one cannot cleanly show a trend either way.
2. **The advantage signal collapses.** Group reward std falls 0.041 → 0.013 as the policy sharpens. Lower variance means weaker advantages means a weaker learning signal — the standard GRPO variance-collapse dynamic, made worse by a tiny group size (4) and a base policy that barely reaches in the first place.

GRPO is downstream of a competent policy. With a base that succeeds even occasionally, sparse task-success GRPO becomes viable. The dense reaching reward was a workaround to make the *machinery* run on a 0% base, not a substitute for the base.

## The bugs (this is the actual work)

Five more, beyond the action-token offset above. Several fail silently, which is the dangerous kind.

**bf16 vs fp32 outside autocast.** Training wraps the forward in `autocast(bf16)`, which silently casts the fp32 projector output. Eval, rollout, and GRPO run *without* autocast → `mat1 Float, mat2 BFloat16`. The local smoke test never caught it because the smoke config runs everything in fp32. Fix: build `inputs_embeds` in the LLM's compute dtype inside the model, independent of any external autocast context.

**A `0 * -inf = NaN` in the GRPO log-probs.** Masking non-action positions with `logp * mask` does `0 * (-inf)` (those vocab columns are −inf), producing NaN, which corrupts the weights and crashes the next rollout's sampler. BC was immune because it uses `cross_entropy(ignore_index=-100)`.

```python
tok_lp = logp.gather(-1, safe[..., None])[..., 0]
tok_lp = torch.where(mask, tok_lp, torch.zeros_like(tok_lp))   # not tok_lp * mask
```

**OOM at batch 16.** ~1,460 vision tokens × 36 Qwen layers exhausts 80 GB. Gradient checkpointing (`use_reentrant=False` + `enable_input_require_grads`) plus `expandable_segments` brought steady state to ~40 GB at batch 8.

**A lost 5.4-hour epoch.** The first BC run had no mid-epoch checkpointing, and the epoch-end *validation* (which runs before the save) hit the bf16 bug and crashed — so nothing was written. Lesson paid in GPU-hours: checkpoint by step from step one.

**h5py + DataLoader fork corruption.** Opening an HDF5 handle in the parent process (during tokenizer fitting) and inheriting it across `num_workers` forks reads corrupt data. Fix: open/close per file locally; let `__getitem__` open lazily inside each worker.

And the environment, which fought back the whole way: LIBERO's `requirements.txt` pins `transformers==4.21.1` and `numpy==1.22.4` (install it `--no-deps`); `transformers` 5.x needs torch ≥2.7 (pinned 4.46.3 for torch 2.4.1); robosuite 1.4 needs `mujoco==2.3.x` (3.x breaks); and EGL headless rendering needs both the `libegl1`/`libglvnd0` system libraries **and** `PYOPENGL_PLATFORM=egl`, not just `MUJOCO_GL=egl`.

## Cost ledger

| Phase | Notes | Cost |
|---|---|---|
| Setup: LIBERO + sim + EGL, dataset | the environment fights above | ~$3 |
| BC (incl. the lost 5.4h epoch) | ~$1.50/hr A100-80GB | ~$10 |
| Closed-loop eval | terminated early at 0/15 | ~$1 |
| GRPO (5 iters) + debugging | dense reward, NaN fix | ~$5 |
| **Total** | ~6 GPU-hours, 1× A100-80GB | **~$19** |

## What would actually fix it

- **Unfreeze the head.** The frozen tied `lm_head` is the bottleneck. Full-finetuning it (or adding trainable action-token embeddings) is the single change most likely to push BC token accuracy past ~0.29 and produce non-zero closed-loop success.
- **Give GRPO a competent base.** Once BC succeeds occasionally, sparse task-success GRPO works and the dense reaching reward becomes unnecessary.
- **Fix the eval confound.** Hold the GRPO evaluation tasks and init states fixed across iterations for a comparable learning curve.

The honest summary: a complete VLA pipeline that runs end-to-end, a behavior-cloned policy that underfits for a diagnosable reason, and a GRPO implementation that trains cleanly but is downstream of a base it never got. The gate worked — it correctly refused to pretend a 0% policy was ready for RL.

Code: [github.com/debtirthasaha/vla-libero-grpo](https://github.com/debtirthasaha/vla-libero-grpo).
