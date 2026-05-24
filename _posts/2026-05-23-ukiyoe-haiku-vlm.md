---
layout: post
title: "A haiku VLM: SFT did the work, KTO collapsed at λ=1.0"
date: 2026-05-23 18:00:00
description: A LLaVA-pattern VLM that writes a 5-7-5 haiku for a ukiyo-e woodblock print. SigLIP (frozen) + trained projector + Qwen2.5-3B (LoRA), on 3,913 Met Museum prints, in English and Japanese. SFT delivered ~95% of the lift; preference optimization only helped where the chosen/rejected gap was real; KTO collapsed at its default λ_U.
tags: vlm multimodal lora orpo kto llava
categories: deep-learning
featured: true
chart:
  plotly: true
toc:
  sidebar: left
---

Build a vision-language model that looks at a Japanese ukiyo-e woodblock print and writes a 5-7-5 haiku about it. The architecture is the standard LLaVA pattern: frozen SigLIP vision encoder, a small trained projector into the LM embedding space, and a frozen Qwen2.5-3B-Instruct with LoRA adapters. ~36M trainable parameters on a ~3.2B backbone. Trained in both English and Japanese on 3,913 Met Museum prints, then graded by Claude Haiku 4.5 on a 4-axis rubric over a held-out 200-image test set.

Three results worth stating up front. **SFT did ~95% of the lift** — the blind text-only baseline scored 3.28, SFT scored 4.18, and image-grounding (`image_fit`) jumped from 1.69 to 3.80. **Preference optimization only helped where the chosen/rejected gap was real** — English ORPO/KTO with same-model pairs moved nothing; Japanese ORPO with Sonnet-vs-Haiku pairs lifted `image_fit` by +0.27. And **KTO collapsed at its default `λ_U=1.0`**, scoring 3.68 (below SFT) until the undesirable weight was knocked down to 0.1.

```plotly
{"data":[{"x":["random proj","blind Qwen","SFT_en","SFT_jp","ORPO_jp","KTO_jp λ=1.0","KTO_jp λ=0.1"],"y":[2.82,3.28,4.18,4.29,4.36,3.68,4.33],"type":"bar","marker":{"color":["#AAB0C0","#AAB0C0","#636EFA","#636EFA","#00CC96","#EF553B","#636EFA"]},"text":["2.82","3.28","4.18","4.29","4.36","3.68","4.33"],"textposition":"outside","hovertemplate":"%{x}<br>avg %{y:.2f}<extra></extra>"}],"layout":{"title":{"text":"Average judge score (1–5), held-out 200-image test set"},"xaxis":{"title":""},"yaxis":{"title":"avg score","range":[0,5]},"height":440,"margin":{"l":60,"r":30,"t":60,"b":80},"showlegend":false}}
```

## Pipeline

The full run, in order. Steps 1–6 are CPU/API work done on the laptop before renting a GPU; 7–10 run on a rented A100.

1. **Scrape** — filter the Met Open Access CSV locally to Japanese woodblock prints (~3.9k objects), then fetch image URLs.
2. **Download** — pull and resize each print to 224×224.
3. **Caption** — Claude writes a rich haiku (chosen) and a bare one (rejected) per image. The Japanese variant uses Sonnet 4.6 for chosen and Haiku 4.5 for rejected.
4. **Self-correct** — Claude critiques and revises its own captions, producing self-correction traces for SFT.
5. **Build preference pairs** — assemble the (chosen, rejected) data for ORPO/KTO.
6. **Split** — hold out 200 test IDs (seed=42), excluded from every training set.
7. **SFT** — train projector + LoRA on chosen captions + self-correct traces.
8. **ORPO** — preference optimization on the pairs, initialized from SFT.
9. **KTO** — the `λ_U` sweep, also initialized from SFT.
10. **Eval** — Claude Haiku 4.5 as judge, 4-axis rubric, on the 200 held-out images.

The rest of this post follows the substance of that flow rather than the script order — architecture first, then where each training stage actually landed.

## Architecture

```
image (224×224)
  → SigLIP-base (frozen, 86M) → 196 patch embeddings (768-d)
  → 2-layer MLP projector (trained, 768→2048→2048, GELU)
  → 196 image tokens (2048-d)
  → injected at <|image|> placeholder positions in Qwen2.5-3B-Instruct
  → LoRA r=16 on q/k/v/o + gate/up/down (~32M trainable)
  → CE loss on the assistant haiku tokens only
```

Trainable: 35.7M of 3.21B (1.11%). Both the vision tower and the LM base are frozen; only the projector and the LoRA adapters learn.

The one genuinely interesting piece is how the image enters the language model. A new `<|image|>` token is added to the tokenizer, and the text prompt contains exactly 196 of them. At forward time the text embeddings are computed normally, then the embeddings at those 196 placeholder positions are *overwritten* with the projected image tokens:

```python
def _inject_image_embeds(self, input_ids, pixel_values):
    embed_layer = self.lm.get_input_embeddings()
    text_embeds = embed_layer(input_ids)              # (B, T, 2048)
    img_embeds = self.encode_image(pixel_values)      # (B, 196, 2048)

    # Boolean mask: True at every <|image|> placeholder
    img_mask = input_ids == self.img_token_id         # (B, T)
    # Each row contains EXACTLY 196 placeholders (the dataset enforces this),
    # so the flat assignment lines up row-by-row.
    text_embeds[img_mask] = img_embeds.reshape(-1, img_embeds.shape[-1])
    return text_embeds
```

That flat masked assignment is the whole trick, and it relies on a hard invariant: every sequence has exactly 196 placeholders, so flattening the masked positions and flattening the image tokens line up one-to-one. The dataset collator enforces it; if a row had 195 or 197 placeholders the assignment would misalign silently.

The image side is a frozen vision pass followed by the trained projector, with one numerical care:

```python
def encode_image(self, pixel_values):
    with torch.no_grad():                       # vision tower is frozen
        feats = self.vision(pixel_values=pixel_values).last_hidden_state  # (B,196,768)
    # Projector runs in fp32 then casts; bf16 random init can underflow on step 0.
    feats = feats.float() if self.projector[0].weight.dtype == torch.float32 else feats
    return self.projector(feats).to(torch.bfloat16)
```

The projector is initialized in fp32 because a bf16 random init can underflow to zero on the very first step and stall learning; it gets cast to bf16 only after the matmul.

## Data: 3,913 prints, and why the chosen/rejected source matters

Images come from the Met Museum Open Access set — filtered locally to Japanese woodblock prints — paired with Claude-generated haiku. The preference data is where the two language runs diverge, and it turned out to be the single most important design choice:

- **English:** chosen = Haiku 4.5 with a rich prompt, rejected = Haiku 4.5 with a bare prompt. Same model on both sides.
- **Japanese:** chosen = **Sonnet 4.6** with a rich prompt, rejected = **Haiku 4.5** with a bare prompt. Genuinely different model capability.

That distinction is the explanation for half the results below.

## SFT, and the two baselines that make the number mean something

SFT trains on the chosen captions plus self-correction traces (Claude critiquing and revising its own captions), with loss masked to the assistant turn. The trained-variant comparison alone (SFT vs ORPO vs KTO) says nothing about whether the *vision pipeline* did anything — for that you need to remove the conditioning. Two baselines:

| Baseline | What it is | avg |
|---|---|---:|
| Random projector | untrained projector, LoRA off — vision pipeline emits noise | 2.82 |
| Blind Qwen | raw Qwen2.5-3B, text-only prompt, no image at all | 3.28 |
| **SFT (trained)** | full pipeline after SFT | **4.18** |

The blind-Qwen → SFT lift is +0.90 on the average, and `image_fit` alone jumps **+2.11 (1.69 → 3.80)**. Without these baselines I'd only have known SFT and ORPO were roughly tied; I wouldn't have known the vision pipeline contributed anything at all. For any conditional-generation model, the "remove the condition" baseline is the one that tells the story.

## Preference optimization needs a real quality gap

With SFT as the floor, did ORPO and KTO add anything? In English — where chosen and rejected came from the same model — no:

| English | SFT | ORPO | KTO |
|---|---:|---:|---:|
| avg | 4.18 | 4.16 | 4.17 |

The rejected captions were 80–90% as good as the chosen ones, so there was no preference signal to learn. In Japanese, with a genuine Sonnet-vs-Haiku capability gap, preference optimization did real work:

| Japanese | SFT | ORPO | KTO (λ_U=0.1) |
|---|---:|---:|---:|
| avg | 4.29 | **4.36** | 4.33 |
| image_fit | 3.61 | **3.88** | 3.81 |

ORPO lifted `image_fit` by +0.27. The lesson is blunt: preference optimization needs the preference to actually exist in the data. Same model with prompt variation is not enough — you need different-capability models, human curation, or genuinely contrasting policies.

## KTO and the λ_U collapse

KTO ("Kahneman-Tversky Optimization") doesn't need pairs; it learns from individual examples labeled desirable or undesirable, weighting the two asymmetrically. The implicit reward is `r = log π_policy − log π_ref`, and `π_ref` is just the model with its LoRA adapter disabled — so policy and reference are the same weights, run twice:

```python
# Forward 1: policy (LoRA active)
policy_logits = model.forward_logits(input_ids, attention_mask, pixel_values)
log_p_policy = response_logprobs(policy_logits, input_ids, response_start, pad_id)

# Forward 2: reference (LoRA disabled — same weights, no second model in VRAM)
with torch.no_grad(), model.lm.disable_adapter():
    ref_logits = model.forward_logits(input_ids, attention_mask, pixel_values)
    log_p_ref = response_logprobs(ref_logits, input_ids, response_start, pad_id)
```

The loss weights desirable and undesirable examples through separate `λ` coefficients:

```python
r = log_p_policy - log_p_ref                          # implicit reward
v_d = lambda_D * torch.sigmoid(beta * (r - z_ref))    # desirable value
v_u = lambda_U * torch.sigmoid(beta * (z_ref - r))    # undesirable value
v = torch.where(label > 0.5, v_d, v_u)
loss = (1.0 - v).mean()
```

At the default `λ_U=1.0`, combined with the *strong* Japanese contrast, this drove the undesirable reward `r_undesirable` to about **−200** by end of training — meaning the policy assigned `e^{−200}×` the reference probability to anything that looked like an undesirable haiku. But undesirable haiku and desirable haiku share most of their tokens (they are both haiku), so suppressing one suppressed the other. The model broke its own ability to generate haiku at all: the average score collapsed to **3.68**, and only 34 of 200 test outputs were even parseable by the judge.

The training logs showed it plainly — `loss → 0.0001`, gradient `norm → 0.000` for hundreds of steps near the end, the model frozen in a degenerate "suppress everything" solution while the loss looked perfectly converged. A sweep on `λ_U` recovered it:

| λ_U | final r_undesirable | test avg | notes |
|---|---:|---:|---|
| 1.0 | ≈ −200 | 3.68 | collapsed; most outputs unparseable |
| 0.3 | ≈ −50 | 4.26 | restored to ~SFT level |
| 0.1 | collapsed late | 4.33 | slight gain; still trails ORPO (4.36) |

Two lessons fell out of this. First, **KTO is far more sensitive to `λ_U` than ORPO is** — under strong contrast the default 1.0 over-suppresses; 0.1–0.3 is the usable range for a task like this. Second, **a converged training loss is not a good signal; eval is the only honest one.** The collapsed model had the cleanest-looking loss curve of the whole run.

(One metric scare worth defusing: `r_undesirable_mean` logged `NaN` on ~20% of batches. With `batch_size=4` and a 33% undesirable rate, `P(no undesirables in a batch) = 0.67⁴ ≈ 20%`, and the mean over an empty slice is `float("nan")` by construction — `r[label < 0.5].mean() if (label < 0.5).any() else float("nan")`. The loss itself computed fine from the desirable examples present; only the printed metric was undefined.)

## Out-of-distribution behavior

Run on a dense Western fantasy painting — nothing like ukiyo-e — both models produced coherent, domain-shifted haiku rather than gibberish or refusals:

{% include figure.liquid loading="eager" path="assets/img/ukiyoe-ood.jpg" class="img-fluid rounded z-depth-1" alt="A dense, grotesque Western fantasy painting — crowded faces, masks, and tangled limbs in dark reds and browns — used as an out-of-distribution test image" %}

English (SFT):
```
Lanterns pierce the dark
Crowded feast in shadow's glow
Night feasts on the night
```

Japanese (ORPO):
```
夜の闇に     into the darkness of night
鬼の面重く   the demon mask is heavy
武者屈む     a warrior crouches
```

The Japanese model caught a real visual feature — the painting genuinely has oni-like grotesque masks, and the model named them (鬼の面, "demon mask"). Neither output is "correct" in a rigorous sense; both are interpretable hallucinations that reach for whatever visual features map to the model's training vocabulary. For a narrow-domain specialist, "tries and reaches for vocabulary it has" is exactly the OOD failure mode you want.

## A note on the checkpoint size

Each published "LoRA adapter" is **1.36 GB**, not the ~30 MB you'd expect from r=16 on a 3B model. Adding the `<|image|>` token resizes Qwen's embedding matrix, and when PEFT serializes the adapter it detects the embedding mismatch and bundles the *entire resized embedding matrix* into the safetensors file: `vocab × hidden × 2 bytes ≈ 621 MB`, stored twice (current + original-shape buffer). New vocabulary tokens are not free at save time — either reuse rare existing tokens as placeholders, or budget for fat checkpoints.

## Cost

| Component | Cost |
|---|---|
| GPU (RunPod A100 80GB, ~12 hr intermittent) | $16.41 |
| Anthropic API (EN+JP captions, self-correct traces, judge) | ~$15 |
| **Total** | **~$31** |

The Anthropic spend came in well under estimate: 224×224 images tokenize to ~80 vision tokens (not the 300+ I'd assumed) and haiku outputs are ~30 tokens, so per-call cost was tiny. Wall-clock was dominated by the tier-1 request-per-minute limit, not by tokens or latency.

Code, scripts (English and Japanese variants), and the full lessons writeup: [github.com/debtirthasaha/ukiyoe-haiku-vlm](https://github.com/debtirthasaha/ukiyoe-haiku-vlm). The two models are on Hugging Face — Japanese ORPO (flagship) at [MR0b0t/ukiyoe-haiku-vlm-jp](https://huggingface.co/MR0b0t/ukiyoe-haiku-vlm-jp) and English SFT at [MR0b0t/ukiyoe-haiku-vlm-en](https://huggingface.co/MR0b0t/ukiyoe-haiku-vlm-en).
