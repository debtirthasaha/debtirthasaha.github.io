---
layout: post
title: "Birkhoff in 8.7 KB"
date: 2026-04-20 10:00:00
description: An 8.71 KB prompt for SAIR's equational-theories competition (Tao + Davis, follow-up to Honda-Murakami-Zhang 2025). Replace free-form LLM reasoning with a 9-magma Birkhoff-sound decision procedure. A 31B model running this prompt beat a 120B one on the hardest set.
tags: prompting llm-reasoning benchmarks equational-logic
categories: nlp
featured: true
chart:
  plotly: true
toc:
  sidebar: left
---

[SAIR's Equational Theories competition (Stage 1)](https://competition.sair.foundation/competitions/mathematics-distillation-challenge-equational-theories-stage1/overview) — organized by Damek Davis (UPenn), Terence Tao (UCLA), and the SAIR Foundation — gives you a pair of equations over a single binary operator `*`:

```
E1:  L1 = R1
E2:  L2 = R2
```

All variables are universally quantified. A *magma* is just a set with a binary operation — no axioms beyond closure. The question: does every magma that satisfies `E1` necessarily satisfy `E2`? Output `true` or `false`.

The dataset is drawn from Tao's [Equational Theories Project](https://teorth.github.io/equational_theories/implications/): 4694 magma laws, giving `4694 × 4693 = 22,028,942` ordered implications. The Stage 1 public subsets are `normal` (1000 problems, 50/50 true/false split), `hard1` (69), `hard2` (200, 50/50), `hard3` (400, 195/205) and an `order5` research subset.

The setup is a follow-up to Honda, Murakami & Zhang (2025), *Distilling Many-Shot In-Context Learning into a Cheat Sheet*: instead of having one model write the cheatsheet, SAIR runs an open competition so the cheatsheet is *discovered* across submissions. You submit a single Markdown file — a prompt template the harness fills in with the two equations and sends to a fixed set of frozen models (GPT-OSS 120B, Llama 3.3 70B, Gemma 4 31B). No fine-tuning, no agents, no tool calls, no chain-of-thought tax beyond what fits in the 8192-token completion budget. Hard cap on cheatsheet size: 10 KB. Scoring on Stage 1 is correctness (accuracy and F1) only — no proof artefacts, no calibrated probabilities. Those come in Stage 2 (Lean proofs, counterexamples, calibration).

The cheatsheet I submitted is **8.71 KB** — under the cap with room to spare. It replaces free-form reasoning with a 9-magma closed-form decision procedure. The headline result: Gemma 4 31B running this prompt beat GPT-OSS 120B by 16 accuracy points on the hardest band.

## The math: Birkhoff completeness

By Birkhoff's theorem, `E1 ⊨ E2` (E1 semantically entails E2) iff E2 is derivable from E1 by equational logic — reflexivity, symmetry, transitivity, congruence, substitution. Equivalently, `E1 ⊨ E2` iff there is *no* magma satisfying E1 but not E2.

This gives you exactly two sound moves:

- **Return `false`**: exhibit a specific magma where E1 holds and E2 fails.
- **Return `true`**: derive E2 from E1 (or argue no counterexample exists).

Anything else is not a proof. In particular, "the equations look similar / share variables / I don't see a derivation" is not sound. This is where LLMs get into trouble.

## Why free-form prompting struggles

Ask an LLM "does E1 imply E2?" and it will produce English reasoning that *looks* like a proof. Sometimes it is one. Often it is structural pattern-matching that happens to be wrong on subtle cases: an implication that needs you to *construct* a specific failing magma, or a non-implication where the equation pair shares enough surface structure that the model is fooled.

The fix is to remove the freedom. Instead of asking the model to reason, give it a *closed-form procedure* over a finite catalog of magmas, and a hard rule that the only way to return `false` is to point to a specific catalog entry that refutes.

## The 9 magmas

A magma is `(M, *)`. For each one, the cheatsheet supplies a *closed-form predicate* on the equation tree that decides whether the magma satisfies a given equation. No enumeration over `M`, no search — a direct formula.

| # | Magma `a*b` | Satisfies `L = R` iff |
|---|---|---|
| 0 | `a*b = b` (right-projection) | `rm(L) == rm(R)` |
| 1 | `a*b = a` (left-projection)  | `lm(L) == lm(R)` |
| 2 | `a*b = c` (constant)         | both depths ≥ 1, or `L`, `R` same bare var |
| 3 | `a*b = a + b` on ℤ/2 (XOR)   | `count(v, L) ≡ count(v, R)` mod 2 ∀v |
| 4 | `a*b = a + b` on ℤ/3         | `count(v, L) ≡ count(v, R)` mod 3 ∀v |
| 5 | `a*b = a + b` on ℤ           | `count(v, L) == count(v, R)` ∀v |
| 6 | `a*b = b + 1` on ℤ/3 (right-successor) | `rm(L) == rm(R)` and `drm(L) ≡ drm(R)` mod 3 |
| 7 | `a*b = a + 1` on ℤ/3 (left-successor)  | `lm(L) == lm(R)` and `dlm(L) ≡ dlm(R)` mod 3 |
| 8 | `a*b = −a − b` on ℤ/3 (negation-sum)   | `signed_count(v, L) ≡ signed_count(v, R)` mod 3 ∀v |

Where the tree primitives are:

```
lm(v) = v,        lm(a*b) = lm(a)               # leftmost leaf
rm(v) = v,        rm(a*b) = rm(b)               # rightmost leaf
dlm(v) = 0,       dlm(a*b) = 1 + dlm(a)         # left-spine length
drm(v) = 0,       drm(a*b) = 1 + drm(b)         # right-spine length
count(v, t) = number of times v appears as a leaf in t
signed_count(v, t) = Σ (-1)^depth(leaf) over leaf occurrences of v
```

Each row of the table is a one-line check the model can run by walking the equation tree once. No symbolic manipulation, no equational rewriting, no induction.

## The decision procedure

For each equation `E`, compute `sig(E) = (h0, h1, …, h8)` — the 9-bit vector of which catalog magmas satisfy E.

```
def implies(E1, E2):
    s1, s2 = sig(E1), sig(E2)
    for i in range(9):
        if s1[i] and not s2[i]:
            return "false"     # magma i refutes
    return "true"
```

If any catalog magma satisfies E1 but falsifies E2, return `false` and the witness is that magma. Otherwise return `true` — the sound default when no catalog magma produces a refutation. Note that this *can* be wrong (the true answer might be `false` via some magma outside the catalog), but it's wrong in the safe direction: the catalog produces no false `false`s.

## Refutation discipline

The hardest part of getting the LLM to be sound is making it stop hallucinating counterexamples. The cheatsheet enforces:

> A refutation must name an index `i ∈ {0..8}` with `sig(E1)[i]=T` and `sig(E2)[i]=F`. Anything else is not a refutation. Do not infer `false` from structural similarity, shared letters, or the absence of an obvious derivation.

And the cheatsheet's last instruction before falling back to `true`:

> If either bit is uncertain after re-check, return `true`. Sound procedure: a hallucinated refutation is worse than a missed one, because a missed refutation at least falls back to the mathematically honest default.

This asymmetric default is the lever. Free-form LLM reasoning falsely says `false` constantly. Constraining false to "name your `i`" cuts those errors almost entirely, at the cost of a few extra false-true answers (which the catalog can't help anyway).

## The mod-3 magmas are where errors happen

Magmas 6, 7, 8 are the ones the cheatsheet spends the most space on, because they're the ones models reliably get wrong. The error pattern: get `lm`/`rm` right, then either skip computing `dlm`/`drm` or compute them wrong, or get the mod-3 arithmetic wrong (signed counts can be negative; `−1 mod 3 = 2`).

The fix is to inline worked examples for each, written so the model *has* to walk the tree. Example for magma 7:

```
Example A (TRUE): x = ((x*y)*z)*w
  lm(L) = x. For R, descend left: R → (x*y)*z → x*y → x. lm(R) = x. Match.
  dlm(L) = 0. dlm(R) counts those 3 left-descents = 3.
  0 mod 3 = 0, 3 mod 3 = 0. Equal. h7 = TRUE.

Example B (FALSE): x = (x*y)*z
  lm(L) = lm(R) = x. Match.
  dlm(L) = 0, dlm(R) = 2.
  0 mod 3 = 0, 2 mod 3 = 2. Differ. h7 = FALSE.
```

The example doesn't teach the model what `dlm` is. It teaches it that *it has to walk the tree* before answering, by being a worked instance with the walk shown.

Magma 8 (signed counts mod 3) gets the most defensive treatment, because depth-parity arithmetic with negative residues is the most error-prone single check in the whole procedure.

## The mandatory PARSE step

Before any rule fires, the cheatsheet requires the model to *explicitly produce* `lm`, `rm`, `dlm`, `drm`, depth, and the per-variable counts for both equation sides. Skipping this step is the single largest source of wrong answers — models guess `lm(R)` instead of descending the tree.

Forcing the structured output reframes the problem. The model isn't reasoning about magma implication anymore; it's filling in a six-slot form, then running nine if-statements, then doing one loop with a hard stop condition. This format is *much* friendlier to the LLM substrate than free-form math.

## Results

The Stage 1 leaderboard has **1,061 participants**. It scores on accuracy and F1 across three difficulty buckets — normal, hard, extra_hard — for each of the three frozen models. Restricted leaderboards (single model, or the order-5 research subset) score the same submission from a different angle.

Cheatsheet size: 8.71 KB. Mean parse success: 100%. Mean per-query cost (across all models, all sets): roughly $0.0004–$0.0009 depending on model.

### Overall leaderboard (all models, all sets) — rank 85

| Model | Set | Accuracy | F1 |
|---|---|---|---|
| GPT-OSS 120B | normal | 77.8% | 81.7% |
| GPT-OSS 120B | hard | 74.0% | 79.3% |
| GPT-OSS 120B | extra_hard | 49.7% | 66.4% |
| Llama 3.3 70B | normal | 61.0% | 63.1% |
| Llama 3.3 70B | hard | 56.5% | 61.6% |
| Llama 3.3 70B | extra_hard | 31.7% | 41.1% |
| Gemma 4 31B | normal | 52.0% | 13.3% |
| Gemma 4 31B | hard | 51.0% | 16.5% |
| Gemma 4 31B | extra_hard | **65.8%** | 48.4% |

Aggregate: **57.7% acc / 52.4% F1 → rank 85**.

```plotly
{"data":[{"x":["normal","hard","extra_hard"],"y":[77.8,74.0,49.7],"name":"GPT-OSS 120B","type":"bar","marker":{"color":"#636efa"},"hovertemplate":"GPT-OSS 120B<br>%{x}: %{y}%<extra></extra>"},{"x":["normal","hard","extra_hard"],"y":[61.0,56.5,31.7],"name":"Llama 3.3 70B","type":"bar","marker":{"color":"#EF553B"},"hovertemplate":"Llama 3.3 70B<br>%{x}: %{y}%<extra></extra>"},{"x":["normal","hard","extra_hard"],"y":[52.0,51.0,65.8],"name":"Gemma 4 31B","type":"bar","marker":{"color":"#00cc96"},"hovertemplate":"Gemma 4 31B<br>%{x}: %{y}%<extra></extra>"}],"layout":{"title":{"text":"Accuracy by model and difficulty"},"barmode":"group","yaxis":{"title":"accuracy (%)","range":[0,100]},"xaxis":{"title":"difficulty"},"height":420,"margin":{"l":60,"r":30,"t":60,"b":50},"legend":{"orientation":"h","x":0.1,"y":-0.15}}}
```

The crossover on the extra_hard band is the result the rest of the post is about. Gemma (the smallest model) lands above GPT-OSS (the largest) for the first time.

### GPT-OSS-only leaderboard — rank 13

Restricting the same submission to the 120B model: **67.2% acc / 75.8% F1**.

### Order-5 research subset (GPT-OSS) — rank 21

Order-5 equations are deeper trees and form a separate research-tier leaderboard. **79.8% acc / 83.2% F1**.

### The Gemma extra-hard anomaly — rank 20

On the extra-hard set specifically, the per-model ranks tell a different story:

| Model | Params | Extra-hard accuracy | Rank on extra-hard |
|---|---|---|---|
| Gemma 4 31B | 31B | 65.8% | **20** |
| GPT-OSS 120B | 120B | 49.7% | 157 |
| Llama 3.3 70B | 70B | 31.7% | 278 |

Gemma 4 31B — the smallest model in the eval — got the *best* extra-hard accuracy with this prompt, by a margin (16 points over GPT-OSS, 34 points over Llama). On a model 4× smaller than GPT-OSS.

The likely explanation: extra-hard problems benefit most from structured procedure-following. GPT-OSS at 120B has more "smart enough to deviate" headroom — it interprets the cheatsheet, decides parts of it are unnecessary, and falls back to free-form reasoning that fails on the hardest cases. Gemma at 31B has less headroom for that kind of agency. It follows the procedure step by step because that's what fits in its working memory, and the procedure is sound. On the easier sets where GPT-OSS's looser interpretation usually gets the right answer anyway, the gap is reversed.

If true, this is a real prompt-engineering result: highly-structured, low-freedom prompts may *prefer* smaller models on the hardest problems, because the largest models will second-guess the structure and lose.

### F1 is low on Gemma — why

Gemma's F1 on normal/hard is low (13.3%, 16.5%) despite reasonable accuracy. This is the asymmetric default at work: Gemma takes the "if uncertain, return true" instruction very literally and answers `true` on most edge cases. Accuracy stays okay because the base rate of `true` in the dataset is high. F1 collapses because Gemma is producing few `false` answers, so the precision/recall on the `false` class is bad. The same instruction that *fixes* GPT-OSS's hallucinated refutations *over-fixes* Gemma's. This is the cost of a single uniform prompt across very different models.

### Sanity check: Qwen 2.5 7B locally

After the submission I ran the same cheatsheet against a *much* smaller model — Qwen 2.5 7B, 4-bit quantized, running locally on a 4 GB GTX 1650 via Ollama — over the first 50 problems of `hard3` (the local dataset roughly aligned with the competition's extra-hard band). Same prompt, same temperature 0, same parse rule (take the last `true`/`false` token).

| Model | Params | Quant | Set | N | Accuracy | F1 (true) | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| Qwen 2.5 7B | 7B | 4-bit | hard3 (first 50) | 50 | 56.0% | 66.7% | 53.8% | 87.5% |
| Gemma 4 31B | 31B | fp16 | extra_hard | 500 | 65.8% | 48.4% | — | — |
| GPT-OSS 120B | 120B | — | extra_hard | 500 | 49.7% | 66.4% | — | — |
| Llama 3.3 70B | 70B | — | extra_hard | 500 | 31.7% | 41.1% | — | — |

Confusion matrix on Qwen: TP=21, TN=7, FP=18, FN=3 — recall on `true` is 87.5%, precision is 53.8%. Same lopsided pattern as Gemma. The model defaults to `true` and is dragged down by the false-positive count on actual-`false` problems.

50 problems is a small slice and the bands aren't a perfect match. Even so: a 7B model at 4-bit, running on a single laptop GPU, comes in *above* GPT-OSS 120B and Llama 3.3 70B on the hardest band. The cheatsheet is doing more of the work than the model is.

A representative failure on Qwen (problem 1 of hard3):

```
E1:  x = x * (x * y)
E2:  x = (x * ((x * x) * x)) * x
actual: true     pred: false
```

The model's trace claims `sig(E1)[2] = T` for the Constant magma. The rule says `depth(L) ≥ 1 ∧ depth(R) ≥ 1` or `L, R same bare var`. Here `L = x` (a leaf — depth 0), so the first conjunct fails; they're not the same bare variable either. `h2` should be `F`, the model wrote `T`, and a non-existent refutation gets reported as a real one.

This is the exact category the cheatsheet warns about: a tree-walking error on the depth primitive, propagating into a wrong bit, propagating into a hallucinated refutation. The fix for a future version is more aggressive worked examples on the Constant rule — the same defensive treatment magmas 6, 7, 8 already get. The PARSE step probably also needs to force the model to emit `depth(L)` and `depth(R)` as explicit lines before any rule fires.

## What I'd change

- **Per-model branches.** The Gemma F1 collapse is fixable: weaken the refutation discipline slightly for smaller models so they produce `false` more often, while keeping it strict for the larger ones. The competition's "one prompt, three models" rule made this not an option for the submission, but it's the obvious next experiment.
- **More magmas.** Extending the catalog past 9 is mostly mechanical — every magma you add comes with its closed-form predicate, plug it into `sig`. The hard part is finding magmas that *actually* refute on the test distribution. An earlier draft had a 4-element bit-swap magma `a*b = 2(a mod 2) + floor(b/2)` (the "C4" magma) which characterized algebraically via slot-pairs; it added measurable coverage on the harder sets in offline testing but I cut it from the submission to keep the cheatsheet at the size where smaller models still parse it reliably.
- **Re-run the PARSE step under model-specific delimiters.** Different model families parse code blocks and bullet lists with slightly different reliability. The PARSE step is the single most important determinant of accuracy on the hard set; getting that step to fire correctly for Gemma vs Llama vs GPT-OSS is worth more than any new magma.

## Apply the procedure to this instance

The submission file ends with the actual placeholder block the harness substitutes into. The full cheatsheet is up at [github.com/debtirthasaha](https://github.com/debtirthasaha).
