---
layout: post
title: "Tiny Shakespeare, tiny GPT"
date: 2026-04-15 10:00:00
description: A 1.83M-parameter decoder-only transformer trained on 1MB of Shakespeare. Architecture is identical to GPT-2, just smaller.
tags: transformer attention gpt
categories: deep-learning
giscus_comments: false
chart:
  plotly: true
toc:
  sidebar: left
---

Same architecture as GPT-2, scaled to fit a 4 GB GPU and trained on 1 MB of Shakespeare. Built one mechanism at a time, measuring val loss after each addition.

| After adding | Val loss |
|---|---|
| Bigram baseline | 2.88 |
| + single-head self-attention | 2.41 |
| + multi-head | 2.32 |
| + feed-forward MLP | 2.23 |
| + residual + LayerNorm (3 blocks) | 2.09 |
| Scaled up (4 layers, 192-d, 6 heads) | **1.59** |

```plotly
{"data":[{"x":["bigram","+ single-head attn","+ multi-head","+ feed-forward","+ residual + LN (x3)","scaled up"],"y":[2.88,2.41,2.32,2.23,2.09,1.59],"type":"bar","marker":{"color":["#bbbbbb","#9b9bff","#7a7aff","#5959ff","#3838ff","#EF553B"]},"text":[2.88,2.41,2.32,2.23,2.09,1.59],"textposition":"outside","hovertemplate":"%{x}<br>val loss %{y}<extra></extra>"}],"layout":{"title":{"text":"Validation loss after each architectural addition"},"yaxis":{"title":"val loss","range":[0,3.2]},"xaxis":{"tickangle":-25},"height":420,"margin":{"l":60,"r":30,"t":60,"b":120},"showlegend":false}}
```

1.83M parameters, val loss 1.59. Output has speaker tags, sentence rhythm, and (mostly) closed quotes.

## Data, in one tensor

Dataset is `input.txt` — every Shakespeare play, 1,115,394 characters, 65 unique characters including `\n`.

```python
chars = sorted(list(set(text)))
stoi  = {ch: i for i, ch in enumerate(chars)}
itos  = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]   # 1,003,854 tokens
val_data   = data[n:]   #   111,540 tokens
```

That's the tokenizer. (A real BPE tokenizer comes in the [next post]({% post_url 2026-04-25-bpe-tokenizer %}).)

## Sampling chunks, not whole documents

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i    : i + block_size    ] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
```

`y` is `x` shifted right by one. Each position in the chunk is one training example: the model predicts position `t+1` given positions `0..t`. A chunk of length 8 produces 8 examples in parallel. Stacking `batch_size` chunks gives independent training signal — sequences in a batch don't communicate.

## The mathematical trick

Before self-attention there's one observation worth dwelling on: how do you let each position aggregate information from all previous positions in *parallel*?

Goal: `xbow[b, t] = mean(x[b, 0..t])`. Naive double loop. The fast version is a matrix multiply.

Build a lower-triangular matrix of equal weights:

```
wei = [[1.00, 0.00, 0.00, 0.00],
       [0.50, 0.50, 0.00, 0.00],
       [0.33, 0.33, 0.33, 0.00],
       [0.25, 0.25, 0.25, 0.25]]
```

Now `wei @ x` produces, for each row of `wei`, a weighted sum over the value vectors. Row `t` only mixes positions `0..t`. Same answer as the loop, one matmul.

Self-attention will turn `wei` from "equal weights" into "weights computed from the data." The matrix-multiply-with-causal-mask structure stays.

## Self-attention, single head

Three linear projections from the residual stream to a smaller `head_size`:

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        return wei @ v
```

Three details worth pointing at:

- `1/sqrt(head_size)` scaling. Without it, large `head_size` produces dot products with large variance. Softmax collapses to near-one-hot and gradients stop flowing. Scaling holds the softmax diffuse at init.
- `register_buffer` for `tril`. The mask is a constant; we don't want it tracked as a learnable parameter, but we *do* want it to move to GPU when `model.to(device)` is called.
- Mask is `−inf` on upper triangle. After softmax, `exp(−inf) = 0`. Future positions get exactly zero weight.

Single head added on top of the bigram baseline drops val loss 2.88 → 2.41.

## Multi-head attention

Run `n_head` heads in parallel, concatenate, then project back to `n_embd`:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)
```

`head_size = n_embd // n_head`, so concatenation gets you back to `n_embd`. The final `proj` is what lets heads interact — concatenation alone just glues outputs together.

`nn.ModuleList` instead of a plain Python list: PyTorch needs to know about these submodules to register their parameters. A plain list is invisible to `model.parameters()`.

Val loss: 2.32.

## Feed-forward: per-token computation

After attention mixes information across positions, the FFN lets each position *think* about what it gathered. Same MLP applied to every token independently:

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd),
)
```

4× expansion is from the original Transformer paper. Val loss: 2.23.

A useful mental model from this point on: **attention is communication, feed-forward is computation.** Tokens talk to each other in attention, then each one updates its own representation in the FFN.

## The block: pre-norm + residual

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

Three things going on:

- **Residual `x + ...`**: the addition creates a gradient highway. Gradients flow back through `+` undisturbed to earlier layers. Without it, deep networks have vanishing gradients.
- **LayerNorm before the sub-block (pre-norm)**: the original 2017 paper put LN *after*; modern practice puts it before. Pre-norm trains more stably at depth.
- **Sub-blocks start near-zero in output**: at init, attention and FFN contribute tiny perturbations to the residual stream. Useful signal accumulates gradually.

Val loss after stacking 3 such blocks: 2.09.

## Scaling up

```python
batch_size    = 64
block_size    = 128
n_embd        = 192
n_head        = 6      # head_size = 192/6 = 32
n_layer       = 4
dropout       = 0.2
```

1.83M parameters. Val loss **1.59**. The drop from 2.09 → 1.59 is mostly capacity — same architecture, just more of it.

Sample output:

```
DUKE VINCENTIO:
Whither dost thou pursue, and what shall be done
To these things that he was forc'd to make?

ANGELO:
My lord, I will entreat your grace's hand.
```

The model is making it up character by character. There is no concept of a word in its vocabulary. It learned word boundaries, speaker tags, and sentence structure from 1 MB of text.

## What's missing vs GPT-2

This is GPT-2's architecture at small scale. The pieces *not* in this build:

- A real tokenizer (we used per-character; GPT-2 uses BPE — [post]({% post_url 2026-04-25-bpe-tokenizer %})).
- Weight tying between `wte` and `lm_head` (the input embedding and output classifier share the same matrix in GPT-2 — `lm_head.weight = wte.weight`).
- Initialization variants (GPT-2 scales the output projection of each residual layer by `1/sqrt(2*n_layer)` to control variance through depth).
- A proper optimizer recipe (cosine LR schedule, weight decay split, warmup) and DDP for multi-GPU.

All of those show up in the [GPT-2 reproduction]({% post_url 2026-05-17-gpt2-124m %}).

Code: [github.com/debtirthasaha](https://github.com/debtirthasaha).
