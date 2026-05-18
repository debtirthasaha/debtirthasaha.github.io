---
layout: post
title: "A transformer that reads C++ and writes Python"
date: 2026-03-01 10:00:00
description: 16.4M-parameter encoder-decoder transformer for C++ → Python code translation, trained on XLCoST on a 4 GB GPU. val_loss 2.0474.
tags: transformer translation code
categories: deep-learning
giscus_comments: false
chart:
  plotly: true
toc:
  sidebar: left
---

Encoder-decoder transformer that takes C++ source and emits the equivalent Python. Trained on XLCoST (a corpus of competitive-programming problems with parallel solutions in seven languages). 16.4M parameters. Best checkpoint at epoch 19, val_loss **2.0474**, sized to fit a GTX 1650 4 GB.

Most "transformer from scratch" implementations are English → French. This is C++ → Python, which makes the input distribution different (structured code, lots of repeated tokens, hard syntax) and exposes a tokenization problem the dataset's stock format papers over.

## Problem setup

XLCoST ships parallel source files. Pairs look like:

```cpp
// C++
int binary_search(vector<int>& a, int x) {
    int lo = 0, hi = a.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (a[mid] == x) return mid;
        if (a[mid] < x) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
```

```python
def binary_search(a, x):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == x: return mid
        if a[mid] < x: lo = mid + 1
        else: hi = mid - 1
    return -1
```

An encoder-decoder transformer is the right shape: the encoder reads the entire C++ source bidirectionally, the decoder generates Python autoregressively with cross-attention into the encoder's output. This is the original 2017 architecture.

## Hyperparameters, sized to 4 GB

```
d_model         = 256
N (layers)      = 4
h (heads)       = 8
d_ff            = 512
dropout         = 0.1
label_smoothing = 0.1
max_seq_len     = 350
batch_size      = 8   (with dynamic padding)
```

16.4M parameters. The constraints are real: with `d_model = 512` and full padding to 350 tokens, the VRAM math doesn't close. Every knob got pulled until the model both fit and trained.

VRAM math, roughly: `B × T × T × h × 4 bytes` for attention scores per layer. At `B=8, T=350, h=8`: `8 × 350² × 8 × 4 = ~31 MB` *per layer*, and you need this for forward and backward. Across 8 layers (4 encoder + 4 decoder) and other activations it adds up fast. Dynamic padding (pad each batch to its longest sequence rather than to `max_seq_len`) is what made training viable.

## The dataset problem

XLCoST is pre-tokenized. The files look like:

```
int NEW_LINE binary_search ( vector < int > & a , int x ) { NEW_LINE INDENT ...
```

`NEW_LINE`, `INDENT`, `DEDENT` are XLCoST's whitespace-preserving tokens. Splitting on whitespace gives you tokens directly — no tokenizer required.

This is fine for training and evaluation on XLCoST. It is not fine for inference on real code, because real code doesn't ship with `NEW_LINE` tokens.

So a second tokenizer had to be built: a raw C++ tokenizer that takes ordinary source and produces the XLCoST tokenization. It handles:

- Comments (`//`, `/* */`) — stripped before tokenization
- String literals — preserved as single tokens (don't split inside quoted strings)
- Multi-char operators (`<<`, `>>`, `==`, `!=`, `<=`, `>=`, `&&`, `||`, `++`, `--`, `->`, `::`, `+=`, etc.) — match greedy
- Numbers, identifiers — match maximally
- Whitespace → `NEW_LINE`, `INDENT`, `DEDENT` based on column position

The inference path is `raw C++ → my tokenizer → XLCoST tokens → model → Python tokens → join`.

## Vocabulary coverage and UNKs

Vocab is built from the training set with `min_freq=2` — any token appearing fewer than 2 times is replaced with `<UNK>`. Final vocab is ~12K source tokens, ~10K target tokens.

This means common things work and uncommon things fail. `binary_search` is in vocab. `Hello, World!\n` is not — the string literal `"Hello, World!\n"` is a single rare token, gets mapped to `<UNK>`, and the model has no signal to translate it. You can confirm this by tokenizing `cout << "Hello, World!" << endl;` and watching the string vanish into an UNK.

For competitive-programming-style code (loops, arrays, recursion, math) coverage is good and translation is fluent. For anything string-heavy it falls apart.

## The architecture, 12 components

All twelve are in `model.py`. Quick map:

1. `InputEmbeddings` — `nn.Embedding(vocab_size, d_model)`, output scaled by `sqrt(d_model)`
2. `PositionalEncoding` — sinusoidal, fixed (not learned)
3. `LayerNormalization` — manual implementation with learnable γ, β
4. `FeedForwardBlock` — `Linear(d_model, d_ff) → ReLU → Dropout → Linear(d_ff, d_model)`
5. `MultiHeadAttentionBlock` — Q/K/V projections, scaled-dot-product, output projection. Stores `attention_scores` as a buffer for later visualization.
6. `ResidualConnection` — `x + dropout(sublayer(norm(x)))` (pre-norm)
7. `EncoderBlock` — self-attention + FFN, each wrapped in residual
8. `Encoder` — stack of `N` encoder blocks + final LayerNorm
9. `DecoderBlock` — masked self-attention + cross-attention + FFN
10. `Decoder` — stack of `N` decoder blocks + final LayerNorm
11. `ProjectionLayer` — `Linear(d_model, vocab_size)`, no softmax (cross-entropy applies it internally)
12. `Transformer` — encoder + decoder + source/target embeddings + source/target positional + projection

Pre-norm everywhere. Output of the projection layer is logits, not log-softmax — `nn.CrossEntropyLoss` expects logits and applies log-softmax internally for numerical stability.

The cross-attention in the decoder is where the two halves meet: decoder `Q` comes from the decoder's own residual stream, but `K` and `V` come from the encoder's final output. Each decoder position queries the encoded source to decide what to translate next.

## Training: warmup + label smoothing

```python
optimizer = Adam(params, lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
scheduler = LambdaLR(optimizer,
                     lambda step: d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5))
criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
```

The LR schedule is from the original transformer paper: linear warmup for `warmup_steps`, then `1/sqrt(step)` decay. `ignore_index=PAD` masks padding tokens from the loss. `label_smoothing=0.1` gives 10% of the probability mass to non-target tokens uniformly — softens the optimization target and regularizes.

Greedy decoding for inference. No beam search.

## Results

20 epochs. Train and val loss every epoch:

| Epoch | Train | Val |
|---|---|---|
| 13 | 1.9109 | 2.0615 |
| 15 | 1.8708 | 2.0545 |
| 16 | 1.8542 | 2.0511 |
| 19 | 1.8103 | **2.0474** ← best |
| 20 | 1.7964 | 2.0576 ← train still dropping, val rising |

Best checkpoint at epoch 19. Train kept dropping past that point but val stopped — classic overfitting signature. The save-best-by-val-loss logic kept epoch 19 as `best_model.pt`.

Sample translation:

```cpp
int binary_search(vector<int>& a, int x) {
    int lo = 0, hi = a.size() - 1;
    ...
}
```

```python
def binary_search(a, x):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        ...
```

Hello-world-style code with rare string literals fails: `cout << "Hello"` produces `print(<UNK>)`. Loops, math, recursion, array indexing all translate cleanly.

## What the model actually learned

Inspired by [Anthropic's circuits work](https://transformer-circuits.pub/2026/nla/index.html#introduction), I loaded the checkpoint and probed two things: the token embedding matrices on each side, and the attention pattern of the last encoder layer.

### Embedding nearest neighbors

For a seed token, find the closest tokens in embedding space by cosine similarity. Both the source-side (C++) and target-side (Python) embedding tables show clean semantic structure even though no one taught the model what these tokens *mean*.

C++ side (top 6 per row):

```
int    -> endl, ll, ;, [EOS], long, <<
for    -> while, memset, getline, case, faces, sortRowWise
if     -> ==, 127, while, case, break, fast
vector -> calloc, begin, multiset, sizeof, NthPostordernode, word_size
<      -> >, ::, %, <=, &, #
==     -> case, 127, <=, if, checkAbundant, !=
true   -> Magic, False, ||, True, slope3, npos
string -> char, "4", chanceA, 122, modifyString, findNumberOfLIS
```

Python side (top 6 per row):

```
def    -> class, divTermCount, for, in, NEW_LINE, Euler
if     -> elif(0.70), while, or, and, isPower2, checkPerfectcube
for    -> while, in, def, [, range, within
range  -> in, while, sqrt, ord, int, xrange
+      -> +=, -, >>=, -=, <<=, >
==     -> !=(0.67), >=, >, <=, <, than
print  -> return, PrintList, format, Squares, cout, round
<      -> >(0.65), >=, <=, ==, <, ->
True   -> False, 82, 3.14159265, 0.25, 4.5, None
str    -> acos, log2, trailingZero, int, string, singlePrimeFactor
```

A few clusters that aren't accidents:

- **Comparison operators**. On the Python side, `<` is nearest to `>` (0.65), then `>=` (0.63), `<=` (0.63), `==` (0.53), `<` (0.51), `->` — a tight cluster of every binary comparison the model has seen.
- **Boolean values**. `True` finds `False` and `None`. `true` (on the C++ side) finds `True`, `False`, and `||`. The model puts truth values close together regardless of casing or language.
- **Control flow**. `if` on the Python side has `elif` as its nearest neighbor at cosine 0.70 — by a clear margin. `for` is nearest to `while`. `def` is nearest to `class`.
- **Cross-language synonyms**. `print` (Python) has `cout` in its top-6. The model learned that the C++ side's `cout` and the Python side's `print` play structurally similar roles, even though they live in different vocabularies and different embedding tables.
- **C++ integer family**. `int` is nearest to `long` and `ll` (the `typedef long long ll` shorthand competitive programmers use). The model picked up that these are interchangeable integer types.

None of this is taught explicitly. The supervision signal is a cross-entropy loss on next-token prediction in a sequence-to-sequence setup. Semantically related tokens end up close together because the loss is lower when interchangeable tokens have similar representations.

### Encoder attention on `int sum = a + b ; NEW_LINE return sum ;`

Pulling out head 0 of the last encoder layer's self-attention on one short example. Rows = query positions, columns = key positions. Hover for exact weights:

```plotly
{"data":[{"z":[[0.02,0.00,0.04,0.00,0.00,0.00,0.16,0.57,0.02,0.00,0.14,0.07],[0.00,0.00,0.02,0.00,0.00,0.00,0.09,0.76,0.01,0.00,0.07,0.05],[0.01,0.00,0.06,0.00,0.00,0.00,0.11,0.64,0.03,0.00,0.12,0.02],[0.01,0.00,0.05,0.00,0.00,0.00,0.16,0.57,0.02,0.00,0.16,0.02],[0.01,0.00,0.03,0.00,0.00,0.00,0.10,0.75,0.01,0.00,0.08,0.03],[0.01,0.00,0.06,0.00,0.00,0.00,0.18,0.51,0.03,0.00,0.18,0.03],[0.03,0.00,0.06,0.00,0.00,0.00,0.19,0.41,0.05,0.00,0.16,0.09],[0.07,0.00,0.08,0.00,0.01,0.00,0.22,0.26,0.11,0.00,0.18,0.08],[0.02,0.00,0.04,0.00,0.00,0.00,0.17,0.52,0.03,0.00,0.15,0.08],[0.01,0.00,0.02,0.00,0.00,0.00,0.09,0.75,0.01,0.00,0.08,0.05],[0.03,0.00,0.06,0.00,0.00,0.00,0.18,0.46,0.04,0.00,0.15,0.08],[0.05,0.00,0.05,0.00,0.00,0.00,0.23,0.31,0.06,0.00,0.18,0.11]],"x":["int","sum","=","a","+","b",";","NEW_LINE","return","sum",";","[EOS]"],"y":["int","sum","=","a","+","b",";","NEW_LINE","return","sum",";","[EOS]"],"type":"heatmap","colorscale":"Viridis","hovertemplate":"q: %{y}<br>k: %{x}<br>weight: %{z:.2f}<extra></extra>","colorbar":{"title":{"text":"attn"}}}],"layout":{"title":{"text":"Encoder self-attention, head 0, last layer"},"xaxis":{"title":"key (attended to)","side":"top"},"yaxis":{"title":"query (attending)","autorange":"reversed"},"height":500,"margin":{"l":80,"r":30,"t":90,"b":50}}}
```

In text form:

```
              int   sum    =     a     +     b     ;   NEW_  ret   sum    ;   [EOS]
   int    [ 0.02  0.00  0.04  0.00  0.00  0.00  0.16  0.57  0.02  0.00  0.14  0.07 ]
   sum    [ 0.00  0.00  0.02  0.00  0.00  0.00  0.09  0.76  0.01  0.00  0.07  0.05 ]
     =    [ 0.01  0.00  0.06  0.00  0.00  0.00  0.11  0.64  0.03  0.00  0.12  0.02 ]
     a    [ 0.01  0.00  0.05  0.00  0.00  0.00  0.16  0.57  0.02  0.00  0.16  0.02 ]
     +    [ 0.01  0.00  0.03  0.00  0.00  0.00  0.10  0.75  0.01  0.00  0.08  0.03 ]
     b    [ 0.01  0.00  0.06  0.00  0.00  0.00  0.18  0.51  0.03  0.00  0.18  0.03 ]
     ;    [ 0.03  0.00  0.06  0.00  0.00  0.00  0.19  0.41  0.05  0.00  0.16  0.09 ]
NEW_LI    [ 0.07  0.00  0.08  0.00  0.01  0.00  0.22  0.26  0.11  0.00  0.18  0.08 ]
return    [ 0.02  0.00  0.04  0.00  0.00  0.00  0.17  0.52  0.03  0.00  0.15  0.08 ]
   sum    [ 0.01  0.00  0.02  0.00  0.00  0.00  0.09  0.75  0.01  0.00  0.08  0.05 ]
     ;    [ 0.03  0.00  0.06  0.00  0.00  0.00  0.18  0.46  0.04  0.00  0.15  0.08 ]
 [EOS]    [ 0.05  0.00  0.05  0.00  0.00  0.00  0.23  0.31  0.06  0.00  0.18  0.11 ]
```

The argmax-key per query position:

```
q[ 0] int       -> k[ 7] NEW_LINE  (0.57)
q[ 1] sum       -> k[ 7] NEW_LINE  (0.76)
q[ 2] =         -> k[ 7] NEW_LINE  (0.64)
q[ 3] a         -> k[ 7] NEW_LINE  (0.57)
q[ 4] +         -> k[ 7] NEW_LINE  (0.75)
q[ 5] b         -> k[ 7] NEW_LINE  (0.51)
q[ 6] ;         -> k[ 7] NEW_LINE  (0.41)
q[ 7] NEW_LINE  -> k[ 7] NEW_LINE  (0.26)
q[ 8] return    -> k[ 7] NEW_LINE  (0.52)
q[ 9] sum       -> k[ 7] NEW_LINE  (0.75)
q[10] ;         -> k[ 7] NEW_LINE  (0.46)
q[11] [EOS]     -> k[ 7] NEW_LINE  (0.31)
```

Every single position is attending most heavily to position 7 — the `NEW_LINE` statement boundary. The mass on that one column ranges from 0.26 (the boundary attending to itself) to 0.76 (`sum` and `+` attending to the boundary). Other columns are near-zero almost everywhere.

This head has specialized into something like a *statement-end aggregator*: route information from anywhere in the current statement to the boundary marker that closes it. Cross-attention into the decoder then has a privileged column at `NEW_LINE` that has gathered everything about the C++ statement, and the decoder can read from it to emit the Python equivalent. The reason XLCoST's pre-tokenization scheme produces these explicit boundary tokens is exactly so that the model has somewhere to put statement-level information. Head 0 of the last encoder layer is using them for that.

Other heads in the same layer attend differently (some local, some diagonal, some on operators) — the specialization isn't uniform. But this one's job is clear, and is the kind of mechanistic finding that motivates the circuits-style probing in the first place.

## What would close the gap

- **Subword tokenization** (BPE on the raw source) instead of per-word vocab + UNKs. The whole "rare string literal" failure goes away.
- **Bigger model** if you have the VRAM. `d_model=512` and 6 layers is the standard small-transformer scale, but doesn't fit at `T=350` on 4 GB.
- **Beam search** at decode time. Greedy is fine for code but a beam of 4 reliably picks better completions for long sequences.

Code: [github.com/debtirthasaha](https://github.com/debtirthasaha). The 16 numbered tests in `test_step*.py` build up each of the 12 components in isolation before the full model is assembled.
