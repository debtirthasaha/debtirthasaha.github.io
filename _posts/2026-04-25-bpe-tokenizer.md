---
layout: post
title: "BPE from scratch, and why your LLM can't count L's"
date: 2026-04-25 10:00:00
description: "Byte-pair encoding implemented in pure Python. Plus SolidGoldMagikarp, the encode/decode asymmetry, and a list of LLM weirdness all caused by the tokenizer."
tags: tokenization bpe gpt
categories: nlp
giscus_comments: false
chart:
  plotly: true
toc:
  sidebar: left
---

A byte-pair-encoding tokenizer in pure Python on byte arrays. No NumPy, no neural net, no gradients. The trained tokenizer is two dicts:

```python
merges: {(int, int): int}    # the parameters of the tokenizer
vocab:  {int: bytes}         # derived from merges, used to decode
```

That's the entire model. Then a long second half on what the tokenizer makes weird about LLMs: SolidGoldMagikarp, spelling failures, arithmetic failures, and the encode/decode asymmetry.

## Why tokenization exists

LLMs eat integers, not text. The tokenizer maps strings ↔ integer sequences.

Two obvious approaches, both bad:

- **Unicode code points as tokens.** ~150K possible code points → vocab too large. The Unicode standard also keeps changing — not stable.
- **Raw UTF-8 bytes.** Vocab is a clean 256, but every text becomes 3-4× longer. Attention is `O(T²)`. Long sequences blow up compute and exhaust context length.

BPE: start at 256 raw bytes, iteratively merge the most frequent adjacent pair into a new token. Sequences shrink, vocab grows in a controlled way, you stop whenever you like. Vocab size is now a tunable hyperparameter.

GPT-2 uses ~50K. GPT-4 uses ~100K. Llama 2 uses ~32K.

## UTF-8 in one paragraph

UTF-8 encodes each Unicode code point as 1-4 bytes. ASCII is 1 byte (compatible with the old world). CJK ideographs are 3 bytes. Most emoji are 4. Crucially, not every byte sequence is valid UTF-8 — `b'\x80'` alone is not a legal start byte. This matters when we look at encode/decode round-tripping.

```python
"Hello".encode('utf-8')   # b'Hello'             (5 bytes)
"안".encode('utf-8')       # b'\xec\x95\x88'      (3 bytes)
"🌊".encode('utf-8')       # b'\xf0\x9f\x8c\x8a'  (4 bytes)
```

GPT-2, GPT-4, and Llama all run BPE on UTF-8 bytes (byte-level BPE). Sentencepiece runs BPE on code points and falls back to bytes only for rare ones — clunkier but you'll meet it in Llama and Mistral.

## The BPE algorithm

Given a sequence of token IDs:

1. Count adjacent pairs.
2. Pick the most frequent pair.
3. Mint a new token ID (256, 257, 258, ...).
4. Replace every occurrence of the pair with the new ID.
5. Record the merge.

Repeat N times. The dict of recorded merges *is* the trained tokenizer.

```
Start (vocab=4):  a a b d a a b d a a b              (length 11)
Most freq: (a,a) → mint Z
Round 1:          Z b d Z b d Z b                    (length 9, vocab 5)
Most freq: (Z,b) → mint Y       ← Z is brand new but already mergeable
Round 2:          Y d Y d Y                          (length 5, vocab 6)
```

Token 256 can participate in the round-2 merge that creates token 257. BPE is hierarchical — merges form a forest where new merges build on top of old ones. This hierarchy is *load-bearing* and shows up again when we build `vocab` from `merges` and again when we encode.

## Implementation

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
```

`zip(ids, ids[1:])` is the idiomatic way to iterate adjacent pairs. The `i < len(ids) - 1` check must come first — otherwise `ids[i+1]` on the last element raises `IndexError`. Python's `and` short-circuits, so the bounds check before the comparison is the fix.

Training is the loop:

```python
tokens = list(text.encode('utf-8'))   # ints in [0, 255]
ids    = list(tokens)
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair  = max(stats, key=stats.get)
    idx   = 256 + i
    ids   = merge(ids, pair, idx)
    merges[pair] = idx
```

`list(text.encode('utf-8'))` because iterating a `bytes` object yields ints in Python 3, so `list(bytes_obj)` is a flat list of ints in `[0, 255]`. We keep `tokens` untouched for the compression-ratio report and mutate `ids`.

20 merges on 20868 bytes of text: down to 16154 tokens, **1.29× compression**. The first pair selected is `(101, 32) = 'e '` — words ending in `e` followed by a space.

```plotly
{"data":[{"x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"y":[20868,20276,19859,19465,19148,18865,18594,18326,18092,17873,17671,17469,17300,17136,16975,16826,16689,16552,16417,16285,16154],"type":"scatter","mode":"lines+markers","line":{"color":"#EF553B","width":2},"marker":{"size":7},"text":["start","'e '","'in'","'s '","'th'","'er'","'t '","'co'","', '","'an'","'d '","'or'","'ar'","'en'","'<g> on '","'<d> on '","'y '","'al'","'on'","'<256><256> '","'ac'"],"hovertemplate":"merge %{x}<br>pair: %{text}<br>tokens: %{y}<extra></extra>","name":"sequence length"}],"layout":{"title":{"text":"Sequence length over 20 BPE merges (20,868 bytes -> 16,154 tokens)"},"xaxis":{"title":"merge step","dtick":2},"yaxis":{"title":"tokens","range":[15500,21500]},"height":420,"margin":{"l":70,"r":30,"t":60,"b":50},"showlegend":false}}
```

The biggest drops come early — `'e '` alone saves 592 tokens. Returns diminish: by merge 20, each new pair removes ~130 tokens. The dominant merges are mostly English space-suffix bigrams (`'e '`, `'s '`, `'t '`, `'d '`, `'y '`, `', '`) and high-frequency root pairs (`'in'`, `'th'`, `'er'`, `'an'`, `'or'`, `'ar'`, `'en'`, `'al'`, `'on'`, `'ac'`). BPE rediscovers the morphological structure of English suffixes and roots from byte frequencies alone.

## Building `vocab` from `merges`

Decode needs to know what each token ID is in bytes. Derive it:

```python
def _build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab
```

The insertion-order requirement is real. Token 258 might be `vocab[256] + vocab[257]` — both must already exist when we look them up. Python 3.7+ guarantees `dict.items()` iterates in insertion order. In Python ≤3.6 this code silently produces wrong vocab.

## Decode

```python
def decode(ids, vocab):
    raw_bytes = b"".join(vocab[i] for i in ids)
    return raw_bytes.decode('utf-8', errors='replace')
```

`errors='replace'` because not every byte sequence is valid UTF-8. If the LLM emits a sequence of token IDs whose concatenated bytes don't form a valid UTF-8 string, `errors='strict'` raises `UnicodeDecodeError` and the inference call crashes. `'replace'` substitutes U+FFFD (the `?`-in-a-diamond character) and keeps going. OpenAI's released code does the same.

## Encode

The trick: apply merges *in the same order they were created during training*. Get this wrong and you produce a different token sequence than the trained vocabulary expects.

```python
def encode(text, merges):
    tokens = list(text.encode('utf-8'))
    if len(tokens) < 2:
        return tokens

    while True:
        stats = get_stats(tokens)
        pair  = min(stats, key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break
        tokens = merge(tokens, pair, merges[pair])
    return tokens
```

Three subtleties:

- **`min`, not `max`.** Training picked the most *frequent* pair. Encoding picks the lowest-rank merge (the one created earliest). Why: later merges depend on earlier ones existing as their building blocks. If the text contains a pair that became token 258 = `(256, 257)`, then tokens 256 and 257 must merge first. Always do the earliest available merge.
- **`float('inf')` as fallback.** Pairs not in `merges` get rank infinity. `min` never picks them. The loop terminates when every remaining pair has rank infinity.
- **`len(tokens) < 2` guard.** Empty or single-char strings give empty `stats` and `min({})` raises `ValueError`.

## The encode(decode(x)) asymmetry

```python
decode(encode("hello world")) == "hello world"   # always
encode(decode([128]))         == [128]            # NOT guaranteed
```

`decode([128])`: byte 128 alone is `b'\x80'`, an invalid UTF-8 start byte. With `errors='replace'`, decode returns the replacement character. Re-encoding the replacement character gives different bytes than `[128]`.

Forward (text → tokens → text) is always lossless. Reverse may not be. If your code ever relies on `encode(decode(x)) == x`, it has a latent bug.

## GPT-2 regex pre-splitting

Plain BPE happily merges across word and punctuation boundaries: `dog.`, `dog!`, `dog?`, `dog,` end up as separate tokens. GPT-2 prevents this by *forcing* a split before BPE runs, using a regex:

```python
import regex as re   # pip install regex — NOT the stdlib re

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

re.findall(GPT2_PATTERN, "Hello world! I'm fine.")
# ['Hello', ' world', '!', " I", "'m", ' fine', '.']
```

Pattern parts:

- `'s|'t|'re|'ve|'m|'ll|'d` — English contraction suffixes
- ` ?\p{L}+` — optional space + Unicode letters (so " world" is one chunk)
- ` ?\p{N}+` — optional space + Unicode numbers
- ` ?[^\s\p{L}\p{N}]+` — optional space + punctuation
- `\s+(?!\S)` and `\s+` — whitespace runs

BPE runs *per chunk* and IDs are concatenated. Two known GPT-2 bugs fixed in GPT-4: only ASCII apostrophe (curly quotes break), and not case-insensitive (`DON'T` doesn't split). GPT-4 uses `(?i:...)` and properly handles Unicode apostrophes.

GPT-4 also caps numbers at 3 digits per chunk. With arbitrary-length number tokens, `12345` might be 1 token but `12346` might be `[12, 346]` — totally inconsistent splits that wreck digit-position arithmetic. The 3-digit cap forces predictable behavior.

## Whitespace in code: a GPT-2 disaster, a GPT-4 fix

GPT-2 makes every space its own token. Four spaces of Python indent = 4 tokens. GPT-4 merges runs of whitespace into single tokens, roughly doubling effective context for indented code. This is one of the largest single improvements in GPT-4's tokenizer.

## Special tokens

Outside BPE entirely. Matched by string before BPE runs, then assigned a hardcoded ID.

```
GPT-2 vocab layout:
  0..255         raw byte tokens
  256..50255     50,000 BPE merges
  50256          <|endoftext|>           ← only special token
```

`<|endoftext|>` marks document boundaries during training:

```
doc1_tokens + [50256] + doc2_tokens + [50256] + doc3_tokens + ...
```

The model *learns* what 50256 means. It's special only in that the tokenizer never produces it from regular text — only when explicitly requested.

GPT-4 adds FIM tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`) for code completion and `<|im_start|>`/`<|im_end|>` for chat boundaries.

Adding a special token to a pretrained model is surgery: resize the embedding table by N rows, resize the LM head by N columns, typically freeze the base and train only the new slices.

Security: `tiktoken.encode(user_input, allowed_special="all")` lets users inject `<|endoftext|>` and confuse boundary logic. Default is strict — opt in only when you know the input is trusted.

## Vocab size: the only two places it shows up

`vocab_size` appears in exactly two places in the model:

1. Token embedding: `nn.Embedding(vocab_size, n_embd)`
2. LM head: `nn.Linear(n_embd, vocab_size)`

Everything else (attention, MLP, LayerNorm) is independent of vocab size.

|  | Small (256) | Large (1M) |
|---|---|---|
| Sequence length | very long | short |
| Embed / head size | tiny | huge |
| Per-token signal | strong (every token seen often) | weak (rare tokens undertrained) |

Sweet spot is **32k–100k**.

## SolidGoldMagikarp

The famous tokenization bug. Mechanism:

1. OpenAI trained the **tokenizer** on a dataset that included Reddit.
2. User `SolidGoldMagikarp` posted enough that BPE merged the username into a single token.
3. OpenAI then trained the **language model** on a different, filtered dataset that didn't include those Reddit posts.
4. The token exists in vocab but its embedding row was never updated by gradient descent. It's still the random initialization.
5. At inference, typing `SolidGoldMagikarp` loads that random embedding into the model. Undefined behavior.

Observed: the model evades, hallucinates, insults the user, bypasses safety, or gets stuck looping.

The C analogy: reading uninitialized memory. The slot exists, but no one ever wrote a meaningful value to it.

Root cause is the dataset mismatch between tokenizer training and LM training. Prevention is to use the same (or strictly overlapping) datasets, or audit per-token activation counts after pretraining and remove zero-activation tokens.

Other examples: `" davidjl"`, `" TheNitromeFan"`, `" RandomRedditorWithNo"`. All Reddit usernames or fragments.

## Spelling, reversal, arithmetic — all tokenization

**Why GPT can't count the L's in `DefaultCellStyle`.** `DefaultCellStyle` is a single token in GPT-4's vocab. The model sees one opaque ID, not the constituent letters. Asking how many L's are inside is like asking how many L's are in the integer `28139`.

Workaround for character-level tasks: prompt the model to first split with spaces (`D e f a u l t C e l l S t y l e`) so each character becomes its own token.

**Why arithmetic is brittle.** Number tokenization in GPT-2 is essentially arbitrary. `1024` might be one token, `123456` might be `[12, 3456]`. Digit position is unaligned across examples. Carrying digits requires aligning positions, which is structurally impossible when chunks are random. GPT-4's 3-digit cap helps. Llama uses `split_digits=True` (one digit per token).

**Why non-English is undertrained.** Korean text takes ~3× more tokens than English for the same content. Less training signal per concept, less effective context.

**Trailing whitespace warning.** In training, `" the"` is a single token (space + "the"). If you end a prompt with a bare space, the last token is a lone space — almost never seen during training. The model is now out-of-distribution. Don't end prompts with spaces.

**Partial token glitches.** Completing `"DefaultCellSty"` (a partial token) can produce immediate end-of-text, garbage, or content-policy warnings, because that exact subsequence rarely appears in training. tiktoken's source has an entire "unstable tokens" module for this case.

## Token economy

Same data, different format, different token count. JSON has overhead from `{`, `}`, `"`, `:`, commas. YAML strips most of it.

```python
'{"name": "Alice", "age": 30}'    # ~12 tokens
'name: Alice\nage: 30'             # ~8 tokens
```

~15% savings going from JSON to YAML for the same content. Multiply across an API bill or a long-context model and it matters.

[tiktokenizer.vercel.app](https://tiktokenizer.vercel.app) shows you how anything tokenizes.

## Bugs to remember

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | `merge()` no bounds check | `IndexError` on last element | check `i < len(ids) - 1` first |
| 2 | `decode()` with `errors='strict'` | crashes on invalid byte sequences | `errors='replace'` |
| 3 | `encode()` on empty/single-char | `min({})` raises `ValueError` | early return for `len < 2` |
| 4 | Wrong-order vocab build | Python ≤3.6 vocab is silently wrong | require 3.7+ |
| 5 | GPT-2 ASCII-only apostrophe | curly-quote contractions break | use GPT-4 pattern |
| 6 | GPT-2 case-sensitive split | `DON'T` doesn't split | use GPT-4 pattern with `(?i:...)` |
| 7 | Special tokens in user input | boundary confusion / jailbreak | restrict `allowed_special` |
| 8 | Trailing whitespace in prompt | OOD final token | don't end with space |
| 9 | Tokenizer dataset ≠ LM dataset | SolidGoldMagikarp glitch tokens | same datasets, audit activations |
| 10 | `encode(decode(x)) == x` assumption | silent breakage | only `decode(encode(x))` is safe |

Code: [github.com/debtirthasaha](https://github.com/debtirthasaha). Reference: [karpathy/minbpe](https://github.com/karpathy/minbpe).
