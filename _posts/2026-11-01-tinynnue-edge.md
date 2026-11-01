---
layout: post
title: "An int8 NNUE with exact incremental updates, and why it is not faster than fp32"
date: 2026-11-01 20:00:00
description: "A 769-128-32-1 chess evaluator trained on Stockfish labels, quantized to int8 weights and int32 accumulators, run by a small C++ core with real integer arithmetic. Incremental updates match full recomputation bit for bit; the apparent 7x int8 speedup disappears once fp32 sums may be reassociated; almost all quantization error comes from the first layer."
tags: quantization int8 nnue inference cpp
categories: deep-learning
chart:
  plotly: true
toc:
  sidebar: left
---

NNUE ("efficiently updatable neural network") is the evaluation network inside Stockfish. Its trick is that the first layer's input is a sparse set of binary features (which piece is on which square), so after a move only 2–4 features change, and the first-layer output can be updated by subtracting and adding a few weight rows instead of being recomputed. Stockfish runs this in integer arithmetic.

This post builds a tiny version end to end: 769 → 128 → 32 → 1, trained on 121k Stockfish-labeled positions, post-training-quantized to int8 weights / int32 accumulators, and evaluated by a ~340-line C++ core. It is an integer-inference case study with chess as the test bed, not an engine (there is no search). Code: [github.com/debtirthasaha/tinynnue-edge](https://github.com/debtirthasaha/tinynnue-edge). Weights: [MR0b0t/tinynnue-edge](https://huggingface.co/MR0b0t/tinynnue-edge). Data: [MR0b0t/tinynnue-stockfish-labels](https://huggingface.co/datasets/MR0b0t/tinynnue-stockfish-labels).

Three results up front. **Integer incremental updates are exact**: across 18,342 make/unmake steps, the incrementally updated accumulator matched full recomputation on every element, and the C++ output matched the Python reference on every step. **int8 is not faster than fp32 on this CPU**: under strict float semantics int8 incremental looks 7.2x faster than fp32 full recomputation, but with `-ffast-math` fp32 incremental runs in 192 ns against 472 ns for int8. The gap came from what the compiler was allowed to vectorize, not from the narrower type. **Quantization error is almost all layer 1**: rounding W1 alone causes 7.15 cp mean error; W2 causes 1.46, W3 0.37.

## Data

Positions come from the Lichess January 2013 database (CC0): up to 3 per game, from ply 10 onward, skipping positions in check. Each is labeled by Stockfish 19 with `Threads=1`, a fixed budget of 10,000 nodes, and the hash cleared before every position, so the same binary reproduces the same labels.

Two filters matter. Mate scores have no finite centipawn value, so they are dropped. Positions whose best move is a capture or promotion are also dropped: the label there reflects a tactic found by search, which a static evaluator looking at piece placement cannot see. Of 170,559 candidates, 121,187 survive.

The split is by game, hashing the Lichess game id 90/5/5. Positions from one game are strongly correlated, so a position-level split would leak.

Scores are stored from White's point of view. The network has a side-to-move input but no board flip, so a single fixed perspective keeps the target consistent.

## Features and network

```python
def feature_index(square: int, piece: chess.Piece) -> int:
    plane = (0 if piece.color == chess.WHITE else 6) + piece.piece_type - 1
    return plane * 64 + square
```

12 planes × 64 squares + 1 side-to-move bit = 769 inputs, at most 33 active (32 pieces + the bit). In PyTorch the first layer is an `EmbeddingBag` in sum mode: summing rows of a [769, 128] table is exactly `W1 @ x` for binary `x`, and it is also the layout the C++ core uses (one contiguous row per feature).

```python
h1 = torch.clamp(self.l1(idx) + self.b1, 0.0, 1.0)  # clipped ReLU
h2 = torch.clamp(self.l2(h1), 0.0, 1.0)
return self.l3(h2).squeeze(-1)                       # in units of 400 cp
```

The loss is MSE in win-probability space, `(sigmoid(pred/400) - sigmoid(cp/400))^2`. +900 vs +1200 barely matters (both are winning); 0 vs +300 matters a lot. Adam, 40 epochs, about a minute on a laptop CPU: the network has 102,721 parameters, and each forward pass touches ~23 rows of W1 plus a 128×32 matmul.

| model (6,072 test positions) | WDL-space MSE | cp MAE (\|label\| ≤ 1000) |
|---|---|---|
| always 0 | 0.05367 | 323.1 |
| material count | 0.01737 | 195.2 |
| TinyNNUE fp32 | 0.01086 | 146.9 |
| TinyNNUE int8 | 0.01089 | 147.1 |

The network beats material counting by a wide margin and is still a weak evaluator. It also overfits: train WDL-MSE ends at 0.0049 against 0.0111 on validation, with validation flat after about epoch 20.

## Quantization

Symmetric, per-tensor, post-training. Each weight tensor gets `s = max|W| / 127` and `Wq = round(W / s)` in [-127, 127]. Activations need no calibration: the clipped ReLU output is always in [0, 1], so it is represented as an integer `a` in [0, 127] with `h ≈ a / 127`. Every scale in the network is a constant.

```
acc1 = b1q + Σ_{active f} W1q[f]           scale s1          (inputs are 0/1)
a1   = clamp(requant(acc1, M1), 0, 127)    M1 = s1 · 127
acc2 = b2q + W2q · a1                      scale s2 / 127
a2   = clamp(requant(acc2, M2), 0, 127)    M2 = s2
acc3 = b3q + W3q · a2                      scale s3 / 127
cp   = requant(acc3, M3)                   M3 = 400 · s3 / 127
```

Biases are quantized with the accumulator's scale (`b2q = round(b2 · 127 / s2)`), so they are added directly as int32. The real-valued multipliers are applied in fixed point: `M` is stored as `round(m · 2^24)` and applied with a widening multiply and a rounding shift.

```cpp
static inline int32_t requant(int32_t x, int32_t M, int S) {
    return (int32_t)(((int64_t)x * M + (int64_t(1) << (S - 1))) >> S);
}
```

`S` was 16 in the first version. That left `M2` at 50 for an early model, a 0.37% relative error in the multiplier itself. At `S = 24` the worst multiplier error is 7e-6.

### Overflow

Bounds are computed over any possible input, not just the data seen. For acc1 that means the 32 largest-magnitude W1 rows per unit plus the side-to-move row plus the bias:

| accumulator | worst case | observed max (test set) | needs |
|---|---|---|---|
| acc1 | 2,534 | 582 | int16 |
| acc2 | 591,379 | 170,100 | int32 |
| acc3 | 264,318 | 111,367 | int32 |

The loose textbook bound for acc1, 33 × 127 + max bias = 4,338, also fits int16, which is why real NNUE implementations keep the first-layer accumulator in int16 and halve its memory traffic. Here it stays int32 for simplicity.

### Where the error comes from

The int8 output differs from fp32 by 7.1 cp on average (p99 25.4, max 39.2), and the accuracy cost against Stockfish is 0.2 cp of MAE. Simulating one rounding step at a time in float:

| quantized step | mean abs error (cp) | max abs error (cp) |
|---|---|---|
| W1 only | 7.15 | 37.85 |
| W2 only | 1.46 | 6.03 |
| W3 only | 0.37 | 1.17 |
| activations only | 2.35 | 12.13 |
| everything (int8 pipeline) | 7.10 | 39.22 |

Layer 1 dominates for two reasons. Each evaluation sums about 23 independently rounded W1 rows, so their rounding errors accumulate into the pre-activation. And the per-tensor scale is set by an outlier: max |W1| is 0.620 while the 99th percentile is 0.343, so most weights use well under half of the int8 range. Per-channel scales (one per hidden unit, Python-only analysis) cut the mean error to 6.7 cp and the max to 30.5.

## Incremental updates

Python plays the moves with python-chess and diffs feature sets before and after each move. Castling removes and adds two pieces, en passant removes a pawn from a third square, promotion swaps planes, and undo is the reverse diff. None of those need special cases, and the C++ side never sees a chess rule:

```cpp
static void update(const Int8Net& n, const std::vector<int>& removed,
                   const std::vector<int>& added, IntAcc& acc) {
    for (int f : removed) {
        const int8_t* row = &n.W1[f * H1];
        for (int i = 0; i < H1; i++) acc.v[i] -= row[i];
    }
    for (int f : added) {
        const int8_t* row = &n.W1[f * H1];
        for (int i = 0; i < H1; i++) acc.v[i] += row[i];
    }
}
```

The side-to-move bit flips on every move, so every update touches it. A move changes 3.1 features on average.

Because acc1 is an integer sum with position-independent scales, applying deltas in any order gives the same bits as recomputing it. The test traces cover all four castles, en passant for both colors, quiet promotions, capture-promotions, under-promotions, and 300 random games with a 30% undo rate (5,316 undos in total):

| check over 18,342 steps | result |
|---|---|
| int acc1, incremental vs full, any element differs | 0 steps |
| int output, incremental vs full | 0 mismatches |
| C++ int output vs Python int reference | 0 mismatches |
| fp32 incremental vs fp32 full, max drift | 0.00085 cp |

The float path drifts because float addition is not associative. The drift is tiny here, but it is not zero, and in a search that makes and unmakes millions of moves it can only grow. The integer path has no drift by construction.

## Latency

All paths run in the same C++ binary over 20,500 positions (500 test positions × 40 random plies), 20 repetitions, single thread, on an AMD Ryzen 5 4600H under WSL2 with g++ 13.3. One evaluation takes well under a microsecond, close to the clock's own overhead, so time is measured over chunks of 64 consecutive evaluations; median and p95 are over chunks.

```plotly
{
  "data": [
    {"type": "bar", "name": "-O2", "x": ["fp32 full", "fp32 incremental", "int8 full", "int8 incremental"],
     "y": [3934, 3072, 2473, 1345]},
    {"type": "bar", "name": "-O3 -march=native", "x": ["fp32 full", "fp32 incremental", "int8 full", "int8 incremental"],
     "y": [3830, 2850, 1572, 535]},
    {"type": "bar", "name": "-O3 -march=native -ffast-math", "x": ["fp32 full", "fp32 incremental", "int8 full", "int8 incremental"],
     "y": [1115, 192, 1415, 472]}
  ],
  "layout": {"barmode": "group", "yaxis": {"title": "median ns per evaluation"},
             "title": "Ryzen 5 4600H, single thread", "height": 400}
}
```

| path (median / p95 ns) | -O2 | -O3 -march=native | + -ffast-math |
|---|---|---|---|
| fp32 full | 3,934 / 4,685 | 3,830 / 4,815 | 1,115 / 1,439 |
| fp32 incremental | 3,072 / 3,533 | 2,850 / 3,300 | 192 / 264 |
| int8 full | 2,473 / 3,663 | 1,572 / 2,022 | 1,415 / 1,813 |
| int8 incremental | 1,345 / 1,947 | 535 / 685 | 472 / 533 |
| int8 acc1 update only | 198 / 224 | 70 / 90 | 71 / 91 |
| int8 layers 2–3 only | 1,146 / 1,428 | 402 / 504 | 409 / 516 |

The first two columns tell a clean quantization story: int8 incremental is 2.9x faster than fp32 full at `-O2` and 7.2x at `-O3 -march=native`, while fp32 barely moves (3%). The third column breaks it.

The fp32 layer-2 loop is `s += W2[o][i] * h1[i]`, a sum of 128 floats. IEEE addition is not associative, so without permission to reorder, each of the 32 output sums must be accumulated term by term in source order; the compiler cannot split one sum across vector lanes. Integer addition is associative, so the int8 loop vectorizes freely. `-ffast-math` grants the reordering; fp32 then vectorizes too, and fp32 incremental drops from 2,850 ns to 192 ns.

At that point int8 has no width advantage left. The C++ widens each int8 weight to int32 before multiplying, so a 256-bit AVX2 register holds 8 lanes on either path, and int8 additionally pays for the widening and for a 64-bit multiply in every requantization. Compiler output (`-fopt-info-vec`) confirms the shape of the earlier gap: at `-O2` GCC vectorizes only the int8 layer-2 dot product (16-byte vectors) and none of the accumulator row adds; at `-O3 -march=native` every loop uses 32-byte vectors.

Real int8 speedups come from instructions that multiply 8-bit pairs and accumulate into wider sums: `vpmaddubsw` on AVX2 (which Stockfish uses), VNNI, ARM `sdot`, or an accelerator's int8 MAC array. Those need hand-written SIMD or a compiler that targets them, which this project leaves out. On this CPU, int8 delivers what it delivers by construction: a 4x smaller model and drift-free incremental updates.

Once layer 1 is incremental, the time is in layers 2–3 (409 of 472 ns on the int8 path). A faster accumulator update would not help; the dense layer and its requantization are the next target.

## Limitations

- 147 cp mean error against Stockfish. 100k quiet positions and no search; accuracy was not the target.
- No hand-written SIMD, so the latency comparison measures compiler auto-vectorization, not int8 hardware throughput.
- Per-tensor PTQ only in C++. The outlier-driven W1 scale suggests clipping the scale to a percentile, or per-channel scales, as the first improvements.
- One x86 laptop CPU under WSL2, no edge device: a CPU prototype.
