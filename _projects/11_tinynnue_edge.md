---
layout: page
title: "TinyNNUE Edge (int8 inference)"
description: "A 769-128-32-1 NNUE-style chess evaluator, post-training-quantized to int8 weights and int32 accumulators and run by a small C++ core with real integer arithmetic. Incremental updates match full recomputation bit for bit; int8 was not faster than fp32 once fp32 sums could vectorize."
img:
importance: 1
category: shipped
giscus_comments: false
---

[Read the writeup]({% post_url 2026-11-01-tinynnue-edge %}) · [Code](https://github.com/debtirthasaha/tinynnue-edge) · [Weights](https://huggingface.co/MR0b0t/tinynnue-edge) · [Data](https://huggingface.co/datasets/MR0b0t/tinynnue-stockfish-labels).
