---
layout: page
title: Math SLM (in progress)
description: A 1.5B math-reasoning model. LoRA SFT of DeepSeek-R1-Distill-Qwen-1.5B on teacher-generated CoT data, BoN-32 inference with code voting. Target benchmarks include MathNet (ICLR 2026 text-only 500).
img:
importance: 1
category: in-progress
giscus_comments: false
---

Pipeline:

- **Base.** DeepSeek-R1-Distill-Qwen-1.5B.
- **Teacher.** DeepSeek-R1-Distill-Qwen-7B for generating chain-of-thought training data.
- **Training.** LoRA SFT on teacher-distilled CoT, single 1× A100 40GB.
- **Inference.** Best-of-N sampling (N=32) with code-execution voting on math problems.
- **Eval.** GSM8K, MATH-500, AIME 2024+25, MathNet text-only 500.

Target: cross 10% on MathNet with a 1.5B model. Frontier 2026 models get 45-78% on the same benchmark, so the small-model gap is the point.

Budget: ~$60 cloud spend. Writeup will follow the same structure as the GPT-2 post.
