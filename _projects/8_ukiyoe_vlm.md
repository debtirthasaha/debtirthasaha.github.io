---
layout: page
title: Ukiyo-e Haiku VLM (planned)
description: Vision-language model that writes haiku about Japanese ukiyo-e woodblock prints. SigLIP vision encoder + Qwen LLM, LoRA fine-tuned on Met Museum API images.
img:
importance: 2
category: in-progress
giscus_comments: false
---

Pipeline:

- **Vision encoder.** SigLIP, frozen.
- **LLM.** Qwen (small), LoRA-adapted.
- **Bridge.** Projection layer from SigLIP embeddings into Qwen's embedding space.
- **Data.** Ukiyo-e print images from the Met Museum Open Access API, paired with generated haiku captions.

Phase 2 of a VLM-from-scratch effort. Phase 1 (nanoVLM) is a prerequisite build. Budget: ~$15-25.

Writeup will follow when the model trains.
