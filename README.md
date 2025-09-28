# contrastive-decoding

This repository implements [Contrastive Decoding](https://arxiv.org/abs/2210.15097) with HuggingFace transformers and PyTorch.

Qwen/Qwen2.5-3B-Instruct is used as the large model and Qwen/Qwen2.5-Coder-0.5B-Instruct is used as the small model.

The token-level algorithm, rather than the beam search algorithm, is implemented.