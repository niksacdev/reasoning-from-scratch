# Use LFM3 with Hugging Face

Use Python 3.11 or newer. Install `transformers>=5.23.0` and `accelerate` in addition to this package. All examples use the LFM3 base architecture and tokenizer. The `reasoning` option selects chat prompting, using the same base weights.

See the [repository training guide](../../LFM3.md) for export, inference and training instructions. The export approach writes official LFM3 safetensors and copies the pinned modeling code. The local wrapper uses the repository's PyTorch model directly and preserves padding masks during generation and training.
