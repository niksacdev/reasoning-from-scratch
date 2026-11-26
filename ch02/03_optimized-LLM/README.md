# LFM3 generation performance

LFM3 uses PyTorch scaled dot product attention, with gated short convolutions and separate sliding/global attention layers. Its cache holds both convolution windows and attention keys/values.

```bash
python ch02/03_optimized-LLM/compare_inference.py --cache
python ch02/03_optimized-LLM/compare_inference.py --cache --compile
python ch02/03_optimized-LLM/compare_inference.py --compare --max_new_tokens 32
```

The legacy `--optimized` flag selects the same LFM3 implementation. Use `--reasoning` to change prompting on the base weights. Measure throughput and memory on your hardware; original Qwen timings do not apply.

The default comparison budget is 64 tokens. Timings use complete warmups, accelerator synchronization, and fixed token counts, excluding token decoding and console output. `--compare` reports medians for both paths and whether their greedy output tokens match. CPU/MPS default to FP32; the observed MPS BF16 rounding difference and measured speedups are documented in [the validation report](../../LFM3_VALIDATION.md).
