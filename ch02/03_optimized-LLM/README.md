# LFM3 generation performance

LFM3 uses PyTorch scaled dot product attention, with gated short convolutions and separate sliding/global attention layers. Its cache holds both convolution windows and attention keys/values.

```bash
python ch02/03_optimized-LLM/compare_inference.py --cache
python ch02/03_optimized-LLM/compare_inference.py --cache --compile
```

The legacy `--optimized` flag selects the same LFM3 implementation. Use `--reasoning` to change prompting on the base weights. Measure throughput and memory on your hardware; original Qwen timings do not apply.
