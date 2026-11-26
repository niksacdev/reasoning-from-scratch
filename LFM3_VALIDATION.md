# LFM3 validation: 2026-11-25

Candidate: `LiquidAI/LFM3-1.7B-Base`, revision `7bca5038823c0d0f14efe06439109354a10e014c`. Hardware: Apple M4 Max, 64 GB RAM; Python 3.14.9, PyTorch 2.10.0. Native pretrained CPU/MPS execution defaults to FP32. Original BF16 weights were never overwritten.

## Chapter 2 cache investigation

The notebook forced CPU and timed token decoding/printing together with inference. An initial BF16 CPU diagnostic improved only from 0.66 to 0.77 tokens/s with caching over 16 tokens. This was a single diagnostic run, not a warmed median benchmark.

A second prompt exposed MPS BF16 cached/uncached decision differences after 20 generated tokens. FP32 resolved the observed divergence. CPU/MPS loaders now select FP32, and notebooks retain the automatically selected device.

The corrected benchmark uses actual weights, prompt `The nematode C. elegans is`, 32 generated tokens, a complete warmup per path and three synchronized timed runs. Printing/decoding are excluded.

| MPS FP32 | Median time | Throughput |
| --- | ---: | ---: |
| Uncached | 2.311 s | 13.84 tokens/s |
| Cached | 1.228 s | 26.07 tokens/s |

Latest speedup: **1.88×** (an earlier independent run measured 2.00×), with identical output tokens. This applies to this workload/machine. Model-weight reads and backend overhead can dominate short CPU workloads; caching does not promise a fixed speedup.

## Actual pretrained-weight checks

| Path | Result |
| --- | --- |
| Native vs pinned official model, FP32 | Maximum logit difference `7.63e-6`; identical 32-token completion. Official cached/uncached generation also agrees. |
| Native cached vs uncached | Identical 32-token output. Small configurations additionally cross sliding-window boundaries with different chunk sizes. |
| Unequal padded batches vs individual generation | Identical tokens with actual weights. |
| Sampling, voting, refinement and answer scoring | Actual pretrained execution; finite scores. Short answers check execution, not accuracy. |
| Full backward, AdamW update, native checkpoint reload | Finite gradients throughout; convolution weights changed; reloaded logits match. |
| HF wrapper generation | Actual full-weight output matches native generation. |
| Full HF export/reload | Actual full-weight output matches native generation; base-weight round-trip maximum logit difference `4.77e-6`. |
| Native CPU compilation | A small hybrid model passes actual Inductor compilation and cached/uncached generation. This does not establish GPU compilation support/speed. |

All five full-weight integration checks passed, including independent convolution fine-tuning before HF export. The trained-export logit difference was `3.81e-6`.

The base model can give wrong or incomplete math answers before fine-tuning. These are execution/parity checks, not model-quality measurements.

## Chapter and entry-point matrix

| Area | Executed checks |
| --- | --- |
| Chapter 1 | Concepts; no model execution. |
| Chapter 2 | Main/exercise notebooks; ten native generation APIs; zero/one/multiple-token budgets; EOS; context/final-forward limits; fair timings; optional CPU compilation. |
| Chapter 3 | Main/exercise/advanced-parser notebooks; arithmetic/LaTeX grading; saved-output evaluation; Python-payload regressions; evaluation script CLI/import paths. |
| Chapters 4–5 | Main/exercise notebooks with actual generation, temperature/top-p, voting, critique/refinement and continuation scoring; current-tokenizer plots/answer lengths. |
| Chapter 6 | Main/exercise notebooks, including a bounded full-weight GRPO step; single/batched/FSDP-module algorithms on small native models; optimizer/checkpoint loops. FSDP helpers run unsharded. |
| Chapter 7 | Main/exercise notebooks; all advanced scripts' samplers, losses/backward, zero advantages and one-step training/checkpoints; clipping, KL, format rewards, GDPO, DeepSeek-style and OLMo-style variants. |
| Chapter 8 | Main/exercise notebooks including bounded actual distillation training; single/batched losses/gradients; checkpoints; full-weight HF wrapper/export; small-model HF Trainer/collator update. |
| Appendices C–E | All notebooks; architecture parity, pretrained loading/resume, padded batches and shrinking both cache state types. |
| Appendix F | Notebook; letter matching, next-token/teacher-forced MMLU scoring with dataset fixtures; leaderboard math. Live Ollama cells explicitly skipped. |
| Appendix G | Single/multi-turn handlers replayed with a Chainlit fixture and real small LFM3 models; encoding, streaming and history. Browser UI not exercised. |
| Teacher data | Ollama/OpenRouter request/retry/response/resume fixtures; MiniMax parsing/request/resume tests. No live hosted calls. |

All **19 notebooks** passed bounded execution, producing **21 rendered plots**. Executed copies retain source and outputs; tracked notebooks keep clean outputs. Caps: eight generated tokens, two samples/rollouts, one training step/epoch/example, short synthetic arithmetic data. Three live-Ollama cells are recorded as skipped. Historical CSV plots retain their Qwen attribution; the missing conditional-run CSV is not fabricated.

The local repository suite passed **439 tests**, with **32 explicitly skipped**. Skips include opt-in full-weight/reference/compiler checks, services and legacy large-Qwen downloads. LFM3 pretrained/reference/compiler checks ran separately. See [the review findings](CODE_REVIEW.md).

## Reproduce

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 python -m pytest
RUN_REAL_DOWNLOAD_TESTS=1 python -m pytest tests/test_lfm3_reference.py
LFM3_PRETRAINED_DIR=/absolute/path/to/lfm3 LFM3_TEST_DEVICE=mps python -m pytest -s tests/test_lfm3_pretrained.py
RUN_LFM3_COMPILE_TESTS=1 python -m pytest tests/test_lfm3.py -k compiled_native
python tests/run_lfm3_notebooks.py --weights /absolute/path/to/lfm3 --output /tmp/lfm3-notebooks
```

Also set `LFM3_REFERENCE_DIR` to the pinned official configuration/modeling-code directory for full-weight reference and fine-tuned-export checks. Full-weight optimizer/export tests need substantial free RAM and temporary disk. Install optional `accelerate` and `antlr4-python3-runtime==4.11.*` for Trainer and LaTeX backend tests.

Local executed notebooks/manifest: `/tmp/reasoning-validation/notebooks/`. See [recorded measurements](tests/validation/lfm3-2026-11-25.json).

## Boundaries

No CUDA/XPU hardware, multi-GPU FSDP collectives, live Chainlit browser session, live Ollama/hosted teachers, complete benchmark datasets or long training schedules were exercised. External contracts use fixtures. These workloads establish no new MATH-500/MMLU scores or convergence claims. Full 256K context was not attempted.
