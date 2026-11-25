# Repository review for the LFM3 migration

Syntax and migration scans covered 100 Python files and 370 code cells in 19 notebooks. The review covered tracked Python scripts, code cells in the chapter notebooks, model-dependent imports and call sites, training/checkpoint paths, packaging, tests and GitHub workflows. The migration changes active chapter paths to LFM3 while retaining explicitly named legacy Qwen library modules and historical teacher data/logs.

## Outstanding finding

**[P1] Python evaluation of answer text in the math verifiers.** The main parser calls `sympy.parsing.sympy_parser.parse_expr` on answer text at [ch03.py](reasoning_from_scratch/ch03.py#L236). The alternative parser calls `sympify` at [parser.py](reasoning_from_scratch/bonus/parser.py#L402) and [parser.py](reasoning_from_scratch/bonus/parser.py#L410). These are existing Python-evaluation paths, also reachable while grading generated answers. A harmless check confirmed the main parser accepts `__import__('builtins').len('abc')` and returns `3`. Replace these paths with a constrained mathematical-expression parser, and evaluate potentially expensive symbolic expressions with resource limits. This pre-existing security issue is outside the model migration and remains unresolved.

## Review dimensions

| Dimension | Rating | Evidence |
| --- | --- | --- |
| Security | Needs follow-up | The verifier issue above remains; native LFM3 loading uses pinned artifacts, strict state loading and no remote model execution. Hugging Face export deliberately includes the pinned model code. |
| Performance | Needs workload measurements | The LFM3 cache retains short convolution windows and bounded sliding-attention state. Global attention, full-model optimizer memory and GPU/FSDP throughput still require measurements with the pretrained model. |
| Correctness | Validated on small configurations | Full/cached and padded/unpadded outputs, state shrinking, supervised and policy-gradient backward passes, checkpoint reloads, reference logits/gradients and Hugging Face round trips pass. |
| Maintainability | Improved | Chapter loaders share the candidate configuration and tokenization. Appendix C matches the shared implementation. Historical measurements retain their Qwen attribution. |

## Validation boundary

The local suite passes 236 tests with 29 skipped. Separate opt-in tests pass against the pinned reference implementation and tokenizer using small randomly initialized configurations. Lint, spell checking, quote checks, notebook line-length checks, package build/metadata checks and lockfile consistency pass. A one-step Hugging Face Trainer smoke test also updated weights successfully. Full pretrained-weight training, GPU/FSDP execution and new benchmark scores were not run. The original Qwen tests remain to check compatibility of the legacy modules.

No executable Git hooks or `core.hooksPath` override were configured in this checkout. The GitHub workflow matrix includes Linux/macOS/Windows tests, pip installation, old/nightly PyTorch, lint, spell checking and link checking, with a required-checks aggregator. The real-download workflow now includes the LFM3 reference tests.
