# Repository review for the LFM3 migration

The follow-up review exercises all chapter notebooks and the shared model, inference, evaluation, GRPO, distillation, checkpoint, export and chat paths. See [the validation report](LFM3_VALIDATION.md) for workloads, evidence and boundaries.

## Findings fixed

| Finding | Fix and regression coverage |
| --- | --- |
| Chapter 2 forced CPU and timed console work | Keep automatic device selection; compare fixed workloads with full warmups, synchronized timers and medians. Check forward lengths, output tokens, zero budgets, EOS and context limits. |
| MPS BF16 cached generation diverged at close token decisions | CPU/MPS loaders default to FP32. Actual full-weight FP32 logits and 32-token generation agree with the official reference. Dtype overrides remain available. |
| Random full-model initialization before pretrained loading | Strict meta-model loading preserves tied embeddings without initializing another full random model. Script resume paths and the HF wrapper share this loader. |
| Qwen token IDs and labels remained in examples | Derive token IDs, plot ranges, answer lengths, thinking markers and shortest examples from LFM3's tokenizer/data. Mark adaptations with inline LFM3-specific comments. |
| Continuation scores received another chat template | Encode raw continuations; MMLU scores condition on the exact existing prompt IDs. Match complete answer letters rather than letters inside words. |
| Top-k sampling scattered along the batch dimension | Scatter along the vocabulary axis; test all rollout samplers with native hybrid caches and greedy/stochastic temperatures. |
| Masked entropy produced NaNs | Avoid multiplying zero probability by negative infinity; test masked entropy and backward passes. |
| Single-rollout GRPO had undefined statistics | Validate group size across every variant; exercise distinct/equal rewards, zero advantages, training loops and checkpoint reloads. |
| Empty distillation inputs produced undefined losses | Validate prompts/targets/evaluation batches; compare single/batched losses and gradients; reduce low-precision probabilities/losses in FP32. |
| Notebook helper required __file__; exercise fetched an unpublished CSV | Resolve source paths through the package; plot conditional-run metrics only when the user supplies a real log. |
| Graders evaluated generated Python expressions | Reject strings, attributes, private names and non-arithmetic operators before parsing with restricted constructors and no Python builtins. Payload regressions cannot create files. |

## Review dimensions

| Dimension | Evidence |
| --- | --- |
| Correctness | Actual pretrained reference parity, full-weight optimizer/checkpoint/HF round trips, all 19 bounded notebooks and algorithm regressions. |
| Performance | Measured cache throughput; complete warmups; synchronized timers excluding console work; explicit precision/device settings. |
| Security | Pinned artifacts, strict native loading, constrained math parsing and fixture-only hosted-service tests. HF export includes pinned official modeling code. Symbolic simplification is not process-isolated and can still be computationally expensive. |
| Maintainability | Shared loaders/configuration, marked adaptations, Appendix C parity, reproducible notebook runner and validation matrix. |

## Delivery checks

No executable Git hooks or core.hooksPath override are configured. Required workflows cover Linux/macOS/Windows, pip installation, old/nightly PyTorch, lint, spelling, links and the required-checks aggregator. Linux additionally checks the pinned reference implementation. Merge only after the latest PR revision passes all applicable workflows.
