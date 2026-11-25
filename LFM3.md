# Train the LFM3 candidate

The pretrained candidate is [LiquidAI/LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base). This edition keeps the reasoning algorithms and training loops from the book, using a native PyTorch implementation of LFM3 in `reasoning_from_scratch/lfm3.py`.

## Install and load

```bash
python -m pip install -e .
```

```python
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer(device=get_device())
```

Downloads go to `lfm3/`. The official safetensors weights, config, tokenizer and license are pinned to revision `7bca5038823c0d0f14efe06439109354a10e014c`. Loading does not execute remote model code. The 1.73B weights need about 3.5 GB in BF16, before activations, gradients and optimizer states. Full AdamW training needs substantially more memory; start with short sequences, small batches and a GPU, then measure usage before a longer run. The advertised 256K context is an architectural limit, not a practical training default.

`which_model="base"` uses raw text completion prompts. `which_model="reasoning"` uses text chat prompts with an assistant header, leaving the thinking tag in the generated target so the chapter 7 format reward can learn it. Both modes load the identical base checkpoint and tokenizer. They do not select a pretrained reasoning model. The LFM3 tokenizer already contains `<think>` and `</think>`; never add or renumber vocabulary entries. Padding is `<|pad|>` and EOS is `<|im_end|>`.

## Chapter map

| Chapter | LFM3 behavior |
| --- | --- |
| 1 | Concepts; no model code |
| 2 | Load weights and tokenizer, generate text, use caches and optional compilation |
| 3 | Establish the MATH-500 baseline and evaluate saved checkpoints |
| 4-5 | Apply prompting, sampling, self-consistency and refinement to the candidate |
| 6 | Train the candidate with GRPO and save LFM3 state dictionaries |
| 7 | Add clipping, KL and format rewards using LFM3 thinking tokens |
| 8 | Fine-tune the candidate on teacher traces and export its trained weights |
| C-D | Inspect LFM3 architecture, load base weights and resume from local checkpoints |
| E-G | Use padding-aware batching, other evaluations and chat interfaces |

## Suggested experiment order

1. Record a baseline using the same prompt mode, dataset split, sampling settings and token budget you will use after training.
2. Distill reasoning traces into LFM3 with chapter 8. Keep MATH-500 as evaluation data, separate from training.
3. Continue the trained checkpoint with chapter 6 or 7 GRPO.
4. Reevaluate and compare correctness, response length, validation loss and resource use.

Run from this repository checkout:

```bash
python ch03/02_math500-verifier-scripts/evaluate_math500.py --dataset_size 10
python ch08/04_train_with_distillation/distill.py --data_path distill_data.json --use_think_tokens
python ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py --steps 10 --num_rollouts 2 --max_new_tokens 128
```

Supply your own teacher data as a JSON array with `problem`, `message_thinking` and `message_content`. Existing DeepSeek and Qwen teacher datasets remain usable: the teacher need not have the student's architecture or tokenizer. Existing Qwen model checkpoints cannot initialize LFM3. To resume, pass `--checkpoint_path` to the relevant training or evaluation script, using the prompt mode used during training.

```bash
python ch03/02_math500-verifier-scripts/evaluate_math500.py --which_model reasoning --checkpoint_path checkpoints/lfm3-1.7B-rlvr-grpo-step00050.pth
```

To continue a thinking-token distillation run, use `ch07/03_rlvr_grpo_scripts_advanced/7_6_plus_format_reward.py --checkpoint_path YOUR_DISTILL_CHECKPOINT.pth`; that script uses the matching chat prompt format.

GRPO and distillation save `lfm3-1.7B-rlvr-grpo-*.pth` and `lfm3-1.7B-distill-*.pth`. There are no bundled pretrained LFM3 reasoning checkpoints. Original CSV logs, benchmark tables and screenshots show historical Qwen experiments and must not be reported as LFM3 results.

## Hugging Face export

Use Python 3.11 or newer for these examples. The native chapter code also supports Python 3.10.

```bash
python -m pip install 'transformers>=5.23.0' accelerate
python ch08/06_use_via_huggingface/export_approach/hf_export.py --model_path checkpoints/lfm3-1.7B-rlvr-grpo-step00050.pth --tokenizer_kind reasoning --output_dir hf-lfm3
python ch08/06_use_via_huggingface/export_approach/hf_inference.py --model_dir hf-lfm3
```

Export retains the official LFM3 state names and tied embeddings, includes tokenizer files and the model license, and copies the pinned reference modeling code. Hugging Face loading uses `trust_remote_code=True` for those model files. The local wrapper examples support Hugging Face generation and answer-only training without exporting first; wrapper generation recomputes the full prefix.

## Validation

The default suite uses tiny LFM3 configurations to check full versus cached decoding, padding, convolution and attention cache shrinking, gradient updates, strict safetensors loading, checkpoint reloads and the Hugging Face wrapper. Reference parity tests opt in to downloading only the pinned configuration, modeling code and tokenizer, never full weights.

```bash
SKIP_EXPENSIVE=1 python -m pytest tests
RUN_REAL_DOWNLOAD_TESTS=1 python -m pytest tests/test_lfm3_reference.py
```

Full pretrained-model training, GPU/FSDP runs and new MATH-500 scores require dedicated compute and are not implied by tiny-model checks.
