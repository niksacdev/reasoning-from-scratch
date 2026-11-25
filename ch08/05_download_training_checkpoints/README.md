# Use your trained LFM3 checkpoint

The original published Qwen checkpoints are incompatible with LFM3. Train the candidate with the chapter scripts, then load a local `.pth` file:

```bash
python ch08/05_download_training_checkpoints/usage_example.py --checkpoint_path checkpoints/lfm3-1.7B-rlvr-grpo-step00050.pth
```

For distillation, select a `lfm3-1.7B-distill-*.pth` file from the training output directory. Use `--which_model reasoning` if the run used chat prompts or thinking-token supervision. There are no bundled trained LFM3 checkpoints.
