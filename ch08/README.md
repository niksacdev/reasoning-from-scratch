# Chapter 8: Distilling Reasoning Models for Efficient Reasoning

This edition uses `LiquidAI/LFM3-1.7B-Base` as the trainable candidate. Base and reasoning prompt modes share the same pretrained weights. Original published metrics and logs describe Qwen runs; rerun evaluation after training LFM3.

&nbsp;
## Main chapter code

- [01_main-chapter-code](01_main-chapter-code): main chapter code and exercise solutions
- [03_logs](03_logs): log files analyzed in this chapter

&nbsp;
## Bonus material

- [02_generate_distillation_data](02_generate_distillation_data): scripts for generating distillation data from teacher models
- [04_train_with_distillation](04_train_with_distillation): Python scripts for training models with distillation data
- [05_download_training_checkpoints](05_download_training_checkpoints): how to load your locally trained LFM3 distillation checkpoints
- [06_use_via_huggingface](06_use_via_huggingface): how to use scratch LFM3 models and checkpoints with Hugging Face
