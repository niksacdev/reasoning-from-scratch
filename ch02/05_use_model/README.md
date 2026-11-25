# Generate and chat with LFM3

Run from the repository root after `python -m pip install -e .`:

```bash
python ch02/05_use_model/generate_simple.py --max_new_tokens 64
python ch02/05_use_model/chat.py --reasoning
python ch02/05_use_model/chat_multiturn.py --reasoning
```

All scripts load `LiquidAI/LFM3-1.7B-Base`. `--reasoning` enables chat prompting with the same weights; fine-tune the candidate to improve its reasoning and instruction following. These are base-model examples, not pretrained assistant checkpoints.
