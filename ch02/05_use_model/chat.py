from reasoning_from_scratch.lfm3 import load_state_dict_file
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

# Runs the model similar to chapter 2 and 3 in streaming mode with the least
# amount of bells and whistles. Uses KV caching by default.
# Similar to generate_simple.py but uses an interactive REPL (without memory).

import argparse
from pathlib import Path
import time
import torch

from reasoning_from_scratch.ch02 import (
    get_device,
    generate_stats
)
from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache
)
from reasoning_from_scratch.lfm3 import (
    download_lfm3,
    LFM3Model,
    LFM3Tokenizer,
    LFM3_CONFIG_1_7B
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Run LFM3 text generation (interactive REPL)")
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Device to run on (e.g. 'cpu', 'cuda', 'mps'). "
         "If not provided, will auto-detect with get_device()."
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="Maximum number of new tokens to generate."
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="Compile PyTorch model."
)
parser.add_argument(
    "--reasoning",
    action="store_true",
    help="Use chat prompting with the same LFM3 base weights."
)

args = parser.parse_args()
device = torch.device(args.device) if args.device else get_device()

if args.reasoning:
    download_lfm3(kind="reasoning", tokenizer_only=False, out_dir="lfm3")
    tokenizer_path = Path("lfm3") / "tokenizer.json"
    model_path = Path("lfm3") / "model.safetensors"
    tokenizer = LFM3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True
    )
else:
    download_lfm3(kind="base", tokenizer_only=False, out_dir="lfm3")
    tokenizer_path = Path("lfm3") / "tokenizer.json"
    model_path = Path("lfm3") / "model.safetensors"
    tokenizer = LFM3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        apply_chat_template=False,
        add_generation_prompt=False,
        add_thinking=False
    )

model = LFM3Model(LFM3_CONFIG_1_7B)
state = load_state_dict_file(model_path, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

if args.compile:
    model = torch.compile(model)

print()
print("=" * 60)
print(f"torch     : {torch.__version__}")
print(f"device    : {device}")
print("cache     : True")
print(f"compile   : {args.compile}")
print(f"reasoning : {args.reasoning}")
print("memory    : False")
print("=" * 60)
print()
print("Interactive REPL (no memory). Type '\\exit' or '\\quit' to quit.\n")


def run_once(prompt: str):
    input_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    start_time = time.time()
    all_token_ids = []

    print("[Model]\n", end="", flush=True)
    for token in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id
    ):
        token_id = token.squeeze(0).item()
        print(tokenizer.decode([token_id]), end="", flush=True)

        all_token_ids.append(token_id)

    end_time = time.time()
    print("\n")

    print("[Stats]")
    generate_stats(
        torch.tensor(all_token_ids),
        tokenizer,
        start_time,
        end_time
    )
    print("-" * 60)


# REPL loop
try:
    while True:
        try:
            user_in = input(">> ").strip()
        except EOFError:
            print("")
            break
        if user_in.lower() in {r"\exit", r"\quit"}:
            break
        if not user_in:
            continue

        print("\n" + "-" * 60)
        print("[User]")
        print(user_in + "\n")
        run_once(user_in)
except KeyboardInterrupt:
    print("\nInterrupted by user.")
