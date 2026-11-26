# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

"""Measure equal LFM3 generation workloads, excluding console I/O."""

import argparse
import json
import time

import torch

from reasoning_from_scratch.ch02 import (
    benchmark_generation, generate_stats, generate_text_basic,
    generate_text_basic_cache, get_device, synchronize_device,
)
from reasoning_from_scratch.lfm3 import load_model_and_tokenizer


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default=None)
parser.add_argument("--cache", action="store_true")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--reasoning", action="store_true",
                    help="Use chat prompting with the same LFM3 base weights")
parser.add_argument("--optimized", action="store_true",
                    help="Compatibility flag: LFM3 always uses native SDPA")
parser.add_argument("--compare", action="store_true",
                    help="Benchmark both cached and uncached generation")
parser.add_argument("--max_new_tokens", type=int, default=64)
parser.add_argument("--prompt", default=None)
args = parser.parse_args()
if args.max_new_tokens < 1:
    parser.error("--max_new_tokens must be positive")
device = torch.device(args.device) if args.device else get_device()
# LFM3-specific: load the hybrid model, its tokenizer, and tied embeddings.
model, tokenizer = load_model_and_tokenizer(
    which_model="reasoning" if args.reasoning else "base", device=device,
)
if args.compile:
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 8):
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
    model = torch.compile(model)

prompt = args.prompt or (
    "Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field."
    if args.reasoning else "The nematode C. elegans is"
)
input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
print(f"device={device}, dtype={next(model.parameters()).dtype}, native SDPA")
if args.compare:
    print(json.dumps(benchmark_generation(model, input_ids, args.max_new_tokens), indent=2))
else:
    generate = generate_text_basic_cache if args.cache else generate_text_basic
    # Fixed token count and complete warmup make timings comparable.
    generate(model, input_ids, args.max_new_tokens)
    for iteration in range(3):
        synchronize_device(device)
        start = time.perf_counter()
        output = generate(model, input_ids, args.max_new_tokens)
        synchronize_device(device)
        end = time.perf_counter()
        print(f"Run {iteration + 1}, cache={args.cache}, tokens={output.numel()}")
        generate_stats(output, tokenizer, start, end)
    print(tokenizer.decode(output[0].tolist()))
