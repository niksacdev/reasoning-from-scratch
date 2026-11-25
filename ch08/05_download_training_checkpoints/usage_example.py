"""Run a locally trained LFM3 checkpoint from chapters 6-8."""

import argparse
import torch

from reasoning_from_scratch.ch02 import get_device, generate_text_basic_stream_cache
from reasoning_from_scratch.ch03 import render_prompt
from reasoning_from_scratch.lfm3 import load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="LFM3 .pth checkpoint saved by training")
    parser.add_argument("--prompt", default="If x + 7 = 19, what is x?")
    parser.add_argument("--which_model", choices=("base", "reasoning"), default="base",
                        help="Prompt format used during training; both use LFM3 base weights")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else get_device()
    model, tokenizer = load_model_and_tokenizer(
        which_model=args.which_model, device=device, checkpoint_path=args.checkpoint_path,
    )
    input_ids = torch.tensor(tokenizer.encode(render_prompt(args.prompt)), device=device).unsqueeze(0)
    for token in generate_text_basic_stream_cache(
        model, input_ids, args.max_new_tokens, eos_token_id=tokenizer.eos_token_id,
    ):
        print(tokenizer.decode(token.squeeze(0).tolist()), end="", flush=True)
    print()


if __name__ == "__main__":
    main()
