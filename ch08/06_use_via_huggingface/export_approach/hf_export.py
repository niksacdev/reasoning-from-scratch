"""Export LFM3 base weights or a chapter 6-8 checkpoint to Hugging Face."""

import argparse

from reasoning_from_scratch.lfm3 import export_to_huggingface, load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=None, help="Local LFM3 .pth checkpoint; omit for pretrained base")
    parser.add_argument("--tokenizer_kind", choices=("base", "reasoning"), default="base",
                        help="Prompt format used during training")
    parser.add_argument("--download_dir", default="lfm3")
    args = parser.parse_args()
    model, _ = load_model_and_tokenizer(
        which_model=args.tokenizer_kind, local_dir=args.download_dir, checkpoint_path=args.model_path,
    )
    output = export_to_huggingface(model, args.output_dir, args.download_dir, args.tokenizer_kind)
    print(f"Exported LFM3 model to {output}")


if __name__ == "__main__":
    main()
