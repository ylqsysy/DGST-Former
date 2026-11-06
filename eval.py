import argparse
import json

from utils.training import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate a DGST-Former checkpoint on the full test partition")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--data-root", help="Override the data location recorded during training")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("batch-size must be positive")
    result = evaluate_checkpoint(args.checkpoint, args.device, args.data_root, args.batch_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
