import argparse

from utils.data import preprocess
from utils.download import DATASETS


def main():
    parser = argparse.ArgumentParser(description="Clean, split, and normalize PEMS flow data")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    for name in args.datasets:
        metadata = preprocess(name, args.data_root)
        print(f"Prepared {name}: {metadata['time_steps']} steps, {metadata['sensors']} sensors", flush=True)


if __name__ == "__main__":
    main()
