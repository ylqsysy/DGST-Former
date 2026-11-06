import argparse

from utils.download import DATASETS, download_dataset


def main():
    parser = argparse.ArgumentParser(description="Download verified standard PEMS data")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    for name in args.datasets:
        download_dataset(name, args.data_root)
        print(f"Verified {name}", flush=True)


if __name__ == "__main__":
    main()
