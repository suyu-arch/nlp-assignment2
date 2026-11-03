import argparse
import json
from pathlib import Path

from nlp_lm.data import prepare_data


def main():
    parser = argparse.ArgumentParser(description="Prepare ordered character-level train/validation data.")
    parser.add_argument("--data", type=Path, default=Path("news.2017.zh.shuffled.deduped"))
    parser.add_argument("--output", type=Path, default=Path("data/processed.pt"))
    parser.add_argument("--val-sentences", type=int, default=1000)
    parser.add_argument("--min-freq", type=int, default=2)
    args = parser.parse_args()
    if args.min_freq < 1:
        parser.error("--min-freq must be positive")
    stats = prepare_data(args.data, args.output, args.val_sentences, args.min_freq)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
