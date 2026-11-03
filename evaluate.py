"""Re-evaluate a saved best checkpoint on exactly the same validation split."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nlp_lm.data import SentenceChunks, collate_chunks
from nlp_lm.engine import choose_device, run_epoch
from nlp_lm.models import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error("batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    bundle = torch.load(args.data, map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    # Compare the complete preprocessing protocol, not just corpus/vocabulary.
    if bundle["vocab"] != checkpoint["vocab"] or bundle["stats"] != config["data_stats"]:
        raise SystemExit("Data or preprocessing differs from the checkpoint's training data.")
    device = choose_device(args.device)
    model = build_model(config["model"], len(bundle["vocab"]), config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = SentenceChunks(bundle["val"], config["seq_len"])
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_chunks)
    metrics = run_epoch(model, loader, device)
    if metrics["tokens"] != bundle["stats"]["val_target_tokens"]:
        raise RuntimeError("Validation target count mismatch")
    print(json.dumps(dict(model=config["model"], checkpoint_epoch=checkpoint["epoch"],
                          full_validation=True, training_was_debug=config["debug_run"], **metrics), indent=2))


if __name__ == "__main__":
    main()
