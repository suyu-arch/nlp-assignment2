"""Train one model or all four models with a common evaluation protocol."""
import argparse
import csv
import json
from pathlib import Path
import platform

import torch
from torch.utils.data import DataLoader

from nlp_lm.data import SentenceChunks, collate_chunks
from nlp_lm.engine import choose_device, run_epoch, seed_everything
from nlp_lm.models import MODEL_NAMES, build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--data", type=Path, default=Path("data/processed.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/main"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads; keep modest on laptops")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--limit-train-sentences", type=int, help="Debug only: use a training prefix")
    parser.add_argument("--max-train-batches", type=int, help="Debug only: cap each training epoch")
    parser.add_argument("--max-val-batches", type=int, help="Debug only: cap validation")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated model results")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "seq_len", "embedding_dim", "hidden_dim", "layers", "heads", "threads",
                 "limit_train_sentences", "max_train_batches", "max_val_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.log_every < 0:
        parser.error("Require 0 <= dropout < 1, lr > 0 and log_every >= 0")
    if args.model in ("all", "transformer") and args.embedding_dim % args.heads:
        parser.error("embedding-dim must be divisible by heads")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    if not args.data.exists():
        raise SystemExit("Prepared data missing. Run: python3 prepare.py")
    bundle = torch.load(args.data, map_location="cpu", weights_only=True)
    names = MODEL_NAMES if args.model == "all" else (args.model,)
    for name in names:
        destination = args.output / name
        if destination.exists() and any(destination.iterdir()) and not args.overwrite:
            raise SystemExit(f"{destination} already contains results. Choose a new --output or pass --overwrite.")
    train_data = SentenceChunks(bundle["train"], args.seq_len, args.limit_train_sentences)
    val_data = SentenceChunks(bundle["val"], args.seq_len)
    debug_run = any(value is not None for value in
                    (args.limit_train_sentences, args.max_train_batches, args.max_val_batches))
    print(f"Device: {device}; train chunks: {len(train_data)}; validation chunks: {len(val_data)}", flush=True)
    if debug_run:
        print("DEBUG RUN: limited data/batches; do not use these results in the final report.", flush=True)
    for name in names:
        seed_everything(args.seed)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(model=name, device=str(device), debug_run=debug_run,
                      torch_version=str(torch.__version__), python_version=platform.python_version(),
                      data_stats=bundle["stats"])
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  generator=generator, collate_fn=collate_chunks)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_chunks)
        model = build_model(name, len(bundle["vocab"]), config).to(device)
        config["parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        destination = args.output / name
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n{name.upper()}: {config['parameters']:,} parameters", flush=True)
        best_loss, best = float("inf"), None
        training_seconds, validation_seconds = 0.0, 0.0
        fields = ["epoch", "train_loss", "train_ppl", "val_loss", "val_ppl",
                  "train_tokens", "val_tokens", "train_seconds", "val_seconds"]
        with (destination / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                train = run_epoch(model, train_loader, device, optimizer,
                                  args.max_train_batches, args.log_every)
                val = run_epoch(model, val_loader, device, max_batches=args.max_val_batches)
                if args.max_val_batches is None and val["tokens"] != bundle["stats"]["val_target_tokens"]:
                    raise RuntimeError("Validation token count does not match the prepared corpus.")
                training_seconds += train["seconds"]
                validation_seconds += val["seconds"]
                writer.writerow(dict(epoch=epoch, train_loss=train["loss"], train_ppl=train["ppl"],
                                     val_loss=val["loss"], val_ppl=val["ppl"],
                                     train_tokens=train["tokens"], val_tokens=val["tokens"],
                                     train_seconds=train["seconds"], val_seconds=val["seconds"]))
                handle.flush()
                print(f"Epoch {epoch}/{args.epochs} | train loss {train['loss']:.4f} | "
                      f"val PPL {val['ppl']:.2f} | train {train['seconds']:.1f}s | val {val['seconds']:.1f}s", flush=True)
                if val["loss"] < best_loss:
                    best_loss = val["loss"]
                    best = dict(model=name, parameters=config["parameters"], best_epoch=epoch,
                                val_loss=val["loss"], val_ppl=val["ppl"], val_tokens=val["tokens"],
                                debug_run=debug_run)
                    torch.save(dict(model_state=model.state_dict(), config=config,
                                    vocab=bundle["vocab"], epoch=epoch, val_metrics=val),
                               destination / "best.pt")
                summary = dict(best, total_train_seconds=training_seconds,
                               total_val_seconds=validation_seconds, completed_epochs=epoch)
                (destination / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Best {name} validation PPL: {best['val_ppl']:.2f} (epoch {best['best_epoch']})", flush=True)


if __name__ == "__main__":
    main()
