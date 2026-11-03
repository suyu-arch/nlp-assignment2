"""Create a comparison table and learning curves from completed runs."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/main"))
    args = parser.parse_args()
    rows, histories, configs = [], {}, []
    for path in sorted(args.runs.glob("*/summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
        configs.append(json.loads((path.parent / "config.json").read_text(encoding="utf-8")))
        with (path.parent / "history.csv").open(encoding="utf-8") as handle:
            histories[rows[-1]["model"]] = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No training results found under {args.runs}")
    shared = ("data_stats", "seq_len", "embedding_dim", "hidden_dim", "layers", "heads",
              "batch_size", "dropout", "lr", "seed", "epochs", "limit_train_sentences",
              "max_train_batches", "max_val_batches")
    for key in shared:
        if any(config[key] != configs[0][key] for config in configs[1:]):
            raise SystemExit(f"Runs have different {key}; use separate output folders for different experiments.")
    fields = ["model", "parameters", "best_epoch", "val_loss", "val_ppl", "val_tokens",
              "total_train_seconds", "total_val_seconds", "completed_epochs", "debug_run"]
    with (args.runs / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    partial = any(row["completed_epochs"] < config["epochs"] for row, config in zip(rows, configs))
    warning = "DEBUG RESULTS: NOT FOR FINAL REPORT\n\n" if any(row["debug_run"] for row in rows) else ""
    if partial:
        warning += "INCOMPLETE TRAINING: at least one run has unfinished epochs.\n\n"
    table = "| Model | Parameters | Best epoch | Validation PPL | Train seconds |\n|---|---:|---:|---:|---:|\n"
    for row in rows:
        table += (f"| {row['model']} | {row['parameters']:,} | {row['best_epoch']} | "
                  f"{row['val_ppl']:.2f} | {row['total_train_seconds']:.1f} |\n")
    (args.runs / "comparison.md").write_text(warning + table, encoding="utf-8")
    print(warning + table)
    # Keep font caches outside the user's home when running in restricted environments.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nlp-assignment2-matplotlib"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Tables saved. Install matplotlib and rerun to generate curves.")
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        epochs = [int(row["epoch"]) for row in history]
        axes[0].plot(epochs, [float(row["train_loss"]) for row in history], marker="o", label=name.upper())
        axes[1].plot(epochs, [float(row["val_ppl"]) for row in history], marker="o", label=name.upper())
    for ax, title, ylabel in zip(axes, ["Training loss", "Validation perplexity"], ["Cross entropy", "PPL (log scale)"]):
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[1].set_yscale("log")
    if warning:
        figure.suptitle(warning.strip(), fontsize=10)
    figure.tight_layout()
    figure.savefig(args.runs / "learning_curves.png", dpi=180)
    plt.close(figure)
    print(f"Saved comparison.csv, comparison.md, learning_curves.png to {args.runs}")


if __name__ == "__main__":
    main()
