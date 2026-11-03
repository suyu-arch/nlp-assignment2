"""Preserve sentence order; learn the vocabulary from training text only."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = range(4)


def build_vocab(sentences, min_freq=2):
    counts = Counter(char for sentence in sentences for char in sentence)
    return SPECIALS + sorted(char for char, count in counts.items() if count >= min_freq)


def encode_sentences(sentences, vocab):
    lookup = {char: i for i, char in enumerate(vocab)}
    tokens, offsets = [], [0]
    unknown = 0
    for sentence in sentences:
        ids = [lookup.get(char, UNK) for char in sentence]
        unknown += ids.count(UNK)
        tokens.extend([BOS, *ids, EOS])
        offsets.append(len(tokens))
    return {
        "tokens": torch.tensor(tokens, dtype=torch.long),
        "offsets": torch.tensor(offsets, dtype=torch.long),
    }, unknown


def prepare_data(source, output, val_sentences=1000, min_freq=2):
    source, output = Path(source), Path(output)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        # Remove line endings only. Whitespace inside the sentence is retained.
        sentences = [line.rstrip("\r\n") for line in handle]
    if not 0 < val_sentences < len(sentences):
        raise ValueError("The corpus must contain more sentences than the validation split.")
    train, val = sentences[:-val_sentences], sentences[-val_sentences:]
    vocab = build_vocab(train, min_freq)
    train_data, train_unk = encode_sentences(train, vocab)
    val_data, val_unk = encode_sentences(val, vocab)
    # Hash decoded text so the .gz and its unpacked file have the same identity.
    digest = hashlib.sha256()
    for sentence in sentences:
        digest.update(sentence.encode("utf-8") + b"\n")
    stats = {
        "source": str(source.resolve()), "corpus_sha256": digest.hexdigest(),
        "train_sentences": len(train), "val_sentences": len(val),
        "train_characters": sum(map(len, train)), "val_characters": sum(map(len, val)),
        "vocab_size": len(vocab), "min_freq": min_freq,
        "train_unknown_characters": train_unk, "val_unknown_characters": val_unk,
        "train_target_tokens": len(train_data["tokens"]) - len(train),
        "val_target_tokens": len(val_data["tokens"]) - len(val),
        "target_definition": "characters (including whitespace) and EOS; excludes BOS and PAD",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": train_data, "val": val_data, "vocab": vocab, "stats": stats}, output)
    output.with_suffix(".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


class SentenceChunks(Dataset):
    """Non-overlapping targets; reset context at sentence and chunk boundaries."""

    def __init__(self, data, seq_len, limit_sentences=None):
        self.tokens = data["tokens"]
        offsets = data["offsets"].tolist()
        count = len(offsets) - 1
        if limit_sentences is not None:
            count = min(count, limit_sentences)
        self.spans = []
        for i in range(count):
            start, stop = offsets[i], offsets[i + 1] - 1
            self.spans.extend((j, min(j + seq_len, stop)) for j in range(start, stop, seq_len))

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, index):
        start, stop = self.spans[index]
        return self.tokens[start:stop], self.tokens[start + 1:stop + 1]


def collate_chunks(batch):
    inputs, targets = zip(*batch)
    return (pad_sequence(inputs, batch_first=True, padding_value=PAD),
            pad_sequence(targets, batch_first=True, padding_value=PAD))
