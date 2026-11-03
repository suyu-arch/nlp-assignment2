import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = ("fnn", "rnn", "lstm", "transformer")


class FeedForwardLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_len, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.context = seq_len
        # A causal convolution equals a shared linear layer over concatenated
        # context embeddings, without allocating every overlapping window.
        self.hidden = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=seq_len)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens):
        embedded = self.embedding(tokens).transpose(1, 2)
        hidden = torch.tanh(self.hidden(F.pad(embedded, (self.context - 1, 0))))
        return self.output(self.dropout(hidden.transpose(1, 2)))


class RecurrentLM(nn.Module):
    def __init__(self, kind, vocab_size, embedding_dim, hidden_dim, layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        recurrent = nn.RNN if kind == "rnn" else nn.LSTM
        self.recurrent = recurrent(embedding_dim, hidden_dim, num_layers=layers,
                                   batch_first=True, dropout=dropout if layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens):
        # No hidden state is passed between batches or chunks.
        hidden, _ = self.recurrent(self.embedding(tokens))
        return self.output(self.dropout(hidden))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_len, layers, heads, dropout):
        super().__init__()
        if embedding_dim % heads:
            raise ValueError("embedding_dim must be divisible by heads")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position = nn.Embedding(seq_len, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=heads, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens):
        length = tokens.size(1)
        positions = torch.arange(length, device=tokens.device)
        hidden = self.dropout(self.embedding(tokens) + self.position(positions))
        causal_mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        # Padding is on the right. The causal mask already prevents valid
        # positions from attending to it; PAD targets are ignored by the loss.
        hidden = self.encoder(hidden, mask=causal_mask)
        return self.output(hidden)


def build_model(name, vocab_size, config):
    common = dict(vocab_size=vocab_size, embedding_dim=config["embedding_dim"],
                  hidden_dim=config["hidden_dim"], dropout=config["dropout"])
    if name == "fnn":
        return FeedForwardLM(**common, seq_len=config["seq_len"])
    if name in ("rnn", "lstm"):
        return RecurrentLM(name, **common, layers=config["layers"])
    if name == "transformer":
        return TransformerLM(**common, seq_len=config["seq_len"],
                             layers=config["layers"], heads=config["heads"])
    raise ValueError(f"Unknown model: {name}")
