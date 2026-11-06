import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUModel(nn.Module):
    """Five-layer GRU text classifier used in assignment 2."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_layers: int,
        n_labels: int,
        padding_idx: int = 0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.classifier(self.dropout(hidden[-1]))
