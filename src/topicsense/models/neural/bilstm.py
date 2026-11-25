"""Bidirectional LSTM model for text classification."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTM(nn.Module):
    """Bidirectional LSTM using trainable word embeddings."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_index: int = 0,
    ) -> None:
        super().__init__()

        if vocabulary_size < 3:
            raise ValueError("vocabulary_size must be at least 3.")

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        if embedding_dim < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError(
                "embedding_dim, hidden_size and num_layers must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.padding_index = padding_index

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_index,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            in_features=hidden_size * 2,
            out_features=num_classes,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return unnormalised class scores for a batch of token sequences."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length].")

        if input_ids.shape[1] < 1:
            raise ValueError("Input sequences cannot be empty.")

        lengths = (input_ids != self.padding_index).sum(dim=1).clamp_min(1)

        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden_state, _) = self.lstm(packed)

        forward_hidden = hidden_state[-2]
        backward_hidden = hidden_state[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1,
        )

        combined = self.dropout(combined)

        return cast(torch.Tensor, self.classifier(combined))
