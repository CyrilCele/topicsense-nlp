"""Convolutional neural network for text classification."""

from __future__ import annotations

import torch
from torch import nn


class TextCNN(nn.Module):
    """TextCNN using trainable embeddings and parallel convolution filters."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
        padding_index: int = 0,
    ) -> None:
        super().__init__()

        if vocabulary_size < 3:
            raise ValueError("vocabulary_size must be at least 3.")

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        if embedding_dim < 1 or num_filters < 1:
            raise ValueError("embedding_dim and num_filters must be positive.")

        if not kernel_sizes or any(size < 1 for size in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive integers.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_index,
        )

        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                )
                for kernel_size in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            in_features=num_filters * len(kernel_sizes),
            out_features=num_classes,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return unnormalised class scores for a batch of token sequences."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length].")

        if input_ids.shape[1] < max(self.kernel_sizes):
            raise ValueError(
                "The input sequence length must be at least the largest "
                "convolution kernel size."
            )

        embedded = self.embedding(input_ids)
        embedded = embedded.transpose(1, 2)

        pooled_features = []

        for convolution in self.convolutions:
            feature_maps = torch.relu(convolution(embedded))
            pooled = torch.amax(feature_maps, dim=2)
            pooled_features.append(pooled)

        combined = torch.cat(pooled_features, dim=1)
        combined = self.dropout(combined)

        return self.classifier(combined)
