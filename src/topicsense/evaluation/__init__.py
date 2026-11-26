"""Evaluation utilities for TopicSense-NLP."""

from topicsense.evaluation.comparison import (
    build_model_comparison,
    write_model_comparison,
)
from topicsense.evaluation.metrics import calculate_classification_metrics

__all__ = [
    "build_model_comparison",
    "calculate_classification_metrics",
    "write_model_comparison",
]
