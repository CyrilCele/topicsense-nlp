"""Evaluation utilities for TopicSense-NLP."""

from topicsense.evaluation.comparison import (
    build_model_comparison,
    write_model_comparison,
)
from topicsense.evaluation.error_analysis import (
    build_error_analysis,
    write_error_analysis,
)
from topicsense.evaluation.metrics import calculate_classification_metrics

__all__ = [
    "build_error_analysis",
    "build_model_comparison",
    "calculate_classification_metrics",
    "write_error_analysis",
    "write_model_comparison",
]
