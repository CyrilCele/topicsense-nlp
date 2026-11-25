"""Factory for classical text-classification estimators."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def build_classifier(
    classifier_name: str,
    config: dict[str, Any],
    *,
    random_seed: int = 42,
) -> MultinomialNB | LogisticRegression | LinearSVC:
    """Create a configured classical classifier."""
    if classifier_name == "multinomial_nb":
        return MultinomialNB(
            alpha=float(config.get("alpha", 1.0)),
        )

    if classifier_name == "logistic_regression":
        return LogisticRegression(
            C=float(config.get("C", 1.0)),
            solver=str(config.get("solver", "lbfgs")),
            max_iter=int(config.get("max_iter", 1000)),
            random_state=random_seed,
        )

    if classifier_name == "linear_svm":
        return LinearSVC(
            C=float(config.get("C", 1.0)),
            max_iter=int(config.get("max_iter", 5000)),
            dual="auto",
            random_state=random_seed,
        )

    raise ValueError(f"Unsupported classifier: {classifier_name}")
