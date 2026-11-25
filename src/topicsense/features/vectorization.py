"""Configuration-driven sparse text vectorization."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_vectorizer(
    feature_type: str,
    config: dict[str, Any],
) -> CountVectorizer | TfidfVectorizer:
    """Construct a count-based or TF-IDF text vectorizer."""
    ngram_range = tuple(config.get("ngram_range", [1, 1]))

    if len(ngram_range) != 2:
        raise ValueError("ngram_range must contain exactly two integers.")

    common_parameters = {
        "ngram_range": ngram_range,
        "min_df": config.get("min_df", 1),
        "max_df": config.get("max_df", 1.0),
        "max_features": config.get("max_features"),
        "lowercase": config.get("lowercase", False),
        "strip_accents": None,
    }

    if feature_type == "count":
        return CountVectorizer(**common_parameters)

    if feature_type == "tfidf":
        return TfidfVectorizer(
            **common_parameters,
            sublinear_tf=config.get("sublinear_tf", False),
        )

    raise ValueError("feature_type must be 'count' or 'tfidf'.")
