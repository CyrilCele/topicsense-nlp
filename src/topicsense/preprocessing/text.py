"""Deterministic text preparation utilities for TopicSense-NLP."""

from __future__ import annotations

import re
import unicodedata

_WORD_PATTERN = re.compile(r"\b[\w]+(?:['’][\w]+)?\b", flags=re.UNICODE)
_HEADER_SEPARATOR = re.compile(r"\r?\n\r?\n")


def extract_message_body(text: str) -> str:
    """Return the message body, or the full text when no header separator exists."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    parts = _HEADER_SEPARATOR.split(text, maxsplit=1)
    return parts[1] if len(parts) == 2 else text


def normalize_text(text: str, *, lowercase: bool = False) -> str:
    """Apply Unicode NFKC and whitespace normalization without deleting content."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text)

    if lowercase:
        normalized = normalized.casefold()

    return " ".join(normalized.split())


def preprocess_text(
    text: str,
    *,
    representation: str = "full_message",
    lowercase: bool = False,
) -> str:
    """Prepare a full-message or body-only representation without fitting on data."""
    if representation == "full_message":
        selected_text = text
    elif representation == "message_body":
        selected_text = extract_message_body(text)
    else:
        raise ValueError("representation must be 'full_message' or 'message_body'")

    return normalize_text(selected_text, lowercase=lowercase)


def tokenize_words(text: str) -> list[str]:
    """Return simple diagnostic word tokens; this is not a model tokenizer."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return _WORD_PATTERN.findall(text)
