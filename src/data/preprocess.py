# src/data/preprocess.py
from __future__ import annotations
import re
import unicodedata
from typing import Iterable
import emoji

ZERO_WIDTH = "\u200b"

_url_re = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_email_re = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")

def normalize_spacing(text: str) -> str:
    text = text.replace(ZERO_WIDTH, "")
    text = unicodedata.normalize("NFKC", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def redact_urls_emails(text: str) -> str:
    text = _url_re.sub(" <URL> ", text)
    text = _email_re.sub(" <EMAIL> ", text)
    return text

def convert_emojis(text: str) -> str:
    # Turn emojis into :smile: style tokens (keeps sentiment signal)
    return emoji.demojize(text, language="en")

def basic_clean_multilingual(
    text: str,
    *,
    keep_case: bool = True,
    convert_emoji: bool = True,
    redact_contacts: bool = True,
) -> str:
    """Conservative cleaning for English/Indonesian with cased BERT."""
    if not isinstance(text, str):
        return ""
    if redact_contacts:
        text = redact_urls_emails(text)
    if convert_emoji:
        text = convert_emojis(text)
    text = normalize_spacing(text)
    # keep_case=True by default (cased model). If ever needed:
    if not keep_case:
        text = text.lower()
    return text

def bulk_clean_texts(texts: Iterable[str]) -> list[str]:
    return [basic_clean_multilingual(t) for t in texts]