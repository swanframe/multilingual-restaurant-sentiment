import re
import unicodedata
from unidecode import unidecode

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "")  # zero-width space
    text = re.sub(r"\s+", " ", text).strip()
    return text

def transliterate_if_weird(text: str) -> str:
    # Optional: keep originals; use for extremely noisy lines
    return unidecode(text)