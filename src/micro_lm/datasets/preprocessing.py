"""
Text preprocessing for Q&A data (cleansing / prep).

Reusable by download scripts and by dataset loaders so cleansing logic
lives in one place and stays consistent.
"""
import re


def strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities. Use for Stack Exchange / web text."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_words(text: str, max_words: int) -> str:
    """Keep at most the first max_words words."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single space and strip."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())
