# utils/file_processor.py
# Helpers for processing and chunking document text

def truncate_text(text: str, max_words: int = 5000) -> str:
    """Truncate text to a maximum number of words."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

def chunk_text(text: str, chunk_words: int = 4000) -> list:
    """Split text into chunks of specified word count."""
    if not text:
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_words):
        chunks.append(" ".join(words[i:i + chunk_words]))
    return chunks
