"""Token vocabulary and text-to-id mapping (SPEC.md §4)."""

import zlib

VOCAB_SIZE = 50000

PAD = 0
USER = 1
ASSISTANT = 2
END = 3

_CONTENT_BASE = 4
_CONTENT_SPAN = VOCAB_SIZE - _CONTENT_BASE  # 49996


def token_id(word):
    """Stable id for a content token."""
    return _CONTENT_BASE + (zlib.crc32(word.encode("utf-8")) % _CONTENT_SPAN)


def content_tokens(text):
    """Split raw turn text into content token ids."""
    return [token_id(w) for w in text.split()]


def role_marker(role):
    return ASSISTANT if role == "assistant" else USER
