"""Length bucketing and micro-batch packing (SPEC.md §7)."""

import numpy as np

from .tokenizer import PAD


def packing_order(examples):
    """Examples ordered for packing: shortest first, ties broken by id."""
    return sorted(examples, key=lambda e: (len(e["tokens"]), e["id"]))


def _materialize(members):
    width = max(len(e["tokens"]) for e in members)
    tokens = np.full((len(members), width), PAD, dtype=np.int64)
    mask = np.zeros((len(members), width), dtype=np.int64)
    for row, e in enumerate(members):
        n = len(e["tokens"])
        tokens[row, :n] = e["tokens"]
        mask[row, :n] = e["mask"]
    return {
        "example_ids": [e["id"] for e in members],
        "tokens": tokens,
        "mask": mask,
    }


def pack_microbatches(examples, max_tokens_per_microbatch):
    """Greedily pack examples into padded micro-batches under the token budget."""
    batches = []
    current = []
    current_width = 0

    for example in packing_order(examples):
        length = len(example["tokens"])
        if current and (len(current) + 1) * current_width > max_tokens_per_microbatch:
            batches.append(_materialize(current))
            current = []
            current_width = 0
        current.append(example)
        current_width = max(current_width, length)

    if current:
        batches.append(_materialize(current))
    return batches
