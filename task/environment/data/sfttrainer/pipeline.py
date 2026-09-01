"""Example assembly, optimizer-step grouping and loss weighting (SPEC.md §5, §6, §8, §9)."""

import numpy as np

from .batching import pack_microbatches
from .schedule import schedule
from .tokenizer import END, content_tokens, role_marker


def _render(conversation):
    """Render one conversation to a token sequence and its supervision mask."""
    turns = conversation["turns"]
    supervised_turn = None
    for index, turn in enumerate(turns):
        if turn["role"] == "assistant":
            supervised_turn = index

    tokens = []
    mask = []
    for index, turn in enumerate(turns):
        supervised = index == supervised_turn
        body = content_tokens(turn["text"])

        tokens.append(role_marker(turn["role"]))
        mask.append(0)

        tokens.extend(body)
        mask.extend([1 if supervised else 0] * len(body))

        tokens.append(END)
        mask.append(1 if supervised else 0)

    return tokens, mask


def _truncate(tokens, mask, max_seq_len):
    """Clip an over-long example to the configured sequence limit."""
    if len(tokens) <= max_seq_len:
        return tokens, mask
    return tokens[-max_seq_len:], mask[:max_seq_len]


def build_examples(conversations, max_seq_len):
    """Assemble surviving examples in input order."""
    examples = []
    for conversation in conversations:
        tokens, mask = _render(conversation)
        tokens, mask = _truncate(tokens, mask, max_seq_len)
        if not any(mask):
            continue
        examples.append(
            {
                "id": conversation["id"],
                "tokens": np.asarray(tokens, dtype=np.int64),
                "mask": np.asarray(mask, dtype=np.int64),
            }
        )
    return examples


def group_steps(microbatches, grad_accum_steps):
    """Group micro-batch indices into optimizer steps."""
    return [
        list(range(start, min(start + grad_accum_steps, len(microbatches))))
        for start in range(0, len(microbatches), grad_accum_steps)
    ]


def build_run(conversations, config):
    """Full data pipeline: conversations in, step-ready micro-batches out."""
    examples = build_examples(conversations, config["max_seq_len"])
    microbatches = pack_microbatches(examples, config["max_tokens_per_microbatch"])
    steps = group_steps(microbatches, config["grad_accum_steps"])

    for members in steps:
        for index in members:
            microbatches[index]["loss_weight"] = 1.0 / len(members)

    learning_rates = schedule(
        len(steps),
        config["base_lr"],
        config["min_lr"],
        config["warmup_steps"],
    )

    return {
        "examples": examples,
        "microbatches": microbatches,
        "steps": [
            {"microbatch_indices": members, "lr": lr}
            for members, lr in zip(steps, learning_rates)
        ],
    }
