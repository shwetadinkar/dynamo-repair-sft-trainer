"""Reference implementation of the sfttrainer SFT data pipeline (SPEC.md).

This module is a direct, section-by-section transcription of the normative
specification.  It is deliberately written for reading rather than for speed:
each function below corresponds to one numbered section of SPEC.md, and the
section number is named in the function's docstring.

Dependencies: numpy and the Python standard library only.  Nothing in this
file imports any project module, and there is no global mutable state.

Entry point:

    build_run_reference(conversations, config) -> dict   # SPEC.md §1, §9
"""

import math
import zlib

import numpy as np

# ---------------------------------------------------------------------------
# §4  Vocabulary
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50000

PAD_ID = 0        # <pad>
USER_ID = 1       # <user>
ASSISTANT_ID = 2  # <assistant>
END_ID = 3        # <end>

NUM_RESERVED_IDS = 4
NUM_CONTENT_IDS = VOCAB_SIZE - NUM_RESERVED_IDS  # 49996

# The set of ASCII whitespace characters that separate content tokens.
# Deliberately ASCII-only: no non-ASCII Unicode space is a separator.
ASCII_WHITESPACE = frozenset(" \t\n\r\v\f")

ROLE_MARKER = {"user": USER_ID, "assistant": ASSISTANT_ID}


def content_token_id(piece):
    """§4 — the token id of a single content token string."""
    return NUM_RESERVED_IDS + (zlib.crc32(piece.encode("utf-8")) % NUM_CONTENT_IDS)


def split_content_tokens(text):
    """§4 — split `text` on runs of ASCII whitespace, discarding empty pieces.

    Byte-exact and case-sensitive: no normalization, stripping or lowercasing.
    """
    pieces = []
    current = []
    for character in text:
        if character in ASCII_WHITESPACE:
            if current:
                pieces.append("".join(current))
                current = []
        else:
            current.append(character)
    if current:
        pieces.append("".join(current))
    return pieces


def tokenize_text(text):
    """§4 — the content token ids of a turn's text, in order."""
    return [content_token_id(piece) for piece in split_content_tokens(text)]


# ---------------------------------------------------------------------------
# §5  Example assembly
# ---------------------------------------------------------------------------

def render_conversation(conversation):
    """§5 — render one conversation to (tokens, mask) as plain Python lists.

    For each turn, in order: the role marker, the turn's content tokens, then
    <end>.  A position is supervised (mask 1) if and only if it is a content
    token of an assistant turn or the <end> that closes an assistant turn.
    Every other position -- all user positions, and every role marker
    including <assistant> itself -- is 0.
    """
    tokens = []
    mask = []
    for turn in conversation["turns"]:
        role = turn["role"]
        supervised = 1 if role == "assistant" else 0

        # 1. role marker (never supervised)
        tokens.append(ROLE_MARKER[role])
        mask.append(0)

        # 2. content tokens
        for token_id in tokenize_text(turn["text"]):
            tokens.append(token_id)
            mask.append(supervised)

        # 3. <end>
        tokens.append(END_ID)
        mask.append(supervised)

    return tokens, mask


# ---------------------------------------------------------------------------
# §6  Truncation and dropping
# ---------------------------------------------------------------------------

def build_examples(conversations, max_seq_len):
    """§5 + §6 — render, truncate, drop; result is in *input* order.

    Each surviving example is a dict with an "id" and int64 "tokens"/"mask"
    arrays of equal length.  An example whose mask contains no 1 after
    truncation is dropped, which also covers conversations with no assistant
    turn.
    """
    examples = []
    for conversation in conversations:
        tokens, mask = render_conversation(conversation)

        # §6 — keep the first max_seq_len entries of both arrays.
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
            mask = mask[:max_seq_len]

        # §6 — drop an example with no supervised position left.
        if 1 not in mask:
            continue

        examples.append({
            "id": conversation["id"],
            "tokens": np.array(tokens, dtype=np.int64),
            "mask": np.array(mask, dtype=np.int64),
        })
    return examples


# ---------------------------------------------------------------------------
# §7  Bucketing and packing
# ---------------------------------------------------------------------------

def packing_order(examples):
    """§7 — order by token length ascending, ties by id in byte-wise order.

    Returns indices into `examples`.  The ordering does not depend on input
    order, because (length, id) is unique: ids are unique within a call.
    """
    return sorted(
        range(len(examples)),
        key=lambda index: (
            int(examples[index]["tokens"].shape[0]),
            examples[index]["id"].encode("utf-8"),
        ),
    )


def group_into_microbatches(examples, order, max_tokens_per_microbatch):
    """§7 — greedy packing over the packing order.

    Returns a list of lists of indices into `examples`.  A candidate of length
    Lc joins the micro-batch under construction iff

        (n + 1) * max(Lmax, Lc) <= max_tokens_per_microbatch

    where n is the current member count and Lmax the greatest member length
    (0 when the micro-batch is empty).  On failure the micro-batch is closed
    and the candidate opens a new one; §3's precondition
    (max_tokens_per_microbatch >= max_seq_len) guarantees a single example
    always fits.
    """
    microbatches = []
    current = []
    current_width = 0  # Lmax, 0 while `current` is empty

    for index in order:
        candidate_length = int(examples[index]["tokens"].shape[0])
        prospective_width = max(current_width, candidate_length)
        fits = (len(current) + 1) * prospective_width <= max_tokens_per_microbatch

        if current and not fits:
            microbatches.append(current)
            current = []
            current_width = 0
            prospective_width = candidate_length

        current.append(index)
        current_width = prospective_width

    if current:
        microbatches.append(current)
    return microbatches


def materialize_microbatch(examples, member_indices):
    """§7 — materialize one closed micro-batch as two (n, L) int64 arrays.

    L is the greatest token length among the members.  Rows follow packing
    order and are right-padded to L with <pad> (0) in the token array and 0 in
    the mask array.
    """
    n = len(member_indices)
    width = max(int(examples[i]["tokens"].shape[0]) for i in member_indices)

    tokens = np.full((n, width), PAD_ID, dtype=np.int64)
    mask = np.zeros((n, width), dtype=np.int64)
    for row, index in enumerate(member_indices):
        example = examples[index]
        length = int(example["tokens"].shape[0])
        tokens[row, :length] = example["tokens"]
        mask[row, :length] = example["mask"]

    return {
        "example_ids": [examples[i]["id"] for i in member_indices],
        "tokens": tokens,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# §8  Optimizer steps and loss weights
# ---------------------------------------------------------------------------

def group_into_steps(num_microbatches, grad_accum_steps):
    """§8 — group micro-batch indices into steps of grad_accum_steps.

    A trailing group with fewer members forms a final, shorter step.
    """
    return [
        list(range(start, min(start + grad_accum_steps, num_microbatches)))
        for start in range(0, num_microbatches, grad_accum_steps)
    ]


def loss_weights(microbatches, steps):
    """§8 — loss_weight_i = u_i / sum(u_j for j in the step containing i).

    u_i is the number of supervised (mask 1) positions in micro-batch i.
    Padding is masked 0, so counting over the padded array is exact.
    Weights within a step sum to 1.
    """
    supervised_counts = [
        float(np.count_nonzero(microbatch["mask"] == 1)) for microbatch in microbatches
    ]

    weights = [0.0] * len(microbatches)
    for step_indices in steps:
        total = math.fsum(supervised_counts[i] for i in step_indices)
        for i in step_indices:
            weights[i] = supervised_counts[i] / total
    return weights


# ---------------------------------------------------------------------------
# §10  Learning-rate schedule
# ---------------------------------------------------------------------------

def learning_rate(step_index, num_steps, base_lr, min_lr, warmup_steps):
    """§10 — the learning rate of optimizer step `step_index` in [0, num_steps).

    Warmup is linear over the first W steps; afterwards the rate follows a
    cosine decay from base_lr to min_lr.  The schedule advances once per
    optimizer step, never per micro-batch.
    """
    if step_index < warmup_steps:
        return base_lr * (step_index + 1) / warmup_steps

    span = max(num_steps - warmup_steps, 1)
    progress = (step_index - warmup_steps) / span
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def learning_rate_schedule(num_steps, base_lr, min_lr, warmup_steps):
    """§10 — the learning rate of every optimizer step, in step order."""
    return [
        learning_rate(t, num_steps, base_lr, min_lr, warmup_steps)
        for t in range(num_steps)
    ]


# ---------------------------------------------------------------------------
# §1 / §9  Entry point and return value
# ---------------------------------------------------------------------------

def build_run_reference(conversations, config):
    """§1 — pure entry point; returns the §9 structure.

    Neither `conversations` nor `config` is read after this function returns,
    and neither is mutated: every array in the result is freshly allocated.
    """
    max_seq_len = int(config["max_seq_len"])
    max_tokens_per_microbatch = int(config["max_tokens_per_microbatch"])
    grad_accum_steps = int(config["grad_accum_steps"])
    base_lr = float(config["base_lr"])
    min_lr = float(config["min_lr"])
    warmup_steps = int(config["warmup_steps"])

    # §5 + §6 — surviving examples, in input order (the §9 order for
    # "examples"); this list is also the identity map used by §7.
    examples = build_examples(conversations, max_seq_len)

    # §7 — packing order, then greedy packing over it.
    order = packing_order(examples)
    member_groups = group_into_microbatches(examples, order, max_tokens_per_microbatch)
    microbatches = [
        materialize_microbatch(examples, group) for group in member_groups
    ]

    # §8 — optimizer steps over micro-batches in packing order.
    steps = group_into_steps(len(microbatches), grad_accum_steps)
    weights = loss_weights(microbatches, steps)
    for microbatch, weight in zip(microbatches, weights):
        microbatch["loss_weight"] = float(weight)

    # §10 — one learning rate per optimizer step.
    rates = learning_rate_schedule(len(steps), base_lr, min_lr, warmup_steps)

    return {
        "examples": [
            {
                "id": example["id"],
                "tokens": example["tokens"].copy(),
                "mask": example["mask"].copy(),
            }
            for example in examples
        ],
        "microbatches": microbatches,
        "steps": [
            {"microbatch_indices": list(step_indices), "lr": float(rate)}
            for step_indices, rate in zip(steps, rates)
        ],
    }


# ---------------------------------------------------------------------------
# Panel driver
# ---------------------------------------------------------------------------

def _describe_panel(path):
    """Run the reference over one panel file and print its results."""
    import json

    with open(path, "r", encoding="utf-8") as handle:
        panel = json.load(handle)

    result = build_run_reference(panel["conversations"], panel["config"])

    print("=== %s ===" % path)
    print("config: %s" % json.dumps(panel["config"], sort_keys=True))

    surviving = [example["id"] for example in result["examples"]]
    print("examples (%d, input order): %s" % (len(surviving), surviving))

    print("microbatches (%d, packing order):" % len(result["microbatches"]))
    for index, microbatch in enumerate(result["microbatches"]):
        print(
            "  [%d] ids=%s shape=%s loss_weight=%.17g"
            % (
                index,
                microbatch["example_ids"],
                tuple(microbatch["tokens"].shape),
                microbatch["loss_weight"],
            )
        )

    print("steps (%d):" % len(result["steps"]))
    for index, step in enumerate(result["steps"]):
        print(
            "  [%d] microbatch_indices=%s lr=%.17g"
            % (index, step["microbatch_indices"], step["lr"])
        )
    print()


def _main(argv):
    import glob
    import os

    paths = list(argv[1:])
    if not paths:
        here = os.path.dirname(os.path.abspath(__file__))
        paths = sorted(glob.glob(os.path.join(here, "p*.json")))

    for path in paths:
        _describe_panel(path)


if __name__ == "__main__":
    import sys

    _main(sys.argv)
