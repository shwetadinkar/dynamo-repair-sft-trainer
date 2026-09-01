"""Learning-rate schedule (SPEC.md §10)."""

import math


def learning_rate(step_index, total_steps, base_lr, min_lr, warmup_steps):
    """LR for one optimizer step."""
    if step_index < warmup_steps:
        return base_lr * (step_index + 1) / warmup_steps
    decay_span = max(total_steps - warmup_steps, 1)
    progress = (step_index - warmup_steps) / decay_span
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def schedule(total_steps, base_lr, min_lr, warmup_steps):
    """LR for every optimizer step in a run."""
    return [
        learning_rate(t, total_steps, base_lr, min_lr, warmup_steps)
        for t in range(total_steps)
    ]
