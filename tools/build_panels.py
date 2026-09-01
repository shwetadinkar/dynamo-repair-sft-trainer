"""Build the ten held-out verifier panels.

Panel inputs only — expected outputs are produced by the independent reference
implementation in tests/, never by the package under test.
"""

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "task" / "tests" / "panels"

POOL = [
    "invoice", "export", "session", "token", "retry", "billing", "portal",
    "webhook", "timeout", "workspace", "seat", "quota", "sandbox", "digest",
    "rollback", "region", "sync", "credential", "audit", "throttle",
]


def words(n, offset=0):
    return " ".join(POOL[(offset + i) % len(POOL)] for i in range(n))


def turn(role, n, offset=0):
    return {"role": role, "text": words(n, offset)}


def conv(cid, spec, offset=0):
    """spec is [(role, word_count), ...]; offsets keep texts distinct."""
    turns = []
    for i, (role, n) in enumerate(spec):
        turns.append(turn(role, n, offset + 3 * i))
    return {"id": cid, "turns": turns}


def rendered_len(spec):
    return sum(2 + n for _, n in spec)


def cfg(max_seq_len, cap, grad_accum_steps, warmup_steps,
        base_lr=2e-4, min_lr=1e-5):
    assert cap >= max_seq_len, "SPEC §3 precondition violated"
    return {
        "max_seq_len": max_seq_len,
        "max_tokens_per_microbatch": cap,
        "grad_accum_steps": grad_accum_steps,
        "base_lr": base_lr,
        "min_lr": min_lr,
        "warmup_steps": warmup_steps,
    }


def panels():
    p = {}

    # §5 assembly and masking on plain single-turn data.
    p["p01_single_turn"] = (
        [{"id": "a-01", "turns": [
            {"role": "user", "text": "invoice Invoice café"},
            {"role": "assistant", "text": "CAFÉ café"},
         ]},
         conv("a-02", [("user", 5), ("assistant", 4)], 5),
         conv("a-03", [("user", 2), ("assistant", 6)], 11)],
        cfg(32, 64, 1, 0),
    )

    # §5 every assistant turn is supervised, not only the last.
    p["p02_multi_turn_pair"] = (
        [conv("b-01", [("user", 3), ("assistant", 3),
                       ("user", 2), ("assistant", 4)], 0)],
        cfg(48, 64, 1, 0),
    )

    # §5 uneven turn counts with a trailing unsupervised user turn.
    p["p03_multi_turn_interleaved"] = (
        [conv("c-01", [("assistant", 2), ("assistant", 5), ("user", 3),
                       ("assistant", 2), ("user", 4)], 0)],
        cfg(64, 64, 1, 0),
    )

    # §6 truncation with supervised content surviving inside the prefix.
    p["p04_truncate_assistant_prefix"] = (
        [conv("d-01", [("user", 2), ("assistant", 20)], 0)],
        cfg(12, 32, 1, 0),
    )

    # §6 truncation cutting into a later turn of a multi-turn example.
    p["p05_truncate_multi_turn"] = (
        [conv("e-01", [("user", 2), ("assistant", 3),
                       ("user", 2), ("assistant", 3)], 0)],
        cfg(12, 32, 1, 0),
    )

    # §6 examples with no surviving supervision are dropped; §9 input order.
    p["p06_drop_unsupervised"] = (
        [conv("f-z-survivor", [("user", 2), ("assistant", 3)], 4),
         conv("f-drop-truncated", [("user", 10), ("assistant", 2)], 0),
         conv("f-drop-user-only", [("user", 4), ("user", 3)], 9),
         conv("f-a-survivor", [("assistant", 2), ("user", 3)], 13)],
        cfg(9, 32, 1, 0),
    )

    # §7 capacity test at exact equality.
    p["p07_pack_exact_boundary"] = (
        [conv(cid, [("user", 2), ("assistant", 2)], 2 * i)
         for i, cid in enumerate(["g-z", "g-a", "g-m", "g-b", "g-y", "g-c"])],
        cfg(32, 40, 1, 0),
    )

    # §7 a long example following short ones.
    p["p08_pack_width_jump"] = (
        [conv("h-long-1", [("user", 2), ("assistant", 18)], 0)]
        + [conv(f"h-short-{i}", [("user", 2), ("assistant", 2)], 2 * i)
           for i in range(4)]
        + [conv("h-long-0", [("user", 2), ("assistant", 18)], 7)],
        cfg(24, 48, 1, 0),
    )

    # §8 one step spanning micro-batches with unequal supervised token counts.
    p["p09_ragged_normalization"] = (
        [conv(f"j-{i:02d}", [("user", 2), ("assistant", 2)], 2 * i)
         for i in range(3)]
        + [conv(f"j-1{i}", [("user", 2), ("assistant", 14)], 5 * i)
           for i in range(2)]
        + [conv("j-tail", [("user", 2), ("assistant", 22)], 9)],
        cfg(28, 40, 2, 0),
    )

    # §10 warmup and cosine across several optimizer steps.
    p["p10_schedule_span"] = (
        [conv(f"k-{i:02d}", [("user", 2), ("assistant", 2)], 3 * i)
         for i in range(10)],
        cfg(16, 24, 1, 1),
    )

    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (conversations, config) in panels().items():
        payload = {"conversations": conversations, "config": config}
        (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        lengths = [sum(2 + len(t["text"].split()) for t in c["turns"])
                   for c in conversations]
        print(f"{name}: {len(conversations)} conversations, rendered lengths {lengths}")


if __name__ == "__main__":
    main()
