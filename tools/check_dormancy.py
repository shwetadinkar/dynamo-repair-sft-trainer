"""Prove the shipped fixtures cannot reveal the pipeline defects.

With the batching defect repaired, the shipped fixtures must produce byte-identical
output whether or not the three pipeline defects are present. If this check fails, the
fixtures leak a signal the task depends on withholding.
"""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1] / "task"
DATA = ROOT / "environment" / "data"

BAIT_FIX = (
    "(len(current) + 1) * current_width > max_tokens_per_microbatch",
    "(len(current) + 1) * max(current_width, length) > max_tokens_per_microbatch",
)

PIPELINE_FIXES = [
    # truncation desync
    (
        "return tokens[-max_seq_len:], mask[:max_seq_len]",
        "return tokens[:max_seq_len], mask[:max_seq_len]",
    ),
    # last-assistant-turn-only masking
    (
        "    supervised_turn = None\n"
        "    for index, turn in enumerate(turns):\n"
        '        if turn["role"] == "assistant":\n'
        "            supervised_turn = index\n\n",
        "",
    ),
    (
        "        supervised = index == supervised_turn",
        '        supervised = turn["role"] == "assistant"',
    ),
    # loss normalized by micro-batch count instead of supervised tokens
    (
        "    for members in steps:\n"
        "        for index in members:\n"
        '            microbatches[index]["loss_weight"] = 1.0 / len(members)',
        "    for members in steps:\n"
        '        total = sum(int(microbatches[i]["mask"].sum()) for i in members)\n'
        "        for index in members:\n"
        '            microbatches[index]["loss_weight"] = (\n'
        '                float(microbatches[index]["mask"].sum()) / total\n'
        "            )",
    ),
]

DUMP = """
import json, pathlib, sys
sys.path.insert(0, {root!r})
from sfttrainer.pipeline import build_run
root = pathlib.Path({root!r})
conversations = json.loads((root / "fixtures" / "conversations.json").read_text())
config = json.loads((root / "fixtures" / "config.json").read_text())
run = build_run(conversations, config)
print(json.dumps({{
    "examples": [{{"id": e["id"], "tokens": e["tokens"].tolist(), "mask": e["mask"].tolist()}}
                 for e in run["examples"]],
    "microbatches": [{{"example_ids": b["example_ids"], "tokens": b["tokens"].tolist(),
                      "mask": b["mask"].tolist(), "loss_weight": b["loss_weight"]}}
                     for b in run["microbatches"]],
    "steps": run["steps"],
}}, sort_keys=True))
"""


def apply(path, edits):
    text = path.read_text()
    for old, new in edits:
        if old not in text:
            raise SystemExit(f"anchor not found in {path.name}: {old[:60]!r}")
        text = text.replace(old, new, 1)
    path.write_text(text)


def variant(tmp, name, pipeline_fixed):
    root = pathlib.Path(tmp) / name
    shutil.copytree(DATA, root)
    apply(root / "sfttrainer" / "batching.py", [BAIT_FIX])
    if pipeline_fixed:
        apply(root / "sfttrainer" / "pipeline.py", PIPELINE_FIXES)
    out = subprocess.run(
        [sys.executable, "-c", DUMP.format(root=str(root))],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"{name} failed to run:\n{out.stderr}")
    return out.stdout


def main():
    with tempfile.TemporaryDirectory() as tmp:
        defective = variant(tmp, "defective", pipeline_fixed=False)
        repaired = variant(tmp, "repaired", pipeline_fixed=True)

    if defective != repaired:
        a, b = json.loads(defective), json.loads(repaired)
        for key in ("examples", "microbatches", "steps"):
            if a[key] != b[key]:
                print(f"DORMANCY BROKEN: shipped fixtures expose a difference in {key!r}")
        raise SystemExit(1)

    print("dormancy holds: shipped fixtures cannot distinguish the pipeline defects")


if __name__ == "__main__":
    main()
