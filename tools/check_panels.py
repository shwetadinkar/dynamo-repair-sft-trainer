"""Check every held-out panel earns its place.

A panel is useful only if it distinguishes the shipped (defective) package from a
repaired one, or is explicitly listed as covering a SPEC rule that no injected defect
violates. Panels that discriminate nothing and are not listed are silently vacuous.
"""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_dormancy as cd  # noqa: E402

PANELS = pathlib.Path(__file__).resolve().parents[1] / "task" / "tests" / "panels"

# Panels that legitimately do not discriminate: they pin SPEC rules the injected
# defects happen to satisfy, guarding a reimplementation against other errors.
CONFORMANCE_ONLY = {
    "p01_single_turn",          # §5 baseline assembly: markers, end tokens, user masking
    "p06_drop_unsupervised",    # §6 drop rule and §9 input ordering
    "p07_pack_exact_boundary",  # §7 capacity test at exact equality
    "p10_schedule_span",        # §10 warmup/cosine, schedule.py is defect-free
}

DUMP = '''
import json, sys
sys.path.insert(0, {root!r})
from sfttrainer.pipeline import build_run
payload = json.loads(sys.stdin.read())
run = build_run(payload["conversations"], payload["config"])
print(json.dumps({{
    "examples": [{{"id": e["id"], "tokens": e["tokens"].tolist(),
                  "mask": e["mask"].tolist()}} for e in run["examples"]],
    "microbatches": [{{"example_ids": b["example_ids"], "tokens": b["tokens"].tolist(),
                      "mask": b["mask"].tolist(), "loss_weight": b["loss_weight"]}}
                     for b in run["microbatches"]],
    "steps": run["steps"],
}}, sort_keys=True))
'''


def build(tmp, name, repaired):
    root = pathlib.Path(tmp) / name
    shutil.copytree(cd.DATA, root)
    if repaired:
        cd.apply(root / "sfttrainer" / "batching.py", [cd.BAIT_FIX])
        cd.apply(root / "sfttrainer" / "pipeline.py", cd.PIPELINE_FIXES)
    return root


def run(root, payload):
    out = subprocess.run(
        [sys.executable, "-c", DUMP.format(root=str(root))],
        input=json.dumps(payload), capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(
            "panel driver failed to run — the comparison below would be meaningless.\n"
            f"interpreter: {sys.executable}\n{out.stderr.strip()[-800:]}"
        )
    return out.stdout


def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        shipped = build(tmp, "shipped", repaired=False)
        repaired = build(tmp, "repaired", repaired=True)

        for path in sorted(PANELS.glob("p*.json")):
            payload = json.loads(path.read_text())
            differs = run(shipped, payload) != run(repaired, payload)
            expected = path.stem not in CONFORMANCE_ONLY
            ok = differs == expected
            note = "discriminates" if differs else "conformance-only"
            print(f"{'ok  ' if ok else 'FAIL'} {path.stem}: {note}")
            if not ok:
                failures.append(path.stem)

    if failures:
        print(f"\n{len(failures)} panel(s) not behaving as designed: {failures}")
        raise SystemExit(1)
    print("\nall panels behave as designed")


if __name__ == "__main__":
    main()
