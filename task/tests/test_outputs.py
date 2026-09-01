"""Behavioral verifier for dynamo/repair-sft-trainer."""

import json
import os
import pathlib
import resource
import signal
import subprocess
import sys

import pytest

TESTS = pathlib.Path(__file__).resolve().parent
PANELS = TESTS / "panels"
APP = pathlib.Path("/app")
RUNNER_UID = 65532
RTOL = 1e-7

sys.path.insert(0, str(TESTS))
import reference  # noqa: E402


DRIVER = r"""
import copy, json, os, sys
sys.path.insert(0, "/app")

def blocked(path, write=False):
    try:
        with open(path, "w" if write else "r"):
            pass
    except (PermissionError, FileNotFoundError, IsADirectoryError):
        return True
    return False

payload = json.loads(sys.stdin.read())
operation = payload.pop("operation", "build")

if operation == "tokenize":
    from sfttrainer.tokenizer import content_tokens
    answer = {"tokens": [content_tokens(text) for text in payload["texts"]]}
elif operation == "schedule":
    from sfttrainer.schedule import schedule
    answer = {"rates": [schedule(**case) for case in payload["cases"]]}
else:
    from sfttrainer.pipeline import build_run
    before = copy.deepcopy(payload)
    first = build_run(payload["conversations"], payload["config"])
    second = build_run(payload["conversations"], payload["config"])

    def serialise(run):
        return {
            "examples": [
                {"id": e["id"], "tokens": e["tokens"].tolist(),
                 "mask": e["mask"].tolist()} for e in run["examples"]
            ],
            "microbatches": [
                {"example_ids": b["example_ids"], "tokens": b["tokens"].tolist(),
                 "mask": b["mask"].tolist(), "loss_weight": float(b["loss_weight"])}
                for b in run["microbatches"]
            ],
            "steps": [
                {"microbatch_indices": list(s["microbatch_indices"]), "lr": float(s["lr"])}
                for s in run["steps"]
            ],
        }

    one = serialise(first)
    two = serialise(second)
    answer = {
        "run": one,
        "inputs_unchanged": payload == before,
        "repeat_equal": one == two,
        "schema": {
            "top": sorted(first),
            "examples": [sorted(item) for item in first["examples"]],
            "microbatches": [sorted(item) for item in first["microbatches"]],
            "steps": [sorted(item) for item in first["steps"]],
        },
        "dtypes": {
            "examples": [[str(e["tokens"].dtype), str(e["mask"].dtype)]
                         for e in first["examples"]],
            "microbatches": [[str(b["tokens"].dtype), str(b["mask"].dtype)]
                             for b in first["microbatches"]],
        },
    }

answer["isolation"] = {
    "tests_hidden": blocked("/tests/reference.py"),
    "reward_protected": blocked("/logs/verifier/child-probe", write=True),
}
print(json.dumps(answer, separators=(",", ":")))
"""


def _drop_privileges():
    """Put untrusted package code in a bounded, unprivileged process group."""
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    os.setgroups([])
    os.setgid(RUNNER_UID)
    os.setuid(RUNNER_UID)


def _run_child(payload, script=DRIVER, cwd="/", timeout=15):
    env = {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        preexec_fn=_drop_privileges,
    )
    try:
        stdout, stderr = proc.communicate(json.dumps(payload), timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        pytest.fail("agent package exceeded the child-process timeout")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    assert proc.returncode == 0, stderr.strip()[-1500:]
    assert len(stdout) <= 4_000_000, "agent package returned excessive output"
    return json.loads(stdout)


def panel(name):
    return json.loads((PANELS / f"{name}.json").read_text())


def _plain(run):
    return json.loads(json.dumps({
        "examples": [
            {"id": e["id"], "tokens": e["tokens"].tolist(), "mask": e["mask"].tolist()}
            for e in run["examples"]
        ],
        "microbatches": [
            {"example_ids": b["example_ids"], "tokens": b["tokens"].tolist(),
             "mask": b["mask"].tolist(), "loss_weight": float(b["loss_weight"])}
            for b in run["microbatches"]
        ],
        "steps": run["steps"],
    }))


_bundles = {}


def bundle(name):
    if name not in _bundles:
        _bundles[name] = _run_child(panel(name))
    return _bundles[name]


def actual(name):
    return bundle(name)["run"]


def expected(name):
    payload = panel(name)
    return _plain(reference.build_run_reference(payload["conversations"], payload["config"]))


ALL_PANELS = sorted(path.stem for path in PANELS.glob("p*.json"))


def test_repro_runs_clean():
    """Shipped reproduction: `/app/repro.py` completes after the repair."""
    script = (
        "import contextlib,io,json,runpy,sys; sys.path.insert(0,'/app'); "
        "sink=io.StringIO(); "
        "ctx=contextlib.redirect_stdout(sink); ctx.__enter__(); "
        "runpy.run_path('/app/repro.py',run_name='__main__'); "
        "ctx.__exit__(None,None,None); print(json.dumps({'ok':True}))"
    )
    _run_child({}, script=script, cwd=str(APP))


def test_entry_point_is_pure_and_isolated():
    """SPEC §1: repeated calls are deterministic, do not mutate inputs, and run isolated."""
    result = bundle("p03_multi_turn_interleaved")
    assert result["inputs_unchanged"] and result["repeat_equal"]
    assert result["isolation"] == {"tests_hidden": True, "reward_protected": True}
    assert set(result["run"]) == {"examples", "microbatches", "steps"}
    assert result["schema"]["top"] == ["examples", "microbatches", "steps"]
    assert all(keys == ["id", "mask", "tokens"] for keys in result["schema"]["examples"])
    assert all(keys == ["example_ids", "loss_weight", "mask", "tokens"]
               for keys in result["schema"]["microbatches"])
    assert all(keys == ["lr", "microbatch_indices"] for keys in result["schema"]["steps"])


def test_tokenization_is_byte_exact_and_case_sensitive():
    """SPEC §4: case and non-ASCII bytes affect deterministic content-token ids."""
    texts = ["invoice Invoice", "café CAFÉ", "mañana mañana"]
    got = _run_child({"operation": "tokenize", "texts": texts})["tokens"]
    want = [[reference.content_token_id(piece) for piece in text.split(" ")] for text in texts]
    assert got == want


def test_example_assembly_and_supervision():
    """SPEC §5: markers and masks are correct for every admitted turn ordering."""
    for name in ("p01_single_turn", "p02_multi_turn_pair", "p03_multi_turn_interleaved"):
        assert actual(name)["examples"] == expected(name)["examples"]


@pytest.mark.parametrize("name", ["p04_truncate_assistant_prefix", "p05_truncate_multi_turn"])
def test_truncation_keeps_aligned_prefix(name):
    """SPEC §6: over-long token and mask arrays retain the same bounded prefix."""
    assert actual(name)["examples"] == expected(name)["examples"]


def test_drop_rule_and_survivor_input_order():
    """SPEC §6/§9: zero-supervision prefixes are dropped and survivors keep input order."""
    name = "p06_drop_unsupervised"
    assert actual(name)["examples"] == expected(name)["examples"]


def test_packing_order_and_id_tiebreak():
    """SPEC §7: packing ignores input order and breaks equal-length ties by id."""
    name = "p07_pack_exact_boundary"
    got = [b["example_ids"] for b in actual(name)["microbatches"]]
    want = [b["example_ids"] for b in expected(name)["microbatches"]]
    assert got == want


def test_packing_at_exact_boundary():
    """SPEC §7: a candidate joins when padded size equals the budget."""
    batch = actual("p07_pack_exact_boundary")["microbatches"][0]
    cap = panel("p07_pack_exact_boundary")["config"]["max_tokens_per_microbatch"]
    assert len(batch["tokens"]) * len(batch["tokens"][0]) == cap


def test_packing_across_width_jump():
    """SPEC §7: the candidate width participates in the prospective capacity check."""
    name = "p08_pack_width_jump"
    assert [b["example_ids"] for b in actual(name)["microbatches"]] == [
        b["example_ids"] for b in expected(name)["microbatches"]
    ]


@pytest.mark.parametrize("name", ALL_PANELS)
def test_microbatch_token_budget(name):
    """SPEC §7: every materialized micro-batch stays within its padded-token budget."""
    cap = panel(name)["config"]["max_tokens_per_microbatch"]
    for batch in actual(name)["microbatches"]:
        assert len(batch["tokens"]) * len(batch["tokens"][0]) <= cap


def test_padding_and_integer_dtypes():
    """SPEC §6/§7/§9: arrays are int64 and right padding is token/mask zero."""
    name = "p08_pack_width_jump"
    assert all(pair == ["int64", "int64"] for pair in bundle(name)["dtypes"]["examples"])
    assert all(pair == ["int64", "int64"] for pair in bundle(name)["dtypes"]["microbatches"])
    assert actual(name)["microbatches"] == expected(name)["microbatches"]


def test_microbatch_materialization():
    """SPEC §7: row order, width, token padding and mask padding match the reference."""
    for name in ("p07_pack_exact_boundary", "p08_pack_width_jump"):
        assert actual(name)["microbatches"] == expected(name)["microbatches"]


def test_optimizer_step_grouping_and_trailing_step():
    """SPEC §8: consecutive micro-batches form full steps plus a shorter trailing step."""
    name = "p09_ragged_normalization"
    got = [step["microbatch_indices"] for step in actual(name)["steps"]]
    want = [step["microbatch_indices"] for step in expected(name)["steps"]]
    assert got == want
    assert len(got[-1]) == 1


def test_loss_weights_use_supervised_token_counts():
    """SPEC §8: ragged-step weights use supervised-token counts, not batch count."""
    name = "p09_ragged_normalization"
    got = [b["loss_weight"] for b in actual(name)["microbatches"]]
    want = [b["loss_weight"] for b in expected(name)["microbatches"]]
    assert got == pytest.approx(want, rel=RTOL, abs=0.0)


@pytest.mark.parametrize("name", ALL_PANELS)
def test_loss_weights_sum_to_one(name):
    """SPEC §8: loss weights sum to one independently within each optimizer step."""
    run = actual(name)
    for step in run["steps"]:
        total = sum(run["microbatches"][i]["loss_weight"] for i in step["microbatch_indices"])
        assert total == pytest.approx(1.0, rel=RTOL, abs=0.0)


def test_learning_rate_schedule_edges():
    """SPEC §10: warmup/cosine behavior covers W=0, the seam, and T<=W."""
    name = "p10_schedule_span"
    got = [step["lr"] for step in actual(name)["steps"]]
    want = [step["lr"] for step in expected(name)["steps"]]
    assert got == pytest.approx(want, rel=RTOL, abs=0.0)

    cases = [
        {"total_steps": 4, "base_lr": 2e-4, "min_lr": 1e-5, "warmup_steps": 0},
        {"total_steps": 3, "base_lr": 2e-4, "min_lr": 1e-5, "warmup_steps": 5},
    ]
    direct = _run_child({"operation": "schedule", "cases": cases})["rates"]
    expected_rates = [
        reference.learning_rate_schedule(case["total_steps"], case["base_lr"],
                                         case["min_lr"], case["warmup_steps"])
        for case in cases
    ]
    for observed, required in zip(direct, expected_rates):
        assert observed == pytest.approx(required, rel=RTOL, abs=0.0)
