#!/bin/bash
# Run from the repository root. Docker is required for the reward checks.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FAILED=0
IMAGE="dynamo-repair-sft-trainer-validate-$$"
ACTIVE_CONTAINERS=""
TMP="$(mktemp -d)"

if python3 -c 'import sys' >/dev/null 2>&1; then
  DYNAMO_PYTHON=python3
elif python -c 'import sys' >/dev/null 2>&1; then
  DYNAMO_PYTHON=python
else
  echo "FAIL: Python 3.11 or newer is required for local validation"
  exit 1
fi

step() { printf '\n=== %s ===\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1"; FAILED=1; }
cleanup() {
  for container in $ACTIVE_CONTAINERS; do
    docker rm -f "$container" >/dev/null 2>&1 || true
  done
  docker image rm -f "$IMAGE" >/dev/null 2>&1 || true
  rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

step "layout, manifest and taxonomy"
"$DYNAMO_PYTHON" - <<'PY' || FAILED=1
import pathlib
import re
import sys
import tomllib

required = {
    "task/task.toml", "task/instruction.md", "task/environment/Dockerfile",
    "task/environment/data/SPEC.md", "task/solution/solve.sh",
    "task/tests/test.sh", "task/tests/test_outputs.py", "task/tests/reference.py",
    "references/check-base-image.sh", "references/diversity-taxonomy.toml",
    "references/dynamo-rubric.toml", "tools/validate.sh",
}
missing = sorted(path for path in required if not pathlib.Path(path).is_file())
if missing:
    print("missing required files:", missing)
    raise SystemExit(1)
if pathlib.Path("task/tests/Dockerfile").exists():
    print("tests/Dockerfile is forbidden in the shared-image layout")
    raise SystemExit(1)
if pathlib.Path("task/README.md").exists():
    print("task/README.md duplicates the root reviewer README")
    raise SystemExit(1)

manifest = tomllib.loads(pathlib.Path("task/task.toml").read_text())
taxonomy = tomllib.loads(pathlib.Path("references/diversity-taxonomy.toml").read_text())
meta = manifest["metadata"]
norm = lambda value: value.lower().replace(" ", "_").replace("-", "_")
ok = True

category = norm(meta["category"])
categories = {norm(key): value for key, value in taxonomy["categories"].items()}
if category not in categories:
    print("invalid category:", meta["category"]); ok = False
if norm(meta["subcategory"]) not in {norm(value) for value in categories.get(category, [])}:
    print("invalid subcategory:", meta["subcategory"]); ok = False
for field in ("task_objective", "artifact_type"):
    allowed = {norm(value) for value in taxonomy[field]}
    if not meta[field] or any(norm(value) not in allowed for value in meta[field]):
        print("invalid", field, meta[field]); ok = False

name = manifest["task"]["name"]
if not re.fullmatch(r"[A-Za-z0-9_.-]+/[a-z0-9]+(?:-[a-z0-9]+){0,2}", name):
    print("invalid task name:", name); ok = False
if not manifest.get("artifacts"):
    print("artifacts is empty"); ok = False
if "/app/repro.py" in manifest["artifacts"]:
    print("immutable repro.py must not be declared as agent-produced"); ok = False
for field in ("difficulty_explanation", "solution_explanation", "verification_explanation"):
    if not meta.get(field, "").strip():
        print(field, "is empty"); ok = False

tests = pathlib.Path("task/tests/test_outputs.py").read_text()
test_count = len(re.findall(r"^def test_", tests, flags=re.MULTILINE))
panel_count = len(list(pathlib.Path("task/tests/panels").glob("p*.json")))
if test_count != 16:
    print(f"expected 16 test functions, found {test_count}"); ok = False
if panel_count != 10:
    print(f"expected 10 panels, found {panel_count}"); ok = False

print("manifest and layout OK" if ok else "manifest or layout failed")
sys.exit(0 if ok else 1)
PY

step "base image and dependency pins"
bash references/check-base-image.sh task || fail "base image policy failed"
"$DYNAMO_PYTHON" - <<'PY' || FAILED=1
from pathlib import Path
text = Path("task/environment/Dockerfile").read_text()
required = ("numpy==2.3.2", "pytest==8.4.1", "pytest-json-ctrf==0.3.5")
missing = [item for item in required if item not in text]
if missing:
    print("missing dependency pins:", missing)
    raise SystemExit(1)
print("dependency pins OK")
PY

step "environment and verifier hygiene"
if grep -qiE '^[[:space:]]*COPY[[:space:]].*(solution|tests)' task/environment/Dockerfile; then
  fail "Dockerfile copies solution or tests"
fi
if grep -qiE '(apt-get|pip install|curl|uvx)' task/tests/test.sh; then
  fail "test.sh installs or downloads verifier tooling"
fi
"$DYNAMO_PYTHON" -m py_compile task/tests/test_outputs.py task/tests/reference.py \
  task/environment/data/sfttrainer/*.py task/solution/*.py || fail "Python syntax check failed"

step "container build"
if ! command -v docker >/dev/null 2>&1; then
  fail "Docker is required; reward checks were not run"
elif ! docker info >/dev/null 2>&1; then
  fail "Docker daemon is unavailable; reward checks were not run"
elif ! docker build --pull=false -t "$IMAGE" task/environment; then
  fail "environment image build failed"
else
  step "fixture dormancy and panel behavior"
  dev_container="dynamo-dev-checks-$$"
  ACTIVE_CONTAINERS="$ACTIVE_CONTAINERS $dev_container"
  if docker create --name "$dev_container" "$IMAGE" \
      bash -lc 'python /tools/check_dormancy.py && python /tools/check_panels.py' \
      >/dev/null \
      && docker cp tools/. "$dev_container:/tools" >/dev/null \
      && docker cp task/. "$dev_container:/task" >/dev/null \
      && docker start -a "$dev_container"; then
    :
  else
    fail "fixture dormancy or panel behavior failed"
  fi

  run_case() {
    label="$1"
    expected_reward="$2"
    apply_oracle="$3"
    container="dynamo-${label}-$$"
    ACTIVE_CONTAINERS="$ACTIVE_CONTAINERS $container"

    if [ "$apply_oracle" = "yes" ]; then
      command='bash /oracle/solve.sh && mkdir -p /tests && cp -a /verifier-src/. /tests/ && bash /tests/test.sh'
    else
      command='mkdir -p /tests && cp -a /verifier-src/. /tests/ && bash /tests/test.sh'
    fi

    docker create --name "$container" "$IMAGE" bash -lc "$command" >/dev/null || return 1
    docker cp task/tests/. "$container:/verifier-src" >/dev/null || return 1
    if [ "$apply_oracle" = "yes" ]; then
      docker cp task/solution/. "$container:/oracle" >/dev/null || return 1
    fi

    docker start -a "$container"
    status=$?
    reward_file="$TMP/${label}-reward.txt"
    docker cp "$container:/logs/verifier/reward.txt" "$reward_file" >/dev/null 2>&1 || return 1
    reward="$(tr -d '[:space:]' < "$reward_file")"
    printf '%s: exit=%s reward=%s\n' "$label" "$status" "$reward"

    if [ "$reward" != "$expected_reward" ]; then
      return 1
    fi
    if [ "$expected_reward" = "1" ]; then
      [ "$status" -eq 0 ]
    else
      [ "$status" -ne 0 ]
    fi
  }

  step "no-op reward"
  run_case no-op 0 no || fail "no-op did not fail with reward 0"
  step "oracle reward"
  run_case oracle 1 yes || fail "oracle did not pass with reward 1"
fi

printf '\n'
if [ "$FAILED" -eq 0 ]; then
  echo "validate.sh: all checks passed"
else
  echo "validate.sh: FAILURES above"
fi
exit "$FAILED"
