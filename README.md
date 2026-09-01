# dynamo/repair-sft-trainer

Reviewer notes for the supervised fine-tuning pipeline repair task.

The shipped reproduction exposes one packing error. Three pipeline errors remain dormant
under the supplied fixtures because those fixtures are short, single-turn and uniform in
length. The task is to audit the package against `SPEC.md`, not merely clear the initial
exception.

The verifier has ten panels. Six distinguish the shipped package from the oracle; four
cover behavior that is already correct and protect against regressions during a rewrite.
`tests/reference.py` derives expected results from the specification without importing
the package under test.

Package code is executed in an unprivileged child process. `tests/test.sh` makes the
verifier and reward directories root-only before pytest starts, so the child receives its
panel through stdin but cannot read `/tests` or write `/logs/verifier`.

Run the complete local gate from the repository root:

```bash
bash tools/validate.sh
```

Docker is required. The gate checks the manifest and controlled taxonomy, the exact base
image and dependency pins, fixture dormancy, panel behavior, and both reward directions
using the real container and `tests/test.sh`:

- shipped package: reward `0`;
- oracle applied: reward `1`.
