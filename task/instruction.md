`/app/sfttrainer` prepares supervised fine-tuning batches. `/app/SPEC.md` is the
normative specification of its behaviour: the package must conform to the specification
for every input the specification admits.

`python /app/repro.py` currently fails. Make `/app/sfttrainer` conform to `/app/SPEC.md`,
changing only files under `/app/sfttrainer/`.

Leave `/app/SPEC.md`, `/app/repro.py` and `/app/fixtures/` unchanged.
