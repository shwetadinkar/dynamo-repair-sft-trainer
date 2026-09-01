#!/bin/bash
# Oracle repair for the four SPEC violations in batching.py and pipeline.py.
set -euo pipefail

cd "$(dirname "$0")"
install -m 644 pipeline.py /app/sfttrainer/pipeline.py
install -m 644 batching.py /app/sfttrainer/batching.py

cd /app
python repro.py
