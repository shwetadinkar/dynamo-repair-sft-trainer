# Single image for both the agent run and the verifier (canonical TB2 — no tests/Dockerfile).
# Harbor overlays tests/ at /tests only at verify time, so never COPY tests/ or solution/.
FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

# Task runtime deps AND verifier deps, pinned, so tests/test.sh installs nothing.
RUN pip install --no-cache-dir \
      numpy==2.3.2 \
      pytest==8.4.1 \
      pytest-json-ctrf==0.3.5

# Agent-visible inputs.
COPY data /app

WORKDIR /app
