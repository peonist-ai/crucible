# Sandbox image for the agent probes (crucible_bench.agent_probes).
#
#   docker build -t crucible-bench-agent-probe -f containers/agent-probe-sandbox.Dockerfile .
#
# Separate from bench-sandbox.Dockerfile on purpose. That image is the EvalPlus
# baseline and is pinned so a score stays comparable over time; adding packages
# to it would quietly change the conditions those numbers were produced under.
# This one is free to carry whatever the probe tasks need.
#
# WHY PYTEST
#
# Two of the three bundled tasks (`feature_add`, `recovery`) have test files that
# re-invoke themselves through pytest. Under a plain python:3.12-slim they exit
# non-zero on the missing import and the probe records a failure the model did
# not cause.
#
# WHAT RUNS IN HERE
#
# Code the model wrote during an agent loop, executed via the `run_python` tool
# and again for the final test pass. The harness runs this image with
# `--network none`, a 512 MB memory cap and `--pids-limit 128`. Unlike the
# benchmark sandbox the workspace is mounted READ-WRITE, because editing it is
# the task — the network and resource caps are what contain it.

FROM python:3.12-slim

RUN pip install --no-cache-dir pytest==9.0.3
