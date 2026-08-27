# Sandbox image for crucible-bench code benchmarks.
#
#   docker build -t crucible-bench-sandbox -f containers/bench-sandbox.Dockerfile .
#   crucible-bench run --sandbox docker --sandbox-image crucible-bench-sandbox ...
#
# (podman works identically. Nothing is COPYed, so the build context is unused —
# `docker build -t crucible-bench-sandbox - < containers/bench-sandbox.Dockerfile`
# is equivalent and avoids sending one.)
#
# WHY THIS IMAGE EXISTS
#
# EvalPlus's *tests* import numpy — 163 of 164 HumanEval+ problems and all 378
# MBPP+ problems. Under a plain `python:3.12-slim` every one of them dies on
# `import numpy` inside the test block, so the task scores 0% no matter what the
# model wrote. Measured 2026-08-22: a correct HumanEval/0 solution failed with a
# ModuleNotFoundError raised by the harness, not by the solution.
#
# That is the worst kind of wrong number, because it looks exactly like
# catastrophic model failure. crucible-bench probes the image up front and
# refuses to start rather than report it.
#
# WHAT RUNS IN HERE
#
# Code a language model wrote, which is arbitrary code execution by construction.
# The harness runs this image with `--network none`, a read-only rootfs, a 512 MB
# memory cap and `--pids-limit 128`. Keep it minimal — every package added here
# is another thing model-generated code can reach.

FROM python:3.12-slim

# Pinned, not floating. This image is part of the measurement apparatus, so a
# silent dependency bump is a silent change to the conditions a score was
# produced under. Record this tag alongside any number you keep, and bump it
# deliberately rather than rebuilding against whatever is current.
RUN pip install --no-cache-dir numpy==2.4.4

# BigCodeBench's tests import considerably more than numpy (pandas, flask, and
# others besides). Extend rather than edit, so the EvalPlus baseline stays fixed:
#
#     FROM crucible-bench-sandbox
#     RUN pip install --no-cache-dir pandas flask ...
#
# then pass --sandbox-image to whatever you tagged it.
