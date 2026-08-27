# Contributing to Crucible

Thanks for looking. This file covers the things about this repo that are not
obvious from reading it, and the two or three rules that CI will enforce whether
or not you knew about them.

## Licensing of contributions

Crucible is Apache-2.0. **There is no CLA to sign.** Apache-2.0 section 5 already
says that anything you deliberately submit for inclusion is licensed under the
same terms as the project, so opening a pull request is all that is required.

If you are porting or adapting code from another project, say so in the pull
request and add the attribution to `NOTICE`. That file exists for code we
redistribute — not for dependencies, which carry their own licenses and are
installed rather than shipped.

## Setup

```sh
uv sync --extra dev
```

That is the whole thing. This is a uv workspace, so it installs both packages in
editable mode along with pytest and ruff. Do not use bare `pip`.

```sh
uv run pytest -q          # 405 tests, all offline, all mocked
uv run ruff check .
uv run ruff format .
```

The unit tests never touch the network and never load a real model. If a test you
write needs either, it belongs somewhere other than the unit suite.

## The one architectural rule

The repo is two packages that **share no imports**:

- `packages/crucible/` — compression. Produces weights. Needs torch.
- `packages/crucible-bench/` — measurement. Talks to an HTTP endpoint. Needs
  neither torch nor transformers.

The only thing crossing that boundary is files: `crucible` writes a model plus
`compression_metadata.json`, `crucible-bench` writes result JSON.

This is load-bearing, not stylistic. It is what lets someone `uv pip install
crucible-bench` on a laptop that only ever talks to a remote server, without
pulling a 2 GB torch wheel. Three tests enforce it and will fail your build:

| Test | Rule |
|---|---|
| `test_registry.py::test_crucible_does_not_benchmark` | `crucible` must not grow a benchmark subcommand |
| `test_testbench.py::test_no_import_of_crucible` | `crucible_bench` must not import `crucible` |
| `test_testbench.py::test_torch_is_not_imported_at_module_scope` | `crucible_bench` must not import torch at module scope |

A benchmark whose grading loop we do not own — Terminal-Bench, SWE-bench — does
not belong in either package. Run it with its own harness and import the result
through `crucible-bench import`.

## Adding things without touching the CLI

Two registries carry the extension points, and an entry declares its own CLI
flags, so neither needs a parser edit:

- **A new model architecture** — one `ModelAttrs` entry in
  `packages/crucible/crucible/models/registry.py`.
- **A new compression or scoring strategy** — one entry in
  `packages/crucible/crucible/methods/registry.py`.

## No silent passes

This is the rule the benchmark harness is built around, and it is the one most
likely to get a pull request sent back.

A check the harness cannot actually perform **raises**. It never scores as
"passed" and never guesses on the model's behalf. Missing dependency, unknown
dataset id, malformed ground truth: preflight it and refuse to start. Unparseable
model output is scored *wrong* and recorded, so a run summary can report how much
of a score rests on formatting rather than knowledge.

Graders are ported from their reference implementation checker by checker, not
reconstructed from memory. An audit in August 2026 found four silent-pass paths
in this repo, and every one of them inflated a compressed model's score. That is
the failure mode: a bug here does not look like a crash, it looks like good news.

## Where host-specific things go

`packages/` and `scripts/` must stay portable — no LAN IPs, no `/Users/...`
paths, no assumptions about a particular GPU.

Anything tied to one machine goes in `contrib/`, which is explicitly unsupported.
`contrib/upstream/` holds standalone reproducers that get pasted into other
projects' issue trackers; some are already published verbatim, so they are
excluded from ruff on purpose. Do not reformat them.

## Running the code benchmarks

Code benchmarks execute code a language model wrote, which is arbitrary code
execution by construction. They run in a container with no network by default.

You need a sandbox image that has numpy, because EvalPlus's *tests* import it —
163 of 164 HumanEval+ problems and all 378 MBPP+ problems. A plain
`python:3.12-slim` scores both tasks 0%, so the harness refuses to run rather
than hand you that zero:

The repo ships one:

```sh
docker build -t crucible-bench-sandbox -f containers/bench-sandbox.Dockerfile .
crucible-bench run --sandbox docker --sandbox-image crucible-bench-sandbox ...
```

`podman` works identically. BigCodeBench needs more than numpy — extend that
image rather than editing it, so the EvalPlus baseline stays fixed.

The agent probes need their own image, because two of their tasks' tests invoke
pytest:

```sh
docker build -t crucible-bench-agent-probe -f containers/agent-probe-sandbox.Dockerfile .
```

## Commit messages

`area: what changed, in lowercase, describing the effect`. Look at `git log` for
the shape. Prefer stating the behaviour that changed over the mechanism:

```
export: stop declaring an MTP head the weights do not have
bench: fix a sandbox mount failure that scored every problem 0%
calib: render the chat template, and stop silently reweighting the mix
```

## Reporting bugs

If it is a security issue, see [SECURITY.md](SECURITY.md) instead.

For a compression or measurement bug, the useful report includes the model, the
method and ratio, and the `compression_metadata.json` from the run. That file is
the replay key — it is enough to regenerate the output weights, so you rarely
need to send anything large.
