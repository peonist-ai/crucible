# Security Policy

## Reporting a vulnerability

Report privately via GitHub's [Report a
vulnerability](https://github.com/peonist-ai/crucible/security/advisories/new)
button, which opens a draft advisory visible only to the maintainers.

Please do not open a public issue for anything that lets someone execute code or
exfiltrate data.

Expect an initial response within a week. This is a small project without a paid
security team, so please size your expectations accordingly — but reports are
genuinely welcome and will be credited unless you ask otherwise.

## Supported versions

Only the `main` branch is supported. There are no long-lived release branches and
no backports.

## What this project actually does

Two parts of Crucible are meaningfully security-relevant, and both are dangerous
by design rather than by accident. Understanding them is more useful than a
generic policy.

### It executes code written by a language model

Coding benchmarks work by running code the model under test generated. That is
arbitrary code execution by construction, and the model is only as trustworthy as
the endpoint you pointed at.

`crucible_bench.testbench.sandbox` runs that code in a container with no network.
The modes are `auto` (prefers podman, then docker, falls back to local execution
**with a warning**), `docker`/`podman` (require that runtime and raise if it is
missing), and `none` (local, no warning).

The container runs with `--network none` (no exfiltration, no callbacks), a
read-only rootfs, a 512 MB memory cap, and `--pids-limit 128` so fork bombs
terminate themselves.

- Use `--sandbox docker` or `--sandbox podman` in CI, and against any model or
  endpoint you do not control. These fail closed.
- `auto` can silently degrade to running generated code as your own user if no
  container runtime is installed. That is a deliberate convenience for local
  work on a model you trust, and a bad default for anything automated.
- `none` is only reasonable for a model you built yourself.

A sandbox escape, or any path that runs generated code outside the container when
a container was requested, is a vulnerability. Please report it.

### It can execute model-repository code

`trust_remote_code` reaches `transformers` and `mlx-lm` from `crucible inspect`,
`crucible quantize`, and `crucible quantize-mlx`. When enabled, loading a model
executes Python that shipped with that model's repository.

It is **off by default and gated behind an explicit flag** in every case. Turning
it on for a checkpoint you did not build is equivalent to running an untrusted
script. Prefer architectures supported natively by the installed `transformers`.

### What is not in scope

- Crucible does not run a network service, so there is no server surface here.
  `crucible-bench` is an HTTP *client*; point it at endpoints you trust.
- Compressed weights inherit their base model's licence and its behaviour. Model
  outputs being wrong, biased, or unsafe is a model-quality question, not a
  vulnerability in this toolkit.
- `contrib/` is unsupported, host-specific material and is not part of the
  security boundary.
