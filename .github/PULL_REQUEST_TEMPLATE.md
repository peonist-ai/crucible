## What this changes

<!-- The effect, not the mechanism. Same shape as a commit message here. -->

## Why

<!-- Link the issue if there is one. If this is a bug fix, what was the symptom? -->

---

Before marking ready — the four things most likely to send a PR back:

- [ ] `uv run pytest -q` and `uv run ruff check .` both pass locally
- [ ] The package boundary holds: `crucible_bench` imports neither `crucible` nor
      torch at module scope, and `crucible` grew no benchmark subcommand
- [ ] No silent passes: any check the harness cannot actually perform **raises**.
      It is never scored as passed, and never guessed on the model's behalf
- [ ] Nothing host-specific in `packages/` or `scripts/` — no LAN IPs, no
      `/Users/...` paths, no assumption about a particular GPU

If this ports or adapts code from another project, say so here and add the
attribution to `NOTICE`. There is no CLA — Apache-2.0 §5 covers the rest.

<!--
If this changes a measured score, say which benchmark and by how much, and
against which baseline. A compression change that moves a number without a
before/after is very hard to review.
-->
