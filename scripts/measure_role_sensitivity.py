"""Measure how much degrading each role moves the model's output distribution.

Produces the table `crucible plan --sensitivity` consumes. This is the term an
importance matrix cannot supply: energy measures how large a tensor's inputs are,
which clusters by position relative to the nearest RMSNorm rather than by
importance, so comparing it across roles is meaningless. Degrading a role and
measuring the result is not.

Method, per role:

    hold every tensor at BASELINE, drop this one role to PROBE, quantize,
    measure KL-divergence against the f16 model, delete the file

The all-BASELINE run is measured once as the reference; each role's contribution
is its KLD minus that. `crucible.sensitivity` converts those deltas into
coefficients.

Lives in scripts/ rather than as a crucible subcommand on purpose: it shells out
to llama-quantize and llama-perplexity, and crucible takes no llama.cpp
dependency. It writes one GGUF at a time and deletes it, so peak disk is one
quantized model, not one per role.

    python scripts/measure_role_sensitivity.py \
        --gguf model-f16.gguf --imatrix model.imatrix.gguf \
        --corpus kld_corpus.txt --logits-base kld-base.logits \
        -o sensitivity.json

CAVEAT worth stating in any writeup that uses the output: sensitivity is measured
with every other role at BASELINE, which is not the mixed allocation a real plan
produces. It is a first-order approximation — far better than a prior, not the
same as exact.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Importable from the repo without installing, and from anywhere else via
# PYTHONPATH (the Halo runs it out of a synced tree with a different layout).
_pkg = Path(__file__).resolve().parent.parent / "packages" / "crucible"
if _pkg.is_dir():
    sys.path.insert(0, str(_pkg))

from crucible.allocate import role_pattern  # noqa: E402
from crucible.gguf import read_header  # noqa: E402

# llama.cpp keeps these F32 whatever a tensor-type file says, so probing them
# measures nothing. Mirrors llama_tensor_get_type in src/llama-quant.cpp.
_NEVER_QUANTIZED = ("_norm.", "ffn_gate_inp.", "ssm_conv1d", "altup", "laurel")


def quantizable_roles(gguf_path: Path) -> dict[str, int]:
    """Role -> total parameter count, for every role llama.cpp will quantize."""
    roles: dict[str, int] = {}
    for t in read_header(gguf_path).tensors:
        if not t.name.endswith("weight") or len(t.shape) < 2:
            continue
        if any(s in t.name for s in _NEVER_QUANTIZED):
            continue
        roles[t.role] = roles.get(t.role, 0) + t.n_params
    return roles


def write_type_file(path: Path, roles: dict[str, int], baseline: str,
                    probe_role: str | None, probe: str) -> None:
    """All roles at `baseline`, except `probe_role` at `probe`.

    No comments: llama-quantize parses this with `while (file >> arg)` and every
    whitespace-separated token must be pattern=TYPE.
    """
    with path.open("w") as f:
        for role in sorted(roles):
            qtype = probe if role == probe_role else baseline
            f.write(f"{role_pattern(role)}={qtype}\n")


def run(cmd: list[str], log: Path) -> str:
    with log.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    out = log.read_text(errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{out[-2000:]}")
    return out


def measure_kld(bin_dir: Path, model: Path, args, log: Path) -> float:
    out = run([
        str(bin_dir / "llama-perplexity"),
        "-m", str(model), "-f", args.corpus,
        "--kl-divergence", "--kl-divergence-base", args.logits_base,
        "--chunks", str(args.chunks),
        # -b and -ub MUST equal -c. Splitting a chunk across ubatches corrupts a
        # quantized hybrid GDN model: measured PPL 1402.95 at the default
        # -ub 512 versus 2.72 with -ub 2048 on identical input.
        "-c", str(args.n_ctx), "-b", str(args.n_ctx), "-ub", str(args.n_ctx),
        "-ngl", "999", "-ctk", "bf16", "-ctv", "bf16",
    ], log)
    m = re.search(r"Mean\s+KLD\s*:\s*([0-9.eE+-]+)", out.replace("\r", "\n"))
    if not m:
        raise SystemExit(f"could not find 'Mean KLD' in {log} — did the run finish?")
    return float(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="f16/bf16 source GGUF")
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--corpus", required=True, help="KLD corpus text")
    ap.add_argument("--logits-base", required=True,
                    help="f16 reference logits from llama-perplexity --kl-divergence-base")
    ap.add_argument("-o", "--output", default="sensitivity.json")
    ap.add_argument("--baseline", default="Q6_K", help="type every role is held at (default Q6_K)")
    ap.add_argument("--probe", default="Q3_K", help="type the probed role drops to (default Q3_K)")
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--bin-dir", default=str(Path.home() / "llama.cpp/build-rocm714/bin"))
    ap.add_argument("--work-dir", default="/tmp/sensitivity")
    ap.add_argument("--only", nargs="*", default=None, help="Probe only these roles")
    args = ap.parse_args()

    bin_dir = Path(args.bin_dir)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)

    roles = quantizable_roles(Path(args.gguf))
    targets = [r for r in sorted(roles) if not args.only or r in args.only]
    print(f"{len(roles)} quantizable roles, probing {len(targets)}", flush=True)

    # Resume: a 90-minute sweep should not restart from zero.
    doc = json.loads(out_path.read_text()) if out_path.exists() else {}
    doc.setdefault("baseline_type", args.baseline)
    doc.setdefault("probe_type", args.probe)
    doc.setdefault("roles", {})
    if doc["baseline_type"] != args.baseline or doc["probe_type"] != args.probe:
        raise SystemExit(f"{out_path} was built with a different baseline/probe pair")

    def quantize_and_measure(probe_role: str | None, label: str) -> float:
        tf = work / f"types-{label}.txt"
        model = work / f"probe-{label}.gguf"
        write_type_file(tf, roles, args.baseline, probe_role, args.probe)
        t0 = time.time()
        try:
            run([str(bin_dir / "llama-quantize"), "--imatrix", args.imatrix,
                 "--tensor-type-file", str(tf), args.gguf, str(model), "Q4_K_M"],
                work / f"quant-{label}.log")
            kld = measure_kld(bin_dir, model, args, work / f"kld-{label}.log")
        finally:
            # ALWAYS, including on failure and on SIGTERM handled below. Each probe
            # model is ~16 GB; a handful of orphans fills the disk and the next run
            # dies for an unrelated-looking reason.
            model.unlink(missing_ok=True)
        print(f"  {label:34s} KLD {kld:.6f}   ({time.time()-t0:.0f}s)", flush=True)
        return kld

    if "baseline_kld" not in doc:
        print(f"baseline: every role at {args.baseline}", flush=True)
        doc["baseline_kld"] = quantize_and_measure(None, "baseline")
        out_path.write_text(json.dumps(doc, indent=2) + "\n")

    failed: list[str] = []
    for role in targets:
        if role in doc["roles"]:
            print(f"  {role:34s} (already measured, skipping)", flush=True)
            continue
        try:
            kld = quantize_and_measure(role, role.replace(".", "_"))
        except SystemExit as exc:
            # One bad role must not cost the other seventeen. Record it and move on;
            # the table loader treats an absent role as neutral, and the summary
            # below names what is missing so it is never silently assumed measured.
            print(f"  {role:34s} FAILED: {exc}", flush=True)
            failed.append(role)
            continue
        doc["roles"][role] = {
            "n_params": roles[role],
            "kld": kld,
            "delta_kld": kld - doc["baseline_kld"],
        }
        out_path.write_text(json.dumps(doc, indent=2) + "\n")

    if failed:
        print(f"\n{len(failed)} role(s) FAILED and are absent from the table: "
              f"{', '.join(failed)}", flush=True)
    print(f"\nwrote {out_path}")
    ranked = sorted(doc["roles"].items(), key=lambda kv: -kv[1]["delta_kld"])
    print(f"\n{'role':34s} {'delta KLD':>12s} {'per 1B params':>14s}")
    for role, e in ranked:
        print(f"{role:34s} {e['delta_kld']:12.6f} {e['delta_kld']/e['n_params']*1e9:14.6f}")


if __name__ == "__main__":
    main()
