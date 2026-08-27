"""crucible plan — choose a quantization type for every tensor, under a budget.

`crucible compress` chooses how many experts survive; this chooses how many bits
each surviving tensor gets. Both answer to the same question — what is the best
model that fits a given machine — and they are the two variables nobody else
gets to move together. A quantizer that starts from a fixed checkpoint can only
pick bits; crucible picks the expert count too, so a tier's budget can be spent
on more experts at lower precision or fewer at higher, and the trade can be
measured rather than assumed.

Output is a `--tensor-type-file` plus the `llama-quantize` invocation that
consumes it. This command never runs llama.cpp and never touches weights: it
reads a GGUF's header, reads an importance matrix, solves for an assignment, and
writes a text file. Converting and quantizing stay the user's step, on whatever
machine holds the model.

    crucible plan model-f16.gguf --target mac-mini-m4-16gb \\
        --imatrix model.imatrix -o plan.txt

Named for what it produces. It plans; it does not build — Peonist's Forge is the
thing that builds, and calls this to find out what to ask llama-quantize for.
"""

from __future__ import annotations

import argparse
import sys

NAME = "plan"
HELP = "Plan a per-tensor GGUF quantization for a memory budget or hardware tier"

# Minimum encoding per role. These are a PRIOR, and they are load-bearing.
#
# Per-role energies are not comparable without σ_W (see allocate.py), so nothing
# in the measurement stops the solver from starving a role whose inputs merely
# run at a small scale. Measured on Qwen 3.6 REAP-48 that is not hypothetical:
# unfloored, the search assigned `attn_output` and `ssm_out` 1.75 bpw because
# their imatrix energies sit five orders of magnitude below the shared expert's.
#
# What each floor encodes:
#   attention          the most quantization-sensitive part of a transformer.
#                      Q6_K rather than Q5_K on the exllamav3 finding that in MoE
#                      models specifically, holding self-attn and shared experts
#                      at 6-8 bit is a large KL-divergence win
#   attn_gate/ssm_out  the GDN path's output projections — structurally the same
#                      job as attn_output, and they were the ones destroyed
#   *_shexp            the shared expert, measured here as the three
#                      highest-energy tensors in the whole model
#   output             logit projection; error lands directly on the distribution
#
# A floor is a minimum, not a pin: the search still spends more where the
# per-layer measurement earns it. Replace this table with measured per-role
# KL-divergence when a machine is free to produce one.
STRUCTURAL_FLOORS: dict[str, str] = {
    "attn_q.weight": "Q6_K",
    "attn_k.weight": "Q6_K",
    "attn_v.weight": "Q6_K",
    "attn_qkv.weight": "Q6_K",
    "attn_output.weight": "Q6_K",
    "attn_gate.weight": "Q5_K",
    "ssm_out.weight": "Q5_K",
    "ffn_down_shexp.weight": "Q6_K",
    "ffn_gate_shexp.weight": "Q6_K",
    "ffn_up_shexp.weight": "Q6_K",
    "output.weight": "Q6_K",
    # The embedding table is a lookup, so its error is not amplified through the
    # stack the way output.weight's is — it is the right place to save bytes, and
    # v2's hand analysis moved it from Q8_0 to Q4_K deliberately. But it also has
    # no imatrix coverage (llama-imatrix does not hook it), so it plans at the
    # neutral default and the solver will squeeze it to the ladder floor. Q4_K is
    # where published recipes put it; below that is uncharted, not clever.
    "token_embd.weight": "Q4_K",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("gguf", help="Source GGUF to plan against (f16/bf16 preferred)")
    parser.add_argument(
        "--target", default=None,
        help="Hardware tier preset. Expands to --max-bytes and --backend; any flag "
             "given explicitly wins over the preset. Use --list-tiers to see them.",
    )
    parser.add_argument(
        "--max-bytes", default=None,
        help="Weight budget, e.g. 11GiB / 8.5G / 900MB. Required unless --target "
             "supplies one. Binary and decimal units are distinguished.",
    )
    parser.add_argument(
        "--backend", default=None, choices=("metal", "cuda", "rocm", "cpu"),
        help="Inference backend. Only affects which quant types are admissible — "
             "some are correct but pathologically slow on some hardware.",
    )
    parser.add_argument(
        "--imatrix", default=None,
        help="Importance matrix from llama-imatrix. Strongly recommended: without "
             "it the search cannot tell a load-bearing tensor from a large one.",
    )
    parser.add_argument(
        "--assume-uniform-importance", action="store_true",
        help="Plan without an imatrix, treating every tensor as equally important. "
             "Refused by default because the resulting objective is known to "
             "starve attention to feed experts; passing this also applies "
             "conservative floors to the tensors that failure mode damages.",
    )
    parser.add_argument(
        "--per-layer", action="store_true",
        help="Plan each tensor separately instead of one type per role. Lets the "
             "budget vary layer by layer, which is where the headroom is once "
             "role-level mistakes are fixed. Needs an imatrix to be worth anything.",
    )
    parser.add_argument(
        "--pin", action="append", default=[], metavar="ROLE=TYPE",
        help="Force a role to a type, e.g. --pin ffn_down_shexp.weight=Q8_0. "
             "Repeatable. Honoured even if it breaks the budget — the plan then "
             "reports as not fitting rather than quietly substituting.",
    )
    parser.add_argument(
        "--reserve", default=None, metavar="SIZE",
        help="Hold back this much before planning, e.g. 500MB. For anything that "
             "shares the budget but is not weights — a vision projector, a larger "
             "KV cache than the tier assumed, another process on the box.",
    )
    parser.add_argument(
        "--with-spec", action="store_true",
        help="Reserve room for a speculative-decoding draft head, using the tier's "
             "own figure. On a capacity-bound tier the draft head competes with the "
             "weights it accelerates, so the trade has to be made explicitly: on "
             "Qwen 3.6 35B-A3B, MTP at n=2 measured +24% decode for ~500 MB.",
    )
    parser.add_argument(
        "--sensitivity", default=None, metavar="PATH",
        help="Measured per-role KL-divergence table from "
             "scripts/measure_role_sensitivity.py. This is the term an imatrix "
             "cannot supply — energy clusters by position relative to the nearest "
             "RMSNorm, not by importance. With a table, cross-role allocation "
             "comes from measurement and --no-floors becomes reasonable.")
    parser.add_argument(
        "--no-floors", action="store_true",
        help="Drop the structural per-role minimums. Diagnostic only — without "
             "them the solver will starve roles whose activations merely run at a "
             "small scale (measured: attn_output and ssm_out at 1.75 bpw).")
    parser.add_argument(
        "--raw-energy", action="store_true",
        help="Compare imatrix energies across roles rather than within each role. "
             "Unsound without per-tensor weight scale, and off by default for that "
             "reason; kept for diagnosing what the raw measurement says.")
    parser.add_argument(
        "--floor", default=None, help="Never assign anything smaller than this type")
    parser.add_argument(
        "--ceiling", default=None, help="Never assign anything larger than this type")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Write the --tensor-type-file here (default: stdout with the report)")
    parser.add_argument("--list-tiers", action="store_true", help="List tiers and exit")


def _resolve_budget(args, tier) -> tuple[int, str | None, tuple[str, ...], int]:
    """Fold tier presets and explicit flags into (budget, backend, exclusions, reserved)."""
    from crucible.tiers import parse_size

    budget = parse_size(args.max_bytes) if args.max_bytes else None
    backend = args.backend
    exclude: tuple[str, ...] = ()
    if tier is not None:
        budget = budget if budget is not None else tier.weight_budget_bytes
        backend = backend or tier.backend
        exclude = tier.exclude_types
    if budget is None:
        raise SystemExit("need a budget: pass --max-bytes or --target")

    reserved = parse_size(args.reserve) if args.reserve else 0
    if args.with_spec:
        if tier is None:
            raise SystemExit("--with-spec needs a --target to read the tier's figure "
                             "(or pass --reserve directly)")
        if tier.spec_overhead_bytes == 0:
            print(f"note: {tier.name} budgets nothing for a draft head "
                  f"({tier.bound}-bound), so --with-spec reserves nothing",
                  file=sys.stderr)
        reserved += tier.spec_overhead_bytes

    if reserved >= budget:
        raise SystemExit(f"reserved {reserved} bytes of a {budget}-byte budget — "
                         "nothing left for weights")
    return budget - reserved, backend, exclude, reserved


# What a tensor with no measured importance is planned at. Not 0 (which would
# strip it) and not the maximum (which would gold-plate it) — the neutral weight
# that leaves such a tensor ranked purely by parameter count.
_NEUTRAL_ENERGY = 1.0


def _quantizable(t) -> bool:
    """Whether llama.cpp will actually quantize this tensor.

    Mirrors the refusals in `llama_tensor_get_type` (src/llama-quant.cpp): it
    keeps these F32 no matter what a tensor-type file asks for, so counting them
    as quantizable would make every projected size too small.
    """
    if not t.name.endswith("weight"):
        return False
    if len(t.shape) < 2:
        return False
    return not any(
        s in t.name for s in ("_norm.", "ffn_gate_inp.", "ssm_conv1d", "altup", "laurel")
    )


def _build_groups(gguf, energies, per_layer):
    from crucible.allocate import Group, anchored, regroup_by_role, role_pattern

    quantizable = _quantizable
    groups = []
    for t in gguf.tensors:
        energy = energies.get(t.name, _NEUTRAL_ENERGY)
        if per_layer:
            groups.append(Group(
                key=t.name, pattern=anchored(t.name), n_params=t.n_params,
                n_tensors=1, energy=energy, quantizable=quantizable(t), role=t.role,
            ))
        else:
            groups.append(Group(
                key=t.role, pattern=role_pattern(t.role), n_params=t.n_params,
                n_tensors=1, energy=energy, quantizable=quantizable(t), role=t.role,
            ))
    return groups if per_layer else regroup_by_role(groups)


def _report_spread(result, *, file) -> None:
    """Flag roles allocated far below the file's own average.

    The IQ1_M plan that started all of this was emitted without complaint — it
    took reading the table by eye to notice `attn_output` had been handed 1.75
    bpw next to a shared expert at 6.56. A plan that starves one role to feed
    another is sometimes correct and sometimes a broken objective, and the
    difference is not something the solver can know. Surfacing it is cheap;
    silently emitting it is how a bad file reaches a benchmark.
    """
    from collections import defaultdict

    by_role: dict[str, list[int]] = defaultdict(list)
    bytes_by_role: dict[str, int] = defaultdict(int)
    params_by_role: dict[str, int] = defaultdict(int)
    for c in result.choices:
        role = c.group.role or c.group.key
        by_role[role].append(0)
        bytes_by_role[role] += c.n_bytes
        params_by_role[role] += c.group.n_params

    mean_bpw = result.bpw
    outliers = []
    for role in sorted(by_role):
        if not params_by_role[role]:
            continue
        bpw = bytes_by_role[role] * 8 / params_by_role[role]
        if bpw < mean_bpw / 2:
            outliers.append((role, bpw))

    if outliers:
        print(f"\nWARNING  {len(outliers)} role(s) allocated below half the file average "
              f"({mean_bpw:.2f} bpw):", file=file)
        for role, bpw in sorted(outliers, key=lambda r: r[1]):
            print(f"           {role:32s} {bpw:5.2f} bpw", file=file)
        print("         Check this is intended. Without a --sensitivity table the "
              "solver cannot\n         tell a genuinely cheap role from one whose "
              "activations merely run small.", file=file)


def run(args) -> None:
    from crucible.allocate import plan, render_tensor_type_file
    from crucible.gguf import read_header, read_imatrix
    from crucible.quant_types import ladder
    from crucible.tiers import TIERS, format_size, get

    if args.list_tiers:
        for t in TIERS.values():
            print(f"{t.name:20s} {t.description}")
            print(f"{'':20s}   {t.bound}-bound, {t.backend}, budget "
                  f"{format_size(t.weight_budget_bytes)} of {format_size(t.usable_bytes)} usable")
            if t.exclude_types:
                print(f"{'':20s}   excludes {', '.join(t.exclude_types)}")
        return

    tier = get(args.target) if args.target else None
    budget, backend, exclude, reserved = _resolve_budget(args, tier)

    if args.imatrix is None and not args.assume_uniform_importance:
        raise SystemExit(
            "refusing to plan without an importance matrix.\n"
            "\n"
            "With every tensor weighted equally the objective becomes parameters x\n"
            "distortion, which on an MoE spends the whole budget on the expert stacks\n"
            "and strips attention to make room. That produces a file that looks\n"
            "correctly sized and is measurably worse than the recipe it replaced.\n"
            "\n"
            "  generate one:  llama-imatrix -m model-f16.gguf -f calibration.txt \\\n"
            "                     -o model.imatrix --n-gpu-layers 999 -c 4096\n"
            "  calibration:   python scripts/gen_imatrix_data.py -o calibration.txt\n"
            "\n"
            "Or pass --assume-uniform-importance to plan blind with conservative floors."
        )

    gguf = read_header(args.gguf)
    energies = read_imatrix(args.imatrix) if args.imatrix else {}
    if args.imatrix and not energies:
        raise SystemExit(f"{args.imatrix}: no usable entries — was it written for this model?")

    floors = {} if args.no_floors else dict(STRUCTURAL_FLOORS)
    pins: dict[str, str] = {}
    for spec in args.pin:
        if "=" not in spec:
            raise SystemExit(f"--pin wants ROLE=TYPE, got {spec!r}")
        role, _, qtype = spec.partition("=")
        pins[role.strip()] = qtype.strip().upper()

    if energies:
        uncovered = sorted({
            t.role for t in gguf.tensors
            if _quantizable(t) and t.name not in energies
        })
        if uncovered:
            print(
                "note: no importance data for "
                + ", ".join(uncovered)
                + f" — planning them at the neutral default ({_NEUTRAL_ENERGY}). "
                  "llama-imatrix does not hook every tensor; pin these explicitly "
                  "if the default misplaces them.",
                file=sys.stderr,
            )

    table = None
    if args.sensitivity:
        from crucible import sensitivity as sens

        table = sens.load(args.sensitivity)

    groups = _build_groups(gguf, energies, args.per_layer)
    if table is not None:
        from dataclasses import replace as _replace

        groups = [_replace(g, sensitivity=table.coefficient(g.role)) for g in groups]
        unmeasured = sorted({g.role for g in groups if g.role not in table.roles})
        if unmeasured:
            print(f"note: no sensitivity measured for {', '.join(unmeasured)} — "
                  "planning them at coefficient 1.0", file=sys.stderr)
    if energies and not args.raw_energy:
        from crucible.allocate import normalize_energy_per_role

        groups = normalize_energy_per_role(groups)
    # A floor naming a role the model does not have is a typo, not a no-op.
    present = {g.role for g in groups} | {g.key for g in groups}
    floors = {k: v for k, v in floors.items() if k in present}
    try:
        rungs = ladder(have_imatrix=bool(energies), exclude=exclude,
                       floor=args.floor, ceiling=args.ceiling)
        result = plan(groups, rungs, budget, pins=pins, floors=floors)
    except ValueError as exc:
        # A bad --pin or an over-filtered ladder is user error, not a crash.
        raise SystemExit(str(exc)) from exc

    where = f"tier {tier.name}" if tier else "explicit budget"
    print(f"model      {args.gguf}", file=sys.stderr)
    print(f"target     {where}, {format_size(budget)} for weights"
          + (f" (after reserving {format_size(reserved)})" if reserved else "")
          + (f", {backend}" if backend else ""), file=sys.stderr)
    scope = "raw (cross-role, unsound)" if args.raw_energy else "per-role normalized"
    print(f"importance {args.imatrix or 'UNIFORM (blind)'}"
          + (f", {scope}" if energies else "")
          + (f", {len(floors)} structural floors" if floors else ", NO FLOORS"),
          file=sys.stderr)
    print(f"sensitivity {table.unmeasured_note if table else 'UNMEASURED (floors are a prior)'}",
          file=sys.stderr)
    print(f"planned    {format_size(result.total_bytes)} "
          f"({result.bpw:.3f} bpw over {result.total_params / 1e9:.2f}B quantized params)",
          file=sys.stderr)
    if not result.fits:
        over = result.total_bytes - budget
        print(f"WARNING    over budget by {format_size(over)} — pins cannot be satisfied "
              f"within {format_size(budget)}", file=sys.stderr)
    print("", file=sys.stderr)
    for c in sorted(result.choices, key=lambda c: -c.n_bytes):
        tag = ""
        if c.group.key in pins or c.group.role in pins:
            tag = " (pinned)"
        elif c.group.role in floors and c.quant.name == floors[c.group.role]:
            tag = " (at floor)"
        print(f"  {c.group.key:32s} {c.quant.name:8s} {format_size(c.n_bytes):>11s}"
              f"  x{c.group.n_tensors}{tag}", file=sys.stderr)

    _report_spread(result, file=sys.stderr)

    body = render_tensor_type_file(result)
    if args.output:
        with open(args.output, "w") as f:
            f.write(body)
        # Provenance goes beside the file: llama-quantize's parser rejects any
        # token that is not `pattern=TYPE`, comments included.
        with open(f"{args.output}.meta", "w") as f:
            f.write(f"crucible plan — {where}, budget {format_size(budget)}\n"
                    f"planned {format_size(result.total_bytes)} at {result.bpw:.3f} bpw\n"
                    f"imatrix: {args.imatrix or 'none'}\n")
        print(f"\nwrote {args.output} (+ .meta)", file=sys.stderr)
        print(f"\n  llama-quantize --imatrix {args.imatrix or '<imatrix>'} \\\n"
              f"      --tensor-type-file {args.output} \\\n"
              f"      {args.gguf} out.gguf Q4_K_M", file=sys.stderr)
    else:
        print(body)
