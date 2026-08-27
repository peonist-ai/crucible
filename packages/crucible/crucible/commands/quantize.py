"""crucible quantize — second pass: pruned weights to compressed-tensors W4A16.

The output contract for a compressed model. `crucible compress` drops expert
count; this drops precision, and the two are deliberately separate `oneshot`
passes rather than one recipe — llm-compressor's own REAP modifier asserts it is
the only modifier during calibration, and the same holds here: saliency and
Hessians want different forward passes. Save between them (see the ordering note
in `compress`) so a failed quantization never costs the prune.

Emits compressed-tensors safetensors, which vLLM serves natively — including on
ROCm/gfx1151, where compressed-tensors W4A16 is the one layout that reaches the
RDNA-tuned linear kernels. AutoGPTQ-layout and AWQ checkpoints do not: AWQ's only
kernel-selecting class is CUDA-gated, and AutoGPTQ's `qzeros` leak past the
strix-native kernel's symmetric-checkpoint guard.

Requires llm-compressor, which crucible does not depend on — it is a heavyweight
quantization stack, needed only for this command:

    uv pip install "llmcompressor>=0.13" "transformers>=5.9"
"""

from __future__ import annotations

import argparse

NAME = "quantize"
HELP = "Quantize a model to compressed-tensors W4A16 (needs llm-compressor)"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Path to HF model (original or REAP-compressed)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: <model>-w4a16)")
    parser.add_argument("--samples", type=int, default=512,
                        help="Number of calibration WINDOWS of --max-seq-length "
                             "tokens (default: 512). Source conversations are "
                             "tokenized whole, split into windows, shuffled, and "
                             "sampled — so this counts windows, not conversations. "
                             "It is the main driver of both runtime and "
                             "activation-cache size: each window costs ~256 "
                             "sequential per-expert GEMM launches per layer, so "
                             "wall-clock scales with this, NOT with sequence "
                             "length.")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Window size for calibration (default: 4096). Do NOT "
                             "lower this for agentic mixes: the first assistant "
                             "turn alone has a median of 1056 tokens (p75 1706) "
                             "because agent traces open with long system prompts "
                             "and tool definitions, so at 512 only 19%% of windows "
                             "reach any generation content. Since long "
                             "conversations are CHUNKED rather than truncated, "
                             "raising this no longer trades away corpus coverage — "
                             "it trades against memory, which is O(S^2) under "
                             "eager attention (8192 -> ~2.1GB per attention call, "
                             "32768 -> ~34GB, infeasible).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--algorithm", default="gptq", choices=("gptq", "awq", "rtn"),
        help="Quantization algorithm (default: gptq). All three emit "
             "compressed-tensors, so all three reach the RDNA-tuned kernel; "
             "they differ only in quality. AWQ is broken on Qwen3.5/3.6 hybrid "
             "attention in llm-compressor 0.13 — see build_recipe().",
    )
    parser.add_argument(
        "--calibration-profile", default="default",
        choices=["default", "code-only", "general"],
        help="Calibration mix (default: 'default' — the gold-only coding-agent "
             "mix shared with the expert scorer)",
    )
    parser.add_argument(
        "--device-map", default="auto_offload",
        help="Passed to from_pretrained (default: auto_offload). MUST NOT be left "
             "at transformers' default of None on a unified-memory box. "
             "`auto_offload` is compressed-tensors' own option: fill CPU RAM up to "
             "`psutil.available - 5GB`, then spill the remainder to disk. Without "
             "it, loading a 2D-expert checkpoint into a fused-3D module class and "
             "then re-linearizing it back to 2D holds both layouts in RAM at once "
             "— 2x the expert weights, which is what OOM-killed every Ornith run "
             "on 2026-08-15. See --offload-folder.",
    )
    parser.add_argument(
        "--offload-folder", default=None,
        help="Scratch directory for weights that spill to disk under "
             "--device-map auto_offload. Needs room for roughly the whole model.",
    )
    parser.add_argument(
        "--max-memory-gb", type=float, default=None,
        help="Hard cap on how many GB of the model may sit in CPU RAM; the "
             "remainder goes to --offload-folder. SET THIS on a unified-memory "
             "box. Left unset, `auto_offload` computes its own budget of "
             "`psutil.available - 5GB` (compressed_tensors/offload/load.py:120), "
             "which on a 124GB Halo with the seat stopped is ~115GB — larger "
             "than the 66GB model, so accelerate spills NOTHING and the "
             "linearization copy still OOMs. Reserve room for that copy plus "
             "~8.25GB/layer of GPTQ hessians: 45-50 is a safe first try.",
    )
    parser.add_argument(
        "--pad-to-max-length", action="store_true",
        help="Pad every calibration sample to --max-seq-length. OFF by default, "
             "against llm-compressor's own default of True, which is a trap on "
             "this model: the median real sample is ~1056 tokens, so padding to "
             "8192 burns ~8x the compute AND builds the GPTQ hessians partly "
             "from pad tokens (llmcompressor/args/dataset_arguments.py warns "
             "'calibrates on padding tokens'). Measured cost of leaving it on: "
             "a dead-constant 13.67 s/sample, ~1h56m per MoE layer.",
    )
    parser.add_argument(
        "--moe-calibrate-all-experts", action="store_true",
        help="Run EVERY expert over EVERY token instead of only its routed "
             "tokens. Upstream defaults this to True; OFF here. With 256 experts "
             "it multiplies expert compute by ~32x, and its purpose — making sure "
             "no expert starves for samples — is already served by a large "
             "calibration set. Arguably it also mismatches inference: an expert "
             "optimised on tokens it never routes to spends capacity where it "
             "will never be used. Turn it on only if the log warns that modules "
             "received too few tokens.",
    )
    parser.add_argument(
        "--quantize-attention", action="store_true",
        help="Quantize attention projections too. Default is to hold them at "
             "full precision — see crucible.quantize.quant_ignore_patterns().",
    )
    parser.add_argument(
        "--quantize-shared-expert", action="store_true",
        help="Quantize the shared expert too. Default holds it full precision.",
    )
    parser.add_argument(
        "--attn-implementation", default="eager",
        help="Attention backend for the calibration forward passes (default: "
             "eager). NOT the transformers default of sdpa, which is broken here: "
             "torch SDPA on ROCm/gfx1151 raises hipErrorInvalidValue inside "
             "sdpa_attention_forward on this model's full-attention layers. "
             "Deterministic at group 5/41 — layer 3, the first `full_attention` "
             "layer; the other 30 are linear_attention and never call SDPA. "
             "Independent of sequence length. Measured directly "
             "with a standalone probe: F.scaled_dot_product_attention raises "
             "hipErrorInvalidValue at EVERY head_dim tried (64/128/192/256) "
             "while the eager matmul+softmax path succeeds — so torch SDPA is "
             "simply non-functional in this image, not a head_dim limit. vLLM "
             "serves the same model fine because it uses its own attention "
             "kernels, not torch SDPA.",
    )
    parser.add_argument(
        "--probe-load", action="store_true",
        help="Load the model, report peak RSS, and exit before calibration. "
             "The cheap way to prove a memory fix without paying for a full run.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Execute custom modeling code shipped with the checkpoint. This "
             "runs arbitrary Python from the model repo — only enable it for "
             "sources you trust.",
    )


def run(args) -> None:
    import json
    import resource
    import shutil
    import time
    from pathlib import Path

    from crucible.quantize import (
        build_calibration_windows,
        build_recipe,
        quant_ignore_patterns,
        realign_ignore_list,
        require_llmcompressor,
    )

    llmc = require_llmcompressor()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model
    output_dir = args.output or (model_path.rstrip("/") + "-w4a16")

    print(f"Loading model: {model_path}")
    t0 = time.time()
    # load_context() combines offloaded loading with MoE linearization — without
    # it, experts stored as fused 3D tensors (Gemma 4, Qwen 3.5/3.6) are not
    # `Linear` modules and get skipped by the quantizer entirely.
    #
    # device_map matters more than it looks. llm-compressor 0.13 only has 2D load
    # converters for {deepseek_v4, hy_v3, qwen2_moe} (qwen3_moe remaps onto
    # qwen2_moe, so it is covered; qwen3_5_moe is NOT — filed upstream as #3037).
    # For an uncovered arch, `load_quantizable_moe` falls back to
    # load-then-convert: transformers fuses the checkpoint's per-expert 2D weights
    # into 3D, then `linearize_moe` splits them back into 2D Linears. Both layouts
    # are resident at the peak, and the freed 3D pages are not returned to the OS.
    # On a UMA box there is no second pool to absorb that, so `auto_offload` —
    # which spills the overflow to disk — is what keeps the peak survivable.
    load_kwargs = {}
    if args.offload_folder:
        load_kwargs["offload_folder"] = args.offload_folder
    if args.max_memory_gb is not None:
        # Honoured over auto_offload's own estimate — it only fills in max_memory
        # when the caller left it out.
        load_kwargs["max_memory"] = {"cpu": int(args.max_memory_gb * 2**30)}
    with llmc.load_context():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype="auto", trust_remote_code=args.trust_remote_code,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation, **load_kwargs,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=args.trust_remote_code
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"  Peak RSS through load + MoE linearization: {peak_gb:.1f} GiB")
    if args.probe_load:
        print("\n--probe-load set: stopping before calibration.")
        return

    print(f"\nLoading calibration data ({args.samples} samples, "
          f"profile={args.calibration_profile})...")
    windows = build_calibration_windows(
        tokenizer,
        profile=args.calibration_profile,
        num_samples=args.samples,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )

    # Persist the calibration input next to the artifact. GPTQ's hessians cannot be
    # saved (llmcompressor/modifiers/gptq/base.py:134 keeps them in an in-memory
    # PrivateAttr, freed per layer), so the calibration *set* is the only reusable
    # record of what shaped this quant. It is deterministic from
    # (profile, num_samples, seed), but recording it means a later A/B — actorder,
    # attention precision, a different mix — can prove it varied one thing only.
    # Written before quantization so a failed run still leaves the record.
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "calibration_samples.jsonl").open("w") as f:
        for i, text in enumerate(windows.texts):
            f.write(json.dumps({"index": i, "text": text}) + "\n")
    stats = dict(windows.stats, algorithm=args.algorithm)
    with (out / "calibration_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Calibration record -> {out}/calibration_stats.json")
    print(f"    {stats['source_conversations']} conversations "
          f"(median {stats['conv_token_median']} tok, "
          f"p95 {stats['conv_token_p95']}, max {stats['conv_token_max']})")
    print(f"    -> {stats['windows_available']} windows of "
          f"<={args.max_seq_length}; using {stats['windows_used']} "
          f"({stats['coverage_pct']}% of corpus), "
          f"{stats['total_calibration_tokens']:,} calibration tokens, 0 truncated")

    ignore = quant_ignore_patterns(
        keep_attention=not args.quantize_attention,
        keep_shared_expert=not args.quantize_shared_expert,
    )
    recipe = build_recipe(args.algorithm, ignore)

    print(f"\nRunning {args.algorithm.upper()} quantization...")
    print("  Scheme: W4A16 (4-bit weights, 16-bit activations)")
    print(f"  Calibration: {args.samples} samples, {args.max_seq_length} max seq len")
    print(f"  Full precision: {', '.join(ignore)}")
    t0 = time.time()

    llmc.oneshot(
        model=model,
        dataset=windows.dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=stats["windows_used"],
        pad_to_max_length=args.pad_to_max_length,
        moe_calibrate_all_experts=args.moe_calibrate_all_experts,
        data_collator=windows.collator,
        # Hand it the tokenizer we already built. Otherwise `pre_process` re-loads a
        # processor from the model dir and dies with "An error occurred when
        # attempting to initialize model processor" on any checkpoint lacking a full
        # processor config.
        processor=tokenizer,
    )

    print(f"  {args.algorithm.upper()} completed in {time.time() - t0:.1f}s")

    print("\n--- Sanity check ---")
    try:
        from compressed_tensors.offload import dispatch_model
        dispatch_model(model)
        input_ids = tokenizer(
            "def fibonacci(n):", return_tensors="pt",
        ).input_ids.to(model.device)
        output = model.generate(input_ids, max_new_tokens=50)
        print(tokenizer.decode(output[0]))
    except Exception as e:
        print(f"  Sanity check skipped: {e}")

    # Save. Note: for multimodal checkpoints (Qwen3_5Moe*, Gemma4*) loaded through
    # AutoModelForCausalLM, the saved config comes back unwrapped — rewrap it into
    # the multimodal shape or vLLM won't load the result.
    # save_original_format=False is deliberate, not a fallback. transformers
    # otherwise tries to *revert* the 2D->3D expert conversion on the way out, and
    # with offloaded weights spread across shards it cannot:
    #   RuntimeError: We could not revert some weight conversions because of
    #   offloading, and several weights needed for a single conversion operation
    #   living in different shard files.
    # That killed a completed 6.5-hour run on 2026-08-15 at the final step, after
    # all 41 groups had quantized. Saving unreverted is also the *correct* layout
    # here: this checkpoint is natively per-expert 2D on disk (30,720 tensors like
    # `mlp.experts.0.down_proj.weight`) — transformers fuses to 3D at load time, so
    # skipping the revert restores the original on-disk shape rather than departing
    # from it.
    print(f"\nSaving to {output_dir}...")
    t0 = time.time()
    try:
        model.save_pretrained(
            output_dir, save_compressed=True, save_original_format=False,
        )
    except TypeError:
        # Older transformers without the kwarg; smaller shards make the revert
        # more likely to find its tensors co-located.
        print("  save_original_format unsupported; retrying with smaller shards")
        model.save_pretrained(
            output_dir, save_compressed=True, max_shard_size="2GB",
        )
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved in {time.time() - t0:.1f}s")

    fixed = realign_ignore_list(out / "config.json")
    if fixed:
        print(f"  Realigned {fixed} ignore entries to saved tensor names")

    src_processor = Path(model_path) / "processor_config.json"
    if src_processor.exists():
        shutil.copy(src_processor, out / "processor_config.json")
        print("  Copied processor_config.json")

    print("\nDone. This is a compressed-tensors checkpoint — serve it directly:")
    print(f"  vllm serve {output_dir} --dtype bfloat16")
