#!/usr/bin/env python3
"""Benchmark the SM103 top-level 4-bit GEMM dispatch alternatives.

The benchmark calls the custom GEMV kernel and dequantize-plus-matmul fallback
directly, alternating their measurement order within each round. For example,
the crossover grid used to calibrate the SM103 dispatch can be reproduced with:

    python benchmarking/gemm_4bit_sm103_dispatch.py \
      --grid crossover --warmup 20 --repetitions 100 --rounds 7 \
      --build-label native-sm103 --output results/sm103-crossover.jsonl
"""

import argparse
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess

import torch

import bitsandbytes
from bitsandbytes.backends.cuda.ops import (
    _dequant_linear_fallback,
    _gemm_4bit_kernel_impl,
    _gemm_4bit_use_custom_cuda,
)
from bitsandbytes.functional import quantize_4bit

PILOT_CASES = (
    ("square_4096", 4096, 4096, (1, 8, 16, 32, 64, 128)),
    ("square_7168", 7168, 7168, (1, 8, 16, 32, 64, 128)),
    ("square_8192", 8192, 8192, (1, 8, 16, 32, 64, 128)),
    ("wide_8192x4096", 8192, 4096, (1, 8, 16, 32, 64, 128)),
    ("tall_8192x11008", 8192, 11008, (1, 8, 16, 32, 64, 128)),
    ("wide_11008x4096", 11008, 4096, (1, 8, 16, 32, 64, 128)),
    ("highwave_32768", 32768, 4096, (1, 16, 32, 64, 128, 256)),
    ("highwave_65536", 65536, 4096, (1, 16, 32, 64, 128, 256)),
)

CROSSOVER_CASES = (
    ("square_6144", 6144, 6144, (5, 6, 7, 8, 16, 28, 31, 32, 33, 36)),
    ("square_7168", 7168, 7168, (5, 6, 7, 8, 16, 28, 31, 32, 33, 36)),
    ("square_8192", 8192, 8192, (5, 6, 7, 8, 16, 28, 31, 32, 33, 36)),
    ("wide_8192x4096", 8192, 4096, (8, 16, 28, 31, 32, 33, 36)),
    ("tall_8192x11008", 8192, 11008, (8, 16, 28, 31, 32, 33, 36)),
    ("wide_11008x4096", 11008, 4096, (8, 16, 28, 31, 32, 33, 36)),
    ("highwave_32768", 32768, 4096, (32, 33, 48, 56, 64, 72, 80, 96, 112, 128)),
    ("highwave_65536", 65536, 4096, (32, 33, 48, 56, 64, 72, 80, 96, 112, 128)),
    ("highwave_128256", 128256, 4096, (32, 33, 48, 56, 64, 72, 80, 96, 112, 128)),
)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_case(value):
    """Parse NAME:N:K:M1,M2,... into a benchmark case."""
    try:
        name, n_text, k_text, m_text = value.split(":")
        n = positive_int(n_text)
        k = positive_int(k_text)
        m_values = tuple(positive_int(item) for item in m_text.split(","))
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("case must be NAME:N:K:M1,M2,... with positive integer dimensions") from error
    if not name or not m_values:
        raise argparse.ArgumentTypeError("case name and M values must not be empty")
    return name, n, k, m_values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="JSONL output path")
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--repetitions", type=positive_int, default=30)
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--grid", choices=("pilot", "crossover"), default="pilot")
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help="custom NAME:N:K:M1,M2,... case; repeat to override the preset grid",
    )
    parser.add_argument(
        "--dtype",
        action="append",
        choices=tuple(DTYPES),
        help="dtype to measure; repeat as needed (default: fp16 and bf16)",
    )
    parser.add_argument("--blocksize", type=positive_int, default=64)
    parser.add_argument("--quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument(
        "--compress-statistics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--build-label", default="unspecified")
    return parser.parse_args()


def current_commit():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def require_sm103():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires a B300 GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible B300 GPU, found {torch.cuda.device_count()}")
    capability = torch.cuda.get_device_capability(0)
    if capability != (10, 3):
        raise RuntimeError(f"expected SM103 (compute capability 10.3), found {capability}")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if "B300" not in properties.name.upper():
        raise RuntimeError(f"expected an NVIDIA B300, found {properties.name}")
    return properties


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": percentile(samples, 0.10),
        "p90_ms": percentile(samples, 0.90),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def collect_pair(custom, fallback, warmup, repetitions, flip_order):
    for index in range(warmup):
        if (index + flip_order) % 2:
            fallback()
            custom()
        else:
            custom()
            fallback()
    torch.cuda.synchronize()

    custom_events = []
    fallback_events = []
    for index in range(repetitions):
        custom_start = torch.cuda.Event(enable_timing=True)
        custom_end = torch.cuda.Event(enable_timing=True)
        fallback_start = torch.cuda.Event(enable_timing=True)
        fallback_end = torch.cuda.Event(enable_timing=True)
        if (index + flip_order) % 2:
            fallback_start.record()
            fallback()
            fallback_end.record()
            custom_start.record()
            custom()
            custom_end.record()
        else:
            custom_start.record()
            custom()
            custom_end.record()
            fallback_start.record()
            fallback()
            fallback_end.record()
        custom_events.append((custom_start, custom_end))
        fallback_events.append((fallback_start, fallback_end))
    torch.cuda.synchronize()
    custom_samples = [start.elapsed_time(end) for start, end in custom_events]
    fallback_samples = [start.elapsed_time(end) for start, end in fallback_events]
    return custom_samples, fallback_samples


def quantized_arguments(weight, args):
    packed, state = quantize_4bit(
        weight,
        blocksize=args.blocksize,
        compress_statistics=args.compress_statistics,
        quant_type=args.quant_type,
    )
    if state.nested:
        positional = (packed, list(weight.shape), state.state2.absmax, args.blocksize, args.quant_type)
        keyword = {
            "absmax_8bit": state.absmax,
            "absmax_code": state.state2.code,
            "absmax_offset": state.offset,
        }
    else:
        positional = (packed, list(weight.shape), state.absmax, args.blocksize, args.quant_type)
        keyword = {}
    return packed, state, positional, keyword


def benchmark_cell(args, case_name, n, k, m, dtype, positional, keyword, num_sms):
    activation = torch.randn(m, k, device="cuda", dtype=dtype)

    def custom():
        return _gemm_4bit_kernel_impl(activation, *positional, **keyword)

    def fallback():
        return _dequant_linear_fallback(activation, *positional, **keyword)

    custom_result = custom()
    fallback_result = fallback()
    torch.cuda.synchronize()
    tolerance = 0.02 if dtype == torch.float16 else 0.08
    torch.testing.assert_close(custom_result, fallback_result, rtol=0.02, atol=tolerance)
    difference = custom_result.float() - fallback_result.float()
    max_abs_error = difference.abs().max().item()
    error_rms = difference.square().mean().sqrt().item()
    reference_rms = fallback_result.float().square().mean().sqrt().item()
    relative_rms_error = error_rms / reference_rms if reference_rms else 0.0
    del custom_result, fallback_result, difference

    custom_samples = []
    fallback_samples = []
    custom_round_medians = []
    fallback_round_medians = []
    for round_index in range(args.rounds):
        custom_round, fallback_round = collect_pair(
            custom,
            fallback,
            args.warmup,
            args.repetitions,
            round_index,
        )
        custom_samples.extend(custom_round)
        fallback_samples.extend(fallback_round)
        custom_round_medians.append(statistics.median(custom_round))
        fallback_round_medians.append(statistics.median(fallback_round))

    custom_stats = summarize(custom_samples)
    fallback_stats = summarize(fallback_samples)
    custom_over_fallback = custom_stats["median_ms"] / fallback_stats["median_ms"]
    record = {
        "type": "measurement",
        "case": case_name,
        "n": n,
        "k": k,
        "m": m,
        "dtype": str(dtype).removeprefix("torch."),
        "baseline_dispatch_custom": _gemm_4bit_use_custom_cuda(
            0,
            dtype,
            m,
            n,
            k,
        ),
        "custom": custom_stats,
        "fallback": fallback_stats,
        "custom_over_fallback": custom_over_fallback,
        "winner_at_5_percent": (
            "custom" if custom_over_fallback < 0.95 else "fallback" if custom_over_fallback > 1.05 else "noise"
        ),
        "correctness": {
            "max_abs_error": max_abs_error,
            "relative_rms_error": relative_rms_error,
        },
        "custom_round_medians_ms": custom_round_medians,
        "fallback_round_medians_ms": fallback_round_medians,
        "custom_samples_ms": custom_samples,
        "fallback_samples_ms": fallback_samples,
        "num_sms": num_sms,
    }
    return record


def main():
    args = parse_args()
    properties = require_sm103()
    cases = tuple(args.case) if args.case else (PILOT_CASES if args.grid == "pilot" else CROSSOVER_CASES)
    dtype_names = tuple(dict.fromkeys(args.dtype or tuple(DTYPES)))

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "type": "metadata",
        "device": properties.name,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": current_commit(),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "num_sms": properties.multi_processor_count,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_arch_list": torch.cuda.get_arch_list(),
        "bitsandbytes_version": bitsandbytes.__version__,
        "bitsandbytes_native_library": str(getattr(getattr(bitsandbytes.cextension.lib, "_lib", None), "_name", None)),
        "grid": "custom" if args.case else args.grid,
        "cases": [{"name": name, "n": n, "k": k, "m_values": list(m_values)} for name, n, k, m_values in cases],
        "dtypes": list(dtype_names),
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "rounds": args.rounds,
        "blocksize": args.blocksize,
        "quant_type": args.quant_type,
        "compress_statistics": args.compress_statistics,
        "seed": args.seed,
        "build_label": args.build_label,
    }

    with args.output.open("w", encoding="utf-8") as output:
        line = json.dumps(metadata, sort_keys=True)
        print(line, flush=True)
        output.write(line + "\n")
        for case_name, n, k, m_values in cases:
            for dtype_name in dtype_names:
                dtype = DTYPES[dtype_name]
                weight = torch.empty((n, k), device="cuda", dtype=dtype)
                weight.normal_(mean=0.0, std=k**-0.5)
                packed, state, positional, keyword = quantized_arguments(weight, args)
                del weight
                for m in m_values:
                    record = benchmark_cell(
                        args,
                        case_name,
                        n,
                        k,
                        m,
                        dtype,
                        positional,
                        keyword,
                        properties.multi_processor_count,
                    )
                    line = json.dumps(record, sort_keys=True)
                    print(line, flush=True)
                    output.write(line + "\n")
                    output.flush()
                del packed, state, positional, keyword
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
