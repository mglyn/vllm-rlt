# SPDX-License-Identifier: Apache-2.0
"""Compare RLT with the official MLP from a trusted local Ouro-1.4B checkpoint.

    python -m benchmarks.kernels.ouro_mlp /path/to/Ouro-1.4B

Executes modeling_ouro.py from that directory and reads model.safetensors.
Use --check-only to skip timing. Any BF16 bit mismatch fails the run.
"""

import argparse
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch
from safetensors import safe_open
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from vllm_rlt.kernels import silu_and_mul
from vllm_rlt.kernels._extension import load_extension
from vllm_rlt.models.ouro import OuroConfig, OuroMLP


def same_bits(actual, expected, context):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{context}: shape or dtype mismatch")
    if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
        count = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
        raise AssertionError(f"{context}: {count}/{expected.numel()} BF16 words differ")


def check_bf16_activation(act_fn):
    # Include signed zero, subnormals and every NaN payload without FP conversions.
    patterns = torch.arange(65536, device="cuda", dtype=torch.int32).short().view(torch.bfloat16)
    up = patterns.expand(128, -1).contiguous()
    for chunk in patterns.split(128):
        gate = chunk[:, None].expand_as(up).contiguous()
        same_bits(silu_and_mul(gate, up), act_fn(gate) * up, "SiLU and multiply")
    print(json.dumps(dict(activation_pairs=65536**2, bit_mismatches=0)), flush=True)


def capture(layers, hidden):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for layer in layers:
                layer(hidden)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for layer in layers:
            layer(hidden)
    return graph


def measure(graph, replays, layer_count):
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (replays * layer_count)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 2, 3, 4, 5, 7, 8, 16, 31, 32, 64, 127, 512, 2048]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 713])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--exhaustive-activation", action="store_true")
    args = parser.parse_args()
    folder = args.checkpoint.resolve(strict=True)
    config = OuroConfig.from_dict(json.loads((folder / "config.json").read_text()))
    hf_mlp = get_class_from_dynamic_module(
        "modeling_ouro.OuroMLP", str(folder), local_files_only=True
    )
    load_extension()
    timing_layers = [[], []]
    checked = 0
    words = 0
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as weights:
        for idx in range(config.num_hidden_layers):
            with torch.device("meta"):
                ref = hf_mlp(SimpleNamespace(**config.to_dict()))
                base, native = OuroMLP(config), OuroMLP(config)
            if args.exhaustive_activation and idx == 0:
                check_bf16_activation(ref.act_fn)
            native.use_native_mlp = True
            state = {
                key: weights.get_tensor(f"model.layers.{idx}.mlp.{key}").to(
                    device="cuda", dtype=torch.bfloat16
                )
                for key in ref.state_dict()
            }
            for layer in (ref, base, native):
                layer.load_state_dict(state, assign=True)
                layer.requires_grad_(False).eval()
            if idx < 4:
                timing_layers[0].append(base)
                timing_layers[1].append(native)
            for seed in args.seeds:
                torch.manual_seed(seed)
                for rows in args.rows:
                    x = torch.randn(rows, config.hidden_size, device="cuda", dtype=torch.bfloat16)
                    expected = ref(x)
                    same_bits(native(x), expected, (idx, rows, seed))
                    checked += 1
                    words += expected.numel()
    print(json.dumps(dict(hf_cases=checked, bf16_output_words=words, bit_mismatches=0)), flush=True)
    if args.check_only:
        return
    for rows in args.rows:
        x = torch.randn(rows, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        graphs = [capture(layers, x) for layers in timing_layers]
        samples = [[], []]
        for repeat in range(10):
            for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                samples[arm].append(
                    measure(graphs[arm], 50 if rows <= 8 else 5, len(timing_layers[arm]))
                )
        print(
            json.dumps(
                dict(
                    rows=rows,
                    base_us=median(samples[0]),
                    native_us=median(samples[1]),
                    samples_us=samples,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
