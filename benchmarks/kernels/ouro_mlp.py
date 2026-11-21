# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph microbenchmark of split vs packed BF16 Ouro-1.4B MLPs.

Run with a local, single-file checkpoint:
    python -m benchmarks.kernels.ouro_mlp /path/to/Ouro-1.4B/model.safetensors
"""

import argparse
import json
from statistics import median

import torch
from safetensors import safe_open

from vllm_rlt.kernels._extension import load_extension
from vllm_rlt.models.ouro import OuroConfig, OuroMLP, PackedOuroMLP


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
    parser.add_argument("checkpoint")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 3, 5, 8, 512, 2048])
    args = parser.parse_args()
    load_extension()
    torch.manual_seed(713)
    layers = []
    with safe_open(args.checkpoint, framework="pt", device="cpu") as weights:
        for idx in range(4):
            with torch.device("meta"):
                layer = OuroMLP(OuroConfig())
            state = {
                key: weights.get_tensor(f"model.layers.{idx}.mlp.{key}").to(
                    device="cuda", dtype=torch.bfloat16
                )
                for key in layer.state_dict()
            }
            layer.load_state_dict(state, assign=True)
            layers.append(layer.eval())
    packed = [PackedOuroMLP(layer) for layer in layers]
    for rows in args.rows:
        hidden = torch.randn(
            rows, layers[0].gate_proj.in_features, device="cuda", dtype=torch.bfloat16
        )
        error = (layers[0](hidden).float() - packed[0](hidden).float()).abs()
        graphs = [capture(group, hidden) for group in (layers, packed)]
        samples = [[], []]
        for repeat in range(10):
            for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                samples[arm].append(measure(graphs[arm], 50 if rows <= 8 else 5, len(layers)))
        print(
            json.dumps(
                dict(
                    rows=rows,
                    split_us=median(samples[0]),
                    packed_us=median(samples[1]),
                    samples_us=samples,
                    max_abs=error.max().item(),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
