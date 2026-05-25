"""
End-to-end latency + throughput bench against a running apex-inference.

Connects via tritonclient.grpc, runs N iterations of ModelInfer against the
doubler fixture, and reports p50 / p99 / p99.9 plus aggregate QPS.

Run locally (after starting the server):

    python tests/python/bench.py --host localhost:9000 --iters 2000

Or against a different model:

    python tests/python/bench.py --model resnet50 --input-name input \\
        --shape 1 3 224 224
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import numpy as np
import tritonclient.grpc as grpcclient


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost:9000")
    p.add_argument("--model", default="doubler")
    p.add_argument("--input-name", default="input")
    p.add_argument("--output-name", default="output")
    p.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1, 4],
        help="leading 1 + the model's per-request shape",
    )
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    args = p.parse_args()

    client = grpcclient.InferenceServerClient(args.host)
    if not client.is_server_ready():
        print(f"server at {args.host} not ready", file=sys.stderr)
        return 1
    if not client.is_model_ready(args.model):
        print(f"model {args.model!r} not ready", file=sys.stderr)
        return 1

    payload = np.random.rand(*args.shape).astype(np.float32)
    inp = grpcclient.InferInput(args.input_name, list(payload.shape), "FP32")
    inp.set_data_from_numpy(payload)

    # Warm-up
    for _ in range(args.warmup):
        client.infer(args.model, inputs=[inp])

    latencies_ms: List[float] = []
    started = time.perf_counter()
    for _ in range(args.iters):
        t0 = time.perf_counter()
        client.infer(args.model, inputs=[inp])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    elapsed_s = time.perf_counter() - started

    latencies_ms.sort()
    n = len(latencies_ms)
    p50 = latencies_ms[n // 2]
    p99 = latencies_ms[min(n - 1, int(n * 0.99))]
    p999 = latencies_ms[min(n - 1, int(n * 0.999))]
    qps = args.iters / elapsed_s

    print(f"model:   {args.model}")
    print(f"shape:   {args.shape}")
    print(f"iters:   {args.iters} (warmup {args.warmup})")
    print(f"latency: p50 {p50:.3f} ms · p99 {p99:.3f} ms · p99.9 {p999:.3f} ms")
    print(f"qps:     {qps:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
