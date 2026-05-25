"""
Minimal apex-inference client using tritonclient.

This is the same code that works against a Triton deployment — the
"drop-in" compatibility claim verified.

    pip install tritonclient[grpc] numpy
    python examples/python_client.py --host localhost:9000 --model resnet50
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import tritonclient.grpc as grpcclient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost:9000")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--output-name", default="output")
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1, 3, 224, 224],
        help="leading 1 + the model's per-request shape",
    )
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    client = grpcclient.InferenceServerClient(args.host)

    if not client.is_server_live():
        raise SystemExit(f"server at {args.host} is not live")
    if not client.is_model_ready(args.model):
        raise SystemExit(f"model {args.model!r} not ready")

    payload = np.random.rand(*args.shape).astype(np.float32)
    inp = grpcclient.InferInput(args.input_name, list(payload.shape), "FP32")
    inp.set_data_from_numpy(payload)

    # Warm-up
    client.infer(args.model, inputs=[inp])

    latencies: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        resp = client.infer(args.model, inputs=[inp])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    out = resp.as_numpy(args.output_name)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    print(
        f"model={args.model}  iters={args.iters}  output_shape={out.shape}  "
        f"p50={p50:.2f}ms  p99={p99:.2f}ms"
    )


if __name__ == "__main__":
    main()
