"""
Verify the drop-in Triton compatibility claim end-to-end.

Connects via the official `tritonclient.grpc` client (the same library teams
use against NVIDIA Triton today) and runs a `ModelInfer` against the doubler
fixture. If this passes, the "tritonclient-compatible" claim in the README
is real, not aspirational.

Run locally:

    pip install tritonclient[grpc] numpy
    python tests/python/test_tritonclient.py             # uses :9000
    python tests/python/test_tritonclient.py host:port   # custom endpoint
"""

from __future__ import annotations

import sys

import numpy as np
import tritonclient.grpc as grpcclient


def main() -> int:
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "localhost:9000"
    client = grpcclient.InferenceServerClient(endpoint)

    if not client.is_server_live():
        print(f"FAIL: server at {endpoint} is not live", file=sys.stderr)
        return 1
    if not client.is_server_ready():
        print(f"FAIL: server at {endpoint} is not ready", file=sys.stderr)
        return 1
    if not client.is_model_ready("doubler"):
        print("FAIL: model 'doubler' not ready", file=sys.stderr)
        return 1

    payload = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    inp = grpcclient.InferInput("input", list(payload.shape), "FP32")
    inp.set_data_from_numpy(payload)

    resp = client.infer("doubler", inputs=[inp])
    out = resp.as_numpy("output")
    expected = payload * 2.0

    if out is None or not np.allclose(out, expected):
        print(f"FAIL: expected {expected.tolist()}, got {out}", file=sys.stderr)
        return 1

    print(f"OK: doubler({payload.tolist()}) = {out.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
