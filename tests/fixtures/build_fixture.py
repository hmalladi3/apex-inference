"""
Build a minimal ONNX model used by integration tests.

The model is a "doubler": output = input * 2.0.
- Input:  {name: "input",  dtype: FP32, shape: [-1, 4]}  (dynamic batch dim)
- Output: {name: "output", dtype: FP32, shape: [-1, 4]}

Run this once to regenerate tests/fixtures/doubler.onnx. The .onnx file is
committed to the repo so CI doesn't need Python to run the Rust e2e test.

Requirements: onnx, numpy (pip install onnx numpy).
"""

from __future__ import annotations

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build() -> onnx.ModelProto:
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [-1, 4]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [-1, 4]
    )
    multiplier = numpy_helper.from_array(
        np.full((1, 4), 2.0, dtype=np.float32),
        name="multiplier",
    )
    mul_node = helper.make_node(
        "Mul", inputs=["input", "multiplier"], outputs=["output"]
    )
    graph = helper.make_graph(
        nodes=[mul_node],
        name="doubler",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[multiplier],
    )
    model = helper.make_model(
        graph,
        producer_name="apex-inference-test",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_path = os.path.join(os.path.dirname(__file__), "doubler.onnx")
    model = build()
    onnx.save(model, out_path)
    size = os.path.getsize(out_path)
    print(f"wrote {out_path} ({size} bytes)")


if __name__ == "__main__":
    main()
