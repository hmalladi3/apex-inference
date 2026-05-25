"""
Build minimal ONNX models used by integration tests.

Two fixtures, both shape [-1, 4] FP32:
- doubler: output = input * 2.0
- tripler: output = input * 3.0

The pair lets multi-model + reload tests route between distinct models
and verify each one is wired correctly.

Run this once to regenerate the .onnx files. The .onnx files are committed
to the repo so CI doesn't need Python to run the Rust integration tests.

Requirements: onnx, numpy (pip install onnx numpy).
"""

from __future__ import annotations

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(scalar: float, name: str) -> onnx.ModelProto:
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [-1, 4]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [-1, 4]
    )
    multiplier = numpy_helper.from_array(
        np.full((1, 4), scalar, dtype=np.float32),
        name="multiplier",
    )
    mul_node = helper.make_node(
        "Mul", inputs=["input", "multiplier"], outputs=["output"]
    )
    graph = helper.make_graph(
        nodes=[mul_node],
        name=name,
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
    here = os.path.dirname(__file__)
    for name, scalar in [("doubler", 2.0), ("tripler", 3.0)]:
        out_path = os.path.join(here, f"{name}.onnx")
        onnx.save(build(scalar, name), out_path)
        size = os.path.getsize(out_path)
        print(f"wrote {out_path} ({size} bytes)")


if __name__ == "__main__":
    main()
