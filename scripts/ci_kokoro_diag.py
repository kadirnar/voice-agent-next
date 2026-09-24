"""TEMP (remove before merge): which node of Kokoro int8 turns non-finite on macOS arm64?"""

import platform
import sys

import numpy as np
import onnx
import onnxruntime as ort

from voice_agent_next.providers.kokoro import KokoroTTS

print("platform", platform.platform(), platform.machine(), sys.version)
print("onnxruntime", ort.__version__)
engine = KokoroTTS(model="v1.0-int8", providers=["CPUExecutionProvider"])._get_engine()
path = engine.sess._model_path
model = onnx.shape_inference.infer_shapes(onnx.load(path))
types = {vi.name: vi.type.tensor_type.elem_type for vi in model.graph.value_info}
FLOAT = onnx.TensorProto.FLOAT
skip = {"Constant", "Shape", "Cast", "Identity", "Reshape", "Unsqueeze", "Squeeze", "Gather",
        "Slice", "Concat", "Transpose", "Expand", "ConstantOfShape", "Range"}  # fmt: skip
nodes = [n for n in model.graph.node if n.op_type not in skip]
existing = {o.name for o in model.graph.output}
names = []
for n in nodes:
    for o in n.output:
        if o and types.get(o) == FLOAT and o not in existing:
            model.graph.output.append(onnx.helper.make_tensor_value_info(o, FLOAT, None))
            names.append(o)
print("watching", len(names), "tensors")
opts = ort.SessionOptions()
opts.log_severity_level = 3
sess = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
text = "Hello."
tokens = engine.tokenizer.tokenize(engine.tokenizer.phonemize(text, "en-us"))
style = engine.get_voice_style("af_heart")[len(tokens)]
engine.sess = sess
inputs_of = {}
for n in model.graph.node:
    for o in n.output:
        inputs_of[o] = (n.op_type, n.name, list(n.input))
out_names = [o.name for o in sess.get_outputs()]
for attempt in range(8):
    dt = engine._input_dtypes
    feed = {
        engine._tokens_input: np.array([[0, *tokens, 0]], dtype=dt[engine._tokens_input]),
        "style": np.asarray(style, dtype=dt["style"]),
        "speed": np.array([engine._speed_value(1.0)], dtype=dt["speed"]),
    }
    values = dict(zip(out_names, sess.run(None, feed), strict=True))
    bad = [n for n in names if not np.isfinite(values[n]).all()]
    audio_ok = bool(np.isfinite(values[out_names[0]]).all())
    print(f"attempt {attempt}: audio finite={audio_ok}, {len(bad)} non-finite tensors")
    for b in bad[:40]:
        op, name, ins = inputs_of[b]
        v = np.asarray(values[b], dtype=np.float64)
        print(f"   {op} {name} {v.shape} nan={int(np.isnan(v).sum())} inf={int(np.isinf(v).sum())}")
        for i in ins:
            w = values.get(i)
            if w is not None:
                f = np.asarray(w, dtype=np.float64)
                fin = f[np.isfinite(f)]
                lo, hi = (fin.min(), fin.max()) if fin.size else (None, None)
                print(f"      in {i} {w.shape} finite range=({lo}, {hi}) "
                      f"nonfinite={int((~np.isfinite(f)).sum())}")  # fmt: skip
    if not audio_ok:
        break
