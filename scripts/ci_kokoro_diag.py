"""TEMP (remove before merge): why does Kokoro int8 produce NaN audio on macOS arm64?"""

import platform
import sys

import numpy as np
import onnxruntime as ort

from voice_agent_next.providers.kokoro import KokoroTTS

print("platform", platform.platform(), platform.machine(), sys.version)
print("onnxruntime", ort.__version__, ort.get_available_providers())
model = sys.argv[1] if len(sys.argv) > 1 else "v1.0-int8"
path = KokoroTTS(model=model, providers=["CPUExecutionProvider"])._get_engine().sess._model_path
engine = KokoroTTS(model=model, providers=["CPUExecutionProvider"])._get_engine()
level = ort.GraphOptimizationLevel


def options(**kw: object) -> ort.SessionOptions:
    o = ort.SessionOptions()
    o.log_severity_level = 3
    for k, v in kw.items():
        setattr(o, k, v)
    return o


CASES = {
    "default": options(),
    "1 thread": options(intra_op_num_threads=1),
    "opt basic": options(graph_optimization_level=level.ORT_ENABLE_BASIC),
    "opt none": options(graph_optimization_level=level.ORT_DISABLE_ALL),
    "opt extended": options(graph_optimization_level=level.ORT_ENABLE_EXTENDED),
}
for name, opts in CASES.items():
    sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    engine.sess = sess
    bad = 0
    for _ in range(10):
        for text in ["Hello.", "Hello! This is Kokoro."]:
            tokens = engine.tokenizer.tokenize(engine.tokenizer.phonemize(text, "en-us"))
            style = engine.get_voice_style("af_heart")[len(tokens)]
            audio, _ = engine._infer(tokens, style, 1.0)
            bad += bool(not np.isfinite(audio).all())
    print(f"{model} cpu {name}: non-finite audio in {bad}/20 runs")
