"""TEMP (remove before merge): why does Kokoro's warm-up fail on macOS CI?"""

import platform
import sys

import numpy as np
import onnxruntime as ort

from voice_agent_next.providers.kokoro import KokoroTTS

print("platform", platform.platform(), platform.machine(), sys.version)
print("onnxruntime", ort.__version__, ort.get_available_providers())
CASES = {
    "auto": None,
    "cpu": ["CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}
model = sys.argv[1] if len(sys.argv) > 1 else "v1.0-int8"
for name, providers in CASES.items():
    if name == "coreml" and "CoreMLExecutionProvider" not in ort.get_available_providers():
        continue
    tts = KokoroTTS(model=model, providers=providers)
    eng = tts._get_engine()
    bad = 0
    for _ in range(10):
        for text in ["Hello.", "Hello! This is Kokoro."]:
            tokens = eng.tokenizer.tokenize(eng.tokenizer.phonemize(text, "en-us"))
            style = eng.get_voice_style("af_heart")[len(tokens)]
            audio, _ = eng._infer(tokens, style, 1.0)
            bad += bool(not np.isfinite(audio).all())
    print(f"{model} {name}: {eng.sess.get_providers()} -> non-finite audio in {bad}/20 runs")
