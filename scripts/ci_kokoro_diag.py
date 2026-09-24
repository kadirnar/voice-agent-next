"""TEMP (remove before merge): why does Kokoro's warm-up fail on macOS CI?"""

import platform
import sys

import numpy as np
import onnxruntime as ort

from voice_agent_next.providers.kokoro import KokoroTTS

print("platform", platform.platform(), platform.machine(), sys.version)
print("onnxruntime", ort.__version__, ort.get_available_providers())
order = sys.argv[1:] or ["kokoro"]
for step in order:
    if step == "dnsmos":
        import asyncio

        from voice_agent_next.bench.mos import DNSMOS

        async def run() -> None:
            mos = DNSMOS()
            await mos.load()
            t = np.sin(np.arange(48000) / 16000 * 2 * np.pi * 220) * 8000
            from voice_agent_next.audio import AudioFrame

            print("dnsmos", await mos.score(AudioFrame(t.astype(np.int16).tobytes(), 16000, 1)))

        asyncio.run(run())
    elif step == "kokoro":
        tts = KokoroTTS(model="v1.0-int8")
        eng = tts._get_engine()
        print("session providers", eng.sess.get_providers())
        for text in ["Hello.", "Hello! This is Kokoro."]:
            print(repr(text), "phonemes", repr(eng.tokenizer.phonemize(text, "en-us")))
            for trim in (False, True):
                try:
                    a, sr, sp = eng.create_timed(text, "af_heart", trim=trim)
                    peak = float(np.nanmax(np.abs(a))) if len(a) else None
                    print(f"  trim={trim} len={len(a)} absmax={peak} nan={int(np.isnan(a).sum())}")
                except Exception as e:
                    print(f"  trim={trim} ERROR {e!r}")
            tokens = eng.tokenizer.tokenize(eng.tokenizer.phonemize(text, "en-us"))
            style = eng.get_voice_style("af_heart")[len(tokens)]
            out = eng._infer(tokens, style, 1.0) if hasattr(eng, "_infer") else None
            if out is not None:
                audio, dur = out
                print(
                    f"  raw len={len(audio)} nan={int(np.isnan(audio).sum())} "
                    f"absmax={float(np.nanmax(np.abs(audio))) if len(audio) else None} "
                    f"dur={None if dur is None else np.asarray(dur).tolist()}"
                )
