# faster-whisper (local STT)

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) runs OpenAI's Whisper models on
[CTranslate2](https://github.com/OpenNMT/CTranslate2), locally, on the CPU (int8) or an
NVIDIA GPU (float16). Multilingual models cover 99 languages. It is the default local
multilingual recognizer of voice-agent-next.

| | |
|---|---|
| Spec | `faster_whisper/<model>` (alias `whisper/<model>`), default model `large-v3-turbo` |
| Class | `voice_agent_next.providers.faster_whisper.FasterWhisperSTT` |
| Extra | `pip install 'voice-agent-next[faster-whisper]'` (or `uv sync --extra faster-whisper`) |
| Credentials | none (models are public on the Hugging Face Hub) |
| Capabilities | batch recognizer; behind a VAD: interim results (opt-in), hallucination guard; word timestamps (opt-in); language detection (multilingual models) |
| Platforms | Linux, Windows (CPU, CUDA); macOS (CPU) |

## Usage

```python
from voice_agent_next import create

stt = create("stt", "faster_whisper/small", language="en")
await stt.warmup()  # download (first run) + load + warm-up inference
transcript = await stt.transcribe(frame)  # any sample rate / channel count
print(transcript.text, transcript.language, transcript.confidence)
```

Whisper transcribes complete utterances. In a cascade it needs a VAD: `CascadeEngine`
automatically wraps batch recognizers in `voice_agent_next.stt.StreamAdapter`, which cuts
the input into utterances at VAD end-of-speech and transcribes each one (one
`FINAL_TRANSCRIPT` per utterance, plus `START_OF_SPEECH` / `END_OF_SPEECH` from the VAD).

```yaml
# agent.yaml
stt: {provider: faster_whisper/small, language: en}
vad: silero          # pip install 'voice-agent-next[silero]'
llm: ...
tts: ...
```

The model is loaded once, lazily, in a worker thread (guarded by a lock), and each
transcription runs in `asyncio.to_thread`, so the event loop never blocks. Call
`warmup()` (the cascade's `warmup()` does it for you) so the first user turn does not pay for
the download, the load and kernel initialization.

## Models

Any model faster-whisper knows by name, any CTranslate2 Whisper repository on the Hub
(`faster_whisper/deepdml/faster-whisper-large-v3-turbo-ct2`), or a local directory holding a
converted model (`model="/models/whisper-ct2"`). Downloads go to the Hugging Face cache
(`~/.cache/huggingface/hub`, or `download_root=`).

| Model | Download | Languages | Notes |
|---|---|---|---|
| `tiny` / `tiny.en` | 78 MB | 99 / English | fastest, least accurate |
| `base` / `base.en` | 148 MB | 99 / English | good CPU default for English |
| `small` / `small.en` | 486 MB | 99 / English | best accuracy that is still interactive on a desktop CPU |
| `medium` / `medium.en` | 1.5 GB | 99 / English | GPU recommended |
| `large-v3` | 3.1 GB | 99 | most accurate Whisper; GPU |
| `large-v3-turbo` (`turbo`) — **default** | 1.6 GB | 99 | large-v3 encoder with a 4-layer decoder: close to large-v3 accuracy, much faster; GPU recommended |
| `distil-large-v3.5` | 1.5 GB | English | distilled large-v3 |

`*.en` and `distil-*` models are English-only (`capabilities.language_detection` is `False`).

## Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `large-v3-turbo` | model name, Hub repository or local directory |
| `language` | `None` | language code (`"en"`, `"de"`...; `"en-US"` becomes `"en"`); `None` detects it per utterance |
| `device` | `"auto"` | `"auto"`, `"cpu"` or `"cuda"` (see below) |
| `device_index` | `0` | CUDA device id, or a list of ids to spread concurrent requests |
| `compute_type` | `"auto"` | `float16` on CUDA, `int8` on CPU; or any [CTranslate2 type](https://opennmt.net/CTranslate2/quantization.html) (`int8_float16`, `float32`...) |
| `beam_size` | `1` | greedy decoding for the lowest latency (Whisper's own default is 5) |
| `word_timestamps` | `False` | fill `Transcript.words` (word, start, end, probability) |
| `vad_filter` | `False` | faster-whisper's own Silero VAD pass; off because the pipeline already segments speech |
| `initial_prompt` | `None` | text that primes the decoder (spelling, style, vocabulary) |
| `hotwords` | `None` | hint phrases (names, product terms) |
| `cpu_threads` | `0` | CTranslate2 threads on CPU (0 = its default of 4, or `OMP_NUM_THREADS`) |
| `num_workers` | `1` | model replicas that can transcribe concurrently (several sessions sharing one instance) |
| `download_root` | `None` | model cache directory (default: the Hugging Face cache) |
| `local_files_only` | `False` | never download; `VAN_OFFLINE=1` (and `HF_HUB_OFFLINE=1`) imply it |
| `transcribe_options` | `{}` | extra `WhisperModel.transcribe()` arguments (`temperature`, `no_speech_threshold`, `task`...), applied last |
| `interim_results` | `False` | behind a VAD, re-decode the utterance while the user speaks and emit interim transcripts (see [Partial transcripts](#partial-transcripts)) |
| `interim_interval` | `None` | seconds of new speech between interim decodes; `None` = 0.25 s on CUDA, 0.5 s on the CPU |
| `hallucination_guard` | `True` | drop segments that are probably not speech (see [Hallucination guard](#hallucination-guard)); `False`, or a mapping / `HallucinationGuard` to configure it |
| `final_from_interim` | `False` | when the utterance ends and the latest interim decode heard all of the speech, use it as the final transcript instead of decoding again (see [Final from the interim](#final-from-the-interim)) |

`Transcript.confidence` is `exp(mean token log-probability)` over the utterance, and
`start_time` / `end_time` are relative to the start of the utterance audio.

## Devices and compute types

`device="auto"` uses CUDA when CTranslate2 reports a CUDA device, the CUDA libraries it
opens at run time load, **and** a warm-up inference on it succeeds; otherwise the model runs
on the CPU. A visible GPU is not enough: CTranslate2's pip wheels (Linux and Windows) open
**cuBLAS 12** at run time and do not ship it. Without it the model loads on the GPU and then
fails on the first inference with `Library libcublas.so.12 is not found or cannot be
loaded`. The provider checks this up front (`voice_agent_next.hardware`): when cuBLAS is
missing the model stays on the CPU and one INFO line names the fix; when the GPU fails later
(no kernels for it, out of memory) a warning is logged and the CPU is used. Forcing
`device="cuda"` turns failures into a `ProviderError` during `warmup()` instead of a
fallback. `resolved_device` and `resolved_compute_type` tell you what was picked, and
`van doctor` shows what `"auto"` picks on this machine and why.

To enable the GPU (Linux x86_64, Windows x64), install the `cuda` extra:

```bash
pip install 'voice-agent-next[faster-whisper,cuda]'
```

It installs NVIDIA's cuBLAS 12 wheel, which the library finds in `site-packages` and loads
before CTranslate2 needs it: no `LD_LIBRARY_PATH` or `PATH` changes. A system CUDA 12
toolkit on the library path works too. macOS wheels are CPU-only (int8 runs well on Apple
Silicon); for Metal use an MLX provider. See [hardware.md](../hardware.md).

GPUs newer than the CTranslate2 build (for example the RTX 50xx series) run through the
driver's PTX JIT: the very first CUDA inference on such a machine took about 11 s here while
kernels were compiled, and about 0.25 s afterwards (the driver caches them in
`~/.nv/ComputeCache`). `warmup()` absorbs this cost.

`compute_type="auto"` picks the first type CTranslate2 supports on the device: `float16`,
`int8`, `float32` on CUDA; `int8`, `float32` on CPU. An explicit type is used as given, and a
type the device cannot run raises `ConfigurationError`.

## Latency

Whisper transcribes after the utterance ends (VAD end-of-speech), so the time to the final
transcript is the transcription time below. Some consequences:

* **Short utterances cost almost as much as long ones.** Whisper always encodes a 30 s
  window, so on the CPU below a 2.5 s utterance took 78-91 % of the time of an 11 s one.
  Per-utterance latency matters more than RTF for turn-taking.
* **Set `language` when you know it.** With `language=None`, multilingual models encode the
  audio twice: once to detect the language and once to transcribe (faster-whisper 1.2.1
  does not reuse the first pass). That made `base` and `small` about 2x slower on the CPU
  below.
* `beam_size=1` (the default) is the fastest; `word_timestamps=True` adds an alignment pass.
* In a local stack the LLM and TTS compete for the same CPU cores; cap them with
  `cpu_threads`.

Measured with `FasterWhisperSTT` defaults (`beam_size=1`, no word timestamps) on the 11 s
public-domain JFK clip (RTF = transcription time / audio duration) and on its first 2.5 s
(one typical voice-agent turn); median of 5 runs after `warmup()`, faster-whisper 1.2.1,
CTranslate2 4.8.2, 2026-09-24.

**CPU:** AMD Ryzen 5 5600 (6 cores / 12 threads), `int8`, CTranslate2's default 4 threads. The
machine was busy with other jobs (load average ~7-10), so treat these as upper bounds.

| Model | `language` | 11 s clip | RTF | 2.5 s utterance |
|---|---|---|---|---|
| `tiny.en` | (English-only) | 0.23-0.32 s | 0.021-0.029 | 200-245 ms |
| `base.en` | (English-only) | 0.52-0.55 s | 0.047-0.050 | 430-480 ms |
| `tiny` | `"en"` / detect | 0.31 / 0.45 s | 0.028 / 0.041 | 245 / 365 ms |
| `base` | `"en"` / detect | 0.45 / 0.88 s | 0.041 / 0.080 | 356 / 778 ms |
| `small` | `"en"` / detect | 1.22 / 2.60 s | 0.111 / 0.237 | 1,018 / 2,360 ms |
| `large-v3-turbo` | — | not measured | — | — |

**GPU:** NVIDIA RTX 5070 Ti (16 GB), `float16`, with cuBLAS 12 (the `cuda` extra),
`language="en"`.

| Model | 11 s clip | RTF | 2.5 s utterance |
|---|---|---|---|
| `tiny.en` | 0.041 s | 0.004 | 19 ms |
| `tiny` | 0.063 s | 0.006 | 35 ms |
| `base` | 0.057 s | 0.005 | 42 ms |
| `small` | 0.108 s | 0.010 | 52 ms |
| `large-v3-turbo` | not measured | — | — |

`large-v3-turbo` was not measured: its 1.6 GB download is over the dev machine's download
budget. Its encoder is the large-v3 one, roughly 5-7x the compute of `small`'s, so on a
CPU like this one expect several seconds per utterance: use `base`/`small` (with `language`
set) for CPU agents and keep `turbo` for GPUs.

Reproduce the table with a loop like this (`clip` is the 11 s JFK recording as an
`AudioFrame`):

```python
stt = FasterWhisperSTT(model="base", device="cpu", language="en")
await stt.warmup()
t0 = time.perf_counter()
await stt.transcribe(clip)
rtf = (time.perf_counter() - t0) / clip.duration
```

## Partial transcripts

Whisper has no streaming decoder, but it is fast enough on short audio to re-decode the
utterance so far a few times per second. With `interim_results=True`, the
`StreamAdapter` that the cascade puts in front of the model (or `van bench asr --vad ...`)
emits `INTERIM_TRANSCRIPT` events while the user speaks, each one the transcript of the
whole utterance so far:

```python
stt = create("stt", "faster_whisper/small", language="en", interim_results=True)
engine = CascadeEngine(stt=stt, vad="silero", llm=..., tts=...)
```

```yaml
stt: {provider: faster_whisper/small, language: en, interim_results: true}
vad: silero
```

How the interim decodes are scheduled:

* **Interval.** A decode starts when `interim_interval` seconds of new audio have
  arrived since the last one started: 0.25 s on CUDA, 0.5 s on the CPU by default. The
  first decode of an utterance starts after half an interval of speech.
* **Back-off.** When a decode takes longer than half the interval, the next one waits
  for twice the last decode's duration of new audio instead, so the model is busy at most
  about half the time, whatever the device, model and utterance length.
* **Speech only.** Decodes start only while the VAD's latest window is voiced (at or
  above its activation threshold). Nothing is decoded during pauses and the trailing
  silence, and a decode that starts on the last voiced window has the VAD's
  `min_silence_duration` (0.25 s by default) to finish before the utterance ends.
* **The final transcript comes first.** Only one interim decode runs at a time. When the
  utterance ends, a decode still running is awaited and its result dropped (a CTranslate2
  call cannot be interrupted), then the final transcript is decoded as in batch mode.
  With `num_workers=2` the final runs on the second model replica right away.
  Interim decodes are cheaper than the final: greedy, no timestamps, no temperature
  fallback, no word alignment.

Measured on LibriSpeech test-clean (50 utterances of the `librispeech-test-clean-smoke`
subset, 1.4-24 s), `van bench asr --mode streaming --vad silero`, audio pushed in real
time in 20 ms chunks, `language="en"`, 2026-09-25:

| Model, device | Interims | WER | TTFS p50 / p90 | First partial p50 / p90 | Interim revision rate | Interims per utterance |
|---|---|---|---|---|---|---|
| `base`, CPU int8 | off | 5.11 % | 337 / 508 ms | – | – | – |
| `base`, CPU int8 | on | 5.02 % | 353 / 590 ms | 1,132 / 1,330 ms | 0.145 | 8.7 |
| `base`, CPU int8, `num_workers=2` | on | 4.93 % | 357 / 557 ms | 1,139 / 1,320 ms | 0.154 | 8.4 |
| `small`, CUDA float16 | off | 3.96 % | 69 / 243 ms | – | – | – |
| `small`, CUDA float16 | on | 3.96 % | 8 / 82 ms | 766 / 875 ms | 0.127 | 22.8 |
| `small`, CUDA float16, `num_workers=2` | on | 3.96 % | 21 / 94 ms | 767 / 875 ms | 0.127 | 22.8 |

CPU: AMD Ryzen 5 5600 shared with other jobs (load average 3-50 during the runs), so the
CPU rows are noisy. GPU: RTX 5070 Ti.

*First partial* is measured from the start of the audio, which begins with 0.3-0.5 s of
silence in LibriSpeech; the VAD also needs `min_speech_duration` (0.1 s) before it reports
speech. The first interim decode starts after half an interval of speech. *TTFS* is the
time from the end of the audio (`end_input()`) to the final transcript. In this benchmark
the input ends right after the last word, often before the VAD has seen 0.25 s of
silence, so an interim decode can still be running at the flush: on the CPU that adds
16 ms at the median and about 80 ms at p90 (a second model replica does not help on a
CPU: the two decodes share the cores). In a cascade the final is requested after the VAD's
end of speech, when no interim decode is running. On the GPU, TTFS went *down* with
interims: the utterance tail is already decoded when the input ends often enough, and a
GPU that decodes every 0.25 s stays at its high clocks (an idle GPU needs time to ramp
up); treat that as a side effect, not a guarantee. The transcripts are the same: interim
decodes do not change the final transcript.

*TTFS* is the time from the end of the audio (`end_input()`) to the final transcript. The
interim revision rate is the share of interim updates that rewrite words already shown
rather than only adding words (the tail of a growing utterance is often a partial word
that the next decode corrects).

Costs: interim decodes take CPU/GPU time from the rest of a local pipeline (LLM, TTS).
On a CPU prefer `base`/`base.en` or `tiny.en` for interims, and set `interim_interval`
higher when the machine is shared. `num_workers=2` lets the final transcript run next to
an interim decode instead of after it, at the cost of a second copy of the model; it
helps on a GPU, not on a CPU.

### Final from the interim

When the input ends while the VAD still reports speech (a forced flush, the end of a
file, `van bench asr`), an interim decode can still be running, and the final transcript
waits for it before it is decoded (on one model replica). With
`final_from_interim=True`, if no voiced VAD window arrived after the latest interim
decode (in flight or finished) took its audio, that decode has heard the whole
utterance: its transcript becomes the final one and no second decode runs. If voiced
audio arrived after it started, the final is decoded as usual. This applies to every end
of utterance, the VAD's end of speech included.

The final is then an interim-grade decode: greedy, without timestamps or temperature
fallback, and filtered by the guard with the VAD confidence of the moment it started.
`word_timestamps=True` disables the option (interim decodes do not align words). The
stream counts these finals in `finals_from_interim` and still reports `STTMetrics` for
them (the time spent waiting for the decode). Off by default, because it trades final
accuracy (no beam search, no temperature fallback) for one decode less.

## Hallucination guard

Whisper was trained on subtitles, and when a VAD lets noise, breathing or silence through
it tends to produce text anyway: "Thank you.", "Thanks for watching!", "you", or a phrase
repeated until the window ends. `hallucination_guard` (on by default) drops such
segments. A segment is dropped when:

1. it has no words (`"..."`, `"♪"`);
2. it is a known subtitle artifact or video outro in one of 16 languages (see
   [Multilingual phrase lists](#multilingual-phrase-lists)): "Thanks for watching!",
   "Untertitel im Auftrag des ZDF für funk, 2017", "Sous-titres réalisés par la
   communauté d'Amara.org", "Altyazı M.K.", "ご視聴ありがとうございました"...;
3. it is a stock phrase that is also a real answer ("Thank you.", "you", "Bye.",
   "Danke.", "Gracias.", "谢谢"...) **and** there is other evidence of non-speech:
   `no_speech_prob >= 0.2`, `avg_logprob < -0.8`, or a weak VAD;
4. `no_speech_prob >= 0.6` and (`avg_logprob < -1.0` or a weak VAD): Whisper's own
   no-speech rule, which also accepts the VAD's doubt instead of a low log-probability;
5. its gzip compression ratio is above 2.4 (a repetition loop that Whisper's temperature
   fallback did not fix).

Repetition loops inside a kept segment (an n-gram of up to 4 words repeated more than 4
times in a row) are cut back to one occurrence.

"Weak VAD" applies behind a `StreamAdapter`: the mean speech probability of the
utterance's speech windows is below 0.5 (`vad_threshold`), i.e. the VAD itself was
unsure. Energy-based VADs report lower probabilities than neural ones on real speech in
noise; with 0.5 no real utterance of the benchmark below was dropped for a weak VAD. A
user who says "Thank you." clearly keeps their transcript; the same words decoded from a
keyboard click that barely crossed the VAD threshold are dropped. When the whole utterance is dropped,
the adapter still reports the VAD's `START_OF_SPEECH` / `END_OF_SPEECH` but no final
transcript, like any utterance without words.

The guard lives in `voice_agent_next.stt_guard` and is shared with
[`mlx_whisper`](mlx.md#stt-whisper-mlx_whisper). Every threshold is configurable, and
`None` disables a rule:

```python
from voice_agent_next.providers.faster_whisper import FasterWhisperSTT
from voice_agent_next.stt_guard import HallucinationGuard

stt = FasterWhisperSTT(
    model="small",
    hallucination_guard=HallucinationGuard(vad_threshold=0.8, suspects=("you", "thank you")),
)
```

```yaml
stt:
  provider: faster_whisper/small
  hallucination_guard: {no_speech_threshold: 0.5, compression_ratio_threshold: null}
```

`hallucination_guard=False` returns Whisper's output unchanged. Dropped segments are
logged at DEBUG level (`voice_agent_next` logger) with the rule that dropped them.

### Multilingual phrase lists

Texts are compared after normalization: case-folded, accents removed (`ı` folded to `i`),
punctuation replaced by spaces. The lists are per language, in
`stt_guard.ARTIFACTS_BY_LANGUAGE` and `stt_guard.SUSPECTS_BY_LANGUAGE`:

| Language | Artifacts (dropped always, examples) | Suspects (dropped on weak evidence) |
|---|---|---|
| `en` | "Thanks for watching!", "Please subscribe", "Transcription by CastingWords" | "you", "Thank you.", "Bye.", "So", "Hmm" |
| `de` | "Untertitel im Auftrag des ZDF für funk, 2017", "Vielen Dank fürs Zuschauen", "Copyright WDR 2021" | "Danke.", "Vielen Dank.", "Tschüss." |
| `es` | "Subtítulos realizados por la comunidad de Amara.org", "Gracias por ver el video", "Suscríbete al canal" | "Gracias.", "Adiós." |
| `fr` | "Sous-titres réalisés para la communauté d'Amara.org", "Merci d'avoir regardé cette vidéo", "Sous-titrage ST' 501" | "Merci.", "Au revoir." |
| `it` | "Sottotitoli e revisione a cura di QTSS", "Grazie per la visione" | "Grazie.", "Ciao." |
| `pt` | "Legendas pela comunidade Amara.org", "Obrigado por assistir" | "Obrigado.", "Tchau." |
| `nl` | "Ondertiteld door de Amara.org gemeenschap", "Bedankt voor het kijken" | "Bedankt." |
| `pl` | "Napisy stworzone przez społeczność Amara.org", "Dziękuję za oglądanie" | "Dziękuję." |
| `tr` | "Altyazı M.K.", "İzlediğiniz için teşekkürler", "Abone olmayı unutmayın" | "Teşekkürler." |
| `ru` | "Продолжение следует...", "Субтитры сделал DimaTorzok", "Спасибо за просмотр" | "Спасибо." |
| `zh` | "字幕由Amara.org社区提供", "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目", "谢谢观看" | "谢谢", "好" |
| `ja` | "ご視聴ありがとうございました", "チャンネル登録をお願いします" | "ありがとう" |
| `ko` | "시청해주셔서 감사합니다", "구독과 좋아요 부탁드립니다" | "감사합니다" |
| `ar` | "ترجمة نانسي قنقر", "شكرا على المشاهدة" | "شكرا" |
| `el`, `no` | "Ευχαριστώ που παρακολουθήσατε", "Tekstet av Nicolai Winther" | – |

Credit lines whose wording varies (a year, a name) are matched by regular expressions
(`stt_guard.ARTIFACT_PATTERNS`, the guard's `patterns`): any segment mentioning
"Amara.org" or "DimaTorzok", "Untertitel im Auftrag des ZDF/WDR/..., <year>",
"Sous-titres par <name>", "Napisy by <name>", "字幕由…提供".

By default the guard uses every language's lists: Whisper detects the language per
utterance, and an artifact can come out in another language than the user's. The
artifacts are credit lines and outros nobody says as a whole utterance to a voice agent,
so the union costs nothing; a suspect is only dropped with other evidence of non-speech.
To restrict the lists:

```python
from voice_agent_next.stt_guard import HallucinationGuard, artifact_phrases, suspect_phrases

guard = HallucinationGuard(
    artifacts=artifact_phrases("en", "de"),
    suspects=suspect_phrases("en", "de"),
)
```

Sources: the subtitle credits and outros reported in openai/whisper discussions
[#928](https://github.com/openai/whisper/discussions/928),
[#1873](https://github.com/openai/whisper/discussions/1873),
[#2412](https://github.com/openai/whisper/discussions/2412) and
[#2608](https://github.com/openai/whisper/discussions/2608), in
[whisperX #230](https://github.com/m-bain/whisperX/issues/230), and Whisper's outputs on a
noise-only corpus in the
[sachaarbonel/whisper-hallucinations](https://huggingface.co/datasets/sachaarbonel/whisper-hallucinations)
dataset (MIT); [Barański et al., ICASSP 2025](https://arxiv.org/abs/2501.11378) describe
the same English outros ("bag of hallucinations"). Only phrases that are clearly not a
voice-agent utterance are artifacts; short thank-yous and goodbyes are suspects.

Measured on the T4 VAD corpus (`van bench vad`'s deterministic corpus: 50 LibriSpeech
utterances laid out with 0.8-2.5 s noise-only gaps, a 2 s lead and a 20 s noise-only tail,
in six noise conditions). Every clip is cut into utterances by the VAD, as the
`StreamAdapter` does, and each utterance is decoded once; an utterance with less than
50 ms of labelled speech is noise-only, and a hallucination is a noise-only utterance
with a non-empty transcript. WER is over the whole clip.

1. **VAD-cut utterances.** Neither Silero nor the energy VAD fired on noise alone in this
   corpus (0 noise-only utterances in 675 / 912 utterances), so there was nothing to
   hallucinate on. The guard dropped no real speech (0 utterances with `small`; 2
   one-word "you" transcripts of utterances holding 0.12-0.25 s of speech with `base` and
   the energy VAD), and WER is unchanged in every condition.
2. **Simulated VAD false alarms.** Every noise-only region of every clip (0.3 s away from
   speech) cut into 1 s windows, 88 per condition, each decoded as if the VAD had reported
   it as an utterance (the VAD's own probabilities over the window feed the guard):

| Condition | `small` without guard | `small` with guard | `base` without / with guard |
|---|---|---|---|
| clean | 24 / 88 | 0 | 0 / 0 |
| pink noise, 20 dB SNR | 21 / 88 | 0 | 0 / 0 |
| pink noise, 10 dB SNR | 74 / 88 | 0 | 0 / 0 |
| pink noise, 5 dB SNR | 88 / 88 | 0 | 0 / 0 |
| white noise, 10 dB SNR | 87 / 88 | 0 | 0 / 0 |
| transient clicks | 62 / 88 | 0 | 0 / 0 |
| **all** | **356 / 528 (67 %)** | **0** | **0 / 0** |

`small` (CUDA float16, `language="en"`) produced "Thank you." on quiet noise and "you" on
loud noise; Whisper rated all of them as likely no-speech (`no_speech_prob` 0.79-0.90) but
with an `avg_logprob` between -0.84 and -0.99, just above the -1.0 that faster-whisper's
own filter requires, so they got through. The guard drops them as suspect phrases on weak
evidence. `base` produced nothing on the same windows. Measured 2026-09-25,
faster-whisper 1.2.1, 50 utterances, 3 min of noise per condition.

## Errors

| Situation | Exception |
|---|---|
| `faster-whisper` not installed | `MissingDependencyError` (at construction, with the install command) |
| unknown model / revision, bad `device`, `compute_type`, language or option | `ConfigurationError` |
| gated or private repository, HTTP 401/403 | `AuthenticationError` (log in with `hf auth login` or set `HF_TOKEN`) |
| HTTP 429 from the Hub | `RateLimitError` |
| network failure, or model not cached while offline | `ProviderConnectionError` |
| CTranslate2 failure (CUDA out of memory, missing CUDA libraries with `device="cuda"`...) | `ProviderError` |

A failed load is retried on the next call.

## Limitations

* Partial transcripts are re-decodes of the whole utterance: each one costs a full Whisper
  pass (30 s window), and the tail of an interim is often a cut-off word that the next
  one corrects. Long utterances make every decode slower, and the back-off spaces them out.
* The hallucination guard works on whole segments: a hallucinated phrase glued to real
  speech inside one segment is kept. Its phrase lists cover 16 languages; for others, add
  to `artifacts` / `suspects` / `patterns`.
* `word_timestamps` still list the words of a repetition loop that the guard cut from the
  text.
* A transcription that is already running finishes in its worker thread even when the
  turn is cancelled (its result is discarded).
