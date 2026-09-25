# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html), and the entries are generated
from the [Conventional Commits](https://www.conventionalcommits.org/) history with
[git-cliff](https://git-cliff.org/) (`cliff.toml`, see `docs/releasing.md`).

## [Unreleased]

### Features

- **audio:** WebRTC APM echo cancellation and half-duplex gate ([#65](https://github.com/kadirnar/voice-agent-next/issues/65))
- **bench:** benchmark harness and voice-to-voice latency track ([#64](https://github.com/kadirnar/voice-agent-next/issues/64))
- **bench:** framework-overhead track and CI regression gate ([#40](https://github.com/kadirnar/voice-agent-next/issues/40)) ([#82](https://github.com/kadirnar/voice-agent-next/issues/82))
- **bench:** T2 ASR track — WER/CER, RTFx, TTFS and final latency ([#41](https://github.com/kadirnar/voice-agent-next/issues/41)) ([#99](https://github.com/kadirnar/voice-agent-next/issues/99))
- **bench:** T3 TTS track — TTFA, RTF, round-trip WER, MOS ([#42](https://github.com/kadirnar/voice-agent-next/issues/42)) ([#102](https://github.com/kadirnar/voice-agent-next/issues/102))
- **bench:** T4 VAD, end-of-turn and turn-taking battery ([#43](https://github.com/kadirnar/voice-agent-next/issues/43)) ([#109](https://github.com/kadirnar/voice-agent-next/issues/109))
- **cascade:** speculative LLM generation on probable end of turn ([#27](https://github.com/kadirnar/voice-agent-next/issues/27)) ([#87](https://github.com/kadirnar/voice-agent-next/issues/87))
- **cascade:** dynamic endpointing and dictation mode ([#31](https://github.com/kadirnar/voice-agent-next/issues/31)) ([#108](https://github.com/kadirnar/voice-agent-next/issues/108))
- **cli:** `van models` model manager — list, download, verify, prune ([#48](https://github.com/kadirnar/voice-agent-next/issues/48)) ([#94](https://github.com/kadirnar/voice-agent-next/issues/94))
- **cli:** presets and `van run --preset` with readiness checks ([#8](https://github.com/kadirnar/voice-agent-next/issues/8)) ([#96](https://github.com/kadirnar/voice-agent-next/issues/96))
- **cli:** `van doctor` deep diagnostics — audio host APIs, mic meter, echo and latency probes ([#50](https://github.com/kadirnar/voice-agent-next/issues/50)) ([#110](https://github.com/kadirnar/voice-agent-next/issues/110))
- **core:** provider failover chains for STT, LLM and TTS ([#30](https://github.com/kadirnar/voice-agent-next/issues/30)) ([#85](https://github.com/kadirnar/voice-agent-next/issues/85))
- **core:** hardware-aware backend auto-selection and CUDA library loading ([#49](https://github.com/kadirnar/voice-agent-next/issues/49)) ([#88](https://github.com/kadirnar/voice-agent-next/issues/88))
- **deploy:** `van serve` worker pool with prewarm, health, metrics and limits ([#37](https://github.com/kadirnar/voice-agent-next/issues/37)) ([#107](https://github.com/kadirnar/voice-agent-next/issues/107))
- **engine:** OpenAI Realtime engine with compatibility profiles ([#66](https://github.com/kadirnar/voice-agent-next/issues/66))
- **engine:** Gemini Live engine with session resumption ([#67](https://github.com/kadirnar/voice-agent-next/issues/67))
- **engine:** Moshi / PersonaPlex full-duplex speech-to-speech engine ([#13](https://github.com/kadirnar/voice-agent-next/issues/13)) ([#100](https://github.com/kadirnar/voice-agent-next/issues/100))
- **engine:** proactive session rotation and reconnect with context carry-over ([#17](https://github.com/kadirnar/voice-agent-next/issues/17)) ([#104](https://github.com/kadirnar/voice-agent-next/issues/104))
- **llm:** Anthropic Claude provider with streaming tools and prompt caching ([#57](https://github.com/kadirnar/voice-agent-next/issues/57))
- **llm:** OpenAI and OpenAI-compatible LLM providers ([#59](https://github.com/kadirnar/voice-agent-next/issues/59))
- **llm:** cache-write tokens in metrics, system-message policy for strict templates ([#73](https://github.com/kadirnar/voice-agent-next/issues/73))
- **providers:** ElevenLabs streaming TTS with alignment and Scribe v2 Realtime STT ([#21](https://github.com/kadirnar/voice-agent-next/issues/21)) ([#80](https://github.com/kadirnar/voice-agent-next/issues/80))
- **providers:** OpenAI streaming STT and gpt-4o-mini-tts TTS ([#24](https://github.com/kadirnar/voice-agent-next/issues/24)) ([#81](https://github.com/kadirnar/voice-agent-next/issues/81))
- **providers:** Gemini LLM and Gemini TTS via google-genai ([#23](https://github.com/kadirnar/voice-agent-next/issues/23)) ([#84](https://github.com/kadirnar/voice-agent-next/issues/84))
- **providers:** sherpa-onnx streaming STT, TTS and VAD ([#9](https://github.com/kadirnar/voice-agent-next/issues/9)) ([#90](https://github.com/kadirnar/voice-agent-next/issues/90))
- **providers:** AssemblyAI Universal-Streaming v3 STT with neural turn detection ([#20](https://github.com/kadirnar/voice-agent-next/issues/20)) ([#91](https://github.com/kadirnar/voice-agent-next/issues/91))
- **providers:** Kyutai Pocket TTS with audio streaming and voice cloning ([#76](https://github.com/kadirnar/voice-agent-next/issues/76)) ([#93](https://github.com/kadirnar/voice-agent-next/issues/93))
- **providers:** Moonshine Streaming STT ([#77](https://github.com/kadirnar/voice-agent-next/issues/77)) ([#95](https://github.com/kadirnar/voice-agent-next/issues/95))
- **providers:** Soniox and Speechmatics real-time STT ([#25](https://github.com/kadirnar/voice-agent-next/issues/25)) ([#117](https://github.com/kadirnar/voice-agent-next/issues/117))
- **server:** serve any engine over the OpenAI Realtime protocol ([#36](https://github.com/kadirnar/voice-agent-next/issues/36)) ([#83](https://github.com/kadirnar/voice-agent-next/issues/83))
- **session:** interruption policy with false-interruption recovery ([#68](https://github.com/kadirnar/voice-agent-next/issues/68))
- **session:** pre-warm the engine at start, keep the user turn before the reply ([#69](https://github.com/kadirnar/voice-agent-next/issues/69))
- **session:** truncate at the listener's real playback position ([#74](https://github.com/kadirnar/voice-agent-next/issues/74))
- **session:** call recording (stereo WAV + JSONL timeline) and OpenTelemetry tracing ([#29](https://github.com/kadirnar/voice-agent-next/issues/29)) ([#89](https://github.com/kadirnar/voice-agent-next/issues/89))
- **session:** tool watchdog fillers, non-blocking tools and progress updates ([#28](https://github.com/kadirnar/voice-agent-next/issues/28)) ([#97](https://github.com/kadirnar/voice-agent-next/issues/97))
- **session:** multi-agent handoffs, shared userdata and conversation flows ([#32](https://github.com/kadirnar/voice-agent-next/issues/32)) ([#112](https://github.com/kadirnar/voice-agent-next/issues/112))
- **stt:** faster-whisper local STT provider ([#54](https://github.com/kadirnar/voice-agent-next/issues/54))
- **stt,tts:** Deepgram Nova-3, Flux and Aura-2 providers ([#58](https://github.com/kadirnar/voice-agent-next/issues/58))
- **stt,tts:** Cartesia Sonic TTS and Ink STT providers ([#61](https://github.com/kadirnar/voice-agent-next/issues/61))
- **transports:** local audio transport with device selection ([#55](https://github.com/kadirnar/voice-agent-next/issues/55))
- **transports:** WebSocket server transport and browser demo ([#60](https://github.com/kadirnar/voice-agent-next/issues/60))
- **transports:** telephony media streams for Twilio, Telnyx, Vonage and Plivo ([#35](https://github.com/kadirnar/voice-agent-next/issues/35)) ([#92](https://github.com/kadirnar/voice-agent-next/issues/92))
- **transports:** WebRTC transport (aiortc) with browser demo ([#34](https://github.com/kadirnar/voice-agent-next/issues/34)) ([#98](https://github.com/kadirnar/voice-agent-next/issues/98))
- **tts:** Kokoro local TTS provider ([#63](https://github.com/kadirnar/voice-agent-next/issues/63))
- **tts:** spoken-form text normalization for TTS input ([#105](https://github.com/kadirnar/voice-agent-next/issues/105)) ([#115](https://github.com/kadirnar/voice-agent-next/issues/115))
- **turn:** Smart Turn v3.2 end-of-turn detector ([#56](https://github.com/kadirnar/voice-agent-next/issues/56))
- **vad:** Silero VAD v6 ONNX provider ([#53](https://github.com/kadirnar/voice-agent-next/issues/53))
- foundation of voice-agent-next (M0)

### Bug fixes

- **cascade:** whole-turn audio for EOT models; overlap audio detectors with STT flush
- **cascade:** STT-owned turns, speech-end timing from STT, word-exact truncation
- **ci:** green main on every OS once Actions ran again
- **cli:** never colorize `van providers --json` output
- **cli:** escape Rich markup in `van providers` so '[extra]' hints are shown
- **engine:** OpenAI Realtime connect timeouts and test races on Windows ([#71](https://github.com/kadirnar/voice-agent-next/issues/71))
- **tests:** deterministic timing on Windows and macOS for rotation, recording and telephony tests ([#114](https://github.com/kadirnar/voice-agent-next/issues/114))
- **transports:** FileTransport stalled forever on float-rounding empty slices

### Performance

- **tts,cascade:** trim TTS silence padding and speak the first clause early

### Documentation

- **bench:** first results — fully local CPU cascade and framework overhead
- **bench:** Pocket TTS vs Kokoro latency
- **bench:** T2 ASR smoke results
- **bench:** Moshi full-duplex results
- **examples:** example gallery with offline-testable scripts ([#47](https://github.com/kadirnar/voice-agent-next/issues/47)) ([#101](https://github.com/kadirnar/voice-agent-next/issues/101))
- mkdocs-material documentation site with API reference ([#46](https://github.com/kadirnar/voice-agent-next/issues/46)) ([#103](https://github.com/kadirnar/voice-agent-next/issues/103))
- add deploy/serving.md to the nav (strict build)

### Tests

- **bench:** one-sided latency tolerances for the mock v2v checks
- **cascade:** one-sided bound for the eager-EOT speculation gain (flaked under load)
- **components:** allow Windows timer slack in the MockTTS ttfb check
- **engine:** make the Gemini Live dropped-connection test timer-independent ([#70](https://github.com/kadirnar/voice-agent-next/issues/70))
- **session:** tolerate CI jitter in the playback-position tests

### Build and CI

- add an all-extras test job so provider SDK tests run in CI
- run Windows on every pull request, keep macOS behind ci:full
- weekly real-model tests on Linux, macOS and Windows ([#72](https://github.com/kadirnar/voice-agent-next/issues/72))

[Unreleased]: https://github.com/kadirnar/voice-agent-next/commits/main

