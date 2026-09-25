"""Benchmark suite: one stimulus-driven harness for every engine (research note 06, §8).

The harness measures what the *user hears*: a simulated caller
(:class:`~voice_agent_next.bench.caller.CallerEmulator`) streams pre-rendered stimuli in
real time over a loopback transport, both sides are recorded on one clock
(:class:`~voice_agent_next.bench.recording.DuplexRecording`) and every user-perceived
number is read off the recording with a reference VAD
(:mod:`~voice_agent_next.bench.onset`). Results are written as ``manifest.json``,
``items.jsonl``, ``summary.json`` and ``report.md`` (:mod:`~voice_agent_next.bench.results`).

Tracks: T1 latency (:mod:`voice_agent_next.bench.tracks.latency`), T5 speech-to-speech
quality (:mod:`voice_agent_next.bench.tracks.quality`), T7 framework overhead
(:mod:`voice_agent_next.bench.tracks.overhead`) with its CI regression gate
(:mod:`voice_agent_next.bench.gate`), T6 tool use (:mod:`voice_agent_next.bench.tracks.tools`).
CLI: ``van bench``.
"""

from __future__ import annotations

from .caller import CallerEmulator, CallResult, TurnTiming
from .gate import Baseline, GateReport, compare_to_baseline, load_baseline, update_baseline
from .onset import OnsetDetector, ProviderReferenceVAD, ReferenceVAD, RMSReferenceVAD
from .recording import DuplexRecording, Label, read_labels, write_labels
from .results import Distribution, RunManifest, RunResults, RunSummary, load_run, write_run
from .stimuli import Scenario, Stimulus, TurnSpec, load_scenario, render_stimuli
from .system import BenchSystem
from .tracks.latency import LatencyItem, LatencyOptions, run_latency_benchmark
from .tracks.overhead import OverheadCondition, OverheadOptions, run_overhead_benchmark

__all__ = [
    "Baseline",
    "BenchSystem",
    "CallResult",
    "CallerEmulator",
    "Distribution",
    "DuplexRecording",
    "GateReport",
    "Label",
    "LatencyItem",
    "LatencyOptions",
    "OnsetDetector",
    "OverheadCondition",
    "OverheadOptions",
    "ProviderReferenceVAD",
    "RMSReferenceVAD",
    "ReferenceVAD",
    "RunManifest",
    "RunResults",
    "RunSummary",
    "Scenario",
    "Stimulus",
    "TurnSpec",
    "TurnTiming",
    "compare_to_baseline",
    "load_baseline",
    "load_run",
    "load_scenario",
    "read_labels",
    "render_stimuli",
    "run_latency_benchmark",
    "run_overhead_benchmark",
    "update_baseline",
    "write_labels",
    "write_run",
]
