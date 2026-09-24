"""Benchmark tracks (research note 06, §8.3). T1: :mod:`.latency`."""

from __future__ import annotations

from .latency import LatencyItem, LatencyOptions, run_latency_benchmark

__all__ = ["LatencyItem", "LatencyOptions", "run_latency_benchmark"]
