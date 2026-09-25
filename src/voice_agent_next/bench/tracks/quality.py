"""T5 speech-to-speech quality: does the agent *say* the right answer?

``van bench quality`` plays spoken questions (Big Bench Audio, VoiceBench subsets or your
own recordings) to **any** system — native speech-to-speech, cascade, omni or full-duplex —
over the T1 harness (real-time :class:`~voice_agent_next.bench.caller.CallerEmulator`,
loopback transport, stereo recording), one fresh session per question so no answer sees
another question. The agent's reply is cut from the recording and transcribed **after**
all sessions by a fixed local ASR (``--asr``, default ``faster-whisper/small.en``), and
that transcript is scored — the audio, not the engine's own text (research note 06, §8.3):

* ``accuracy`` — share of closed questions answered correctly by rule
  (:mod:`voice_agent_next.bench.quality_scoring`: extracted label / number / letter,
  reference contained, or refusal), with a 95 % bootstrap CI of the mean, per dataset and
  per category;
* ``judge_accuracy`` / ``judge_score`` — the optional LLM judge's verdicts (closed and
  reference answers) and 1–5 ratings (open questions); the judge (model, temperature,
  prompts and their SHA-256) is recorded in the manifest and skipped with a note when it
  cannot be reached;
* ``answer_latency_ms`` — agent onset (reference VAD) − end of the question, i.e. T1's
  ``v2v_ms`` on long questions; ``answer_speech_ms`` — how long the agent spoke;
* ``refusal_rate`` (the agent declined: "I'm sorry, but I can't..."), ``empty_rate`` (no
  audible answer or an empty transcript) and ``missed_rate`` (no agent speech within
  ``reply_timeout``);
* ``text_accuracy`` and ``fidelity_wer`` when the engine exposes its own text (cascades,
  most realtime APIs): the same rules on that text, and the word error rate of the
  transcribed audio against it (*speech fidelity*: 0 = the agent said what it wrote).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...audio.frame import AudioFormat, AudioFrame
from ...audio.resample import resample
from ...config import ComponentSpec
from ...errors import ConfigurationError
from ...registry import create
from ...stt import STT
from ...utils.clock import now
from ..asr_datasets import load_audio
from ..environment import collect_environment
from ..onset import OnsetDetector
from ..quality_datasets import QualityDataset, QualityItem
from ..quality_scoring import Judge, is_refusal, score_answer
from ..report import ReportSpec, fmt, markdown_table, render_report
from ..results import (
    Distribution,
    RunManifest,
    RunResults,
    RunSummary,
    new_run_id,
    utc_timestamp,
    write_run,
)
from ..stimuli import Scenario, Stimulus, _pad_to_chunks, annotate_speech, normalize_loudness
from ..system import BenchSystem
from ..text_norm import get_normalizer
from ..wer import EditCounts, corpus_rate, word_counts
from .latency import (
    _AGENT_FORMAT,
    LatencyOptions,
    _analyze_session,
    _reserve_directory,
    _run_session,
    _SessionAnalysis,
    _SessionRun,
    _write_artifacts,
)

__all__ = [
    "DEFAULT_ASR",
    "TRACK",
    "QualityOptions",
    "QualityRecord",
    "quality_markdown_table",
    "render_quality_report",
    "run_quality_benchmark",
    "summarize_quality",
]

TRACK = "quality"
DEFAULT_ASR = "faster-whisper/small.en"
_NORMALIZER = "whisper-english"


@dataclass
class QualityOptions:
    """How the quality track runs."""

    limit: int | None = None
    """Keep the first N questions of every dataset."""
    reply_timeout: float = 20.0
    """No agent speech this long after the question ended -> missed (s)."""
    gap_after_reply: float = 1.5
    """The answer is over once the agent is idle and quiet this long (s)."""
    max_reply: float = 120.0
    """Upper bound for one answer (s)."""
    lead_in: float = 0.5
    """Silence streamed before the question (s)."""
    loudness_dbfs: float | None = -20.0
    """Speech RMS of every question (``None``: keep the source level)."""
    sample_rate: int = 16_000
    chunk: float = 0.02
    language: str | None = "en"
    """Language passed to the ASR."""
    save_audio: bool = True
    """Write ``artifacts/session-NNN/`` (stereo recording, labels, timeline) per question."""
    warmup_engine: bool = True
    judge_temperature: float = 0.0
    transcribe_questions: bool = True
    """Give the judge an ASR transcript of the question when the dataset has no text."""
    seed: int = 0
    bootstrap_resamples: int = 2000

    def validate(self) -> None:
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be >= 1")
        if self.reply_timeout <= 0 or self.max_reply <= 0:
            raise ValueError("reply_timeout and max_reply must be > 0")
        if self.gap_after_reply < 0 or self.lead_in < 0:
            raise ValueError("gap_after_reply and lead_in must be >= 0")


class QualityRecord(BaseModel):
    """One spoken question and the scored answer (a line of ``items.jsonl``)."""

    model_config = ConfigDict(extra="forbid")

    index: int
    """Session index (``artifacts/session-NNN``)."""
    dataset: str
    item: str
    category: str | None = None
    scoring: str
    reference: str | None = None
    question: str | None = None
    question_s: float | None = None
    """Duration of the question's speech (s)."""
    transcript: str | None = None
    """The fixed ASR's transcript of the agent's answer audio."""
    engine_text: str | None = None
    """The engine's own text for the answer, when it exposes one."""
    extracted: str | None = None
    correct: bool | None = None
    """Rule-based verdict on the transcript (``None``: open question, not scored)."""
    text_correct: bool | None = None
    """The same rule on ``engine_text``."""
    refusal: bool = False
    empty: bool = False
    missed: bool = False
    answer_latency_ms: float | None = None
    answer_speech_ms: float | None = None
    fidelity_wer: float | None = None
    """WER (%) of ``transcript`` against ``engine_text``."""
    judge_kind: str | None = None
    judge_verdict: bool | None = None
    judge_score: float | None = None
    judge_raw: str | None = None
    judge_error: str | None = None
    question_transcript: str | None = None
    """ASR transcript of the question (given to the judge when there is no text)."""
    errors: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------------ stimuli


def _stimulus(item: QualityItem, options: QualityOptions) -> Stimulus:
    audio = resample(load_audio(item.audio).to_mono(), options.sample_rate)
    if not audio:
        raise ConfigurationError(f"{item.id}: empty audio file {item.audio}")
    try:
        span = annotate_speech(audio)
    except ValueError as exc:
        raise ConfigurationError(f"{item.id}: {exc}") from exc
    gain_db = 0.0
    if options.loudness_dbfs is not None:
        audio, gain_db = normalize_loudness(audio, options.loudness_dbfs, span)
    return Stimulus(
        id=item.id,
        text=item.prompt,
        audio=_pad_to_chunks(audio, options.chunk),
        speech_start=span[0],
        speech_end=span[1],
        source="wav",
        gain_db=gain_db,
        category=item.category,
    )


def _scenario(name: str, options: QualityOptions) -> Scenario:
    """Harness timing for one question per session (the turn list is a placeholder)."""
    return Scenario(
        name=name,
        sample_rate=options.sample_rate,
        chunk=options.chunk,
        loudness_dbfs=options.loudness_dbfs,
        lead_in=options.lead_in,
        reply_timeout=options.reply_timeout,
        gap_after_reply=options.gap_after_reply,
        max_reply=options.max_reply,
        turns=[{"id": "question", "duration": 1.0}],  # type: ignore[list-item]
    )


# ----------------------------------------------------------------------- scoring


def _pct(counts: EditCounts) -> float | None:
    rate = counts.rate
    return None if rate is None else round(100.0 * rate, 3)


def _clip_answer(run: _SessionRun) -> AudioFrame:
    """Agent channel from the user's speech onset to the end of the session."""
    rec = run.call.recording
    agent = rec.agent_audio()
    if not run.call.turns:
        return agent
    start = max(0.0, rec.to_offset(run.call.turns[0].speech_start))
    return agent.slice(min(start, agent.duration), None)


def summarize_quality(
    records: Sequence[QualityRecord], *, seed: int = 0, n_resamples: int = 2000
) -> tuple[dict[str, Distribution], dict[str, float | None], dict[str, int], dict[str, Any]]:
    """Distributions, rates, counts and per-dataset / per-category breakdowns."""

    def dist(values: Any, digits: int = 3, ci: tuple[str, ...] = ("mean", "p50")) -> Distribution:
        return Distribution.of(values, ci=ci, seed=seed, n_resamples=n_resamples, digits=digits)

    def share(flags: Sequence[bool]) -> float | None:
        return round(sum(flags) / len(flags), 6) if flags else None

    def block(rs: Sequence[QualityRecord]) -> dict[str, Any]:
        scored = [r for r in rs if r.correct is not None]
        acc = dist((float(bool(r.correct)) for r in scored), 4, ("mean",))
        judged = [r for r in rs if r.judge_verdict is not None]
        jacc = dist((float(bool(r.judge_verdict)) for r in judged), 4, ("mean",))
        jscore = dist((r.judge_score for r in rs), 3, ("mean",))
        lat = dist(r.answer_latency_ms for r in rs)
        out: dict[str, Any] = {
            "n": len(rs),
            "scored": len(scored),
            "accuracy": acc.mean,
            "accuracy_ci95": acc.ci95.get("mean"),
            "refusal_rate": share([r.refusal for r in rs]),
            "empty_rate": share([r.empty for r in rs]),
            "answer_latency_p50_ms": lat.p50,
        }
        if judged:
            out.update(judge_accuracy=jacc.mean, judge_accuracy_ci95=jacc.ci95.get("mean"))
        if jscore.n:
            out.update(judge_score=jscore.mean, judge_score_ci95=jscore.ci95.get("mean"))
        return out

    scored = [r for r in records if r.correct is not None]
    judged = [r for r in records if r.judge_verdict is not None]
    text_scored = [r for r in records if r.text_correct is not None]
    metrics: dict[str, Distribution] = {
        "accuracy": dist((float(bool(r.correct)) for r in scored), 4, ("mean",)),
        "answer_latency_ms": dist(r.answer_latency_ms for r in records),
        "answer_speech_ms": dist(r.answer_speech_ms for r in records),
    }
    optional = {
        "judge_accuracy": dist((float(bool(r.judge_verdict)) for r in judged), 4, ("mean",)),
        "judge_score": dist((r.judge_score for r in records), 3, ("mean",)),
        "text_accuracy": dist((float(bool(r.text_correct)) for r in text_scored), 4, ("mean",)),
        "fidelity_wer": dist(r.fidelity_wer for r in records),
    }
    metrics.update({k: d for k, d in optional.items() if d.n})

    normalize = get_normalizer(_NORMALIZER)
    fidelity = [
        word_counts(normalize(r.engine_text or ""), normalize(r.transcript or ""))
        for r in records
        if r.fidelity_wer is not None
    ]
    rates: dict[str, float | None] = {
        "accuracy": metrics["accuracy"].mean,
        "judge_accuracy": optional["judge_accuracy"].mean,
        "text_accuracy": optional["text_accuracy"].mean,
        "refusal_rate": share([r.refusal for r in records]),
        "empty_rate": share([r.empty for r in records]),
        "missed_rate": share([r.missed for r in records]),
        "fidelity_wer": corpus_rate(fidelity) if fidelity else None,
    }
    counts = {
        "items": len(records),
        "scored": len(scored),
        "correct": sum(bool(r.correct) for r in scored),
        "judged": len(judged) + sum(r.judge_score is not None for r in records),
        "judge_errors": sum(r.judge_error is not None for r in records),
        "refusals": sum(r.refusal for r in records),
        "empty": sum(r.empty for r in records),
        "missed": sum(r.missed for r in records),
        "errors": sum(len(r.errors) for r in records),
    }
    datasets: dict[str, Any] = {}
    for name in dict.fromkeys(r.dataset for r in records):
        rs = [r for r in records if r.dataset == name]
        entry = block(rs)
        cats = dict.fromkeys(r.category for r in rs if r.category)
        if cats:
            entry["categories"] = {c: block([r for r in rs if r.category == c]) for c in cats}
        datasets[name] = entry
    return metrics, rates, counts, {"datasets": datasets}


# ------------------------------------------------------------------------ report

_METRIC_LABELS = {
    "answer_latency_ms": "answer latency (end of question → agent onset)",
    "answer_speech_ms": "answer length (agent speech)",
    "judge_score": "judge score (1–5, open questions)",
    "fidelity_wer": "speech fidelity WER % (audio vs engine text)",
}
_RATE_LABELS = {
    "accuracy": "**accuracy** (rule-based, transcribed audio)",
    "judge_accuracy": "judge accuracy",
    "text_accuracy": "accuracy of the engine's own text",
    "refusal_rate": "refusals",
    "empty_rate": "empty answers",
    "missed_rate": "missed (no agent speech)",
    "fidelity_wer": "speech fidelity WER (corpus)",
}
_ITEM_COLUMNS = (
    ("item", "item"),
    ("category", "category"),
    ("reference", "reference"),
    ("extracted", "read"),
    ("correct", "correct"),
    ("judge_verdict", "judge"),
    ("judge_score", "score"),
    ("answer_latency_ms", "latency ms"),
    ("transcript", "answer (ASR)"),
)

_METHOD = """\
* Every question is played to a **fresh session** on one engine instance (warmed up
  once) by the real-time caller over the loopback transport ({chunk_ms:g} ms chunks,
  questions normalized to {loudness} and trimmed to their speech). The answer ends when
  the agent is idle and quiet for {gap:g} s; no agent speech {timeout:g} s after the
  question ends is a missed answer.
* The agent's audio (from the question onset to the end of the session) is transcribed
  after all sessions by **{asr}** and scored: {normalizer} normalizer; closed answers are
  read from the transcript (the first label/number/letter after the last "answer", else
  the label the reply starts with, else the last one mentioned); `contains`: the
  reference appears as whole words; `refusal`: VoiceBench's AdvBench refusal phrases.
  Empty answers count as wrong.
* Judge: {judge}.
* Accuracy CIs: 95 % percentile bootstrap of the mean ({resamples} resamples, seed
  {seed}). `answer latency` = agent onset (reference VAD) − end of the question.
* Smoke subsets are regression canaries (a 48-question accuracy has a CI of about ±14
  points), not capability scores.
"""


def _dataset_rows(results: RunResults) -> list[list[str]]:
    rows: list[list[str]] = []

    def pct(v: Any) -> str:
        return "–" if v is None else f"{100 * float(v):.1f}%"

    def ci(v: Any) -> str:
        return "" if not v else f" [{100 * v[0]:.0f}, {100 * v[1]:.0f}]"

    for name, d in results.summary.extra.get("datasets", {}).items():
        entries = [(name, d)] + [
            (f"  · {cat}", c) for cat, c in (d.get("categories") or {}).items()
        ]
        for label, e in entries:
            rows.append([
                label, fmt(e.get("n")), pct(e.get("accuracy")) + ci(e.get("accuracy_ci95")),
                pct(e.get("judge_accuracy")), fmt(e.get("judge_score"), 2),
                pct(e.get("refusal_rate")), pct(e.get("empty_rate")),
                fmt(e.get("answer_latency_p50_ms")),
            ])  # fmt: skip
    return rows


_DATASET_HEADERS = ["dataset / category", "n", "accuracy [95% CI]", "judge acc.",
                    "judge score", "refusals", "empty", "latency p50 ms"]  # fmt: skip


def quality_markdown_table(results: RunResults) -> str:
    """One row per dataset and category (for PR descriptions)."""
    return markdown_table(_DATASET_HEADERS, _dataset_rows(results), ["l"] + ["r"] * 7)


def render_quality_report(results: RunResults) -> str:
    opts = results.manifest.options
    judge = opts.get("judge") or {}
    judge_desc = (
        f"`{judge.get('provider')}/{judge.get('model')}` ({judge.get('version')}, temperature "
        f"{judge.get('temperature')}, 1 sample; prompts and SHA-256 in `manifest.json`)"
        if judge.get("used")
        else "not used" + (f" ({judge['skipped']})" if judge.get("skipped") else "")
    )
    asr = opts.get("asr") or {}
    loudness = opts.get("loudness_dbfs")
    method = _METHOD.format(
        chunk_ms=1000 * opts.get("chunk", 0.02),
        loudness="the source level" if loudness is None else f"{loudness:g} dBFS",
        gap=opts.get("gap_after_reply", 1.5),
        timeout=opts.get("reply_timeout", 20.0),
        asr=f"{asr.get('provider')}/{asr.get('model')}" if asr else "?",
        normalizer=opts.get("normalizer", _NORMALIZER),
        judge=judge_desc,
        resamples=opts.get("bootstrap_resamples", 2000),
        seed=opts.get("seed", 0),
    )
    spec = ReportSpec(
        title=f"Speech-to-speech quality (T5) · {results.summary.system}",
        metric_labels=_METRIC_LABELS,
        rate_labels=_RATE_LABELS,
        item_columns=_ITEM_COLUMNS,
        sections=[("Results by dataset", quality_markdown_table(results)), ("Method", method)],
    )
    return render_report(results, spec)


# ------------------------------------------------------------------------- entry


async def run_quality_benchmark(
    system: BenchSystem,
    datasets: QualityDataset | Sequence[QualityDataset],
    options: QualityOptions | None = None,
    *,
    asr: STT | ComponentSpec = DEFAULT_ASR,
    judge: ComponentSpec | Any | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    detector: OnsetDetector | None = None,
    on_item: Callable[[QualityRecord], None] | None = None,
) -> RunResults:
    """Run the T5 quality track and (if ``out_dir``) write ``<out_dir>/<run_id>/``.

    Args:
        system: engine or cascade under test.
        datasets: one or more spoken-question datasets
            (:func:`~voice_agent_next.bench.quality_datasets.load_quality_dataset`).
        options: timing, limit, loudness...
        asr: the fixed ASR that transcribes the agent's audio (instance or spec).
        judge: an LLM spec or instance for the judge (``None``: rules only). If it cannot
            be created or reached, the run continues without it and says so.
        out_dir: parent of the run directory (``None``: nothing is written).
        run_id: defaults to ``<UTC time>-quality-<system label>``.
        detector: agent onset detector (default: RMS reference VAD).
        on_item: progress callback after each question's session (not yet scored).
    """
    options = options or QualityOptions()
    options.validate()
    sets = [datasets] if isinstance(datasets, QualityDataset) else list(datasets)
    if not sets:
        raise ValueError("no dataset")
    sets = [d.limit(options.limit) for d in sets]
    final_id = run_id or new_run_id(TRACK, system.label)
    directory: Path | None = None
    if out_dir is not None:
        directory = _reserve_directory(Path(out_dir), final_id, unique=run_id is None)
        final_id = directory.name
    own_asr = not isinstance(asr, STT)
    stt: STT | None = None
    try:
        stt = create("stt", asr) if own_asr else asr  # type: ignore[assignment]
        assert stt is not None
        return await _run_quality(
            system, sets, options, stt, None if not own_asr else asr, judge,
            directory=directory, run_id=final_id, detector=detector or OnsetDetector(),
            on_item=on_item,
        )  # fmt: skip
    except BaseException:
        if directory is not None and directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
        raise
    finally:
        if own_asr and stt is not None:
            await stt.aclose()


async def _make_judge(
    judge: ComponentSpec | Any | None, options: QualityOptions, notes: list[str]
) -> tuple[Judge | None, dict[str, Any]]:
    if judge is None:
        return None, {"used": False}
    from ...llm import LLM

    spec = None if isinstance(judge, LLM) else judge
    try:
        llm = judge if isinstance(judge, LLM) else create("llm", judge)
    except Exception as exc:
        notes.append(f"Judge {judge!r} unavailable, skipped: {exc}")
        return None, {"used": False, "spec": spec, "skipped": f"cannot create: {exc}"}
    j = Judge(llm, spec=spec, temperature=options.judge_temperature)
    try:
        await j.check()
    except Exception as exc:
        notes.append(f"Judge {judge!r} unreachable, skipped: {exc!r}")
        if not isinstance(judge, LLM):
            await llm.aclose()
        return None, {**j.describe(), "used": False, "skipped": f"unreachable: {exc!r}"}
    return j, {**j.describe(), "used": True}


async def _run_quality(
    system: BenchSystem,
    sets: list[QualityDataset],
    options: QualityOptions,
    stt: STT,
    asr_spec: Any,
    judge_spec: Any,
    *,
    directory: Path | None,
    run_id: str,
    detector: OnsetDetector,
    on_item: Callable[[QualityRecord], None] | None,
) -> RunResults:
    created = utc_timestamp()
    t_start = now()
    notes: list[str] = []
    pairs = [(ds, item) for ds in sets for item in ds.items]
    stimuli = [await asyncio.to_thread(_stimulus, item, options) for _, item in pairs]
    scenario = _scenario("+".join(d.name for d in sets), options)
    lat = LatencyOptions(
        turns=1, sessions=1, warmup_turns=0, reply_timeout=options.reply_timeout,
        gap_after_reply=options.gap_after_reply, save_audio=options.save_audio,
        warmup_engine=False, seed=options.seed,
    )  # fmt: skip

    # ---- timed phase: one session per question
    records: list[QualityRecord] = []
    runs: list[_SessionRun] = []
    analyses: list[_SessionAnalysis] = []
    answers: list[AudioFrame | None] = []
    t0 = now()
    engine = system.build_engine()
    engine_init_ms = (now() - t0) * 1000.0
    engine_warmup_ms: float | None = None
    try:
        if options.warmup_engine:
            t0 = now()
            await engine.warmup()
            engine_warmup_ms = (now() - t0) * 1000.0
        for index, ((ds, item), stim) in enumerate(zip(pairs, stimuli, strict=True)):
            record = QualityRecord(
                index=index, dataset=ds.name, item=item.id, category=item.category,
                scoring=item.scoring, reference=item.answer, question=item.prompt,
                question_s=round(stim.speech_duration, 3),
            )  # fmt: skip
            try:
                run = await _run_session(index, engine, system, [stim], scenario, lat, None)
            except Exception as exc:
                record.errors.append(f"session failed: {exc!r}")
                record.missed = record.empty = True
                answers.append(None)
            else:
                analysis = await asyncio.to_thread(_analyze_session, run, detector, lat)
                turn = analysis.items[0] if analysis.items else None
                if turn is not None:
                    record.answer_latency_ms = turn.v2v_ms
                    record.answer_speech_ms = turn.agent_speech_ms
                    record.engine_text = turn.agent_transcript
                    record.missed = turn.missed
                    record.errors += turn.errors
                runs.append(run)
                analyses.append(analysis)
                answers.append(_clip_answer(run) if turn is not None and turn.agent_audio
                               else None)  # fmt: skip
            records.append(record)
            if on_item is not None:
                on_item(record)
    finally:
        await engine.aclose()

    # ---- untimed phase: transcribe, score, judge
    await stt.warmup()
    normalize = get_normalizer(_NORMALIZER)
    judge, judge_info = await _make_judge(judge_spec, options, notes)
    try:
        for record, (_, item), stim, answer in zip(records, pairs, stimuli, answers, strict=True):
            transcript = ""
            if answer is not None and answer.duration > 0:
                try:
                    transcript = (await stt.transcribe(answer, language=options.language)).text
                except Exception as exc:
                    record.errors.append(f"ASR failed: {exc!r}")
            record.transcript = transcript.strip()
            record.empty = not normalize(record.transcript).strip()
            record.refusal = is_refusal(record.transcript)
            result = score_answer(record.transcript, scoring=item.scoring, answer=item.answer,
                                  choices=item.choices)  # fmt: skip
            record.correct, record.extracted = result.correct, result.extracted
            if record.engine_text:
                text_result = score_answer(record.engine_text, scoring=item.scoring,
                                           answer=item.answer, choices=item.choices)  # fmt: skip
                record.text_correct = text_result.correct
                ref_n = normalize(record.engine_text)
                if ref_n.strip():
                    record.fidelity_wer = _pct(word_counts(ref_n, normalize(record.transcript)))
            if judge is not None:
                question = item.prompt
                if question is None and options.transcribe_questions:
                    try:
                        speech = stim.audio.slice(stim.speech_start, stim.speech_end)
                        text = (await stt.transcribe(speech, language=options.language)).text
                        record.question_transcript = question = text.strip() or None
                    except Exception as exc:
                        record.errors.append(f"question ASR failed: {exc!r}")
                verdict = await judge.judge(
                    scoring=item.scoring, question=question, response=record.transcript,
                    reference=item.answer,
                )  # fmt: skip
                if verdict is not None:
                    record.judge_kind = verdict.kind
                    record.judge_verdict, record.judge_score = verdict.verdict, verdict.score
                    record.judge_raw, record.judge_error = verdict.raw, verdict.error
    finally:
        if judge is not None and not _is_llm(judge_spec):
            await judge.llm.aclose()

    metrics, rates, counts, extra = summarize_quality(
        records, seed=options.seed, n_resamples=options.bootstrap_resamples
    )
    extra["engine_init_ms"] = round(engine_init_ms, 3)
    extra["engine_warmup_ms"] = None if engine_warmup_ms is None else round(engine_warmup_ms, 3)
    extra["sessions"] = [a.info for a in analyses]
    if counts["missed"]:
        notes.append(
            f"{counts['missed']} question(s) got no agent speech within {options.reply_timeout:g} s"
            " (--reply-timeout)."
        )
    if counts["judge_errors"]:
        notes.append(f"{counts['judge_errors']} judge call(s) failed or were unreadable.")
    if any(r.scoring == "open" for r in records) and judge is None:
        notes.append("Open questions are only scored by a judge (--judge): not scored here.")

    manifest = RunManifest(
        run_id=run_id,
        track=TRACK,
        created=created,
        system=system.describe(),
        scenario={
            "name": scenario.name,
            "datasets": [d.describe() for d in sets],
            "stimuli": [s.describe() for s in stimuli],
        },
        transport={
            "type": "loopback",
            "realtime_playout": True,
            "input_format": str(AudioFormat(options.sample_rate, 1)),
            "output_format": str(_AGENT_FORMAT),
            "chunk_ms": round(options.chunk * 1000, 3),
        },
        options={
            **asdict(options),
            "normalizer": _NORMALIZER,
            "asr": {
                "spec": asr_spec,
                "provider": getattr(stt, "provider", None),
                "model": getattr(stt, "model", None),
                "class": type(stt).__qualname__,
            },
            "judge": judge_info,
            "onset": detector.describe(),
        },
        environment=await asyncio.to_thread(collect_environment),
        notes=notes,
    )
    summary = RunSummary(
        run_id=run_id,
        track=TRACK,
        system=system.label,
        transport="loopback",
        dataset="+".join(d.id for d in sets),
        n=len(records),
        metrics=metrics,
        rates=rates,
        counts=counts,
        extra=extra,
        duration_s=round(now() - t_start, 3),
    )
    results = RunResults(manifest, [r.model_dump(mode="json") for r in records], summary)
    results.report = render_quality_report(results)
    if directory is not None:
        write_run(directory, results)
        if options.save_audio:
            await asyncio.to_thread(_write_artifacts, directory, runs, analyses)
    return results


def _is_llm(value: Any) -> bool:
    from ...llm import LLM

    return isinstance(value, LLM)
