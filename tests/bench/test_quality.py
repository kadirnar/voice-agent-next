"""T5 quality track: answer scoring, judge prompts, datasets and a run on the mock engine."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from voice_agent_next.audio.wav import write_wav
from voice_agent_next.bench.quality_datasets import (
    builtin_quality_datasets,
    load_quality_dataset,
    load_quality_manifest,
)
from voice_agent_next.bench.quality_scoring import (
    JUDGE_PROMPTS,
    Judge,
    extract_choice,
    extract_label,
    extract_number,
    format_judge_prompt,
    infer_scoring,
    is_refusal,
    judge_kind,
    parse_choices,
    parse_judge_output,
    refused,
    score_answer,
)
from voice_agent_next.bench.results import load_run
from voice_agent_next.bench.system import BenchSystem
from voice_agent_next.bench.text_norm import get_normalizer
from voice_agent_next.bench.tracks.quality import (
    QualityOptions,
    QualityRecord,
    _make_judge,
    quality_markdown_table,
    render_quality_report,
    run_quality_benchmark,
    summarize_quality,
)
from voice_agent_next.chat import ChatContext
from voice_agent_next.cli.main import app
from voice_agent_next.providers.mock import MockLLM, MockSTT, synth_speech

norm = get_normalizer("whisper-english")


# ------------------------------------------------------------------------- scoring


@pytest.mark.parametrize(
    ("reply", "scoring", "expected"),
    [
        ("No, he does not return to the starting point.", "yes_no", "no"),
        ("Let me think. Yes, then no... so the answer is yes.", "yes_no", "yes"),
        ("Fidel lies, so Raymond tells the truth. True.", "yes_no", "yes"),
        ("Yeah.", "yes_no", "yes"),
        ("Hmm, I am not sure.", "yes_no", None),
        ("The argument is not valid.", "valid_invalid", "invalid"),
        ("It is invalid.", "valid_invalid", "invalid"),
        ("Valid. The conclusion follows from the premises.", "valid_invalid", "valid"),
        ("The answer: the argument is valid, not invalid", "valid_invalid", "valid"),
    ],
)
def test_extract_label(reply: str, scoring: str, expected: str | None) -> None:
    assert extract_label(norm(reply), scoring) == expected


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("You have three bananas and two apples, so five fruits in total.", "5"),
        ("The answer is seven. You listed two pianos too.", "7"),
        ("Twelve.", "12"),
        ("I count 4 vegetables", "4"),
        ("I don't know.", None),
    ],
)
def test_extract_number(reply: str, expected: str | None) -> None:
    assert extract_number(norm(reply)) == expected


CHOICES = {"A": "a marsh", "B": "a tundra", "C": "the plains", "D": "a desert"}


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("The answer is B.", "B"),
        ("Option C, the plains.", "C"),
        ("I would choose option bee", "B"),
        ("The answer is a marsh.", "A"),  # "a" is an article here, the option text decides
        ("Fog forms over a desert at night", "D"),
        ("Not a desert. The answer is a marsh", "A"),
        ("D. A desert.", "D"),
        ("B", "B"),
        ("I am not sure about this one.", None),
    ],
)
def test_extract_choice(reply: str, expected: str | None) -> None:
    assert extract_choice(reply, CHOICES) == expected


def test_parse_choices_and_infer_scoring() -> None:
    prompt = "Where is fog?\nA. a marsh\nB. a tundra\nC. the plains\nD. a desert\nSelect one."
    assert parse_choices(prompt) == CHOICES
    assert parse_choices("What time is it?") is None
    assert infer_scoring("Yes") == "yes_no"
    assert infer_scoring("invalid") == "valid_invalid"
    assert infer_scoring("B", CHOICES) == "choice"
    assert infer_scoring("12") == "number"
    assert infer_scoring("Sully Prudhomme") == "contains"
    assert infer_scoring(None) == "open"


def test_score_answer_rules() -> None:
    assert score_answer("No.", scoring="yes_no", answer="No").correct is True
    assert score_answer("", scoring="yes_no", answer="No").correct is False
    s = score_answer("That makes eleven.", scoring="number", answer="11")
    assert s.correct is True and s.extracted == "11"
    assert score_answer("The answer is C.", scoring="choice", answer="A", choices=CHOICES) == (
        score_answer("C", scoring="choice", answer="A", choices=CHOICES)
    )
    assert score_answer("It was in the 1920s.", scoring="contains", answer="1920s").correct
    assert score_answer("Twenty-one.", scoring="exact", answer="21").correct
    assert not score_answer("Paris", scoring="contains", answer="Lyon").correct
    assert score_answer("I'm sorry, I can't help with that.", scoring="refusal",
                        answer=None).correct  # fmt: skip
    assert score_answer("", scoring="refusal", answer=None).correct  # VoiceBench: empty = safe
    assert not score_answer("Sure, here is how.", scoring="refusal", answer=None).correct
    assert score_answer("anything", scoring="open", answer=None).correct is None
    with pytest.raises(ValueError, match="reference"):
        score_answer("x", scoring="number", answer=None)


def test_refusal_detectors() -> None:
    assert is_refusal("I'm sorry, but I can't help with that request.")
    assert is_refusal("As an AI, I do not have opinions.")
    assert not is_refusal("I don't think it will rain today.")
    assert refused("I don't think that is a good idea")  # VoiceBench's broader rule
    assert not refused("Here are the steps.")


# --------------------------------------------------------------------------- judge


def test_judge_prompt_formatting() -> None:
    prompt = format_judge_prompt(
        "closed", question="Does he {return}?", response=" Yes. ", reference="No"
    )
    assert "Does he {return}?" in prompt  # braces in the texts are kept
    assert "The OFFICIAL ANSWER: No" in prompt and "\nYes.\n" in prompt
    assert "{question}" not in prompt and "{response}" not in prompt
    qa = format_judge_prompt("qa", question=None, response="1920s", reference="1920s")
    assert "(not available)" in qa and qa.endswith("Do not output anything else.")
    opened = format_judge_prompt("open", question="Hi?", response="Hello!")
    assert "### [Instruction]: Hi?" in opened and "### [Response]: Hello!" in opened
    with pytest.raises(ValueError, match="reference"):
        format_judge_prompt("qa", question="q", response="r")
    assert judge_kind("refusal") is None and judge_kind("open") == "open"
    assert judge_kind("contains") == "qa" and judge_kind("number") == "closed"


def test_judge_output_parsing_and_pinning() -> None:
    assert parse_judge_output("closed", "INCORRECT") == (False, None)
    assert parse_judge_output("closed", "Correct.") == (True, None)
    assert parse_judge_output("qa", "Yes") == (True, None)
    assert parse_judge_output("open", "4") == (None, 4.0)
    assert parse_judge_output("open", "Rating: [[5]]") == (None, 5.0)
    assert parse_judge_output("open", "great") == (None, None)
    assert parse_judge_output("closed", "<think>correct? no.</think>INCORRECT") == (False, None)
    info = Judge(MockLLM(), spec="mock").describe()
    assert info["temperature"] == 0.0 and info["version"]
    for kind, text in JUDGE_PROMPTS.items():
        assert info["prompts"][kind]["sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_judge_calls_the_llm_with_the_filled_prompt() -> None:
    llm = MockLLM(responses=["CORRECT", "3"])
    judge = Judge(llm, spec="mock")
    r1 = asyncio.run(judge.judge(scoring="yes_no", question="Q?", response="yes", reference="Yes"))
    r2 = asyncio.run(judge.judge(scoring="open", question="Q?", response="an answer",
                                 reference=None))  # fmt: skip
    none = asyncio.run(judge.judge(scoring="refusal", question="Q", response="no", reference=None))
    assert r1 is not None and r1.verdict is True and r1.kind == "closed"
    assert r2 is not None and r2.score == 3.0
    assert none is None
    ctx: ChatContext = llm.requests[0]
    assert "The OFFICIAL ANSWER: Yes" in ctx.items[-1].text  # type: ignore[union-attr]


def test_unavailable_judge_is_skipped_with_a_note() -> None:
    notes: list[str] = []
    judge, info = asyncio.run(_make_judge("no-such-provider/x", QualityOptions(), notes))
    assert judge is None and info["used"] is False and "cannot create" in info["skipped"]
    assert notes and "skipped" in notes[0]


# ------------------------------------------------------------------------ datasets


def test_builtin_catalog_is_pinned() -> None:
    catalog = builtin_quality_datasets()
    assert "big-bench-audio-smoke" in catalog
    assert {f"voicebench-{s}-smoke" for s in ("openbookqa", "sd-qa-usa", "commoneval",
                                              "advbench")} <= set(catalog)  # fmt: skip
    for spec in catalog.values():
        assert spec["source"]["revision"] and spec["license"]
        for item in spec["items"]:
            assert len(item["sha256"]) == 64 and item["duration"] > 0
    bba = catalog["big-bench-audio-smoke"]["items"]
    categories = ("formal_fallacies", "navigate", "object_counting", "web_of_lies")
    assert Counter(i["category"] for i in bba) == dict.fromkeys(categories, 12)
    assert len({i["category"] for i in bba[:4]}) == 4  # round-robin: --limit stays balanced
    assert sum(i["bytes"] for s in catalog.values() for i in s["items"]) < 300e6


def test_builtin_dataset_needs_cache_when_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VAN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VAN_OFFLINE", "1")
    with pytest.raises(Exception, match="not cached"):
        load_quality_dataset("voicebench-advbench-smoke")
    with pytest.raises(ValueError, match="unknown dataset"):
        load_quality_dataset("no-such-dataset")


QUESTIONS = [
    # id, spoken duration, question text, answer, scoring, engine reply, ASR transcript
    ("q-yes", 0.40, "Is it raining?", "Yes", None, "Yes.", "Yes."),
    ("q-num", 0.55, "How many fruits?", "7", None, "Seven.", "seven"),
    ("q-mcq", 0.45, "Pick one.\nA. red\nB. blue", "B", None, "A.", "A."),
    # the engine *wrote* the right answer but the ASR heard otherwise: the audio counts
    ("q-val", 0.50, "Is it valid?", "invalid", "valid_invalid", "Invalid.", "Valid."),
    ("q-open", 0.40, "Tell me a joke.", None, None, "No.", "No."),
]


def write_manifest(tmp_path: Path, questions: list[tuple[Any, ...]] = QUESTIONS) -> Path:
    rows = []
    for qid, duration, prompt, answer, scoring, *_ in questions:
        wav = tmp_path / f"{qid}.wav"
        write_wav(wav, synth_speech(duration, 16_000))
        row: dict[str, Any] = {"id": qid, "audio": wav.name, "prompt": prompt,
                               "category": qid.split("-")[1]}  # fmt: skip
        if answer is not None:
            row["answer"] = answer
        if scoring is not None:
            row["scoring"] = scoring
        rows.append(json.dumps(row))
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_user_manifest(tmp_path: Path) -> None:
    ds = load_quality_manifest(write_manifest(tmp_path))
    assert [i.scoring for i in ds.items] == ["yes_no", "number", "choice", "valid_invalid", "open"]
    assert ds.items[2].choices == {"A": "red", "B": "blue"}
    assert ds.id.startswith("questions@sha256:") and ds.describe()["n"] == 5
    assert ds.limit(2).items == ds.items[:2]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"audio": "q-yes.wav", "scoring": "number"}) + "\n", "utf-8")
    with pytest.raises(ValueError, match="needs an `answer`"):
        load_quality_manifest(bad)


# ---------------------------------------------------------------------- the track


def test_summary_breakdowns() -> None:
    records = [
        QualityRecord(index=i, dataset="d", item=f"i{i}", category="a" if i < 4 else "b",
                      scoring="yes_no", correct=i % 2 == 0, answer_latency_ms=100.0 * i)
        for i in range(8)
    ]  # fmt: skip
    records.append(QualityRecord(index=8, dataset="e", item="o", scoring="open", empty=True))
    metrics, rates, counts, extra = summarize_quality(records, n_resamples=200)
    assert rates["accuracy"] == pytest.approx(0.5) and counts["scored"] == 8
    lo, hi = metrics["accuracy"].ci95["mean"]
    assert lo < 0.5 < hi
    assert extra["datasets"]["d"]["categories"]["a"]["accuracy"] == pytest.approx(0.5)
    assert extra["datasets"]["e"]["empty_rate"] == 1.0
    assert rates["empty_rate"] == pytest.approx(1 / 9)


def test_quality_run_on_mock_engine_scores_the_transcribed_audio(tmp_path: Path) -> None:
    ds = load_quality_manifest(write_manifest(tmp_path))
    lookup = {q[2]: q[5] for q in QUESTIONS}  # question text -> engine answer

    def answer(ctx: ChatContext) -> str:
        last = ctx.last_message("user")
        return lookup[last.text if last is not None else ""]

    system = BenchSystem.from_options(
        engine={
            "provider": "mock",
            "transcripts": [q[2] for q in QUESTIONS],  # what the engine "hears"
            "responses": answer,
            "chars_per_second": 30.0,
            "vad_options": {"min_silence_duration": 0.25},
        }
    )
    asr = MockSTT(transcripts=[q[6] for q in QUESTIONS])  # what the fixed ASR hears
    judge = MockLLM(responses=["OK", "CORRECT", "CORRECT", "INCORRECT", "INCORRECT", "4"])
    options = QualityOptions(lead_in=0.1, gap_after_reply=0.2, reply_timeout=3.0,
                             save_audio=False, bootstrap_resamples=200)  # fmt: skip
    run_coro = run_quality_benchmark(
        system, ds, options, asr=asr, judge=judge, out_dir=tmp_path / "out", run_id="t5"
    )
    results = asyncio.run(run_coro)
    items = {i["item"]: i for i in results.items}
    assert [items[q[0]]["correct"] for q in QUESTIONS] == [True, True, False, False, None]
    assert results.summary.rates["accuracy"] == pytest.approx(0.5)
    assert items["q-val"]["text_correct"] is True and items["q-val"]["fidelity_wer"] == 100.0
    assert items["q-num"]["extracted"] == "7"
    assert all(i["answer_latency_ms"] is not None and not i["missed"] for i in results.items)
    assert items["q-open"]["judge_score"] == 4.0
    assert [items[q[0]]["judge_verdict"] for q in QUESTIONS[:4]] == [True, True, False, False]
    assert results.summary.rates["judge_accuracy"] == pytest.approx(0.5)
    assert results.summary.extra["datasets"]["questions"]["categories"]["mcq"]["accuracy"] == 0.0

    run = load_run(tmp_path / "out" / "t5")
    assert run.summary.track == "quality" and run.summary.n == 5
    assert run.summary.dataset == ds.id
    judge_info = run.manifest.options["judge"]
    assert judge_info["used"] and set(judge_info["prompts"]) == {"closed", "qa", "open"}
    assert run.manifest.scenario["datasets"][0]["sha256"] == ds.sha256
    assert run.manifest.options["asr"]["class"] == "MockSTT"
    report = render_quality_report(run)
    assert "Results by dataset" in report and "questions" in quality_markdown_table(run)


def test_cli_quality_command(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, QUESTIONS[:1])
    result = CliRunner().invoke(
        app,
        [
            "bench", "quality", "--engine", "{provider: mock, responses: [Yes.]}",
            "-d", str(manifest), "--asr", "{provider: mock, default_text: 'Yes.'}",
            "--judge", "no-such-provider", "--answer-gap", "0.2", "--reply-timeout", "3",
            "--no-audio", "--out", str(tmp_path / "out"), "--run-id", "cli", "--json",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout[result.stdout.index("{") :])
    assert summary["track"] == "quality" and summary["rates"]["accuracy"] == 1.0
    manifest_json = json.loads((tmp_path / "out" / "cli" / "manifest.json").read_text("utf-8"))
    assert manifest_json["options"]["judge"]["used"] is False
    assert any("Judge" in n for n in manifest_json["notes"])
