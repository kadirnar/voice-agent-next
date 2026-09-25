# Semantic end of turn: `lm_turn`, `llm_turn` and `fused`

An audio end-of-turn model such as [Smart Turn](smart_turn.md) hears whether an utterance
*sounds* finished. It cannot read it: "I would like to book a table," followed by a pause
sounds complete (Smart Turn: 0.88). A **text** detector reads the transcript and the
agent's last turn. The two make different mistakes, so the cascade fuses them.

| Provider | What it is | Size / cost | Use |
|---|---|---|---|
| `lm_turn` | a small causal LM's probability that the user's message ends here, calibrated | SmolLM2-135M int8: 137 MB, ~30–80 ms on one CPU core | text half of `fused` (default) |
| `llm_turn` | asks any registered LLM "has the user finished?" (yes/no, logprobs when available) | one 1-token LLM call | text half with a capable LLM |
| `fused` | an audio and a text detector, combined in log-odds | both halves, run concurrently | the cascade's `turn_detector` |

## Setup

```bash
pip install 'voice-agent-next[smart-turn,text-turn]'   # onnxruntime + tokenizers
van models download lm_turn/smollm2-135m              # optional: fetch now (137 MB)
```

```yaml
# agent.yaml
vad: silero
turn_detector: {provider: fused, audio: smart_turn, text: lm_turn}
```

```python
from voice_agent_next import AgentSession
from voice_agent_next.turn import FusedTurnDetector

session = AgentSession(
    stt=..., llm=..., tts=..., vad="silero",
    turn_detector=FusedTurnDetector(audio="smart_turn", text="lm_turn"),
)
```

`van models for` and Docker images pick up both halves of a `fused` detector.

## `fused`: how the cascade runs it

At each candidate pause (VAD `END_OF_SPEECH`) the cascade:

1. starts the **audio half** on the turn's audio, concurrently with the STT flush (as it
   does for Smart Turn alone);
2. starts the **text half** on the transcript it already has (finals + interim), also
   concurrently with the flush. When the final transcript arrives it is reused if the
   text did not change, and run again otherwise;
3. fuses both probabilities and passes the result to the
   [endpointing policy](../concepts/endpointing.md) like any detector's.

The text half has a latency budget (`text_timeout`, 0.5 s). Past it, or when it fails, the
audio verdict is used alone. Both halves' `EOTMetrics` are forwarded, and each
`EndpointingMetrics` carries `audio_probability` and `text_probability`.

### Fusion

`fuse_end_of_turn(audio, text, method=...)`:

| `method` | Formula | Notes |
|---|---|---|
| `logit` (default) | `sigmoid(wa·logit(audio) + wt·logit(text) + bias)` | logistic regression over the two scores |
| `product` | `audio^wa · text^wt` | both must agree the user is done; slowest |
| `min` | `min(audio, text)` | the less confident one |
| `mean` | weighted mean | |

Defaults: `logit`, `audio_weight` 0.4, `text_weight` 0.9, `bias` 0.1, `threshold` 0.5,
fitted by logistic regression on eot-bench English (Smart Turn v3.2 int8 + `lm_turn`
SmolLM2-135M). The weights are the same for the 360M model within ±0.02.

## `lm_turn`: end-of-message probability

The detector writes the conversation in ChatML, the agent's last turn followed by the
user's open message, and reads the probability that the next token is `<|im_end|>`:

```text
<|im_start|>assistant
Sure, what would you like to know?<|im_end|>
<|im_start|>user
I would like to book a table,          <- p(<|im_end|>) ≈ 0.0000
```

This is the idea behind TurnGPT and LiveKit's first text detector (a fine-tuned
Qwen2.5-0.5B), here with an off-the-shelf instruct model and no fine-tuning. The raw
probability spans many orders of magnitude, so it is calibrated with
`sigmoid(a·ln p + b)`, fitted on eot-bench English. Punctuation carries much of the
signal, and STTs differ: sherpa-onnx NeMo writes `where is my order`, Kroko writes
`Where is my order?`. Transcripts without any punctuation are therefore lowercased and use
a second calibration fitted on eot-bench with punctuation removed.

| Model | File | Size | License |
|---|---|---|---|
| `smollm2-135m` (default) | `HuggingFaceTB/SmolLM2-135M-Instruct`, `onnx/model_int8.onnx` | 137 MB | Apache-2.0 |
| `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M-Instruct`, `onnx/model_int8.onnx` | 365 MB | Apache-2.0 |

Both are pinned to a revision with SHA-256 checksums (`providers/lm_turn.py`).

| Option | Default | Meaning |
|---|---|---|
| `model` | `smollm2-135m` | model id |
| `model_path`, `tokenizer_path` | – | a local ChatML causal-LM ONNX file (transformers.js layout) and its `tokenizer.json` |
| `calibration`, `calibration_unpunctuated` | fitted | `(a, b)` of `sigmoid(a·ln p + b)` |
| `threshold` | 0.5 | on the calibrated probability |
| `max_tokens` | 128 | the prompt keeps its last N tokens |
| `num_threads` | 1 | ONNX Runtime intra-op threads |

## `llm_turn`: ask an LLM

```yaml
turn_detector:
  provider: fused
  audio: smart_turn
  text: {provider: llm_turn, model: "ollama/qwen3.5:4b", timeout: 0.4}
```

A short system prompt asks whether the user has finished (yes) or will keep talking (no).
With OpenAI-compatible LLMs (OpenAI, Ollama, vLLM, llama.cpp, Groq…) it makes one
non-streamed 1-token completion with `logprobs` and returns `p(yes) / (p(yes) + p(no))`;
other LLMs answer in text (`yes` → `confidence`, 0.9). Past `timeout` it returns
`fallback` (1.0: no objection). Pass `llm=` an instance to share the agent's LLM; it is
not closed with the detector.

With LFM2.5-1.2B (the local presets' LLM) it is **not** useful: ROC-AUC 0.68 on eot-bench
English, below `lm_turn`, and small models answer the question inconsistently. It is
there for larger local or fast cloud models.

## Measurements

### eot-bench (`van bench turns`)

LiveKit's eot-bench, 400 real turns per language; every span scored 0.2 s into the
silence with the words spoken before it (`--transcript-lag 0`: a streaming STT's final
transcript arrives at the pause). Smart Turn v3.2 int8 vs the default fused detector
(English-fitted parameters in every language). False cut-offs and mean latency use the
local presets' policy (0.5 / 1.5 s); the other columns are eot-bench's operating points
(best policy within the budget).

EOTBENCH_TABLE

* Fusion lowers the false cut-offs at the presets' policy in every language measured,
  and improves every operating point: it ranks turn ends better than either half.
* The cost is latency where the text half disagrees with a confident audio verdict (+60
  to +160 ms mean end-of-turn latency at the presets' policy).
* SmolLM2 is English-first but scored better on German and Spanish text (ROC-AUC 0.81,
  0.83) than on English (0.74).

### Candidates we evaluated

| Text detector | eot-bench en ROC-AUC (text only) | Verdict |
|---|---:|---|
| punctuation only (`.?!` = done) | 0.74 | baseline |
| NAMO v1 English (DistilBERT, Apache-2.0) | 0.62 | int8 file returns ~0.13 for every input |
| NAMO v1 Multilingual (mmBERT, Apache-2.0, 309 MB) | 0.58 | |
| LFM2.5-1.2B yes/no via Ollama (`llm_turn`) | 0.68 | |
| Qwen2.5-0.5B-Instruct int8, end-of-message | 0.68 | 512 MB, 200 ms |
| **SmolLM2-135M-Instruct int8, end-of-message** (`lm_turn`) | **0.74** | fused: 0.88 |
| SmolLM2-360M-Instruct int8, end-of-message | 0.76 | fused: 0.88, 3× slower |
| LiveKit turn detector, TEN, Vogent | – | licenses rule them out (research note 04) |

### T4 battery and T1 latency

T4T1_TABLE
