# Text normalization

LLM replies are full of digits, prices, times, dates, e-mail addresses and URLs. Many TTS
models read them badly. Pocket TTS, for example, reads the order number "58213" as
"five tu o thiurt". A **spoken-form normalizer** rewrites such text as words before
synthesis:

| The LLM writes | The TTS gets |
| --- | --- |
| `The total is $42.50.` | The total is forty-two dollars and fifty cents. |
| `Your order number is 58213.` | Your order number is five eight two one three. |
| `See you March 3rd at 4:30 PM.` | See you March third at four thirty P M. |
| `Signed on July 14, 2025.` | Signed on July fourteenth, twenty twenty-five. |
| `Call 555-0142.` | Call five five five, zero one four two. |
| `Mail anna.lee@example.com.` | Mail anna dot lee at example dot com. |
| `Dr. Smith, e.g. at 7 AM` | Doctor Smith, for example at seven A M |

The chat history, the transcript events and barge-in truncation still use the text the
LLM wrote. The normalized text only goes to the TTS model.

## Turning it on

Every TTS takes a `normalize` option:

```python
from voice_agent_next.registry import create

tts = create("tts", "pocket-tts")  # provider default: on
tts = create("tts", "pocket-tts", normalize=False)  # off
tts = create("tts", "openai/gpt-4o-mini-tts", normalize=True)  # force it on
tts = create("tts", "kokoro", normalize="en")  # a language code
```

In a config file:

```yaml
tts:
  provider: pocket-tts
  normalize: false
```

`normalize` accepts:

| Value | Meaning |
| --- | --- |
| `None` (default) | the provider's default (`TTS.normalize_by_default`) |
| `True` | on, in the language of the voice (`TTS.text_language(voice)`) |
| `False` | off |
| `"en"`, `"fr"`... | on, with the normalizer of that language |
| a `TextNormalizer` | on, with this object |

### Provider defaults

Normalization is on by default for local models that read raw symbols badly. It is off
for cloud services, which normalize text on their side.

| Provider | Default | Language |
| --- | --- | --- |
| `pocket-tts` | on | `language` (English built in; others need `num2words`) |
| `kokoro` | on | from `lang`, or from the voice name (`af_heart` is English) |
| `sherpa-onnx` (Piper, Kokoro, Matcha) | on | `lang`, the model's locale (`piper-en_US-...`), or the Kokoro voice. Off when unknown |
| `openai`, `azure_openai`, `elevenlabs`, `cartesia`, `deepgram`, `google` (Gemini), `kokoro_fastapi`, `speaches`, `localai` | off | |
| `mock` | off | |

Why Kokoro is on: espeak-ng reads plain numbers correctly, but not amounts, times,
e-mail addresses, URLs or "Dr.". See the [measurements](#measurements).

## What the English normalizer covers

`EnglishNormalizer` has no dependencies. It handles:

- **Numbers:** cardinals (`1,234`), decimals (`3.14`, `.5`), negatives (`-7`), ordinals
  (`21st`), ranges (`3-5`, `10–12`), fractions (`1/2`, `3/4`, `24/7`), leading zeros
  (`007`, digit by digit).
- **Money and quantities:** currency (`$42.50`, `€12.50`, `£1.01`, `¥500`, `$1.5 million`,
  `$5k`, `20 USD`), percentages (`12%`, `10-20%`), units (`5km`, `2.5 kg`, `60 mph`,
  `-5°C`, `512 GB`, `25 min`, `2x`).
- **Time and dates:** times (`4:30 PM`, `9:05 am`, `7 AM`, `5 p.m.`, `16:45`), dates
  (`March 3rd`, `July 14, 2025`, `3 March 2024`, `2025-03-04`, `3/4/2025`), years
  (`in 1999`, `since 2005`, `by 2030`) and decades (`1990s`, `'80s`).
- **Identifiers, read digit by digit:** phone numbers (`555-0142`, `(415) 555-0142`,
  `+1 415-555-0142`). Numbers after words such as *order number*, *zip code*, *PIN*,
  *account*, *confirmation* or *#*. Bare numbers of 7 or more digits. Codes mixing letters
  and digits are split (`B12`, `A320`, `COVID-19`). Versions are read with "point"
  (`v1.2.3`).
- **Addresses:** e-mail addresses (`anna.lee@example.com`) and URLs
  (`docs.example.org/setup`, `https://www.example.com`). URLs are recognized by their
  top-level domain.
- **Abbreviations:** titles (`Dr.`, `Mr.`, `Mrs.`, `Prof.`...). `St.` becomes *Saint* or
  *Street*, and `Dr.` becomes *Doctor* or *Drive*, depending on the words around them.
  Also `e.g.`, `i.e.`, `etc.`, `vs`, `approx.`, `Inc.`, `No. 5`, `U.S.`, and initialisms
  that are spelled out (`NYC`, `AI`, `API`, `FAQ`). Words that are pronounced as words
  are left alone (`NASA`, and Roman numerals like `IV`).

The normalizer leaves text it does not recognize unchanged. A four-digit number is read
as a year only in a year context (`in 1999`). Elsewhere it is a cardinal (`1500 points`
becomes "one thousand five hundred points").

## Other languages

Normalizers are registered per language (ISO 639-1 code):

```python
from voice_agent_next.text import NormalizedText, register_normalizer
from voice_agent_next.text.normalize import RuleNormalizer


class GermanNormalizer(RuleNormalizer):
    language = "de"

    def __init__(self) -> None:
        super().__init__([(r"\bz\.\s?B\.", lambda m, text: "zum Beispiel")])


register_normalizer("de", GermanNormalizer)
```

A `RuleNormalizer` is a list of `(regex, handler)` rules, applied in one left-to-right
pass. At each position the earliest match wins; on a tie, the rule listed first wins. A
handler returns the spoken form, or `None` to decline. Any object with a `language`
attribute and a `normalize(text, *, context="") -> NormalizedText` method works too.

For languages without a registered normalizer, `get_normalizer()` falls back to
`Num2WordsNormalizer` when the optional [`num2words`](https://pypi.org/project/num2words/)
package is installed (`pip install num2words`). It reads cardinals, decimals (with a
comma where the language uses one) and percentages. Without `num2words`, text in those
languages is left unchanged.

## Streaming and word timings

**Sentence by sentence.** Most local TTS engines use the `SentenceStreamAdapter`. It
normalizes each sentence inside `TTS.synthesize()`. The segmenter never cuts a sentence
inside a number group: `July 14, | 2025`, `4:30 | PM` and `5 | kg` stay together, and
`Dr.`, `p.m.` and `No.` do not end a sentence.

**Native streaming TTS** (with `normalize=True`). `StreamNormalizer` releases pushed text
only at safe points: between two plain words, or after a word that closes a clause or a
sentence. It never splits a number, date or address across two normalization calls. The
last words already released are passed as `context`, so "order number is" still affects
a number that arrives in the next chunk.

**Word timings.** Each `NormalizedText` keeps an offset map back to the original. A
`WordAligner` maps the word timings the TTS reports on the spoken text back to the
original words. It merges "forty-two dollars and fifty cents" into a single `$42.50`
timing that starts with the first spoken word. When the user interrupts, the truncated
assistant message says `The total is $42.50,`, never `The total is forty-two dollars`.
With a sentence-at-a-time TTS, the segment text (`SynthesizedAudio.text`) is the
original sentence too.

## As a text filter

`normalize_text()` is a plain `str -> str` function. It can also run in the cascade's
`text_filter`:

```python
from voice_agent_next import CascadeOptions
from voice_agent_next.text import normalize_text, tts_clean

options = CascadeOptions(text_filter=lambda s: normalize_text(tts_clean(s)))
```

The transcript and the chat history then show the spoken form too, so
`TTS(normalize=True)` is usually the better choice.

## Measurements

`van bench tts --texts smoke --stt faster-whisper/small.en --mode both`, on 20 texts, 8
of them "hard text" with 17 entities. The same runs with `normalize: false` and
`normalize: true`. `hardtext_acc` counts the entities that the round-trip ASR transcript
got right. `rt_wer` is the round-trip word error rate on the original text, after Whisper
English normalization.

MEASUREMENTS_TABLE

## Limitations

- English only in the core. Other languages need `num2words` (numbers and percentages
  only) or a registered normalizer.
- Heuristics can guess wrong in ambiguous cases: a 4-digit number with no year context,
  `St.` between two capitalized words, US `month/day` order in numeric dates, and
  `1-2` read as a range.
- Units that are also ordinary words (`m`, `s`, `in`, `L`) are only read when attached
  to the number (`5m`, not `5 m`).
- With native streaming TTS, the segment text (`SynthesizedAudio.text`) is the spoken
  form. Word timings still map back to the original text.
