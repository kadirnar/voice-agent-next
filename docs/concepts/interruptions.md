# Interruptions (barge-in)

When the user starts talking while the agent speaks, the agent must stop — but only for a
*real* barge-in. Voice activity alone is a poor signal: coughs, noise, echo and
backchannels ("uh-huh", "right") trigger the VAD too. Krisp measured **66% false
positives** for VAD-only barge-in (research note 04 §5). `AgentSession` therefore
**pauses** the agent on possible barge-ins and only **stops** it once the interruption is
confirmed. If the user was not really taking the floor, the agent **resumes** where it
stopped.

The policy lives in `voice_agent_next.session.interruptions` (pure logic, no asyncio). The
session drives it and handles pausing, truncation and events.

## What happens

| From | When | To | What the session does |
|---|---|---|---|
| agent speaking | the user starts speaking | overlap | pause playback |
| overlap | speech ≥ `min_interruption_duration` and ≥ `min_interruption_words` non-backchannel words | interrupted | stop, cancel the response, truncate it to what was heard |
| overlap | the user goes quiet | paused | start the `false_interruption_timeout` timer |
| paused | the user speaks again | overlap | the timer stops; speech time keeps adding up |
| paused | the transcript has only backchannels | agent speaking | resume at once; emit `agent_false_interruption(resumed=True)` |
| paused | quiet for `false_interruption_timeout`, no meaningful words | agent speaking | resume; emit `agent_false_interruption(resumed=True)` |
| paused | quiet for `false_interruption_timeout`, meaningful words | interrupted | stop, cancel, truncate |
| overlap / paused | the engine commits the user's turn or cancels the response itself | interrupted | truncate to what was heard (no cancel: the engine has moved on) |

The agent state stays `speaking` (or `thinking`) while paused. It becomes `listening` only
once the interruption is confirmed, because clients treat `listening` as "the agent's turn
is over". A false interruption therefore leaves no trace in the state.

1. **Overlap.** The engine reports `InputSpeechStarted` while a response is generating or
   playing. The session pauses playback. Queued audio is kept, and so is the
   played position.
2. **Confirm.** The interruption is real once the user has spoken for
   `min_interruption_duration` seconds *and* said `min_interruption_words` non-backchannel
   words (interim transcripts count). The session then clears the transport, cancels the
   response and trims the history to what the user heard, as before (`interrupted` event,
   `ChatMessage.interrupted`, `TurnMetrics.interrupted`).
3. **Resume.** The user goes quiet, and one of these holds:
   * the transcript of what they said has only backchannels ("uh-huh", "okay"): the agent
     resumes at once;
   * they stay quiet for `false_interruption_timeout` seconds without meaningful words (a
     cough, noise): the agent resumes then.

   Either way the session emits `agent_false_interruption` with `resumed=True`. Nothing
   was dropped, so the conversation history is unchanged.
4. **The engine moved on.** If the engine commits the user's turn, or cancels the paused
   response itself (native engines with server-side VAD do this), resuming is impossible.
   The session treats this as a confirmed interruption and truncates the agent's turn to
   what was heard.

## Options

All options are fields of `SessionOptions` (and of the `session:` block in config files).

| Option | Default | Meaning |
|---|---|---|
| `allow_interruptions` | `True` | Let the user barge in at all. `False` makes every response uninterruptible. |
| `min_interruption_duration` | `0.5` | Seconds of user speech needed to confirm a barge-in. |
| `min_interruption_words` | `0` | Non-backchannel words needed as well, counted in interim transcripts. `0` means duration only. |
| `backchannel_words` | `None` | Words and phrases that never count as an interruption. `None` means the built-in list for the agent's language. |
| `false_interruption_timeout` | `2.0` | Seconds of silence without meaningful words before paused speech resumes. `None` disables pause-and-resume. |
| `resume_false_interruption` | `True` | Pause while the verdict is pending. `False` keeps the agent talking until the barge-in is confirmed. |
| `discard_audio_if_uninterruptible` | `True` | Send silence instead of the user's audio while uninterruptible speech plays. |

`min_interruption_duration=0` together with `min_interruption_words=0` turns the policy
off. The agent is then interrupted at the first sign of speech, without pausing (the
behaviour before this policy existed).

```python
from voice_agent_next import AgentSession, SessionOptions

session = AgentSession(
    stt="deepgram/nova-3",
    llm="openai/gpt-4.1-mini",
    tts="cartesia/sonic-2",
    vad="silero",
    options=SessionOptions(min_interruption_words=1),  # streaming STT: ignore coughs and noise
)


@session.on("agent_false_interruption")
async def on_false_interruption(ev):
    if not ev.resumed:  # an interruption stopped the agent, then nothing was said
        await session.generate_reply(instructions="Continue where you left off.")
```

### Choosing values

* **With a streaming STT** (interim transcripts), `min_interruption_words=1` is the most
  robust setting. Noise and coughs produce no words, so they never interrupt, however long
  they last. Backchannels do not count as words. The cost is latency: the interruption
  is confirmed once the first real word is transcribed. The agent is already paused
  before that, so the user does not hear the delay.
* **Duration only** (`min_interruption_words=0`, the default) works with every engine.
  Sounds shorter than `min_interruption_duration` never interrupt. Longer ones do, even
  if they are only "mmm-hmmm". A short but meaningful "stop!" still interrupts: when the
  engine commits it as a turn, or at the timeout.
* **Duration is measured from where the user's speech started.** While the engine still
  reports speech, the count includes the VAD's trailing-silence hangover, as in LiveKit.
  A cough of `c` seconds is filtered when
  `c + VAD min_silence_duration < min_interruption_duration`. With the default VAD
  (0.25 s) and 0.5 s, that means coughs up to about 0.25 s.

## Backchannels

`backchannel_words` entries are matched on normalized words. Matching ignores case,
punctuation and hyphens, and collapses elongations ("Mmmm" → "m", "yeahhh" → "yeah"). An
entry may be a phrase: "all right", "je vois", "hı hı". Chinese and Japanese are split per
character, so `min_interruption_words` counts characters there, like LiveKit.

The default comes from `backchannel_words_for(agent.language)`:

* English (or no language): uh-huh, mm-hmm, mhm, hmm, uh, um, yeah, yep, yes, ok, okay,
  right, sure, alright, "i see", "got it", cool, "go on", ...
* `de`, `es`, `fr`, `it`, `pt`, `tr`, `ja`, `zh`: the language's own list plus the
  English reactions heard everywhere ("okay", "mm-hmm", "yeah").

Words that are also answers or corrections ("no", "wait", "but") are deliberately not
backchannels. Pass your own list to change this:
`SessionOptions(backchannel_words=["uh-huh", "okay", "d'accord"])`.

## The `agent_false_interruption` event

`AgentFalseInterruption` (in `voice_agent_next.session`):

| Field | Meaning |
|---|---|
| `resumed` | `True`: the paused speech resumed. `False`: an interruption had already stopped the agent — confirmed on duration, or the engine cancelled the response — and the user then said nothing meaningful for `false_interruption_timeout` seconds. The app may continue the conversation. |
| `reason` | `"noise"` (nothing transcribed), `"backchannel"` (only backchannels), `"too_few_words"` (fewer than `min_interruption_words`) |
| `response_id`, `item_id` | the agent response that was talked over |
| `transcript` | what was transcribed while the user spoke (`""` if nothing) |
| `speech_duration` | seconds of user speech |
| `paused` | seconds the agent was paused before it resumed (`0.0` if it did not resume) |

## Pausing and transports

* **Transports with `capabilities.pause`** (the loopback transport, and devices that
  support it): the session calls `pause_audio()` / `resume_audio()`. The session's playback
  clock stops too, so the truncation position stays exact.
* **Other transports:** the session stops sending audio and keeps the rest queued. The short
  look-ahead already handed to the transport (`output_lookahead`, 150 ms) plays out. On
  resume, sending continues from where it stopped. Nothing is lost or repeated.

`LoopbackTransport(pausable=False)` simulates the second kind in tests and benchmarks.
`pause_times` and `resume_times` record when the session paused and resumed it.

## Engines

* **Cascade** (`CascadeEngine`): VAD speech events drive the policy. Interim and final STT
  results provide the words. The cascade commits a user turn, and answers it, after
  endpointing. That would cancel the paused response, so when the whole utterance turns out
  not to be meaningful the session calls `clear_input()` first. The backchannel is dropped
  instead of being answered.
* **Native engines** follow the same rules when they report speech. Engines with
  server-side VAD often cancel the response themselves as soon as speech starts (for
  example OpenAI Realtime with `interrupt_response`). The session then receives
  `ResponseDone(status="cancelled")`, confirms the interruption and truncates to what was
  heard. Resuming is not possible in that case. Tune the engine's own turn detection for
  false-interruption handling there.

## Uninterruptible speech

```python
await session.say("This call may be recorded.", allow_interruptions=False)
```

While uninterruptible speech plays, the user cannot barge in. With
`discard_audio_if_uninterruptible` (the default), the engine receives silence instead of
the user's audio, so it cannot queue a turn either. The same applies to every response
when `allow_interruptions=False`. The flag applies to the next response the engine
starts after `say()`.

## Limitations

* **Echo:** without echo cancellation, the agent's own voice can trigger the VAD. The agent
  then pauses and resumes repeatedly. Enable an echo canceller (`processors=...`) when
  playing through speakers.
* **Duration** includes the VAD's trailing-silence hangover while speech is ongoing (see
  above).
* **Rule-based only.** There is no audio interruption classifier yet, and no special
  backchannel window at the start or end of an agent turn (research note 04 §8.2).
* **Tool calls:** a tool round that finishes while the agent is paused starts its follow-up
  response. That response supersedes the paused one, which is then truncated.
