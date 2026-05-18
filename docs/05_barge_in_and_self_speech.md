# Barge-in and self-speech rejection

These are the two hard problems of "always listening" voice. Solutions below
are layered — start simple, add the next layer only if the previous isn't
enough.

## Problem 1 — Self-speech: don't transcribe ourselves

If the assistant says "the time is 3:42", we do not want STT to commit
"the time is 3:42" as a user utterance and the LLM to reply to itself.

### Layer A — Mic-pause during TTS (the cheap fix)

What `MockingAgent/voice_assistant.py:285-294` already does:

```python
mic.paused = True
sd.play(audio, samplerate=sr); sd.wait()
time.sleep(0.12)         # speaker drain
mic.drain()              # discard whatever made it through
mic.paused = False
```

This is enough for ~95% of cases when the user lets the assistant finish.
It also disables barge-in entirely (the cost we're paying for simplicity).

In our bus design this is two messages:

```python
bus.publish("mic.pause", True)   # audio_io drops frames
# ... TTS plays ...
bus.publish("mic.pause", False)
```

### Layer B — AEC (acoustic echo canceller)

When we *want* barge-in, the mic has to stay open while we speak, so
mic-pause is wrong. We need to subtract our own audio from the mic stream.
The scaffold is already there in `audio/aec.py`; the runner that will wire it
for M4 is `core/runners/orchestrator.py`:

```python
self.aec = AECWrapper(sample_rate=SAMPLE_RATE, frame_ms=10)
tts.on_playback_chunk = lambda chunk: self.ring.push_far(chunk)
```

We need to:
- Confirm `speexdsp` (already in requirements) is the AEC engine and that
  it survives macOS audio quirks.
- Make sure `tts.audio_chunk` chunks are published *before* `sd.play()`
  starts so the far-end ring leads slightly in time.
- Tune the per-frame size — AEC wants 10 ms blocks, mic also runs at 10 ms.

### Layer C — Reply-text similarity filter (belt and suspenders)

After A/B, occasionally a fragment of our own audio still leaks through.
Last-chance gate at `stt.text` time:

```python
if SequenceMatcher(None, recent_assistant_reply, candidate_text).ratio() > 0.75:
    drop  # treat as self-speech bleed
```

We already have `recent_assistant_reply` in the LLM history.
**Implemented** in [core/runners/orchestrator.py:_sounds_like_self](../core/runners/orchestrator.py)
with `SELF_SPEECH_SIMILARITY_THRESHOLD = 0.75`.

### Layer D — LLM gate (the catch-all)

If A/B/C all let something through, the LLM itself decides whether to
respond. Every reply must begin with `<ignore>` or `<reply>` (system
prompt instruction); the orchestrator suppresses TTS on `<ignore>`.

This catches not just self-speech but also TV, ambient room talk,
keystroke noise, and Whisper hallucinations — any input that isn't a
directed turn. See [01_architecture.md §LLM gate](01_architecture.md#llm-gate)
for the full design.

The layers compose: audio-side filtering catches the obvious cases cheaply;
the LLM gate catches the subtle cases that no heuristic would. The
audio-side filtering also keeps prompt cost low — better not to feed
fifty `[BLANK_AUDIO]`s per minute to the LLM if we don't have to.

## Problem 2 — Barge-in: stop talking when the user starts talking

Behavior: while assistant is mid-reply, if the user starts speaking, the
assistant cuts off mid-sentence and listens.

### Detection

`vad.speech_start` event published by VAD whenever it transitions from
non-speech to speech. The orchestrator already tracks `state` — if
`state == RESPONDING` when `vad.speech_start` fires, we're in a barge-in.

### Action

1. `bus.publish("tts.cancel", None)` — TTS clears its audio queue, calls
   `sd.stop()`.
2. `bus.publish("mic.pause", False)` — un-gate mic if Layer A was active.
3. Set `state = LISTENING`.
4. Keep the in-flight LLM request alive, but **cancel** it: orchestrator
   calls `backend.cancel()` so we stop generating tokens nobody hears.
5. The new user utterance flows through STT as normal and becomes the next
   `llm.request`.

### Guarding against false barge-in

VAD on the mic *is also hearing the speaker* unless mic-pause is on. So
when we're using AEC instead of mic-pause:
- Run barge-in detection on the **AEC-cleaned** audio, not the raw mic.
- Require sustained voice (≥150 ms) before declaring barge-in, so a cough
  doesn't kill the reply.
- Do not allow barge-in for the first ~250 ms of a TTS turn — otherwise the
  click of `sd.play()` starting can self-trigger.

These are heuristics from prior systems; tune them after first end-to-end test.

## Decision matrix for v1

| Mode | Self-speech | Barge-in |
|---|---|---|
| M2 | Mic-pause during TTS (Layer A) | OFF — TTS finishes before mic re-opens |
| **M3 (current)** | Mic-pause + similarity filter + LLM gate | OFF; mid-reply utterances are queued (single-slot, last-write-wins) |
| M4 (next) | AEC + similarity filter + LLM gate | ON, with sustained-voice + start-grace guards |

The LLM gate (Layer D) was added during M3 and stays on for M4. It's
prompt-only, so no architectural cost; the orchestrator already does the
token buffering.
