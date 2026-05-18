# Kokoro TTS — sentence-streaming and mic coordination

Kokoro setup is the easy part. The work is *coordinating* it with the mic
and the LLM stream so we sound responsive instead of typed-and-read-back.

## Reference implementations

- `MockingAgent/voice_assistant.py:274-304` — minimal Kokoro load + speak +
  mic-pause-during-speak. This is the proven pattern.
- `MockingAgent/PywisperCpp/pywhispercpp_examples/local_assistant/assistant_lmstudio_kokoro_tts.py:130-136`
  — a one-shot synth showing the basic `KPipeline(...)` call.
- `MockingAgent/kokoro_tts/kokoro_tts_node/` — a more complete node with
  `audio.py`, `engine.py`, `node.py` if we want to go further than the
  inline approach.

## Behavior we want

1. **Sentence-streaming.** As soon as the LLM emits a `.`, `?`, or `!`,
   synthesize and start playback. Don't wait for `llm.done`.
2. **Mic-pause coordination.** While we're speaking, set `mic.pause = True`
   so we don't transcribe ourselves. Drain the mic queue when we resume.
3. **Cancel on barge-in.** A `tts.cancel` bus message stops playback within
   one audio buffer (≈100 ms) and clears the queue.
4. **Optional AEC.** Even with mic-pause, if the user wants to barge in we
   need to be hearing while speaking. Feed the synthesized audio chunks as
   the AEC reference signal (the existing `AECWrapper` in
   `audio/aec.py` already accepts a far-end stream).

## The node

```python
class KokoroNode:
    def __init__(self, bus, voice="af_heart", lang="a"):
        from kokoro import KPipeline
        self.pipe = KPipeline(lang_code=lang)
        list(self.pipe("Ready.", voice=voice))   # warm-up; ~1 s
        self.voice = voice
        self.bus = bus
        self.text_buf = ""
        self.audio_q = queue.Queue(maxsize=32)
        self.cancel_event = threading.Event()
        self.player = threading.Thread(target=self._play_loop, daemon=True)
        self.player.start()
        bus.subscribe("llm.token", self.feed_text)
        bus.subscribe("tts.cancel", lambda _: self._cancel())

    def feed_text(self, delta: str):
        self.text_buf += delta
        for sentence in self._pop_sentences(self.text_buf):
            self.text_buf = self.text_buf[len(sentence):]
            self._synth_into_queue(sentence)

    def _synth_into_queue(self, text: str):
        chunks = [r.audio for r in self.pipe(text, voice=self.voice) if r.audio is not None]
        if not chunks: return
        audio = np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
        self.audio_q.put(audio)

    def _play_loop(self):
        while True:
            audio = self.audio_q.get()
            if self.cancel_event.is_set():
                continue                                   # drain & drop
            self.bus.publish("mic.pause", True)
            self.bus.publish("tts.audio_chunk", audio)     # for AEC reference
            sd.play(audio, samplerate=24000); sd.wait()
            if self.audio_q.empty():
                self.bus.publish("mic.pause", False)
                self.bus.publish("tts.done", None)

    def _cancel(self):
        self.cancel_event.set()
        with self.audio_q.mutex: self.audio_q.queue.clear()
        sd.stop()
        self.text_buf = ""
        self.cancel_event.clear()
```

That's a sketch — the real version lives in `plugins/kokoro_tts/node.py`.

## Voice and language

Defaults from MockingAgent:

```python
KOKORO_VOICE = "af_heart"   # warm, conversational US English female
KOKORO_LANG  = "a"          # American English
```

Other voices worth trying when we want a different feel: `am_michael`,
`af_bella`, `bf_emma` (British). All ship with the `kokoro` pip package, no
extra downloads.

## Sentence boundary heuristics

Match the LLM stream cadence — fire on the first sentence end *or* after
~60 chars without one (so a long sentence doesn't stall). The regex from
the existing `plugins/kokoro_tts/node.py` heuristic is fine; loosen the trailing
whitespace requirement so we catch end-of-stream punctuation:

```python
_SENT_END = re.compile(r'[.!?](?:\s|$)')
```

Also strip Kokoro-hostile artifacts before synth (re-use
`clean_for_tts()` from `voice_assistant.py:265-271`).

## Latency notes

- **First synth call after load is slow** (~3-5 s) — that's the warm-up call
  in `__init__` paying it once.
- After warm, ~150-300 ms to start playback for a short sentence on M-series.
- Don't forget the ~120 ms speaker drain (`time.sleep(0.12)` in
  `voice_assistant.py:291`) before un-pausing the mic — otherwise we
  re-trigger on the tail of our own audio.

## Open questions for TTS

- Sentence-streaming with hard cancel: if we cancel mid-sentence, do we
  finish the current Kokoro chunk (clean cutoff) or hard-stop with `sd.stop()`?
  Hard-stop wins for responsiveness; we accept a small click.
- Multiple voices per assistant turn (e.g. quoting someone)? Out of scope v1.
- Should Kokoro run in a process pool to avoid blocking the audio thread?
  Probably overkill; current single-thread queue is fine on M-series.
