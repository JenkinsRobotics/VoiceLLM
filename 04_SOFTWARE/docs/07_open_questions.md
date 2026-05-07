# Open questions

Things we should answer before locking the v1 design. Ranked roughly by
how much of the codebase the answer changes.

## 1. Bus pattern: poll vs. subscribe

`core/bus.py` currently exposes `get(timeout)` — single-consumer polling.
The orchestrator owns the only consumer thread. Multiple nodes wanting to
react to the same topic (e.g. `tts.audio_chunk` going to playback *and*
AEC reference) needs a real fanout.

**Options:**
- **A.** Add `subscribe(topic, callback)` on top of the queue. Each
  publish dispatches to all callbacks in order. Cheap.
- **B.** Multi-queue: each subscriber gets its own queue, publish enqueues
  on all. Cleaner backpressure per consumer.

**Lean:** A for v1. We don't have backpressure pressure yet.

## 2. History ownership

Where does the chat history live?
- In the LLM node (it owns the chat template) — clean, but the
  orchestrator has to ask it for history to log to metrics.
- In the orchestrator — easier to log/inspect, but two places have to
  agree on chat format.

**Lean:** in the LLM node. Expose `node.history_snapshot()` for logging.

## 3. Wake-word: gone, soft, or always-on?

Three modes worth supporting:
- **Always-on**: every committed phrase → LLM (M3 default).
- **Soft hotword**: a short phrase opens an "engaged" window where every
  utterance counts; window expires after silence. Best of both.
- **Strict wake**: every turn requires a wake phrase (the M1 baseline).

**Lean:** ship M3 as soft-hotword once `01_architecture.md` is realized.
Gives us the natural feel without false-firing on TV in the background.

## 4. Default STT model

Trade-off between latency and accuracy:
- `base.en` — ~250 ms per phrase on M-series, makes errors on hard words.
- `medium.en` — ~700 ms, much better.
- `large-v3-turbo` — closer to `medium` in speed, much closer to `large`
  in quality. Worth trialing.

We need an actual head-to-head before locking. Keep MockingAgent's
two-pass strategy available if we can't pick one — fast `base.en` to
gate, accurate model only when we're committing.

## 5. Sentence-streaming chunk size

When does TTS fire?
- On first `.`/`?`/`!` after the LLM emits one (lowest latency, most
  awkward at clause boundaries).
- After N tokens regardless (avoids stalls on long sentences).
- On phrase-level NLP (overkill).

**Lean:** sentence-end OR 60 chars, whichever comes first. Matches the
existing kokoro_node.py heuristic.

## 6. Should `stt.partial` exist?

A live transcript is useful for a UI and for `recent_assistant_reply`
similarity filtering, but it's wasted work if no one's reading it. Decide
based on whether we ship a UI.

**Lean:** publish it but no one subscribes by default. Cheap to add later.

## 7. Do we need the `Lilith-AI/` repo at the same level?

There's a sibling `Lilith-AI/` directory in `GITHUB/`. Unclear if it's a
separate project, an earlier iteration, or assets we should pull from.
Not in scope for VoiceLLM v1, but worth a quick look before assuming
nothing in there matters for this work.

## 8. macOS sandboxing / TCC microphone

Running through VS Code's terminal sometimes inherits the editor's mic
permission, sometimes asks fresh, sometimes silently fails. Not a code
problem but worth a one-line note in the README so future-us doesn't
re-debug it.
