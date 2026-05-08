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

## 2. History ownership ✅ RESOLVED

Lives in the LLM node ([llm/llm_node.py](../llm/llm_node.py)). The
orchestrator reads via `node.history_snapshot()` for the self-speech
similarity filter. History is also capped at `MAX_HISTORY_TURNS = 8`
user/assistant pairs. Closed.

## 3. Wake-word: gone, soft, or always-on? ✅ RESOLVED

Three modes were on the table; we ended up with two:
- **Always-on** (`REQUIRE_WAKE_WORD = False`, default) — every committed
  phrase reaches the LLM, which gates via `<ignore>`/`<reply>` (see
  [01_architecture.md §LLM gate](01_architecture.md#llm-gate)).
- **Strict wake** (`REQUIRE_WAKE_WORD = True`) — original Google-Home flow.

The "soft hotword / engaged mode" idea got replaced by the LLM gate. The
LLM understands context (phrasing, recent history) better than any wake
phrase heuristic, so we let it decide directly. The audio pipeline doesn't
gatekeep, the LLM does. Closed.

## 4. Default STT model

Trade-off between latency and accuracy:
- `base.en` — ~250 ms per phrase on M-series, makes errors on hard words.
- `medium.en` — ~700 ms, much better.
- `large-v3-turbo` — closer to `medium` in speed, much closer to `large`
  in quality. Worth trialing.

**Current shipped state:** two_pass mode loads both `base.en` and
`medium.en` eagerly at startup with real warm passes; every phrase runs
medium.en in M3 mode. Continuous mode uses `STT_CONTINUOUS_MODEL`
(default `base.en`). Head-to-head against `large-v3-turbo` is still owed
when accuracy matters more than speed.

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

## 7. Lilith-AI sibling repo ✅ DEFERRED

Sibling `Lilith-AI/` directory is unrelated to VoiceLLM. Not pulling
anything from it. Closed.

## 8. macOS sandboxing / TCC microphone

Running through VS Code's terminal sometimes inherits the editor's mic
permission, sometimes asks fresh, sometimes silently fails. Not a code
problem but worth a one-line note in the README so future-us doesn't
re-debug it.
