# Architecture

## Goal of the layout

Each concern is a node. Nodes never call each other directly — they
publish/subscribe to a message bus. That way we can swap any one
(LLM backend, STT strategy, TTS engine) without rewriting the others.

All code lives at the repo root. The cognitive layer is under `agent/`,
peripheral I/O nodes are under `nodes/`, the bus is under `transport/`,
shared infrastructure (metrics, state) is under `core/`, and historical
demos live in `references/`.

## Module map

```
VoiceLLM/                      # repo root
├── config.py                  # central tunables (sample rate, device IDs,
│                              # backend selection, model paths, prompts)
├── main.py                    # build bus + nodes, start orchestrator
├── smoke_test.py              # 13 fast scenarios, no model loads
│
├── agent/                     # cognitive layer
│   ├── orchestrator.py        # state machine, wires bus topics to nodes
│   ├── llm/
│   │   ├── node.py            # LLM bus adapter; owns history + ctx guard
│   │   └── backend_base.py    # BackendBase ABC (load/warm/stream/cancel)
│   └── adapters/
│       ├── llama_cpp/backend.py   # llama-cpp-python backend (default)
│       └── mlx/backend.py         # mlx-lm backend
│
├── nodes/                     # peripheral I/O
│   ├── audio_session/
│   │   ├── mic_stream.py      # sounddevice InputStream → mic frames
│   │   └── chimes.py          # wake / follow-up earcons
│   ├── stt/
│   │   ├── two_pass.py        # VAD-gated fast→accurate cascade (default)
│   │   └── continuous.py      # rolling re-transcription (opt-in, unverified)
│   └── tts/node.py            # Kokoro KPipeline synth + playback threads
│
├── transport/bus.py           # in-process pub/sub fanout
├── core/
│   ├── state.py               # IDLE / THINKING / RESPONDING
│   └── metrics.py             # per-turn timing log to metrics.csv
│
└── docs/                      # this folder
```

## Bus topics

All cross-node communication goes through
`transport.bus.Bus.publish(topic, payload)`. Subscribers call
`bus.subscribe(topics)` for their own queue; the orchestrator holds the
default catch-all subscription (`bus.get()`). Raw mic audio never goes on
the bus — it stays on dedicated queues (`MicStream.q`, `phrase_q`,
`audio_q`).

| Topic | Payload | Producer → Consumer |
|---|---|---|
| `mic.pause` | `bool` | TTS → orchestrator → STT (also called directly for synchronous pause) |
| `stt.text` | `dict` (text + timing) | STT → orchestrator (committed phrase) |
| `llm.token` | `str` (delta) | LLM → orchestrator (gate, then TTS) |
| `llm.done` | `str` (cleaned reply) | LLM → orchestrator |
| `llm.error` | `str` | LLM → orchestrator (speaks a fallback) |
| `tts.audio_chunk` | `np.float32` PCM | TTS → *(planned)* M4 AEC reference |
| `tts.done` | `None` | TTS → orchestrator |
| `tts.cancel` | `None` | *(planned)* orchestrator → TTS (M4 barge-in) |

## Lifecycle

```
mic frames ──► VAD worker ──► stt.text {text, timing}
                              │
                              ▼
              orchestrator (state machine)
                              │
                              ▼
                      LLM (stream tokens)
                              │
                              ▼
                  TTS (sentence buffering → audio chunks → speaker)
                              │
                              └──► mic.pause(True/False), tts.audio_chunk
```

States in `core/state.py`:
- `IDLE` — hearing but no active turn.
- `THINKING` — request sent to LLM, awaiting first token. Also the gate
  decision phase (see "LLM gate" below).
- `RESPONDING` — gate decided `<reply>`; LLM streaming, TTS speaking.

*(planned, M4)*: `LISTENING` (VAD-active accumulation) and `INTERRUPTED`
(barge-in detected; cancel TTS) — neither exists in code yet.

## LLM gate

The most important architectural decision in M3: **the audio pipeline does
not decide what's directed speech. The LLM does.** When `REQUIRE_WAKE_WORD
= False`, every committed phrase reaches the LLM, including transcription
artifacts (`[BLANK_AUDIO]`), keystroke noise, ambient TV, and overheard
conversation. We could filter at the audio side with regex / energy / VAD
thresholds, but those heuristics drift and miss the intent. So the system
prompt in [config.py:SYSTEM_PROMPT](../config.py) requires the LLM to
prefix every reply with one of two tags:

- `<ignore>` — the input is not addressed to the assistant. Output the tag
  and nothing else.
- `<reply>` — the input is a real directed turn. Follow the tag with a
  1-3 sentence answer.

The orchestrator buffers the first `LLM_GATE_BUFFER_CHARS = 30` chars of
each streaming reply in `_on_llm_token`
([agent/orchestrator.py](../agent/orchestrator.py)) and
checks for the tags:

- `<ignore>` found → mark the turn `_gate_ignore = True`, discard all
  subsequent tokens, log `llm_ignored` to `outputs/m3_eval.jsonl`. On
  `llm.done` the orchestrator calls `_on_tts_done()` directly to do the
  IDLE handoff (TTS never started, so `tts.done` would never fire).
- `<reply>` found → forward everything *after* the tag to TTS. State
  transitions `THINKING → RESPONDING` exactly when the first post-tag
  delta hits.
- 30 chars elapsed with no tag → fallback: treat as `<reply>`. The LLM
  forgot the protocol; we'd rather speak the response than silently drop it.

**Latency cost:** typically ~50-150 ms (1-2 streaming tokens to see the
tag) on the reply path. Zero audible cost on the ignore path. The buffer
size is the only knob ([config.py:LLM_GATE_BUFFER_CHARS](../config.py));
shrink it for faster fallback when the LLM goes off-protocol.

**Why this design wins:** the same gate handles all of (a) self-speech
the similarity filter missed, (b) ambient TV / room conversation,
(c) Whisper hallucinations on noise, (d) keystroke artifacts. Future
work doesn't need to write more filters; we just give the LLM more
context in the system prompt about when to choose `<ignore>`.

## Why a bus and not direct calls

1. We want to swap STT/LLM/TTS independently. Direct calls would couple them.
2. We want barge-in *(planned, M4)*: a single `tts.cancel` message has to
   reach TTS without the orchestrator knowing what implementation is
   currently running.
3. It makes a future GUI / Slack / log sink trivial — just
   `bus.subscribe()`.

One deliberate exception to "everything over the bus": TTS also calls
`stt.set_paused` directly (wired in `main.py`) so the mic pause is
synchronous with playback start — the bus round-trip only lands after the
orchestrator's next dispatch.

## Threading model

All orchestrator state is **dispatch-thread-only**: every handler
(`_on_stt_text`, `_on_llm_token`, `_on_llm_done`, `_on_tts_done`) runs on
the single thread inside `Orchestrator.run()`. Worker threads (mic
callback, VAD worker, STT main loop, per-turn LLM thread, TTS synth/play
threads) communicate with it exclusively via the bus or locked/queued
structures. Keep it that way — don't mutate orchestrator fields from a
node thread.
