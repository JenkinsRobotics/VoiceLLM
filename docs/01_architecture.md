# Architecture

## Goal of the layout

Each concern is a node. Nodes never call each other directly — they
publish/subscribe to a message bus. That way we can swap any one
(LLM backend, STT strategy, TTS engine) without rewriting the others.

All code lives at the repo root: `core/bus.py`, `orchestrator/`, `llm/`,
`stt/`, `tts/`, `audio/`. We are *keeping* that scaffold and tightening
the contracts.

## Module map

```
VoiceLLM/                      # repo root
├── config.py                  # central tunables (sample rate, device IDs,
│                              # backend selection, model paths, prompts)
├── main.py                    # build bus + nodes, start orchestrator
│
├── core/
│   ├── bus.py                 # pub/sub queue (already present)
│   ├── state.py               # IDLE / LISTENING / THINKING / RESPONDING
│   └── metrics.py             # per-turn timing log to metrics.csv
│
├── audio/
│   ├── audio_io.py            # sounddevice InputStream → mic frames
│   ├── vad.py                 # WebRTC VAD segmenter
│   ├── aec.py                 # optional acoustic echo canceller
│   └── wakeword.py            # (optional) soft hotword trigger
│
├── stt/
│   ├── stt_node.py            # currently: VAD-segmented file-based whisper
│   ├── stt_continuous.py      # NEW: rolling-window always-listening
│   └── stt_two_pass.py        # NEW: fast/accurate cascade (MockingAgent style)
│
├── llm/
│   ├── llm_node.py            # bus-facing node (publishes llm.token / llm.done)
│   ├── backend_base.py        # NEW: BackendBase ABC (load, warm, stream_chat)
│   ├── backend_llamacpp.py    # NEW: llama-cpp-python implementation
│   └── backend_mlx.py         # NEW: mlx-lm implementation
│
├── tts/
│   ├── kokoro_node.py         # streams sentences to Kokoro KPipeline,
│   │                          #   coordinates mic-pause and barge-in
│   └── playback.py            # (optional split) sounddevice player
│
├── orchestrator/
│   └── orchestrator.py        # state machine, wires bus topics to nodes
│
└── docs/                      # this folder
```

## Bus topics

All cross-node communication goes through `core.bus.Bus.publish(topic, payload)`.

| Topic | Payload | Producer → Consumer |
|---|---|---|
| `mic.frame` | `np.int16` PCM block | audio_io → STT, AEC ref |
| `mic.pause` | `bool` | TTS → audio_io (gate captures while we speak) |
| `vad.speech_start` | timestamp | VAD → orchestrator (used for barge-in) |
| `vad.speech_end` | timestamp | VAD → STT (commit phrase) |
| `stt.partial` | `str` | STT → orchestrator (live transcript, optional) |
| `stt.text` | `str` | STT → orchestrator (committed phrase) |
| `llm.request` | `str` (user text) | orchestrator → LLM |
| `llm.token` | `str` (delta) | LLM → TTS, orchestrator |
| `llm.done` | `None` | LLM → orchestrator |
| `tts.sentence` | `str` | TTS internal (sentence boundary detected) |
| `tts.audio_chunk` | `np.int16` PCM | TTS → playback, AEC ref |
| `tts.done` | `None` | TTS → orchestrator |
| `tts.cancel` | `None` | orchestrator → TTS (barge-in) |
| `state.change` | new state | orchestrator → everyone (UI/log) |

## Lifecycle

```
mic.frame ──► VAD ──► stt.partial / stt.text
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
- `LISTENING` — VAD says user is currently speaking, we're accumulating audio.
- `THINKING` — request sent to LLM, awaiting first token. Also the gate
  decision phase (see "LLM gate" below).
- `RESPONDING` — gate decided `<reply>`; LLM streaming, TTS speaking.
- `INTERRUPTED` — barge-in detected; cancel TTS, return to `LISTENING`.

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
([orchestrator/orchestrator.py](../orchestrator/orchestrator.py)) and
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
2. We want barge-in: a single `tts.cancel` message has to reach TTS without
   the orchestrator knowing what implementation is currently running.
3. It makes a future GUI / Slack / log sink trivial — just subscribe.

## What changes from the existing code

The existing `orchestrator.py` already imports STT, AEC, RingAudio. We will:

1. Move LLM out of `llm_node.py`'s hard-coded Ollama HTTP call into a
   `BackendBase`-shaped interface; `llm_node.py` becomes a thin bus
   adapter that owns one backend instance.
2. Add a continuous-mode STT alongside the existing VAD-segmented one,
   selected by `config.STT_MODE`.
3. Add a `tts.cancel` path through `KokoroNode` so we can interrupt mid-reply.
4. Replace the synth stub in `kokoro_node.py` with the real `KPipeline`
   (from MockingAgent's `voice_assistant.py`).
