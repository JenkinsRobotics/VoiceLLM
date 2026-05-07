# VoiceLLM — Status & Handoff

**For a fresh LLM picking this up cold.** Read this first, then
[06_milestones.md](06_milestones.md) for the milestone definitions and
[01_architecture.md](01_architecture.md) for the module/bus layout.

Last updated: **2026-05-07**, end of M2.

---

## TL;DR

Modular, bus-driven local voice assistant on Apple Silicon. Final code lives
in [04_SOFTWARE/](../). Demo/reference code that informed the design lives in
the sibling [MockingAgent/](../../../MockingAgent/) repo (notably
[voice_assistant.py](../../../MockingAgent/voice_assistant.py) — the
proven Google-Home-style baseline we ported from).

**Stack:** sounddevice + WebRTC VAD → pywhispercpp (whisper.cpp) → swappable
LLM (mlx-lm *or* llama-cpp-python, both running **Gemma 4 26B-A4B 4-bit**) →
Kokoro TTS. State + routing through a Pub/Sub `Bus`.

**M2 status: complete.** A wake-word voice assistant runs end-to-end with the
new modular layout, and the LLM backend swaps between MLX and llama.cpp via
one config flag.

---

## What works right now (M2 complete)

```
cd VoiceLLM/04_SOFTWARE
python main.py
# Say: "okay jaeger, what time is it?"
```

This reproduces [MockingAgent/voice_assistant.py](../../../MockingAgent/voice_assistant.py)'s
behavior, but every concern is now its own node communicating over the bus.

### Modules in place

| File | Role |
|---|---|
| [config.py](../config.py) | All tunables. Flip `LLM_BACKEND` between `"mlx"` and `"llamacpp"`. |
| [core/bus.py](../core/bus.py) | Single-queue pub/sub (poll-based via `get(timeout)`). |
| [core/state.py](../core/state.py) | `SysState`: `IDLE`/`THINKING`/`RESPONDING`. |
| [core/metrics.py](../core/metrics.py) | Per-turn timing → `metrics.csv`. |
| [audio/mic_stream.py](../audio/mic_stream.py) | `MicStream` with `paused` flag (ported from voice_assistant.py:96-125). |
| [audio/vad.py](../audio/vad.py), [audio/aec.py](../audio/aec.py), [audio/wakeword.py](../audio/wakeword.py) | Existing — used in M4 (AEC) and the legacy `stt_node.py`; **not yet wired** into the new flow. |
| [stt/stt_two_pass.py](../stt/stt_two_pass.py) | Full port of voice_assistant.py: VAD worker, fast→accurate Whisper cascade, wake-word + follow-up window. Publishes `stt.text`. |
| [stt/stt_node.py](../stt/stt_node.py) | Old VAD-segmented Whisper node. **Unused now** but left in tree. |
| [llm/backend_base.py](../llm/backend_base.py) | `BackendBase` ABC: `load`, `warm`, `stream_chat`, `cancel`. |
| [llm/backend_mlx.py](../llm/backend_mlx.py) | mlx-lm impl. Registers Gemma 4 EOT token. |
| [llm/backend_llamacpp.py](../llm/backend_llamacpp.py) | llama-cpp-python impl. |
| [llm/llm_node.py](../llm/llm_node.py) | Owns history, streams `llm.token` deltas; publishes cleaned reply on `llm.done`. `clean_for_tts()` ported. |
| [tts/kokoro_node.py](../tts/kokoro_node.py) | Real `KPipeline`. Synth thread + play thread. Sentence-streams. Cancellable. Publishes `mic.pause`, `tts.audio_chunk`, `tts.done`. |
| [orchestrator/orchestrator.py](../orchestrator/orchestrator.py) | Single bus consumer; state machine; spawns LLM thread per turn. |
| [main.py](../main.py) | `make_backend()` + `make_stt()` factory funcs, then `Orchestrator(...).run()`. |

### Bus topics in use

- `stt.text` (str) — committed user phrase, post-wake-word.
- `llm.token` (str) — streaming reply delta.
- `llm.done` (str) — full cleaned reply, fired after the stream ends.
- `mic.pause` (bool) — TTS toggles this around playback.
- `tts.audio_chunk` (np.float32) — published before `sd.play()`; nobody
  consumes it yet (subscriber is **M4** — AEC reference + similarity filter).
- `tts.done` (None) — TTS audio queue drained.

### Models & paths (verified on disk)

```
LMSTUDIO_MODELS = ~/.lmstudio/models/
MLX_PATH        = LMSTUDIO_MODELS/mlx-community/gemma-4-26b-a4b-4bit/
GGUF_PATH       = LMSTUDIO_MODELS/lmstudio-community/gemma-4-26B-A4B-it-GGUF/
                  gemma-4-26B-A4B-it-Q4_K_M.gguf
```

STT: `base.en` (fast) → `medium.en` (accurate, lazy-loaded on first wake match).

---

## Repo layout right now

```
GITHUB/
├── MockingAgent/                       # working Google-Home-style baseline
│   ├── voice_assistant.py              # the canonical reference for STT/TTS plumbing
│   ├── ollamacpp/                      # chat_mlx.py, chat_llama.py, bench.py
│   ├── kokoro_tts/                     # standalone Kokoro experiments
│   ├── PywisperCpp/                    # all the always-listening STT demos
│   └── legacy_voicellm_drafts/         # old loose demos that used to live in VoiceLLM/
│
└── VoiceLLM/                           # final clean code lives here
    ├── 00_REFERENCES/                  # research notes, organized by topic
    ├── 04_SOFTWARE/                    # ← THE CODE
    │   ├── config.py
    │   ├── main.py
    │   ├── audio/  core/  llm/  stt/  tts/  orchestrator/
    │   ├── docs/                       # ← these planning docs
    │   ├── requirements.txt
    │   └── metrics.csv                 # auto-written by MetricsLog
    ├── models/                         # local model files (mostly symlinks)
    ├── LICENSE
    └── README.md
```

---

## What's next (build order)

### M3 — Continuous hearing (the actual goal)

**Drop the wake word.** STT runs always-on; every committed phrase becomes
a turn unless we filter it out. This is the "ChatGPT Voice" feel.

1. **Port a continuous pipeline** to `stt/stt_continuous.py`. Default to the
   **hybrid phrase/word** strategy from
   [always_listening_hybrid_phrase_word_pipeline.py](../../../MockingAgent/PywisperCpp/pywhispercpp_examples/llm_listener/always_listening_hybrid_phrase_word_pipeline.py)
   per [02_stt_pipelines.md](02_stt_pipelines.md). Same node interface as
   `STTTwoPassNode` (publishes `stt.text`, has `start`/`stop`/`set_paused`).
2. **`make_stt()` in `main.py`** already has the dispatch hook — add the
   `STT_MODE == "continuous"` branch.
3. **Self-speech similarity filter** on `stt.text` ingress in the
   orchestrator: if the candidate text is `>0.75` similar to the most recent
   `assistant` turn in `LLMNode.history_snapshot()`, drop it. (The mic-pause
   already handles 95% of cases; this is for the residual 5%.)
4. **Cooldown**: if a second `stt.text` arrives within ~600 ms of one already
   being processed, queue it instead of dropping it. Prevents double-fires
   while we're still synthesizing the first reply.
5. **Verification**: run alongside a YouTube video for 5 minutes; the LLM
   should not fire on background dialogue. Log to `outputs/m3_eval.jsonl`.

The current orchestrator drops new `stt.text` messages while a turn is in
flight ([orchestrator.py:80-83](../orchestrator/orchestrator.py)) — that's
a placeholder; M4's barge-in path replaces it with cancellation.

### M4 — Barge-in

Talk over the assistant; it cuts off and listens.

1. **Wire AEC**: `audio/aec.py` exists and `AECWrapper` is already
   constructed in the *old* orchestrator. The new orchestrator doesn't use
   it yet. Subscribe to `tts.audio_chunk` for the far-end reference, run
   the mic frames through AEC before passing them to the VAD.
2. **VAD on cleaned audio** while `state == RESPONDING`: when VAD says
   speech for ≥150 ms, publish `tts.cancel`, call `llm.cancel()`,
   transition `state = LISTENING`. Add a 250 ms start-grace at the top of
   each TTS turn so the speaker click doesn't self-trigger.
3. **Add `tts.cancel` topic** to the bus contract; route it in the
   orchestrator's `_dispatch`. `KokoroNode.cancel()` already exists and
   does the right thing.
4. **`config.BARGE_IN_ENABLED` and `AEC_ENABLED`** are already wired; flip
   them on once 1-3 are in.

### M5 — Polish

- Latency dashboard: `metrics.csv` is already being written; add a tiny
  live print of TTFT/first-audio per turn.
- Voice picker (`config.KOKORO_VOICE`).
- System-prompt presets.
- Optional GUI (PySide6 demo exists in MockingAgent).

---

## Known gotchas

1. **`core/bus.py` is single-consumer.** Only the orchestrator calls
   `bus.get()`. If we ever want a second subscriber on the same topic
   (likely in M4: AEC and similarity-filter both need `tts.audio_chunk`),
   add a `subscribe(topic, cb)` fanout to `Bus`. See
   [07_open_questions.md §1](07_open_questions.md).
2. **TTS publishes `mic.pause` *before* `sd.play()` returns.** The
   orchestrator forwards it to `STTTwoPassNode.set_paused()` which calls
   `MicStream.set_paused()`. Check the exact ordering in
   [tts/kokoro_node.py:_play_loop](../tts/kokoro_node.py) before tightening
   barge-in timing — there's a `tail_sleep_s = 0.12` to let speakers drain
   before un-pausing the mic.
3. **The legacy `stt/stt_node.py` and `audio/audio_io.py`** are still in
   tree but unused. They use `tempfile`-based whisper transcription and a
   different mic abstraction. Don't import them from new code; either
   remove or leave as historical reference. Decision deferred to M5.
4. **`webrtcvad` vs `webrtcvad-wheels`**: requirements.txt asks for
   `-wheels` (prebuilt). The old root requirements named bare `webrtcvad`
   which builds from source. Consistent now.
5. **Gemma 4 in `mlx-lm`** needs the EOT token added explicitly or it'll
   over-generate; handled in
   [backend_mlx.py:load()](../llm/backend_mlx.py).
6. **First-run latency**: `KokoroNode.__init__` does a 1-line warm-up synth
   that takes ~3-5 s. `BackendBase.warm()` does a 1-token gen, also a few
   seconds. Both run before the mic opens.
7. **macOS mic permission**: launching from VS Code's terminal sometimes
   inherits the editor's TCC grant, sometimes prompts. If `MicStream`
   silently captures zeros, that's the issue.

---

## Sanity-check commands

```bash
# Compile-check all M2 modules:
cd VoiceLLM/04_SOFTWARE
python -m py_compile config.py main.py \
  llm/backend_base.py llm/backend_mlx.py llm/backend_llamacpp.py llm/llm_node.py \
  tts/kokoro_node.py audio/mic_stream.py stt/stt_two_pass.py \
  orchestrator/orchestrator.py core/bus.py core/state.py core/metrics.py

# Confirm models exist:
python -c "import config as c; print('mlx:', c.MLX_PATH.exists(), 'gguf:', c.GGUF_PATH.exists())"

# Run end-to-end (loads ~5 GB into memory):
python main.py
```

---

## If you're picking this up cold

1. Read this file.
2. Read [00_overview.md](00_overview.md) and [01_architecture.md](01_architecture.md).
3. Read [voice_assistant.py](../../../MockingAgent/voice_assistant.py) — that
   is the canonical reference for *every* STT/TTS/LLM glue decision in M2.
4. Read [02_stt_pipelines.md](02_stt_pipelines.md) before touching M3.
5. Don't refactor the legacy files (`stt/stt_node.py`,
   `audio/audio_io.py`, the old `kokoro_node.py` was already replaced) —
   delete them in M5 if they're still unused.
