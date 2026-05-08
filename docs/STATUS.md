# VoiceLLM — Status & Handoff

**For a fresh LLM picking this up cold.** Read this first, then
[06_milestones.md](06_milestones.md) for the milestone definitions and
[01_architecture.md](01_architecture.md) for the module/bus layout.

Last updated: **2026-05-07**. M2 complete; M3 quick path shipped; M3.5
continuous STT in tree (opt-in via `STT_MODE`); LLM-gated speech protocol
shipped; both Whisper sizes eager-loaded with real warm-up.

---

## TL;DR

Modular, bus-driven local voice assistant on Apple Silicon. All code lives
at the repo root (this `docs/` folder is a sibling of `config.py`,
`main.py`, `core/`, `stt/`, `llm/`, `tts/`, `audio/`, `orchestrator/`).
Demo/reference code that informed the design lives in the sibling
[MockingAgent/](../../MockingAgent/) repo (notably
[voice_assistant.py](../../MockingAgent/voice_assistant.py) — the
proven Google-Home-style baseline we ported from).

**Stack:** sounddevice + WebRTC VAD → pywhispercpp (whisper.cpp) → swappable
LLM (`llama-cpp-python` default *or* `mlx-lm`, both running **Gemma 4
26B-A4B 4-bit**) → Kokoro TTS. State + routing through a Pub/Sub `Bus`.

**M2 status: complete.** Modular voice loop runs end-to-end. Both LLM
backends supported; `LLM_BACKEND = "llamacpp"` is the default since
llama-cpp-python handles Gemma's `<end_of_turn>` natively. The MLX backend
now stops correctly via an in-stream marker detector
([backend_mlx.py](../llm/backend_mlx.py)) — the old `eot_token` getattr was
a no-op and let Gemma run to `max_tokens` and loop.

**M3 status (quick path): shipped.** No wake word in the default mode.
The orchestrator runs a self-speech similarity filter, a single-slot
pending-turn queue, and the **LLM gate** — every reply must begin with
`<ignore>` or `<reply>`; ignored replies are suppressed from TTS without
audible cost. Every decision is logged to `outputs/m3_eval.jsonl`.

**M3.5 status: in tree, opt-in.** [stt/stt_continuous.py](../stt/stt_continuous.py)
is a drop-in replacement for `STTTwoPassNode` with rolling re-transcription
and energy-based phrase segmentation. Activate via
`STT_MODE = "continuous"` in [config.py](../config.py). Default stays
`"two_pass"` so the proven baseline is unchanged.

**Polish since M2:** all four model loads (fast STT, accurate STT, LLM,
TTS) happen at startup with real warm passes — first user turn pays only
inference time, not setup. Conversation history is capped at
`MAX_HISTORY_TURNS = 8` user/assistant pairs to keep prompt size bounded.

---

## What works right now (M2 + M3 quick path)

```
cd VoiceLLM
python main.py
# REQUIRE_WAKE_WORD=False is the default — just talk.
# Flip back to True in config.py for the wake-word "okay jaeger" flow.
```

This reproduces [MockingAgent/voice_assistant.py](../../MockingAgent/voice_assistant.py)'s
behavior, but every concern is now its own node communicating over the bus.

### Modules in place

| File | Role |
|---|---|
| [config.py](../config.py) | All tunables. Flip `LLM_BACKEND` between `"llamacpp"` (default) and `"mlx"`; flip `STT_MODE` between `"two_pass"` (default) and `"continuous"`. |
| [core/bus.py](../core/bus.py) | Single-queue pub/sub (poll-based via `get(timeout)`). |
| [core/state.py](../core/state.py) | `SysState`: `IDLE`/`THINKING`/`RESPONDING`. |
| [core/metrics.py](../core/metrics.py) | Per-turn timing → `metrics.csv`. |
| [audio/mic_stream.py](../audio/mic_stream.py) | `MicStream` with `paused` flag (ported from voice_assistant.py:96-125). |
| [audio/vad.py](../audio/vad.py), [audio/aec.py](../audio/aec.py), [audio/wakeword.py](../audio/wakeword.py) | Existing — used in M4 (AEC) and the legacy `stt_node.py`; **not yet wired** into the new flow. |
| [stt/stt_two_pass.py](../stt/stt_two_pass.py) | Two-pass cascade ported from voice_assistant.py: VAD worker, fast (`base.en`) + accurate (`medium.en`) eager-loaded with silence warm-up, wake-word + follow-up window. Publishes `stt.text`. |
| [stt/stt_continuous.py](../stt/stt_continuous.py) | M3.5 hybrid pipeline. Energy-based phrase segmentation, rolling re-transcription. Drop-in interface match for `STTTwoPassNode`. |
| [stt/stt_node.py](../stt/stt_node.py) | Old VAD-segmented Whisper node. **Unused** — slated for deletion in M5. |
| [llm/backend_base.py](../llm/backend_base.py) | `BackendBase` ABC: `load`, `warm`, `stream_chat`, `cancel`. |
| [llm/backend_mlx.py](../llm/backend_mlx.py) | mlx-lm impl. In-stream stop-marker detector for `<end_of_turn>` / `<eos>` / `<im_end>` (replaces the old broken `eot_token` getattr). |
| [llm/backend_llamacpp.py](../llm/backend_llamacpp.py) | llama-cpp-python impl. Default backend; chat completion handles Gemma's stop tokens natively. |
| [llm/llm_node.py](../llm/llm_node.py) | Owns history; streams `llm.token` deltas; publishes cleaned reply on `llm.done`. Caps history at `MAX_HISTORY_TURNS = 8` pairs. `clean_for_tts()` ported. |
| [tts/kokoro_node.py](../tts/kokoro_node.py) | Real `KPipeline`. Synth thread + play thread. Sentence-streams. Cancellable. Publishes `mic.pause`, `tts.audio_chunk`, `tts.done`. |
| [orchestrator/orchestrator.py](../orchestrator/orchestrator.py) | Single bus consumer; state machine; spawns LLM thread per turn; **LLM-gate token buffer** (`<ignore>` suppresses TTS, `<reply>` forwards the tail). |
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

STT: `base.en` (fast) and `medium.en` (accurate) — **both eager-loaded at
startup** with a 1.5 s silence warm pass, so first user turn pays only
inference cost, not setup. Old behavior lazy-loaded `medium.en` on the
first wake match (a hangover from M2's wake-word path); pointless now that
M3 fires the accurate model on every phrase.

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
└── VoiceLLM/                           # ← THE CODE (flat at the repo root)
    ├── config.py
    ├── main.py
    ├── audio/  core/  llm/  stt/  tts/  orchestrator/
    ├── docs/                           # ← these planning docs
    ├── outputs/                        # m3_eval.jsonl etc.
    ├── requirements.txt
    ├── metrics.csv                     # auto-written by MetricsLog
    ├── models/                         # local model files (mostly symlinks)
    ├── LICENSE
    └── README.md
```

---

## What's next (build order)

### M3 — Continuous hearing (the actual goal)

**Drop the wake word.** STT runs always-on; every committed phrase becomes
a turn unless we filter it out. This is the "ChatGPT Voice" feel.

We split this into a **quick path** (lean on the existing two-pass STT) and
**M3.5** (build the hybrid pipeline node for lower latency / better feel).

#### M3 quick path — shipped

1. ✅ **`REQUIRE_WAKE_WORD = False`** in [config.py](../config.py). The
   existing `STTTwoPassNode` already has the no-wake-word branch
   ([stt_two_pass.py](../stt/stt_two_pass.py)) — every phrase becomes a turn.
2. ✅ **Self-speech similarity filter** on `stt.text` ingress in the
   orchestrator. Compares incoming text against the most recent `assistant`
   turn from `LLMNode.history_snapshot()` via `difflib.SequenceMatcher`;
   drops if `>= cfg.SELF_SPEECH_SIMILARITY_THRESHOLD` (default 0.75).
3. ✅ **Pending-turn queue** replaces the old "drop while busy" placeholder.
   Single-slot, last-write-wins; fires when `_on_tts_done` returns to IDLE,
   provided the queued utterance is younger than `cfg.PENDING_TURN_MAX_AGE_S`
   (default 3.0 s).
4. ✅ **LLM gate** — the user explicitly didn't want the audio pipeline
   deciding what's directed speech. So the LLM does. Every reply must begin
   with `<ignore>` or `<reply>` (instruction in
   [config.py:SYSTEM_PROMPT](../config.py)). The orchestrator buffers the
   first `LLM_GATE_BUFFER_CHARS = 30` chars of the streaming reply, decides
   based on the tag, and either:
   - `<ignore>` → suppresses TTS, transitions straight to IDLE, logs
     `llm_ignored` to the eval JSONL.
   - `<reply>` → forwards the post-tag tail to TTS as normal.
   Falls back to "treat as reply" if the tag never appears within the
   buffer. See `_on_llm_token` and `_gate_check` in
   [orchestrator/orchestrator.py](../orchestrator/orchestrator.py).
5. ✅ **Eval logging** — every STT decision (`accepted` /
   `dropped_self_echo` / `queued_pending` / `pending_fired` /
   `pending_stale` / `llm_ignored`) is appended to
   `outputs/m3_eval.jsonl` for offline review. Disable by setting
   `cfg.M3_EVAL_LOG = None`.

#### M3 quick path — verification still owed

- [ ] **Run alongside a YouTube video for 5 minutes.** The LLM should not
  fire on background dialogue. Inspect `outputs/m3_eval.jsonl` afterwards;
  expect `dropped_self_echo` for assistant playback and a few `accepted`
  for real user turns. Tune `SELF_SPEECH_SIMILARITY_THRESHOLD` if false
  positives slip through.
- [ ] **Sanity check on the barge-in placeholder** — until M4 ships, talking
  over the assistant queues your follow-up rather than interrupting; that
  reads in the log as `queued_pending` → `pending_fired`.

#### M3.5 — Hybrid phrase/word STT node

Only do this once the quick path is verified and we hit a quality wall the
filter+queue can't paper over (e.g. trailing-word loss on long sentences).

1. ✅ **Port** [always_listening_hybrid_phrase_word_pipeline.py](../../MockingAgent/PywisperCpp/pywhispercpp_examples/llm_listener/always_listening_hybrid_phrase_word_pipeline.py)
   to [stt/stt_continuous.py](../stt/stt_continuous.py). Same node
   interface as `STTTwoPassNode` (publishes `stt.text`, has
   `start`/`stop`/`set_paused`/`open_followup`). Energy-based phrase
   segmentation; rolling re-transcription every
   `cfg.STT_TRANSCRIBE_EVERY_S`; phrase commits on
   `cfg.STT_PHRASE_TIMEOUT_S` of quiet. Single Whisper model
   (`cfg.STT_CONTINUOUS_MODEL`, default `base.en`).
2. ✅ **`make_stt()` in [main.py](../main.py)** — `STT_MODE == "continuous"`
   branch wired with the M3.5 tunables.
3. **Verification still owed** — flip `STT_MODE = "continuous"` in
   [config.py](../config.py), run `python main.py`, and compare:
   - First-token latency vs. two_pass (rolling re-transcription should
     cut the wait at phrase end).
   - False-positive rate on background dialogue (rerun the YouTube test
     against `outputs/m3_eval.jsonl`). M3.5 uses base.en by default; if
     accuracy lags, raise `STT_CONTINUOUS_MODEL` to `small.en` or
     `medium.en`.

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
5. **Gemma 4 in `mlx-lm`** doesn't stop on `<end_of_turn>` by default —
   the tokenizer wrapper's API for additional EOS tokens varies by mlx-lm
   version. We solved it via an in-stream stop-marker detector in
   [backend_mlx.py:stream_chat()](../llm/backend_mlx.py) that buffers the
   last 32 chars of streamed text and stops when it sees `<end_of_turn>`,
   `<eos>`, or `<|im_end|>`. Without this fix, MLX runs to `max_tokens`
   and loops.
6. **First-run latency**: all four model loads happen at startup before
   the mic opens — Kokoro 1-line warm synth (~3-5 s), `BackendBase.warm()`
   1-token gen (~1-2 s), both Whisper sizes plus a 1.5 s silence transcribe
   each (~1-2 s combined). First user turn then pays only inference time.
7. **Whisper warm-up needs ≥1000 ms of audio**; we use 1.5 s of silence
   in [stt_two_pass.py:_warm_stt()](../stt/stt_two_pass.py) and
   [stt_continuous.py](../stt/stt_continuous.py). Sub-1000-ms warm
   transcribes get rejected by Whisper with `input is too short - ... ms`
   and silently skip inference, defeating the whole point.
8. **macOS mic permission**: launching from VS Code's terminal sometimes
   inherits the editor's TCC grant, sometimes prompts. If `MicStream`
   silently captures zeros, that's the issue.

---

## Sanity-check commands

```bash
# Compile-check all M2 + M3.5 modules:
cd VoiceLLM
python -m py_compile config.py main.py \
  llm/backend_base.py llm/backend_mlx.py llm/backend_llamacpp.py llm/llm_node.py \
  tts/kokoro_node.py audio/mic_stream.py \
  stt/stt_two_pass.py stt/stt_continuous.py \
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
3. Read [voice_assistant.py](../../MockingAgent/voice_assistant.py) — that
   is the canonical reference for *every* STT/TTS/LLM glue decision in M2.
4. Read [02_stt_pipelines.md](02_stt_pipelines.md) before touching M3.
5. Don't refactor the legacy files (`stt/stt_node.py`,
   `audio/audio_io.py`, the old `kokoro_node.py` was already replaced) —
   delete them in M5 if they're still unused.
