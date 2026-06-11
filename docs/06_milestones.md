# Milestones — build order

Each milestone is a single runnable command that *demonstrates* a working
behavior. Don't move on until the previous demo runs reliably.

**Status pointer:** M0–M3 shipped (see [STATUS.md](STATUS.md) for the
authoritative current state); M3.5 is in tree but unverified live; M4/M5
are planned.

## M0 — Repo wiring (no behavior change) — done

Goal: docs in place, requirements installable, models verified.

- [x] `docs/` written.
- [x] `pip install -r requirements.txt` succeeds in a fresh venv.
- [x] Model paths verified (`python -c "import config as c; print(c.GGUF_PATH.exists())"`
      replaced the planned `docs/check_models.py` script).

## M1 — Two CLI demos, ported in-place — superseded

The planned `demos/` folder was never created; the known-good scripts
live as historical copies in [references/](../references/) instead
(`voice_assistant.py` is the regression anchor), and the live backends
were ported directly into `agent/adapters/`.

## M2 — Modular voice assistant, no barge-in — done

Goal: the proven `voice_assistant.py` flow re-expressed through the bus +
nodes, with **MLX as the default LLM** and Gemma 4 26B-A4B-4bit.

- [x] `agent/llm/backend_base.py` — `BackendBase` ABC.
- [x] `agent/adapters/mlx/backend.py` — extracted from `chat_mlx.py:39-78`.
- [x] `agent/adapters/llama_cpp/backend.py` — extracted from `chat_llama.py:39-77`.
- [x] `agent/llm/node.py` — rewrite to consume a `BackendBase` instance.
- [x] `nodes/tts/node.py` — replace stub synth with real `KPipeline`,
      port `clean_for_tts()` and the mic-pause coordination.
- [x] `nodes/stt/two_pass.py` — port `voice_assistant.py:127-213` (the
      VAD worker + 2-pass cascade) onto the bus.
- [x] `config.py` — add `LLM_BACKEND`, `MLX_PATH`, `GGUF_PATH`,
      `STT_MODE = "two_pass"`, `KOKORO_VOICE`, voice prompts.
- [x] `main.py` — build bus, instantiate backend by config flag,
      start orchestrator.

Demo: `python main.py` reproduces the MockingAgent voice assistant
behavior, but switching `LLM_BACKEND="llamacpp"` swaps the backend
without touching anything else.

## M3 — Continuous hearing — done (quick path)

Goal: drop the wake word. STT transcribes constantly; any committed phrase
becomes a turn.

- [x] `nodes/stt/continuous.py` — port `always_listening_hybrid_phrase_word_pipeline.py`
      onto the bus (publishes `stt.text` on each commit).
- [x] `config.py` flip: `STT_MODE = "continuous"`, `REQUIRE_WAKE_WORD = False`.
- [x] `agent/orchestrator.py` — when `REQUIRE_WAKE_WORD = False`, every `stt.text`
      becomes an `llm.request`. Add cooldown so a too-quick second commit
      doesn't double-fire while we're still synthesizing the first reply.
- [x] Add `recent_assistant_reply` similarity filter (Layer C in
      `05_barge_in_and_self_speech.md`).

Demo: speak naturally, get a reply, keep talking, get another reply, no
"hey eve" needed. Background TV doesn't trigger the LLM (verified by
running it alongside a YouTube video for 5 minutes — `outputs/` log).

## M4 — Barge-in

Goal: interrupt the assistant by talking over it.

- [ ] AEC turned on by default (`AEC_ENABLED = True`).
- [ ] Run VAD on AEC-cleaned audio, not raw mic.
- [ ] `tts.cancel` path through `KokoroNode` (clear queue, `sd.stop()`).
- [ ] `backend.cancel()` plumbed through `LLMNode` so token generation
      stops too.
- [ ] Sustained-voice guard (≥150 ms) and start-grace (250 ms) before
      declaring barge-in.

Demo: while the assistant is mid-reply, talk over it. It cuts off within
~150 ms and processes the new utterance.

**AEC engine choice:** the old speexdsp wrapper sketch was deleted as dead
code (2026-06-10; recoverable from git history, and
[references/voice_chat.py](../references/voice_chat.py) preserves the
working experiment). Candidates remain
[`speexdsp`](https://pypi.org/project/speexdsp/) /
[`pyaec`](https://pypi.org/project/pyaec/) (easy macOS wheels) or
[`webrtc-audio-processing`](https://pypi.org/project/webrtc-audio-processing/)
(WebRTC APM with AEC + NS + AGC, but harder to build). Wire the far-end
reference via `bus.subscribe(("tts.audio_chunk",))`.

## M5 — Polish

- [ ] Latency dashboard (`metrics.csv` already exists; add a tiny live
      printout: VAD-close → TTFT → first-audio).
- [ ] Voice picker (`config.KOKORO_VOICE`).
- [ ] System-prompt presets (assistant, narrator, "thinking out loud").
- [ ] Optional GUI: there's a PySide6 demo in MockingAgent we can adapt
      if a GUI is wanted. Not required for daily use.
