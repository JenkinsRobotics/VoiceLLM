# Milestones — build order

Each milestone is a single runnable command that *demonstrates* a working
behavior. Don't move on until the previous demo runs reliably.

## M0 — Repo wiring (no behavior change)

Goal: docs in place, requirements installable, models verified.

- [x] `docs/` written.
- [ ] `pip install -r requirements.txt` succeeds in a fresh venv.
  Add to that file: `mlx-lm`, `llama-cpp-python`, `kokoro>=0.9.4`, `scipy`.
- [ ] Sanity script `docs/check_models.py` (small) prints which model paths
      exist on this machine and their sizes. Useful before debugging.

Demo: `python -c "from llm.backend_base import BackendBase; print('ok')"`.

## M1 — Two CLI demos, ported in-place

Goal: the *known good* MockingAgent demos run from inside VoiceLLM with
the new module names. No behavior change.

- [ ] `demos/cli_chat_mlx.py` — copy of
      `MockingAgent/ollamacpp/chat_mlx.py`, points at the same MLX path.
- [ ] `demos/cli_chat_llamacpp.py` — copy of
      `MockingAgent/ollamacpp/chat_llama.py`, points at the same GGUF path.
- [ ] `demos/voice_assistant_baseline.py` — copy of
      `MockingAgent/voice_assistant.py`. Untouched — this is our regression
      anchor.

Demo: each script runs and produces a sensible reply.

## M2 — Modular voice assistant, no barge-in

Goal: the proven `voice_assistant.py` flow re-expressed through the bus +
nodes, with **MLX as the default LLM** and Gemma 4 26B-A4B-4bit.

- [ ] `plugins/llm_core/backend_base.py` — `BackendBase` ABC.
- [ ] `plugins/mlx_llm/backend.py` — extracted from `chat_mlx.py:39-78`.
- [ ] `plugins/llama_cpp_llm/backend.py` — extracted from `chat_llama.py:39-77`.
- [ ] `plugins/llm_core/node.py` — rewrite to consume a `BackendBase` instance.
- [ ] `plugins/kokoro_tts/node.py` — replace stub synth with real `KPipeline`,
      port `clean_for_tts()` and the mic-pause coordination.
- [ ] `plugins/whisper_stt/two_pass.py` — port `voice_assistant.py:127-213` (the
      VAD worker + 2-pass cascade) onto the bus.
- [ ] `config.py` — add `LLM_BACKEND`, `MLX_PATH`, `GGUF_PATH`,
      `STT_MODE = "two_pass"`, `KOKORO_VOICE`, voice prompts.
- [ ] `main.py` — build bus, instantiate backend by config flag,
      start orchestrator.

Demo: `python main.py` reproduces the MockingAgent voice assistant
behavior, but switching `LLM_BACKEND="llamacpp"` swaps the backend
without touching anything else.

## M3 — Continuous hearing

Goal: drop the wake word. STT transcribes constantly; any committed phrase
becomes a turn.

- [ ] `plugins/whisper_stt/continuous.py` — port `always_listening_hybrid_phrase_word_pipeline.py`
      onto the bus (publishes `stt.text` on each commit).
- [ ] `config.py` flip: `STT_MODE = "continuous"`, `REQUIRE_WAKE_WORD = False`.
- [ ] `core/runners/orchestrator.py` — when `REQUIRE_WAKE_WORD = False`, every `stt.text`
      becomes an `llm.request`. Add cooldown so a too-quick second commit
      doesn't double-fire while we're still synthesizing the first reply.
- [ ] Add `recent_assistant_reply` similarity filter (Layer C in
      `05_barge_in_and_self_speech.md`).

Demo: speak naturally, get a reply, keep talking, get another reply, no
"hey jaeger" needed. Background TV doesn't trigger the LLM (verified by
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

**AEC engine choice:** [audio/aec.py](../audio/aec.py) is sketched against
[`pyaec`](https://pypi.org/project/pyaec/) (SpeexDSP-based, easy macOS
wheels). Alternative is [`webrtc-audio-processing`](https://pypi.org/project/webrtc-audio-processing/)
(WebRTC APM with AEC + NS + AGC, but harder to build). Add the chosen
package to [requirements.txt](../requirements.txt) when wiring M4.

## M5 — Polish

- [ ] Latency dashboard (`metrics.csv` already exists; add a tiny live
      printout: VAD-close → TTFT → first-audio).
- [ ] Voice picker (`config.KOKORO_VOICE`).
- [ ] System-prompt presets (assistant, narrator, "thinking out loud").
- [ ] Optional GUI: there's a PySide6 demo in MockingAgent we can adapt
      if a GUI is wanted. Not required for daily use.
