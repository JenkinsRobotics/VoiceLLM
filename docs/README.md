# VoiceLLM planning docs

**Picking this up cold? Start with [STATUS.md](STATUS.md)** — it summarizes
where we are (M2 done), what's next (M3 continuous hearing, M4 barge-in),
and known gotchas.

Then read in order:

1. [00_overview.md](00_overview.md) — what we're building and why it's
   different from the MockingAgent baseline.
2. [01_architecture.md](01_architecture.md) — module layout, bus topics,
   state machine.
3. [02_stt_pipelines.md](02_stt_pipelines.md) — the four pywhispercpp
   listening strategies, with which we'll default to and why.
4. [03_llm_backends.md](03_llm_backends.md) — `BackendBase` interface,
   MLX vs llama.cpp, Gemma 4 26B-A4B configuration.
5. [04_tts_kokoro.md](04_tts_kokoro.md) — Kokoro streaming, mic-pause
   coordination, cancellation.
6. [05_barge_in_and_self_speech.md](05_barge_in_and_self_speech.md) — how
   we stop talking when interrupted, and how we don't transcribe ourselves.
7. [06_milestones.md](06_milestones.md) — concrete build order
   M0 → M5, each ending in a runnable demo.
8. [07_open_questions.md](07_open_questions.md) — unresolved design
   choices. Worth scanning before starting M2.

The MockingAgent reference files these plans cite live in:

- [MockingAgent/voice_assistant.py](../../MockingAgent/voice_assistant.py)
- [MockingAgent/ollamacpp/chat_mlx.py](../../MockingAgent/ollamacpp/chat_mlx.py)
- [MockingAgent/ollamacpp/chat_llama.py](../../MockingAgent/ollamacpp/chat_llama.py)
- [MockingAgent/PywisperCpp/pywhispercpp_examples/llm_listener/](../../MockingAgent/PywisperCpp/pywhispercpp_examples/llm_listener/)
- [MockingAgent/PywisperCpp/pywhispercpp_examples/local_assistant/](../../MockingAgent/PywisperCpp/pywhispercpp_examples/local_assistant/)
