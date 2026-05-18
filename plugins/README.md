# Plugins

Plugins are drop-in integrations for external runtimes, libraries, services, or
hardware-facing capabilities. They are framework-owned code in VoiceLLM today;
they do not imply the app has an agent-callable tool surface yet.

Current plugins:

- `whisper_stt/` — pywhispercpp STT pipelines.
- `kokoro_tts/` — Kokoro speech synthesis and playback.
- `llama_cpp_llm/` — llama-cpp-python GGUF backend.
- `mlx_llm/` — Apple MLX backend.

`llm_core/` is shared adapter code for LLM plugins, not itself an external
integration.
