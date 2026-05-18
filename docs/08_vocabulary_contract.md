# VoiceLLM Vocabulary Contract

This document adapts the AgenticLLM vocabulary contract to VoiceLLM. It is the
source of truth for what new components are called and where future components
should live.

VoiceLLM is currently a local voice loop, not a full agent tool framework. This
contract keeps the simpler repo organized with the same vocabulary: plugins for
external integrations, runners for framework-owned loops, and reserved homes
for future tools, skills, and memory.

## Categories

### Tool

An atomic LLM-callable function.

Current status: not first-class in VoiceLLM yet. The assistant does not expose
tool calls to the model today; it streams plain language through the LLM gate.

Future examples:

- `speak(text)`
- `listen()`
- `transcribe_audio(path)`
- `set_voice(voice_id)`

### Skill

A composite capability bundle with `SKILL.md`, code, and a smoke test.

Current status: not used in VoiceLLM yet. Skills become relevant if this repo
grows into a self-improving or agent-extensible assistant. Until then, avoid
calling ordinary STT/TTS modules "skills."

Future examples:

- `expressive_speak_v1/` wrapping the TTS plugin with prosody preprocessing
- `conversation_repair_v1/` improving transcripts before they reach the LLM
- `audio_scene_filter_v1/` classifying ambient audio before turn routing

### Plugin

A drop-in integration for an external service, model runtime, library-backed
capability, or hardware-facing capability. Plugins may register tools once
VoiceLLM has a tool surface.

Current production plugins:

- `plugins/kokoro_tts/` — Kokoro speech synthesis and playback
- `plugins/whisper_stt/` — two-pass and continuous pywhispercpp STT
- `plugins/llama_cpp_llm/` — llama-cpp-python GGUF backend
- `plugins/mlx_llm/` — Apple MLX backend
- future full-duplex audio/AEC -> `plugins/coreaudio_duplex/` or
  `plugins/speex_aec/`

`plugins/llm_core/` is shared LLM adapter code used by the LLM plugins. It is
kept under `plugins/` because it exists to support plugin backends, but it is
not itself an external integration.

### Runner

Framework-owned background work that the model does not call directly.

Current examples:

- The orchestrator loop in `core/runners/orchestrator.py`
- TTS synth/play threads inside `plugins/kokoro_tts/node.py`
- STT capture/transcription loops inside `plugins/whisper_stt/`

If runners become reusable framework infrastructure, place them under
`core/runners/`.

## Infrastructure Terms

- **Library**: importable packages such as `sounddevice`, `pywhispercpp`,
  `llama_cpp`, `mlx_lm`, `kokoro`, `numpy`, and `webrtcvad`.
- **Model / Artifact**: `.gguf` files, MLX model folders, Whisper weights,
  Kokoro voice files, eval logs, and trained adapters.
- **Transport / Protocol**: the in-process `Bus`, future MCP, HTTP, WebSocket,
  or ZMQ links. A protocol itself is not a plugin.
- **Hardware**: microphone, speaker, GPU, camera, or robot devices. A plugin
  owns access to hardware; hardware itself is not a tool.

## VoiceLLM Trust Zones

VoiceLLM does not currently let the agent write executable capability code at
runtime. If that changes, use the AgenticLLM zones:

| Zone | Path | Runtime agent permission |
|---|---|---|
| Framework plugins | `plugins/` | read-only |
| Framework skills | `skills/` | read-only |
| Instance skills | `<instance>/skills/` | append-only versioning |

For now, all code in this repo is human/framework code. The `references/`
folder is historical material and is not imported by `main.py`.

## Migration Layout

Use this as the target shape for new plugin/skill work:

```text
VoiceLLM/
├── core/
│   ├── tools/              # future framework tools
│   ├── runners/            # reusable background loops
│   │   └── orchestrator.py
│   └── ...
├── plugins/
│   ├── README.md
│   ├── kokoro_tts/
│   │   ├── plugin.yaml
│   │   └── node.py
│   ├── whisper_stt/
│   ├── llama_cpp_llm/
│   ├── mlx_llm/
│   ├── llm_core/
│   └── coreaudio_duplex/
├── skills/
│   └── example_v1/
│       ├── SKILL.md
│       ├── example.py
│       └── tests/smoke_test.py
└── references/
```

## Current Simpler Shape

VoiceLLM's current production shape is:

```text
VoiceLLM/
├── core/
│   ├── bus.py
│   ├── metrics.py
│   ├── state.py
│   ├── runners/orchestrator.py
│   └── tools/README.md
├── plugins/
│   ├── whisper_stt/
│   ├── kokoro_tts/
│   ├── llama_cpp_llm/
│   ├── mlx_llm/
│   └── llm_core/
├── audio/
├── memory/
├── references/
└── main.py
```

## Naming Rules

- Do not call Kokoro a skill. Kokoro is a library; `kokoro_tts` is the plugin;
  `speak()` is the future tool.
- Do not call Whisper a tool. Whisper is a library/model family;
  `whisper_stt` is the plugin; `listen()` or `transcribe_audio()` is the
  future tool.
- Do not call the in-process `Bus` a plugin. It is transport/framework
  infrastructure.
- Do not call `orchestrator` a plugin. It is a runner/state machine.
- Do not call `references/voice_chat.py` production code. It is a reference
  for the future full-duplex plugin.

## Decision Tree

When naming a new component, apply these in order:

1. Physical device? Hardware.
2. Data file, weights, voice, dataset, or adapter? Model / Artifact.
3. Wire format or message substrate? Transport / Protocol.
4. `pip install`-able import? Library.
5. Background loop the model does not call? Runner.
6. Drop-in bridge to an external runtime, service, or hardware-backed
   capability? Plugin.
7. Folder with `SKILL.md` that bundles a composite capability? Skill.
8. Single function the LLM calls directly? Tool.

If none match, the component probably needs a design pass before it lands.
