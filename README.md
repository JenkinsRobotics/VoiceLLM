# VoiceLLM

A continuously-listening, low-latency local voice assistant for Apple
Silicon. Speak naturally; the assistant streams a spoken reply back; the
conversation keeps going without a wake word. Think ChatGPT Voice, fully
offline.

> Status: **current voice-loop milestone complete**. The modular local
> assistant, plugin-style STT/TTS/LLM organization, continuous hearing,
> LLM-gated speech, self-speech rejection, streaming Kokoro TTS, and chimes are
> in place. Further advanced work is unlikely to continue here; the likely path
> is carrying ideas such as full-duplex AEC/barge-in, persistent memory, and
> LLM-callable tools into newer AgenticLLM-style frameworks. See
> [docs/STATUS.md](docs/STATUS.md) for the handoff notes.

## What it does today

- **Continuous hearing.** No wake word in the default mode — every directed
  utterance becomes a turn.
- **LLM-gated speech.** Every reply begins with `<ignore>` or `<reply>`;
  the orchestrator suppresses TTS when the LLM judges the input as not
  addressed to it (background TV, keystroke noise, transcription artifacts,
  ambient conversation). The audio pipeline does not gatekeep — the LLM
  does.
- **Two interchangeable LLM backends.** Same model behavior (Gemma 4
  26B-A4B 4-bit), one config flag — `llama.cpp` (default, robust chat
  template handling) and `mlx-lm` (Apple MLX, faster on M-series). Swap
  with one line in [config.py](config.py).
- **Two interchangeable STT pipelines.** `two_pass` (proven Google-Home-style
  fast→accurate Whisper cascade) and `continuous` (rolling re-transcription
  hybrid). `STT_MODE` flag selects.
- **Self-speech rejection.** Mic-pause during TTS plus a similarity filter
  against the most recent reply. Full-duplex AEC/barge-in is preserved as
  carry-forward design work rather than expected development in this repo.
- **Friendly chimes.** A short wake chime acknowledges wake-only prompts, and
  a double chime marks that the assistant is ready for a follow-up.
- **Streaming TTS** via [Kokoro](https://github.com/hexgrad/kokoro). Audio
  starts at the first sentence boundary, not after the full reply.

## Stack

| Layer | Choice |
|---|---|
| Audio I/O | `sounddevice` (CoreAudio) |
| VAD | `webrtcvad` |
| STT | `pywhispercpp` (whisper.cpp), `base.en` + `medium.en` |
| LLM | `llama-cpp-python` (default) or `mlx-lm`, both running Gemma 4 26B-A4B 4-bit |
| TTS | `kokoro` (`KPipeline`) |

State and routing run through a single in-process pub/sub `Bus` consumed by
the runner in [core/runners/orchestrator.py](core/runners/orchestrator.py)
(`IDLE → THINKING → RESPONDING → IDLE`).

## Current status

I would call this project **complete for the current local voice assistant
milestone**. It is not intended to become the main open-ended assistant
platform; advanced continuation work is expected to move into newer frameworks
such as AgenticLLM.

Done:

- Runnable local voice loop through `python main.py`.
- Plugin-style layout for STT, TTS, and LLM integrations.
- Runner classification for the orchestrator.
- Continuous-hearing default with optional wake-word mode.
- LLM gate for ignoring ambient/non-directed speech.
- Mic-pause plus similarity filtering for self-speech rejection.
- Streaming Kokoro TTS and configurable wake/follow-up chimes.

Carry-forward ideas:

- True full-duplex AEC/barge-in.
- Persistent memory beyond in-session conversation history.
- LLM-callable tools or skills.
- GUI or device/voice picker.

## Quick start

Apple Silicon Mac (M1/M2/M3/M4), 24 GB+ unified memory recommended for the
26B 4-bit model.

```bash
git clone https://github.com/<your-fork>/VoiceLLM.git
cd VoiceLLM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Gemma 4 26B-A4B model files (or symlink them) at the locations
in [config.py](config.py):

```
~/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf
~/.lmstudio/models/mlx-community/gemma-4-26b-a4b-4bit/
```

Then run:

```bash
python main.py
```

The first run loads ~5 GB into unified memory (LLM, both Whisper sizes,
Kokoro). Once you see `[ready] VoiceLLM running. Ctrl-C to quit.` start
talking — no wake word needed.

## Configuration

All tunables live in [config.py](config.py). The flags you'll touch most:

| Flag | Default | What it does |
|---|---|---|
| `LLM_BACKEND` | `"llamacpp"` | `"llamacpp"` (proven) or `"mlx"` (faster on M-series) |
| `STT_MODE` | `"two_pass"` | `"two_pass"` (default) or `"continuous"` (M3.5 rolling) |
| `REQUIRE_WAKE_WORD` | `False` | `True` reverts to "okay jaeger" gating |
| `LLM_TEMPERATURE` | `0.6` | LLM sampling temperature |
| `MAX_HISTORY_TURNS` | `8` | rolling user/assistant pair cap |
| `KOKORO_VOICE` | `"af_heart"` | Kokoro voice id |
| `CHIMES_ENABLED` | `True` | master switch for wake/follow-up audio cues |

See [config.py](config.py) for the full list (VAD aggressiveness, phrase
timeouts, energy thresholds, etc.).

## Project layout

```
VoiceLLM/
├── config.py                  # all tunables
├── main.py                    # build bus + nodes, start orchestrator
├── core/                      # bus, state, metrics, runners, future tools
│   ├── runners/               # framework-owned loops (orchestrator)
│   └── tools/                 # future LLM-callable tools
├── audio/                     # MicStream, VAD, AEC (M4)
├── plugins/                   # STT, TTS, and LLM integrations
│   ├── whisper_stt/           # two_pass.py, continuous.py
│   ├── kokoro_tts/            # Kokoro playback node
│   ├── llama_cpp_llm/         # GGUF backend
│   ├── mlx_llm/               # MLX backend
│   └── llm_core/              # shared LLM bus adapter/base class
├── memory/                    # future persistent memory
├── references/                # pasted historical scripts, not imported
├── docs/                      # architecture / milestones / status
├── outputs/                   # m3_eval.jsonl (runtime decision log)
└── metrics.csv                # per-turn timing log
```

Vocabulary-wise: STT/TTS/LLM are **plugins**, the orchestrator is a
framework-owned **runner**, `core/bus.py` is transport infrastructure, and
`core/tools/` is intentionally empty until VoiceLLM grows model-callable tools.

## Documentation

- [docs/STATUS.md](docs/STATUS.md) — completion state and handoff notes.
- [docs/00_overview.md](docs/00_overview.md) — design intent.
- [docs/01_architecture.md](docs/01_architecture.md) — module/bus layout.
- [docs/02_stt_pipelines.md](docs/02_stt_pipelines.md) — STT strategies.
- [docs/03_llm_backends.md](docs/03_llm_backends.md) — backend selection.
- [docs/04_tts_kokoro.md](docs/04_tts_kokoro.md) — Kokoro coordination.
- [docs/05_barge_in_and_self_speech.md](docs/05_barge_in_and_self_speech.md) — barge-in + self-speech.
- [docs/06_milestones.md](docs/06_milestones.md) — build order M0–M5.
- [docs/07_open_questions.md](docs/07_open_questions.md) — design questions.
- [docs/08_vocabulary_contract.md](docs/08_vocabulary_contract.md) — component naming and future plugin/skill layout.

## Acknowledgments

- [Gemma 4](https://ai.google.dev/gemma) — Google's open-weight model family
  (26B-A4B 4-bit at the heart of this).
- [Kokoro](https://github.com/hexgrad/kokoro) — natural TTS that runs in
  real-time on M-series.
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) via
  [pywhispercpp](https://github.com/absadiki/pywhispercpp) — fastest local
  Whisper on Apple Silicon.
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — Apple MLX inference for
  Gemma.
- [llama.cpp](https://github.com/ggerganov/llama.cpp) /
  [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — the
  cross-platform fallback backend.

## License

See [LICENSE](LICENSE).
