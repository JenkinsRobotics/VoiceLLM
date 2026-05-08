# VoiceLLM — Overview

A continuously-listening, low-latency local voice assistant. The user speaks
naturally, the system streams a quick spoken reply back, and the conversation
keeps going without any "ok google" between turns. Think ChatGPT Voice, but
fully local on Apple Silicon.

## What is different from MockingAgent

MockingAgent is the proven Google-Home-style baseline:
- wake word required ("ok jaeger"),
- 2-pass STT (fast for wake match, accurate for the command),
- short follow-up window after a reply,
- mic paused while TTS speaks (self-speech rejection).

VoiceLLM keeps the proven plumbing but changes the *behavior model*:
- **continuous hearing**: every committed phrase becomes a turn — no
  "okay jaeger" between turns. (M3 — shipped.)
- **LLM-gated speech**: every reply begins with `<ignore>` or `<reply>`;
  the orchestrator suppresses TTS when the LLM judges the input wasn't
  addressed to it (TV, keystrokes, hallucinated transcriptions, ambient
  conversation). The audio pipeline stays dumb; the LLM decides. See
  [01_architecture.md](01_architecture.md#llm-gate) and
  [05_barge_in_and_self_speech.md](05_barge_in_and_self_speech.md).
- **first-token streaming**: TTS starts speaking at the first sentence
  boundary after the gate decides `<reply>`. ~50-150 ms gate buffer
  cost on the reply path; zero audible cost on the ignore path.
- **swappable LLM backends**: `llama-cpp-python` (default — robust chat
  template handling and `<end_of_turn>` stop) and `mlx-lm` (Apple MLX —
  faster on M-series, with our hand-rolled stop-marker detector to handle
  Gemma's EOT). Same model on both: Gemma 4 26B-A4B 4-bit. One config flag.
- **swappable STT pipelines**: `two_pass` (proven Google-Home-style cascade)
  and `continuous` (rolling re-transcription hybrid, M3.5). One config flag.
- **barge-in (M4, planned)**: AEC + VAD-on-cleaned-audio so the user can
  cut the assistant off mid-reply.

## Hardware target

- Apple Silicon (M-series), macOS.
- 24 GB+ unified memory comfortable for Gemma 4 26B-A4B at 4-bit.
- Built-in mic + speakers, or any USB headset.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Audio I/O | `sounddevice` | already used in MockingAgent, low-latency callback API |
| VAD | `webrtcvad` | proven, cheap, 30 ms frames |
| STT | `pywhispercpp` (whisper.cpp) | fastest local Whisper, GGML quantized |
| LLM (default) | `llama-cpp-python` + `gemma-4-26B-A4B-it-Q4_K_M.gguf` | mature chat-template + stop-token handling; proven path |
| LLM (alt) | `mlx-lm` + `mlx-community/gemma-4-26b-a4b-4bit` | fastest decode on Apple Silicon; needs hand-rolled EOT stop |
| TTS | `kokoro` (`KPipeline`) | natural voice, real-time on M-series |
| Echo control | mic-pause during TTS + (optional) AEC | self-speech rejection |

## Documents in this folder

- `00_overview.md` — this file.
- `01_architecture.md` — module/node layout, message bus topics, lifecycle.
- `02_stt_pipelines.md` — the four pywhispercpp listening strategies, what
  each one does well/poorly, which we'll try first for "always listening".
- `03_llm_backends.md` — the unified LLM interface, MLX vs llama.cpp tradeoffs,
  Gemma 4 26B-A4B configuration.
- `04_tts_kokoro.md` — Kokoro setup, streaming-by-sentence, voice selection,
  mic-pause coordination.
- `05_barge_in_and_self_speech.md` — how we stop TTS when the user interrupts,
  and how we reject the assistant's own audio when AEC isn't enough.
- `06_milestones.md` — concrete build order: M1 demo → M2 streaming →
  M3 barge-in → M4 backend switch → M5 polish.
- `07_open_questions.md` — design questions we still need to answer before
  locking the v1 layout.
