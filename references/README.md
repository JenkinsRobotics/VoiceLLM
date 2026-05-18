# Reference Scripts

These scripts are preserved as historical references for the modular VoiceLLM
app. They are not imported by `main.py`.

- `voice_assistant.py` — the proven wake-word, two-pass STT, Kokoro TTS loop.
- `voice_chat.py` — the full-duplex AEC/barge-in experiment that will inform M4.

Useful behavior has been pulled into the main app where it is safe to do so.
Keep these around until M4/M5 has absorbed the remaining full-duplex audio work.
