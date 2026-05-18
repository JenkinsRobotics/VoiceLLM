# Tools

VoiceLLM does not expose LLM-callable tools yet.

Today, the model receives text and streams text back. Runtime coordination uses
bus topics such as `stt.text`, `llm.token`, `mic.pause`, and `tts.done`; those
topics are framework plumbing, not tools.

If VoiceLLM grows an agent-callable surface, atomic functions such as
`speak(text)`, `listen()`, `transcribe_audio(path)`, or `set_voice(voice_id)`
belong here or in a plugin/skill that registers them.
