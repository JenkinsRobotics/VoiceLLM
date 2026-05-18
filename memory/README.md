# Memory

VoiceLLM currently has only in-session conversational memory inside
`plugins/llm_core/node.py`.

Use this directory for future persistent memory:

- `facts.json` or `facts.sqlite` for durable user/project facts.
- `episodic.jsonl` for timestamped conversation events.
- `embeddings/` for semantic recall indexes.

Persistent memory should stay separate from plugins. Plugins provide
capabilities; memory stores what the assistant knows across sessions.
