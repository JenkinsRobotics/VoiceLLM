"""VoiceLLM bus — same lightweight pub/sub the voice loop uses for
STT → LLM → TTS coordination.  Mirrors JROS's transport/ layer
so the two codebases share one mental model.
"""

from .bus import Bus  # noqa: F401
