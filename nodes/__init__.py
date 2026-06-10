"""VoiceLLM peripheral nodes — STT, TTS, audio_session.

Mirrors JROS's nodes/ layer: each peripheral subsystem (mic + AEC +
VAD + STT, speech synthesis, etc.) is a self-contained node behind
a bus topic.  Brain side is in agent/.
"""

from .tts.node import KokoroNode  # noqa: F401
from .stt.continuous import STTContinuousNode  # noqa: F401
from .stt.two_pass import STTTwoPassNode  # noqa: F401
