"""Central tunables for VoiceLLM.

Edit values here; everything else imports from this module.
"""

from __future__ import annotations

from pathlib import Path

# ── LLM backend selection ──────────────────────────────────────────────
LLM_BACKEND = "mlx"  # "mlx" or "llamacpp"

LMSTUDIO_MODELS = Path("/Users/jonathanjenkins/.lmstudio/models")
MLX_PATH = LMSTUDIO_MODELS / "mlx-community" / "gemma-4-26b-a4b-4bit"
GGUF_PATH = (
    LMSTUDIO_MODELS
    / "lmstudio-community"
    / "gemma-4-26B-A4B-it-GGUF"
    / "gemma-4-26B-A4B-it-Q4_K_M.gguf"
)

LLM_CTX = 4096
LLM_MAX_TOKENS = 220
LLM_TEMPERATURE = 0.6
LLM_TOP_P = 0.9
LLM_GPU_LAYERS = -1  # llama.cpp: -1 = all on GPU

SYSTEM_PROMPT = (
    "You are a voice assistant having a real-time spoken conversation. "
    "Reply in 1 to 3 short sentences of natural conversational English. "
    "No markdown, no lists, no code blocks, no emoji. If unsure, say so briefly."
)

# ── STT ────────────────────────────────────────────────────────────────
STT_MODE = "two_pass"          # "two_pass" (M2) | "continuous" (M3)
STT_FAST_MODEL = "base.en"
STT_ACCURATE_MODEL = "medium.en"

# ── Wake words (used in two_pass mode) ─────────────────────────────────
REQUIRE_WAKE_WORD = True
WAKE_PREFIXES = ("ok", "okay", "hey")
ASSISTANT_NAMES = ("jaeger", "yeager", "yager", "jager")
WAKE_PHRASES = tuple(f"{p} {n}" for p in WAKE_PREFIXES for n in ASSISTANT_NAMES)
WAKE_MATCH_THRESHOLD = 0.78
FOLLOWUP_WINDOW_S = 10.0

# ── Audio capture (mic) ────────────────────────────────────────────────
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 @ 16 kHz, 30 ms
VAD_AGGRESSIVENESS = 2

PRE_ROLL_MS = 240
POST_PADDING_MS = 250
SILENCE_HANGOVER_MS = 1000
MIN_SPEECH_MS = 400
MAX_SPEECH_MS = 12000

INPUT_DEVICE = None   # None = system default; index or name string also OK
OUTPUT_DEVICE = None

# ── TTS (Kokoro) ───────────────────────────────────────────────────────
KOKORO_VOICE = "af_heart"
KOKORO_LANG = "a"
TTS_SAMPLE_RATE = 24000
TTS_MIN_CHARS = 60      # synthesize partial when sentence is long but no period yet
TTS_TAIL_SLEEP_S = 0.12 # let speakers drain before un-pausing mic

# ── Behavior flags (M2 defaults; M3+M4 flip these) ─────────────────────
BARGE_IN_ENABLED = False  # M4
AEC_ENABLED = False       # M4

# ── Debug ──────────────────────────────────────────────────────────────
AUDIO_DEBUG = False
PRINT_LLM_TIMING = True
