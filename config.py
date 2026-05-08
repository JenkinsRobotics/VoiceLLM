"""Central tunables for VoiceLLM.

Edit values here; everything else imports from this module.
"""

from __future__ import annotations

from pathlib import Path

# ── LLM backend selection ──────────────────────────────────────────────
# llamacpp uses chat_completion which stops correctly on Gemma's
# <end_of_turn>. mlx-lm needed a hand-rolled stop check (see backend_mlx.py).
LLM_BACKEND = "llamacpp"  # "mlx" or "llamacpp"

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

# Cap history at this many user/assistant pairs (system prompt is preserved).
# Ported from voice_assistant.py:258-263 — keeps prompt size bounded so
# per-turn latency doesn't drift up over a long session.
MAX_HISTORY_TURNS = 8

SYSTEM_PROMPT = (
    "You are Jaeger, a local voice assistant running on this Mac. "
    "You are powered by Google's open-weight Gemma model running fully "
    "offline — not GPT, not ChatGPT, not Claude, no cloud API. "
    "If asked what you are, say you are Jaeger running on Gemma.\n\n"
    "You receive transcriptions from an always-on microphone, so much of "
    "what you hear is NOT directed at you — ambient speech, transcription "
    "artifacts, single-word fragments, conversations between other "
    "people, keystroke noise.\n\n"
    "Begin EVERY reply with exactly one tag:\n"
    "  <ignore>  — input is not addressed to you. Output the tag and "
    "nothing else.\n"
    "  <reply>   — input is a real directed turn for you. Follow the tag "
    "with 1 to 3 short sentences of natural conversational English. No "
    "markdown, lists, code blocks, or emoji. If unsure, say so briefly.\n\n"
    "Examples:\n"
    "  '[BLANK_AUDIO]' → <ignore>\n"
    "  'test' → <ignore>\n"
    "  'so anyway like I was telling her' → <ignore>\n"
    "  'what time is it' → <reply>I can't read the system clock, but the "
    "menu bar shows it.\n"
    "  'hey jaeger tell me a joke' → <reply>Why did the developer go "
    "broke? Their cache was empty."
)

# ── LLM-gated speech control ──────────────────────────────────────────
# How many characters of LLM output to buffer before deciding the gate.
# If neither <ignore> nor <reply> is seen by this length, default to
# treating the output as a reply (the LLM forgot the protocol).
LLM_GATE_BUFFER_CHARS = 30

# ── STT ────────────────────────────────────────────────────────────────
STT_MODE = "two_pass"          # "two_pass" (M2/M3 quick path) | "continuous" (M3.5)
STT_FAST_MODEL = "base.en"
STT_ACCURATE_MODEL = "medium.en"

# ── M3.5 continuous-mode STT (only read when STT_MODE == "continuous") ─
# Single Whisper model for the rolling pipeline. base.en is the speed pick;
# bump to "small.en" or "medium.en" if accuracy lags. Faithful port of
# MockingAgent/.../always_listening_hybrid_phrase_word_pipeline.py defaults.
STT_CONTINUOUS_MODEL = "base.en"

# Mic chunk size for the continuous pipeline. Larger than the two_pass
# FRAME_MS=30 because energy detection runs per-chunk and we want fewer
# tiny chunks. 100 ms = 1600 samples @ 16 kHz.
STT_BLOCK_MS = 100

# Phrase boundaries.
STT_PHRASE_TIMEOUT_S = 1.2     # quiet seconds that close a phrase
STT_MAX_PHRASE_S = 12.0        # hard ceiling; matches MAX_SPEECH_MS=12000

# Rolling re-transcription cadence. Smaller = snappier feel, more CPU.
STT_TRANSCRIBE_EVERY_S = 1.0
STT_MIN_TRANSCRIBE_S = 0.6     # don't transcribe shorter buffers

# Activity detection — refreshes the phrase timer when RMS exceeds this.
STT_ENERGY_THRESHOLD = 0.008

# Duplicate-suppression on rolling and committed text.
STT_DUPLICATE_SIMILARITY = 0.92

# ── Wake words (used in two_pass mode when REQUIRE_WAKE_WORD = True) ───
# M3 default: continuous hearing — every committed phrase becomes a turn.
# Set to True to restore Google-Home-style wake-word gating.
REQUIRE_WAKE_WORD = False
WAKE_PREFIXES = ("ok", "okay", "hey")
ASSISTANT_NAMES = ("jaeger", "yeager", "yager", "jager")
WAKE_PHRASES = tuple(f"{p} {n}" for p in WAKE_PREFIXES for n in ASSISTANT_NAMES)
WAKE_MATCH_THRESHOLD = 0.78
FOLLOWUP_WINDOW_S = 10.0

# ── M3 continuous-mode guards ──────────────────────────────────────────
# Self-speech filter: if an STT commit is >= this similar to the most
# recent assistant reply, drop it. Catches the residual cases where the
# mic picks up the assistant's own voice through the speaker.
SELF_SPEECH_SIMILARITY_THRESHOLD = 0.75

# Pending-turn cooldown: if a second stt.text arrives while a turn is in
# flight, store it and fire it once we go idle (instead of dropping). The
# stored phrase is discarded if it's older than this many seconds when the
# current turn finishes — by then the user has likely moved on.
PENDING_TURN_MAX_AGE_S = 3.0

# M3 evaluation log. Set to None to disable. When set, every stt.text
# decision (accepted / dropped-self-echo / queued-as-pending / dropped-stale)
# is appended as one JSON line for offline review.
M3_EVAL_LOG = "outputs/m3_eval.jsonl"

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

# Keep mic capture bounded if STT falls behind during a transcription pass.
# At 30 ms frames, 200 frames is about 6 seconds of buffered audio.
MIC_QUEUE_MAX_FRAMES = 200

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
