#!/usr/bin/env python3
"""Google-Home-style local voice assistant.

Pipeline:
  mic ─► audio_queue ─► VAD worker thread ─► phrase_queue
                                               │
                                               ▼
        main loop (state machine) ──► wake check / 2-pass STT ──► LLM ──► TTS

Designed off PywisperCpp/pywhispercpp_examples/local_assistant/
continuous_lmstudio_command_listener.py with these additions:
  • mic.pause flag during TTS for self-speech rejection
  • short tone instead of spoken "Yes?" so the user isn't clipped
  • 5-second follow-up window after each reply (no wake word required)
  • LLM (llama.cpp + Gemma 4 26B-A4B) and TTS (Kokoro) wired in

Just hit Run.  Loads can take ~15–25s the first time.
"""

from __future__ import annotations

import collections
import queue
import re
import sys
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad


# ── config ─────────────────────────────────────────────────────────────
LLM_MODEL_PATH = Path(
    "/Users/jonathanjenkins/.lmstudio/models/lmstudio-community/"
    "gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf"
)

# Two-pass STT: fast model runs every phrase to detect wake words. Accurate
# model only re-transcribes when wake matches or we're in follow-up mode,
# so we don't pay its cost on background noise.
STT_FAST = "base.en"
STT_ACCURATE = "medium.en"

KOKORO_VOICE = "af_heart"
KOKORO_LANG = "a"

# Whisper often mishears "jaeger" — covering common phonetic transcriptions
# so any of yeager/yager/jager/jaeger triggers the wake.
_WAKE_PREFIXES = ("ok", "okay", "hey")
_ASSISTANT_NAMES = ("jaeger", "yeager", "yager", "jager")
WAKE_PHRASES = tuple(f"{p} {n}" for p in _WAKE_PREFIXES for n in _ASSISTANT_NAMES)
WAKE_MATCH_THRESHOLD = 0.78
FOLLOWUP_WINDOW_S = 10.0      # listen this long after a reply, no wake needed

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 480 samples
VAD_AGGRESSIVENESS = 2

PRE_ROLL_MS = 240             # capture speech onset
POST_PADDING_MS = 250         # capture trailing word — fixes "what time is it" → "time is in"
SILENCE_HANGOVER_MS = 700     # match the working command listener
MIN_SPEECH_MS = 400
MAX_SPEECH_MS = 8000

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Answer in 1–2 short sentences "
    "in plain conversational English. No markdown, no code blocks, no "
    "emojis, no lists. If you don't know, say so briefly."
)

# Cap rolling chat history so prompt size (and latency) stays bounded.
MAX_HISTORY_TURNS = 8


# ── short chime so the user knows we're listening ──────────────────────
def make_beep(freq: float = 880.0, duration_ms: int = 110,
              sr: int = 24000, amp: float = 0.25) -> np.ndarray:
    n = int(sr * duration_ms / 1000)
    t = np.arange(n) / sr
    # short fade-in/out to avoid clicks
    env = np.minimum(np.minimum(t / 0.01, 1.0), (duration_ms / 1000 - t) / 0.01).clip(0, 1)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


BEEP = make_beep()
DOUBLE_BEEP = np.concatenate([
    make_beep(freq=660, duration_ms=80),
    np.zeros(int(24000 * 0.05), dtype=np.float32),
    make_beep(freq=880, duration_ms=80),
])


# ── audio capture ──────────────────────────────────────────────────────
class MicStream:
    """sounddevice InputStream → frame queue, with a pause flag for TTS."""

    def __init__(self) -> None:
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.paused = False
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=FRAME_SAMPLES, callback=self._cb,
        )

    def _cb(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        if self.paused or frames != FRAME_SAMPLES:
            return
        self.q.put(indata.copy())

    def __enter__(self) -> "MicStream":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stream.stop()
        self._stream.close()

    def drain(self) -> None:
        with self.q.mutex:
            self.q.queue.clear()


# ── VAD worker thread ──────────────────────────────────────────────────
class VadWorker(threading.Thread):
    """Reads audio blocks, runs VAD, accumulates phrases, fast-transcribes,
    then pushes (audio_float32, fast_transcript) onto phrase_queue.
    """

    def __init__(self, mic: MicStream, fast_model,
                 phrase_queue: queue.Queue, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.mic = mic
        self.fast_model = fast_model
        self.phrase_queue = phrase_queue
        self.stop_event = stop_event
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        self.silence_blocks_to_end = max(1, SILENCE_HANGOVER_MS // FRAME_MS)
        self.min_speech_blocks = max(1, MIN_SPEECH_MS // FRAME_MS)
        self.max_speech_blocks = max(self.min_speech_blocks, MAX_SPEECH_MS // FRAME_MS)
        self.pre_roll_blocks = max(0, PRE_ROLL_MS // FRAME_MS)
        self.post_pad_samples = int(SAMPLE_RATE * POST_PADDING_MS / 1000)

        # Exposed so the main loop can avoid expiring the follow-up window
        # while the user is still mid-sentence.
        self.in_speech = False

    def _is_speech(self, chunk: np.ndarray) -> bool:
        pcm = (chunk[:, 0] * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        return self.vad.is_speech(pcm, SAMPLE_RATE)

    def _finalize(self, chunks: list[np.ndarray]) -> None:
        audio = np.concatenate(chunks, axis=0).astype(np.float32).reshape(-1)
        audio = np.concatenate([audio, np.zeros(self.post_pad_samples, dtype=np.float32)])
        try:
            segments = self.fast_model.transcribe(audio, language="en")
            text = " ".join(s.text for s in segments).strip()
        except Exception as exc:
            print(f"[stt-fast] {exc}", file=sys.stderr)
            text = ""
        if text:
            self.phrase_queue.put((audio, text))

    def run(self) -> None:
        pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=self.pre_roll_blocks)
        speech: list[np.ndarray] = []
        speech_blocks = 0
        silent_blocks = 0
        in_speech = False

        while not self.stop_event.is_set():
            try:
                chunk = self.mic.q.get(timeout=0.3)
            except queue.Empty:
                continue

            is_speech = self._is_speech(chunk)

            if is_speech:
                if not in_speech:
                    speech = list(pre_roll)
                    speech_blocks = len(speech)
                    silent_blocks = 0
                    in_speech = True
                speech.append(chunk)
                speech_blocks += 1
                silent_blocks = 0
            elif in_speech:
                speech.append(chunk)
                silent_blocks += 1
            else:
                pre_roll.append(chunk)

            # Publish speech state once we've seen enough sustained voice
            # to be confident this isn't a noise blip.
            self.in_speech = in_speech and speech_blocks >= self.min_speech_blocks

            phrase_done = in_speech and speech_blocks >= self.min_speech_blocks and (
                silent_blocks >= self.silence_blocks_to_end
                or speech_blocks >= self.max_speech_blocks
            )
            if phrase_done:
                self._finalize(speech)
                speech = []
                speech_blocks = 0
                silent_blocks = 0
                in_speech = False
                self.in_speech = False
                pre_roll.clear()


# ── wake-word logic ────────────────────────────────────────────────────
def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def find_wake(text: str) -> tuple[bool, str]:
    """Return (matched, remainder_after_wake_phrase)."""
    norm = normalize(text)
    for phrase in WAKE_PHRASES:
        idx = norm.find(phrase)
        if idx != -1:
            return True, norm[idx + len(phrase):].strip()
    tokens = norm.split()
    for phrase in WAKE_PHRASES:
        n = len(phrase.split())
        for i in range(0, max(0, len(tokens) - n + 1)):
            window = " ".join(tokens[i:i + n])
            if SequenceMatcher(None, window, phrase).ratio() >= WAKE_MATCH_THRESHOLD:
                return True, " ".join(tokens[i + n:]).strip()
    return False, ""


# ── LLM ────────────────────────────────────────────────────────────────
def load_llm():
    from llama_cpp import Llama
    print(f"[llm] loading {LLM_MODEL_PATH.name}...", flush=True)
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(LLM_MODEL_PATH),
        n_ctx=4096, n_gpu_layers=-1, verbose=False,
    )
    print(f"[llm] loaded in {time.perf_counter()-t0:.1f}s, warming up...", flush=True)
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}], max_tokens=1, temperature=0.0,
    )
    print("[llm] ready", flush=True)
    return llm


def trim_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    # history[0] is system; the rest is user/assistant pairs appended by think().
    # Slicing from the tail in pairs keeps the boundary on a user message.
    if len(history) <= 1 + max_turns * 2:
        return history
    return history[:1] + history[-max_turns * 2:]


def think(llm, history: list[dict], user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    out = llm.create_chat_completion(
        messages=history, max_tokens=200, temperature=0.7, top_p=0.95,
    )
    reply = out["choices"][0]["message"]["content"].strip()
    history.append({"role": "assistant", "content": reply})
    return clean_for_tts(reply)


def clean_for_tts(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^[\-\*\d\.\)]+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# ── TTS ────────────────────────────────────────────────────────────────
def load_tts():
    from kokoro import KPipeline
    print("[tts] loading Kokoro...", flush=True)
    t0 = time.perf_counter()
    pipe = KPipeline(lang_code=KOKORO_LANG)
    list(pipe("Ready.", voice=KOKORO_VOICE))     # warm-up
    print(f"[tts] ready ({time.perf_counter()-t0:.1f}s)", flush=True)
    return pipe


def drain_phrase_queue(q: queue.Queue) -> None:
    """Discard phrases the VAD finalized while we were speaking — otherwise the
    follow-up window would treat stale buffered speech as a fresh command."""
    with q.mutex:
        q.queue.clear()


def play_audio_with_mic_paused(mic: MicStream, audio: np.ndarray, sr: int = 24000) -> None:
    """Play to speakers; mic capture is suppressed so we don't transcribe ourselves."""
    mic.paused = True
    try:
        sd.play(audio, samplerate=sr)
        sd.wait()
        time.sleep(0.12)              # let the speaker drain
    finally:
        mic.drain()
        mic.paused = False


def speak(pipe, mic: MicStream, text: str, sr: int = 24000) -> None:
    """Stream Kokoro chunks: play chunk N while chunk N+1 is still generating.
    Mic stays paused across the whole stream so we never re-capture our own voice.
    """
    if not text:
        return
    mic.paused = True
    started = False
    try:
        for r in pipe(text, voice=KOKORO_VOICE):
            if r.audio is None:
                continue
            chunk = np.asarray(r.audio, dtype=np.float32)
            if started:
                sd.wait()  # block until previous chunk finishes
            sd.play(chunk, samplerate=sr)
            started = True
        if started:
            sd.wait()
            time.sleep(0.12)              # let the speaker drain
    finally:
        mic.drain()
        mic.paused = False


# ── main loop ──────────────────────────────────────────────────────────
def warm_stt(model, label: str) -> None:
    """Prime pywhispercpp once so the first real phrase avoids setup latency."""
    warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    print(f"[{label}] warming up...", flush=True)
    t0 = time.perf_counter()
    try:
        list(model.transcribe(warm_audio, language="en"))
    except Exception as exc:
        # Some Whisper builds dislike pure silence. Startup should continue;
        # the model is still loaded and ready for normal speech.
        print(f"[{label}] warm-up skipped: {exc}", file=sys.stderr, flush=True)
    else:
        print(f"[{label}] primed ({time.perf_counter()-t0:.1f}s)", flush=True)


def main() -> int:
    from pywhispercpp.model import Model as STTModel

    print(f"[stt-fast] loading {STT_FAST}...", flush=True)
    t0 = time.perf_counter()
    fast_stt = STTModel(
        STT_FAST, print_realtime=False, print_progress=False,
        single_segment=True, no_context=True,
    )
    print(f"[stt-fast] ready ({time.perf_counter()-t0:.1f}s)", flush=True)
    warm_stt(fast_stt, "stt-fast")

    print(f"[stt-accurate] loading {STT_ACCURATE}...", flush=True)
    t0 = time.perf_counter()
    accurate_stt = STTModel(
        STT_ACCURATE, print_realtime=False, print_progress=False,
        single_segment=True, no_context=True,
    )
    print(f"[stt-accurate] ready ({time.perf_counter()-t0:.1f}s)", flush=True)
    warm_stt(accurate_stt, "stt-accurate")

    def transcribe_accurate(audio: np.ndarray) -> str:
        segments = accurate_stt.transcribe(audio, language="en")
        return " ".join(s.text for s in segments).strip()

    llm = load_llm()
    tts = load_tts()
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    phrase_queue: queue.Queue[tuple[np.ndarray, str]] = queue.Queue()
    stop_event = threading.Event()

    print(f"\n[ready] say one of: {', '.join(WAKE_PHRASES)} — Ctrl-C to quit.\n")

    state = "WAKE"           # "WAKE" or "FOLLOWUP"
    followup_deadline = 0.0

    with MicStream() as mic:
        worker = VadWorker(mic, fast_stt, phrase_queue, stop_event)
        worker.start()
        try:
            while True:
                # Follow-up timeout?  Don't expire while the user is still
                # mid-sentence — wait for them to finish, then we'll see the
                # phrase on the queue and treat it as a follow-up command.
                if (
                    state == "FOLLOWUP"
                    and time.time() > followup_deadline
                    and not worker.in_speech
                ):
                    print("[follow-up window expired — say wake word again]")
                    state = "WAKE"

                try:
                    audio, fast_text = phrase_queue.get(timeout=0.3)
                except queue.Empty:
                    continue

                print(f"[heard]  {fast_text!r}")

                # Decide whether to act
                if state == "FOLLOWUP":
                    # In follow-up window any utterance counts as a command.
                    command = transcribe_accurate(audio).strip() or fast_text
                    print(f"[follow-up command]  {command!r}")
                else:
                    matched, remainder = find_wake(fast_text)
                    if not matched:
                        continue

                    # Re-transcribe the same audio with the accurate model
                    accurate_text = transcribe_accurate(audio)
                    a_matched, a_remainder = find_wake(accurate_text)
                    if a_matched and (a_remainder or not remainder):
                        remainder = a_remainder
                        print(f"[heard*] {accurate_text!r}")

                    if remainder:
                        command = remainder
                    else:
                        # Wake-only utterance: chime, then wait for the command.
                        play_audio_with_mic_paused(mic, BEEP)
                        drain_phrase_queue(phrase_queue)
                        try:
                            cmd_audio, cmd_fast = phrase_queue.get(timeout=6.0)
                        except queue.Empty:
                            print("[no command — back to wake]")
                            continue
                        print(f"[heard]  {cmd_fast!r}")
                        command = transcribe_accurate(cmd_audio).strip() or cmd_fast

                if not command:
                    continue

                # Think
                print(f"[think]  {command!r}")
                t0 = time.perf_counter()
                reply = think(llm, history, command)
                history = trim_history(history)
                print(f"[reply]  {reply!r}  ({time.perf_counter()-t0:.2f}s)")

                # Speak (mic paused inside)
                speak(tts, mic, reply)
                drain_phrase_queue(phrase_queue)

                # Open follow-up window
                play_audio_with_mic_paused(mic, DOUBLE_BEEP)
                drain_phrase_queue(phrase_queue)
                state = "FOLLOWUP"
                followup_deadline = time.time() + FOLLOWUP_WINDOW_S
                print(f"[follow-up open for {FOLLOWUP_WINDOW_S:.0f}s — keep talking]")

        except KeyboardInterrupt:
            print("\n[bye]")
            return 0
        finally:
            stop_event.set()
            worker.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
