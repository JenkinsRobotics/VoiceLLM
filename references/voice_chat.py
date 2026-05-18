#!/usr/bin/env python3
"""Full-duplex local voice chat.

Pipeline:
  mic + speaker -> AudioIO/AEC -> VAD worker -> phrase_queue
                                               |
                                               v
                 continuous chat loop -> 2-pass STT -> local LLM -> streaming TTS

This is intentionally not an advanced agent/tool framework. It is a focused
testbed for the improved voice loop: full-duplex audio, echo cancellation,
barge-in, two-pass STT, and streamed Kokoro TTS, while keeping the simple
llama.cpp conversation brain from voice_assistant.py.

Run it, then just talk. Ctrl-C to quit.
"""

from __future__ import annotations

import collections
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyaec
import sounddevice as sd
import webrtcvad
from scipy.signal import resample_poly


# -- config -----------------------------------------------------------------
LLM_MODEL_PATH = Path(
    "/Users/jonathanjenkins/.lmstudio/models/lmstudio-community/"
    "gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf"
)

# Fast STT runs first so short false positives are cheap. Accurate STT is used
# for the final user text that goes to the model.
STT_FAST = "base.en"
STT_ACCURATE = "medium.en"

KOKORO_VOICE = "af_heart"
KOKORO_LANG = "a"

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# Speex AEC works on 10ms int16 frames. Three AEC frames become one VAD frame.
AEC_FRAME_MS = 10
AEC_FRAME_SAMPLES = SAMPLE_RATE * AEC_FRAME_MS // 1000
AEC_FRAMES_PER_VAD = FRAME_MS // AEC_FRAME_MS
AEC_FILTER_LENGTH = 3200

TTS_NATIVE_SR = 24000

VAD_AGGRESSIVENESS = 2
PRE_ROLL_MS = 240
POST_PADDING_MS = 250
SILENCE_HANGOVER_MS = 700
MIN_SPEECH_MS = 400
MAX_SPEECH_MS = 12000
BARGE_IN_MS = 200

SYSTEM_PROMPT = (
    "You are a helpful voice chat companion. Answer in 1-2 short sentences "
    "in plain conversational English. No markdown, no code blocks, no emojis, "
    "and no lists unless the user explicitly asks for one."
)

MAX_HISTORY_TURNS = 8


# -- audio I/O with AEC ------------------------------------------------------
class AudioIO:
    """Full-duplex 16kHz audio with Speex acoustic echo cancellation."""

    def __init__(self) -> None:
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self._play_q: queue.Queue[np.ndarray] = queue.Queue()
        self._current_chunk = np.zeros(0, dtype=np.int16)
        self._mic_pending: list[np.ndarray] = []
        self._aec = pyaec.Aec(
            frame_size=AEC_FRAME_SAMPLES,
            filter_length=AEC_FILTER_LENGTH,
            sample_rate=SAMPLE_RATE,
            enable_preprocess=True,
        )
        self._playing = False
        self._barged = False
        self._stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            channels=(1, 1),
            dtype="int16",
            blocksize=AEC_FRAME_SAMPLES,
            callback=self._cb,
            latency=("low", "low"),
        )

    def _cb(self, indata, outdata, frames, time_info, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        if frames != AEC_FRAME_SAMPLES:
            outdata.fill(0)
            return

        out = np.zeros(frames, dtype=np.int16)
        n_filled = 0
        while n_filled < frames:
            if len(self._current_chunk) == 0:
                try:
                    self._current_chunk = self._play_q.get_nowait()
                except queue.Empty:
                    break
            take = min(frames - n_filled, len(self._current_chunk))
            out[n_filled : n_filled + take] = self._current_chunk[:take]
            self._current_chunk = self._current_chunk[take:]
            n_filled += take

        outdata[:, 0] = out
        self._playing = (
            n_filled > 0 or len(self._current_chunk) > 0 or not self._play_q.empty()
        )

        mic_frame = indata[:, 0].astype(np.int16).tolist()
        ref_frame = out.tolist()
        cleaned = self._aec.cancel_echo(mic_frame, ref_frame)
        cleaned_arr = np.asarray(cleaned, dtype=np.int16)

        self._mic_pending.append(cleaned_arr)
        if len(self._mic_pending) >= AEC_FRAMES_PER_VAD:
            bundled = np.concatenate(self._mic_pending[:AEC_FRAMES_PER_VAD])
            self._mic_pending = self._mic_pending[AEC_FRAMES_PER_VAD:]
            f32 = (bundled.astype(np.float32) / 32767.0).reshape(-1, 1)
            self.q.put(f32)

    def play_chunk(self, chunk_int16: np.ndarray) -> None:
        self._play_q.put(chunk_int16)

    def wait_until_drained(self, poll_s: float = 0.02, max_s: float = 30.0) -> None:
        deadline = time.time() + max_s
        while time.time() < deadline:
            if self._play_q.empty() and len(self._current_chunk) == 0:
                return
            if self._barged:
                return
            time.sleep(poll_s)

    def is_playing(self) -> bool:
        return self._playing

    def interrupt_playback(self) -> None:
        with self._play_q.mutex:
            self._play_q.queue.clear()
        self._current_chunk = np.zeros(0, dtype=np.int16)
        self._barged = True

    def clear_barge(self) -> None:
        self._barged = False

    def was_barged(self) -> bool:
        return self._barged

    def drain_mic(self) -> None:
        with self.q.mutex:
            self.q.queue.clear()

    def __enter__(self) -> "AudioIO":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stream.stop()
        self._stream.close()


# -- VAD worker --------------------------------------------------------------
class VadWorker(threading.Thread):
    """Accumulates VAD-detected phrases and supports barge-in during TTS."""

    def __init__(
        self,
        audio_io: AudioIO,
        fast_model,
        phrase_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.audio_io = audio_io
        self.fast_model = fast_model
        self.phrase_queue = phrase_queue
        self.stop_event = stop_event
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        self.silence_blocks_to_end = max(1, SILENCE_HANGOVER_MS // FRAME_MS)
        self.min_speech_blocks = max(1, MIN_SPEECH_MS // FRAME_MS)
        self.max_speech_blocks = max(self.min_speech_blocks, MAX_SPEECH_MS // FRAME_MS)
        self.pre_roll_blocks = max(0, PRE_ROLL_MS // FRAME_MS)
        self.post_pad_samples = int(SAMPLE_RATE * POST_PADDING_MS / 1000)
        self.barge_blocks = max(1, BARGE_IN_MS // FRAME_MS)
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
                chunk = self.audio_io.q.get(timeout=0.3)
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

            self.in_speech = in_speech and speech_blocks >= self.min_speech_blocks

            if (
                in_speech
                and speech_blocks >= self.barge_blocks
                and self.audio_io.is_playing()
                and not self.audio_io.was_barged()
            ):
                self.audio_io.interrupt_playback()
                print("[barge-in] user speech detected, stopping TTS")

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


# -- local LLM ---------------------------------------------------------------
def load_llm():
    from llama_cpp import Llama

    print(f"[llm] loading {LLM_MODEL_PATH.name}...", flush=True)
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(LLM_MODEL_PATH),
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(f"[llm] loaded in {time.perf_counter()-t0:.1f}s, warming up...", flush=True)
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
        temperature=0.0,
    )
    print("[llm] ready", flush=True)
    return llm


def trim_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    if len(history) <= 1 + max_turns * 2:
        return history
    return history[:1] + history[-max_turns * 2:]


def clean_for_tts(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^[\-\*\d\.\)]+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def think(llm, history: list[dict], user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    out = llm.create_chat_completion(
        messages=history,
        max_tokens=200,
        temperature=0.7,
        top_p=0.95,
    )
    reply = out["choices"][0]["message"]["content"].strip()
    history.append({"role": "assistant", "content": reply})
    return clean_for_tts(reply)


# -- TTS ---------------------------------------------------------------------
def load_tts():
    from kokoro import KPipeline

    print("[tts] loading Kokoro...", flush=True)
    t0 = time.perf_counter()
    pipe = KPipeline(lang_code=KOKORO_LANG)
    list(pipe("Ready.", voice=KOKORO_VOICE))
    print(f"[tts] ready ({time.perf_counter()-t0:.1f}s)", flush=True)
    return pipe


def _resample_to_mic_rate(audio_f32: np.ndarray) -> np.ndarray:
    if audio_f32.size == 0:
        return np.zeros(0, dtype=np.int16)
    resampled = resample_poly(audio_f32, up=SAMPLE_RATE, down=TTS_NATIVE_SR)
    return np.clip(resampled * 32767.0, -32768, 32767).astype(np.int16)


def speak(pipe, audio_io: AudioIO, text: str) -> bool:
    """Stream TTS into the duplex output queue.

    Returns False when barge-in stopped playback.
    """
    if not text:
        return True
    audio_io.clear_barge()
    for r in pipe(text, voice=KOKORO_VOICE):
        if audio_io.was_barged():
            break
        if r.audio is None:
            continue
        chunk_24k = np.asarray(r.audio, dtype=np.float32)
        audio_io.play_chunk(_resample_to_mic_rate(chunk_24k))
    audio_io.wait_until_drained()
    return not audio_io.was_barged()


# -- utilities ---------------------------------------------------------------
def warm_stt(model, label: str) -> None:
    warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    print(f"[{label}] warming up...", flush=True)
    t0 = time.perf_counter()
    try:
        list(model.transcribe(warm_audio, language="en"))
    except Exception as exc:
        print(f"[{label}] warm-up skipped: {exc}", file=sys.stderr, flush=True)
    else:
        print(f"[{label}] primed ({time.perf_counter()-t0:.1f}s)", flush=True)


def main() -> int:
    from pywhispercpp.model import Model as STTModel

    print(f"[stt-fast] loading {STT_FAST}...", flush=True)
    t0 = time.perf_counter()
    fast_stt = STTModel(
        STT_FAST,
        print_realtime=False,
        print_progress=False,
        single_segment=True,
        no_context=True,
    )
    print(f"[stt-fast] ready ({time.perf_counter()-t0:.1f}s)", flush=True)
    warm_stt(fast_stt, "stt-fast")

    print(f"[stt-accurate] loading {STT_ACCURATE}...", flush=True)
    t0 = time.perf_counter()
    accurate_stt = STTModel(
        STT_ACCURATE,
        print_realtime=False,
        print_progress=False,
        single_segment=True,
        no_context=True,
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

    print("\n[ready] continuous voice chat is listening. Ctrl-C to quit.\n")

    with AudioIO() as audio_io:
        worker = VadWorker(audio_io, fast_stt, phrase_queue, stop_event)
        worker.start()
        try:
            while True:
                try:
                    audio, fast_text = phrase_queue.get(timeout=0.3)
                except queue.Empty:
                    continue

                print(f"[heard-fast] {fast_text!r}")
                user_text = transcribe_accurate(audio).strip() or fast_text
                if not user_text:
                    continue
                print(f"[heard]      {user_text!r}")

                print("[think]")
                t0 = time.perf_counter()
                reply = think(llm, history, user_text)
                history = trim_history(history)
                print(f"[reply]     {reply!r} ({time.perf_counter()-t0:.2f}s)")

                completed = speak(tts, audio_io, reply)
                if not completed:
                    print("[reply cut short by barge-in]")

        except KeyboardInterrupt:
            print("\n[bye]")
            return 0
        finally:
            stop_event.set()
            worker.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
