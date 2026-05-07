"""Two-pass STT node: VAD-segmented, fast model gates, accurate model commits.

Ported from MockingAgent/voice_assistant.py (the proven Google-Home-style
baseline). Runs a VAD worker thread that:
  1. accumulates audio while VAD says speech,
  2. on phrase close, transcribes with the fast model,
  3. checks for a wake phrase (or the follow-up window),
  4. re-transcribes with the accurate model when a match is confirmed,
  5. publishes ``stt.text`` with the final command for the LLM.

The follow-up window means that after each LLM reply the user has
``FOLLOWUP_WINDOW_S`` seconds to speak again without saying the wake word.
"""

from __future__ import annotations

import collections
import queue
import re
import sys
import threading
import time
from difflib import SequenceMatcher

import numpy as np
import webrtcvad

from audio.mic_stream import MicStream


class _VadWorker(threading.Thread):
    """Reads MicStream frames, runs VAD, finalizes phrases through fast STT."""

    def __init__(
        self,
        mic: MicStream,
        fast_model,
        phrase_q: queue.Queue,
        stop_event: threading.Event,
        *,
        sample_rate: int,
        frame_ms: int,
        vad_aggressiveness: int,
        pre_roll_ms: int,
        post_padding_ms: int,
        silence_hangover_ms: int,
        min_speech_ms: int,
        max_speech_ms: int,
    ) -> None:
        super().__init__(daemon=True)
        self.mic = mic
        self.fast_model = fast_model
        self.phrase_q = phrase_q
        self.stop_event = stop_event
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        self.silence_blocks_to_end = max(1, silence_hangover_ms // frame_ms)
        self.min_speech_blocks = max(1, min_speech_ms // frame_ms)
        self.max_speech_blocks = max(self.min_speech_blocks, max_speech_ms // frame_ms)
        self.pre_roll_blocks = max(0, pre_roll_ms // frame_ms)
        self.post_pad_samples = int(sample_rate * post_padding_ms / 1000)

        # Exposed so the main loop can avoid expiring the follow-up window
        # while the user is still mid-sentence.
        self.in_speech = False

    def _is_speech(self, chunk: np.ndarray) -> bool:
        pcm = (chunk[:, 0] * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        return self.vad.is_speech(pcm, self.sample_rate)

    def _finalize(self, chunks: list[np.ndarray]) -> None:
        audio = np.concatenate(chunks, axis=0).astype(np.float32).reshape(-1)
        # Trailing silence so Whisper doesn't clip the last word.
        audio = np.concatenate([audio, np.zeros(self.post_pad_samples, dtype=np.float32)])
        try:
            segments = self.fast_model.transcribe(audio, language="en")
            text = " ".join(s.text for s in segments).strip()
        except Exception as exc:
            print(f"[stt-fast] {exc}", file=sys.stderr)
            text = ""
        if text:
            self.phrase_q.put((audio, text))

    def run(self) -> None:
        pre_roll: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.pre_roll_blocks
        )
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

            self.in_speech = in_speech and speech_blocks >= self.min_speech_blocks

            phrase_done = (
                in_speech
                and speech_blocks >= self.min_speech_blocks
                and (
                    silent_blocks >= self.silence_blocks_to_end
                    or speech_blocks >= self.max_speech_blocks
                )
            )
            if phrase_done:
                self._finalize(speech)
                speech = []
                speech_blocks = 0
                silent_blocks = 0
                in_speech = False
                self.in_speech = False
                pre_roll.clear()


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


class STTTwoPassNode:
    """Public node — owns the mic, the VAD worker, and both Whisper models."""

    def __init__(
        self,
        bus,
        *,
        fast_model_name: str,
        accurate_model_name: str,
        require_wake_word: bool,
        wake_phrases: tuple[str, ...],
        wake_match_threshold: float,
        followup_window_s: float,
        sample_rate: int,
        frame_samples: int,
        frame_ms: int,
        vad_aggressiveness: int,
        pre_roll_ms: int,
        post_padding_ms: int,
        silence_hangover_ms: int,
        min_speech_ms: int,
        max_speech_ms: int,
        input_device=None,
    ) -> None:
        from pywhispercpp.model import Model as STTModel

        self.bus = bus
        self.require_wake_word = require_wake_word
        self.wake_phrases = wake_phrases
        self.wake_match_threshold = wake_match_threshold
        self.followup_window_s = followup_window_s

        print(f"[stt-fast] Loading {fast_model_name}...", flush=True)
        t0 = time.perf_counter()
        self._fast = STTModel(
            fast_model_name,
            print_realtime=False,
            print_progress=False,
            single_segment=True,
            no_context=True,
        )
        print(f"[stt-fast] Ready ({time.perf_counter() - t0:.1f}s).", flush=True)

        # Lazy-load: don't pay the accurate model's cost unless we actually
        # confirm a wake match.
        self._STTModel = STTModel
        self._accurate_name = accurate_model_name
        self._accurate = None

        self.mic = MicStream(
            sample_rate=sample_rate,
            frame_samples=frame_samples,
            device=input_device,
        )
        self.phrase_q: queue.Queue[tuple[np.ndarray, str]] = queue.Queue()
        self._stop = threading.Event()
        self._worker = _VadWorker(
            self.mic,
            self._fast,
            self.phrase_q,
            self._stop,
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            vad_aggressiveness=vad_aggressiveness,
            pre_roll_ms=pre_roll_ms,
            post_padding_ms=post_padding_ms,
            silence_hangover_ms=silence_hangover_ms,
            min_speech_ms=min_speech_ms,
            max_speech_ms=max_speech_ms,
        )

        self._loop_thread: threading.Thread | None = None
        self._state = "WAKE"        # "WAKE" | "FOLLOWUP"
        self._followup_deadline = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        self.mic.start()
        self._worker.start()
        self._loop_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.mic.stop()
        except Exception:
            pass

    def set_paused(self, paused: bool) -> None:
        self.mic.set_paused(paused)

    def open_followup(self) -> None:
        """Called by the orchestrator after a TTS reply finishes."""
        if self.require_wake_word:
            self._state = "FOLLOWUP"
            self._followup_deadline = time.time() + self.followup_window_s

    # ── Wake-word matching ─────────────────────────────────────────────

    def _find_wake(self, text: str) -> tuple[bool, str]:
        norm = _normalize(text)
        for phrase in self.wake_phrases:
            idx = norm.find(phrase)
            if idx != -1:
                return True, norm[idx + len(phrase):].strip()
        tokens = norm.split()
        for phrase in self.wake_phrases:
            n = len(phrase.split())
            for i in range(0, max(0, len(tokens) - n + 1)):
                window = " ".join(tokens[i:i + n])
                if SequenceMatcher(None, window, phrase).ratio() >= self.wake_match_threshold:
                    return True, " ".join(tokens[i + n:]).strip()
        return False, ""

    def _accurate_transcribe(self, audio: np.ndarray) -> str:
        if self._accurate is None:
            print(f"[stt-accurate] Loading {self._accurate_name} (one-time)...", flush=True)
            t0 = time.perf_counter()
            self._accurate = self._STTModel(
                self._accurate_name,
                print_realtime=False,
                print_progress=False,
                single_segment=True,
                no_context=True,
            )
            print(f"[stt-accurate] Ready ({time.perf_counter() - t0:.1f}s).", flush=True)
        segments = self._accurate.transcribe(audio, language="en")
        return " ".join(s.text for s in segments).strip()

    # ── Main loop ──────────────────────────────────────────────────────

    def _main_loop(self) -> None:
        while not self._stop.is_set():
            # Expire the follow-up window unless the user is mid-utterance.
            if (
                self._state == "FOLLOWUP"
                and time.time() > self._followup_deadline
                and not self._worker.in_speech
            ):
                self._state = "WAKE"

            try:
                audio, fast_text = self.phrase_q.get(timeout=0.3)
            except queue.Empty:
                continue

            print(f"[heard]  {fast_text!r}", flush=True)

            command: str | None = None

            if not self.require_wake_word:
                # M3 path: every phrase is a turn (caller is expected to
                # supply a similarity filter against recent assistant replies).
                command = self._accurate_transcribe(audio).strip() or fast_text

            elif self._state == "FOLLOWUP":
                command = self._accurate_transcribe(audio).strip() or fast_text
                print(f"[follow-up] {command!r}", flush=True)

            else:
                matched, remainder = self._find_wake(fast_text)
                if not matched:
                    continue
                accurate_text = self._accurate_transcribe(audio)
                a_matched, a_remainder = self._find_wake(accurate_text)
                if a_matched and (a_remainder or not remainder):
                    remainder = a_remainder
                    print(f"[heard*] {accurate_text!r}", flush=True)

                if remainder:
                    command = remainder
                else:
                    # Wake-only utterance — wait briefly for the actual command.
                    try:
                        cmd_audio, cmd_fast = self.phrase_q.get(timeout=6.0)
                    except queue.Empty:
                        print("[no command — back to wake]", flush=True)
                        continue
                    print(f"[heard]  {cmd_fast!r}", flush=True)
                    command = self._accurate_transcribe(cmd_audio).strip() or cmd_fast

            if not command:
                continue

            # Going into THINKING — pre-emptively close the follow-up window
            # so a slow TTS doesn't get a stale "still listening" feel.
            self._state = "WAKE"
            self.bus.publish("stt.text", command)
