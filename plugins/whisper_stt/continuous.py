"""Continuous-mode STT node (M3.5): hybrid phrase/word pipeline.

Ported from
    MockingAgent/PywisperCpp/pywhispercpp_examples/llm_listener/
    always_listening_hybrid_phrase_word_pipeline.py

Architecture vs. ``stt_two_pass.py``:
  * Energy-based phrase segmentation (RMS threshold) instead of WebRTC VAD.
  * One Whisper model (``base.en`` by default), not a fast→accurate cascade.
  * Rolling re-transcription of the growing phrase buffer every
    ``transcribe_every_s`` seconds, so by the time the phrase closes we
    already have a near-final transcription cached — no big Whisper hit at
    phrase end.

Public contract matches ``STTTwoPassNode``: ``start``/``stop``/
``set_paused``/``open_followup``, publishes ``stt.text`` on the bus once
per phrase close.

Wake-word handling:
  * ``require_wake_word=False`` (M3 default) — every closed phrase becomes
    a turn.
  * ``require_wake_word=True`` — find a wake phrase in the committed text;
    publish only the remainder. ``open_followup`` opens a brief window
    where the wake phrase is not required, mirroring the two-pass node.
"""

from __future__ import annotations

import queue
import re
import sys
import threading
import time
from collections import deque
from difflib import SequenceMatcher

import numpy as np

import config as cfg
from audio.chimes import ChimePlayer
from audio.mic_stream import MicStream


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _segments_text(result) -> str:
    if isinstance(result, str):
        return result.strip()
    return " ".join(s.text.strip() for s in result if s.text.strip()).strip()


class STTContinuousNode:
    def __init__(
        self,
        bus,
        *,
        model_name: str,
        require_wake_word: bool,
        wake_phrases: tuple[str, ...],
        wake_match_threshold: float,
        followup_window_s: float,
        sample_rate: int,
        block_ms: int,
        phrase_timeout_s: float,
        max_phrase_s: float,
        transcribe_every_s: float,
        min_transcribe_s: float,
        energy_threshold: float,
        post_padding_ms: int,
        duplicate_similarity: float,
        input_device=None,
    ) -> None:
        from pywhispercpp.model import Model as STTModel

        self.bus = bus
        self.require_wake_word = require_wake_word
        self.wake_phrases = wake_phrases
        self.wake_match_threshold = wake_match_threshold
        self.followup_window_s = followup_window_s

        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.phrase_timeout_s = phrase_timeout_s
        self.max_phrase_s = max_phrase_s
        self.transcribe_every_s = transcribe_every_s
        self.energy_threshold = energy_threshold
        self.duplicate_similarity = duplicate_similarity

        self._frame_samples = int(sample_rate * block_ms / 1000)
        self._max_phrase_chunks = max(1, int(max_phrase_s * 1000 / block_ms))
        self._min_transcribe_samples = int(sample_rate * min_transcribe_s)
        self._post_pad_samples = int(sample_rate * post_padding_ms / 1000)

        print(f"[stt-cont] Loading {model_name}...", flush=True)
        t0 = time.perf_counter()
        self._model = STTModel(
            model_name,
            print_realtime=False,
            print_progress=False,
            single_segment=True,
            no_context=True,
        )
        print(f"[stt-cont] Ready ({time.perf_counter() - t0:.1f}s).", flush=True)

        # Prime with 1.5 s of silence so the first real phrase doesn't pay
        # model setup cost. Whisper rejects audio under 1000 ms (it skips
        # inference and warns), so 1.5 s makes sure the warm pass runs.
        warm_audio = np.zeros(int(sample_rate * 1.5), dtype=np.float32)
        print("[stt-cont] warming up...", flush=True)
        tw = time.perf_counter()
        try:
            list(self._model.transcribe(warm_audio, language="en"))
        except Exception as exc:
            print(f"[stt-cont] warm-up skipped: {exc}", file=sys.stderr, flush=True)
        else:
            print(f"[stt-cont] primed ({time.perf_counter() - tw:.1f}s).", flush=True)

        self.mic = MicStream(
            sample_rate=sample_rate,
            frame_samples=self._frame_samples,
            max_queue_frames=cfg.MIC_QUEUE_MAX_FRAMES,
            device=input_device,
        )

        self._stop = threading.Event()
        self._loop_thread: threading.Thread | None = None

        self._phrase_chunks: deque[np.ndarray] = deque(maxlen=self._max_phrase_chunks)
        self._last_activity = time.monotonic()
        self._current_text = ""
        self._last_committed_text = ""
        self._chimes = ChimePlayer()

        # Wake-word follow-up window (only used when require_wake_word=True).
        self._followup_deadline = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        self.mic.start()
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
        if paused:
            # Drop in-flight phrase state so we don't merge across the gap.
            self._phrase_chunks.clear()
            self._current_text = ""
            self._last_activity = time.monotonic()

    def open_followup(self) -> None:
        if self.require_wake_word:
            self._followup_deadline = time.time() + self.followup_window_s

    def play_chime(self, kind: str) -> None:
        """Play a tiny UI earcon while suppressing mic capture."""
        if not self._chimes.enabled(kind):
            return
        was_paused = self.mic.paused
        self.mic.set_paused(True)
        try:
            self._chimes.play(kind, output_device=cfg.OUTPUT_DEVICE)
        finally:
            self.mic.set_paused(was_paused)

    # ── Wake-word matching (only used when require_wake_word=True) ─────

    def _extract_command(self, text: str) -> str | None:
        norm = _normalize(text)
        if not norm:
            return None

        for phrase in sorted(self.wake_phrases, key=len, reverse=True):
            phrase_norm = _normalize(phrase)
            idx = norm.find(phrase_norm)
            if idx != -1:
                tail = norm[idx + len(phrase_norm):].strip()
                return tail or text.strip()

        tokens = norm.split()
        for phrase in self.wake_phrases:
            phrase_tokens = _normalize(phrase).split()
            n = len(phrase_tokens)
            for i in range(0, max(0, len(tokens) - n + 1)):
                window = " ".join(tokens[i:i + n])
                if SequenceMatcher(None, window, " ".join(phrase_tokens)).ratio() \
                        >= self.wake_match_threshold:
                    tail = " ".join(tokens[i + n:]).strip()
                    return tail or text.strip()

        return None

    # ── Main loop ──────────────────────────────────────────────────────

    def _main_loop(self) -> None:
        next_transcribe_at = time.monotonic() + self.transcribe_every_s

        while not self._stop.is_set():
            self._drain_audio()
            now = time.monotonic()

            if self.mic.paused:
                # Hold the activity timer high so the post-resume audio gets
                # treated as the start of a fresh phrase.
                self._last_activity = now
                time.sleep(0.05)
                continue

            if now >= next_transcribe_at:
                self._rolling_transcribe()
                next_transcribe_at = now + self.transcribe_every_s

            if self._phrase_is_closing(now):
                self._close_phrase()

            time.sleep(0.02)

    def _drain_audio(self) -> None:
        try:
            while True:
                chunk = self.mic.q.get_nowait()
                mono = chunk[:, 0].astype(np.float32).reshape(-1)
                self._phrase_chunks.append(mono)
                rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                if rms >= self.energy_threshold:
                    self._last_activity = time.monotonic()
        except queue.Empty:
            return

    def _current_phrase_audio(self) -> np.ndarray | None:
        if not self._phrase_chunks:
            return None
        audio = np.concatenate(list(self._phrase_chunks)).astype(np.float32)
        if audio.size < self._min_transcribe_samples:
            return None
        padding = np.zeros(self._post_pad_samples, dtype=np.float32)
        return np.concatenate([audio, padding])

    def _transcribe(self, audio: np.ndarray) -> str:
        try:
            return _segments_text(self._model.transcribe(audio, language="en"))
        except Exception as exc:
            print(f"[stt-cont] {exc}", file=sys.stderr)
            return ""

    def _rolling_transcribe(self) -> None:
        audio = self._current_phrase_audio()
        if audio is None:
            return
        text = self._transcribe(audio)
        if not text:
            return
        if SequenceMatcher(None, text, self._current_text).ratio() \
                >= self.duplicate_similarity:
            return
        self._current_text = text

    def _phrase_is_closing(self, now: float) -> bool:
        if not self._phrase_chunks:
            return False
        phrase_seconds = len(self._phrase_chunks) * self.block_ms / 1000.0
        quiet_seconds = now - self._last_activity
        return (
            quiet_seconds >= self.phrase_timeout_s
            or phrase_seconds >= self.max_phrase_s
        )

    def _close_phrase(self) -> None:
        # One last transcription pass — captures any words spoken between
        # the most recent rolling pass and the silence that closed the phrase.
        audio = self._current_phrase_audio()
        text = self._transcribe(audio) if audio is not None else self._current_text
        text = (text or "").strip()

        self._phrase_chunks.clear()
        self._current_text = ""

        if not text:
            return

        if SequenceMatcher(None, text, self._last_committed_text).ratio() \
                >= self.duplicate_similarity:
            return
        self._last_committed_text = text

        self._publish(text)

    def _publish(self, text: str) -> None:
        print(f"[heard]  {text!r}", flush=True)

        if not self.require_wake_word:
            self.bus.publish("stt.text", text)
            return

        if time.time() <= self._followup_deadline:
            self._followup_deadline = 0.0
            self.bus.publish("stt.text", text)
            return

        command = self._extract_command(text)
        if command:
            self.bus.publish("stt.text", command)
