"""Kokoro TTS node.

Two daemon threads:
  - synth_loop: drains the text-delta queue, buffers until a sentence
    boundary or min-chars threshold, calls Kokoro KPipeline, enqueues
    audio.
  - play_loop: drains the audio queue, publishes ``mic.pause`` around
    playback, plays via sounddevice, publishes ``tts.done`` when idle.

Cancellation drains both queues and stops sounddevice.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


_SENT_END = re.compile(r"[.!?][\"')\]\}»”’]*(?=\s|$)")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_FALLBACK_MIN_CHARS = 240
_SOFT_SEPARATORS = (
    re.compile(r"\n+"),
    re.compile(r"[,;:][\"')\]\}»”’]*(?=\s|$)"),
    re.compile(r"\s[-–—]\s"),
    re.compile(r"\s+"),
)


@dataclass
class _TextEvent:
    kind: str          # "delta" | "flush" | "cancel"
    payload: str = ""


class KokoroNode:
    def __init__(
        self,
        bus,
        *,
        voice: str,
        lang: str,
        sr: int = 24000,
        min_chars: int = _FALLBACK_MIN_CHARS,
        tail_sleep_s: float = 0.12,
        output_device=None,
    ) -> None:
        from kokoro import KPipeline

        print("[tts] Loading Kokoro...", flush=True)
        t0 = time.perf_counter()
        self.pipe = KPipeline(lang_code=lang)
        # Warm-up: first synth pays graph compile tax (~3-5 s).
        list(self.pipe("Ready.", voice=voice))
        print(f"[tts] Ready in {time.perf_counter() - t0:.1f}s.", flush=True)

        self.bus = bus
        self.voice = voice
        self.sr = sr
        # Older config used 60 chars, which can fire mid-sentence. Keep the
        # public knob, but only allow it to raise the fallback threshold.
        self.min_chars = max(min_chars, _FALLBACK_MIN_CHARS)
        self.tail_sleep_s = tail_sleep_s
        self.output_device = output_device

        self.text_q: queue.Queue[_TextEvent] = queue.Queue()
        self.audio_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=32)

        self._cancelled = threading.Event()

        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    # ── Public API (called by orchestrator) ────────────────────────────

    def feed_text(self, delta: str) -> None:
        if delta:
            self.text_q.put(_TextEvent("delta", delta))

    def flush(self) -> None:
        """LLM finished — synthesize whatever's left in the buffer."""
        self.text_q.put(_TextEvent("flush"))

    def cancel(self) -> None:
        """Barge-in: drop pending text, drop pending audio, stop playback."""
        self._cancelled.set()
        self.text_q.put(_TextEvent("cancel"))
        with self.audio_q.mutex:
            self.audio_q.queue.clear()
        sd.stop()

    # ── Internals ──────────────────────────────────────────────────────

    def _pop_sentence(self, buf: str, *, force: bool) -> tuple[str, str]:
        """Return (text, remaining_buf); text='' if nothing to pop yet."""
        cut = self._boundary_cut(buf)
        if cut is None and len(buf) >= self.min_chars:
            cut = self._soft_fallback_cut(buf)
        if cut is None:
            if not force:
                return "", buf
            cut = len(buf)

        text = buf[:cut].strip()
        return text, buf[cut:]

    def _boundary_cut(self, buf: str) -> int | None:
        sentence = _SENT_END.search(buf)
        paragraph = _PARAGRAPH_BREAK.search(buf)
        cuts = [m.end() for m in (sentence, paragraph) if m]
        if not cuts:
            return None
        cut = min(cuts)
        while cut < len(buf) and buf[cut].isspace():
            cut += 1
        return cut

    def _soft_fallback_cut(self, buf: str) -> int | None:
        window = buf[:self.min_chars]
        for pattern in _SOFT_SEPARATORS:
            matches = list(pattern.finditer(window))
            if matches:
                return matches[-1].end()
        return None

    def _synth_loop(self) -> None:
        buf = ""
        while True:
            ev = self.text_q.get()

            if ev.kind == "cancel":
                buf = ""
                # Drop the cancel marker; play_loop also sees the cleared queue.
                self._cancelled.clear()
                continue

            if ev.kind == "delta":
                buf += ev.payload
                while True:
                    sentence, buf = self._pop_sentence(buf, force=False)
                    if not sentence:
                        break
                    self._synthesize(sentence)
                continue

            if ev.kind == "flush":
                if buf.strip():
                    sentence, buf = self._pop_sentence(buf, force=True)
                    if sentence:
                        self._synthesize(sentence)
                # Sentinel so play_loop knows the stream ended.
                self.audio_q.put(None)
                continue

    def _synthesize(self, text: str) -> None:
        if self._cancelled.is_set():
            return
        try:
            chunks = [
                r.audio for r in self.pipe(text, voice=self.voice)
                if r.audio is not None
            ]
        except Exception as exc:
            print(f"[tts] synthesis failed: {exc}", flush=True)
            return
        if not chunks:
            return
        audio = np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
        self.audio_q.put(audio)

    def _play_loop(self) -> None:
        speaking = False
        while True:
            audio = self.audio_q.get()

            if audio is None:
                # Stream-end sentinel.
                if speaking:
                    time.sleep(self.tail_sleep_s)
                    self.bus.publish("mic.pause", False)
                    speaking = False
                self.bus.publish("tts.done", None)
                continue

            if self._cancelled.is_set():
                # Drain any backlog without playing it.
                if self.audio_q.empty() and speaking:
                    self.bus.publish("mic.pause", False)
                    self.bus.publish("tts.done", None)
                    speaking = False
                continue

            if not speaking:
                self.bus.publish("mic.pause", True)
                speaking = True

            # Make the played audio available to AEC / similarity-filter
            # subscribers as a far-end reference.
            self.bus.publish("tts.audio_chunk", audio)
            try:
                sd.play(audio, samplerate=self.sr, device=self.output_device)
                sd.wait()
            except Exception as exc:
                print(f"[tts] playback failed: {exc}", flush=True)
