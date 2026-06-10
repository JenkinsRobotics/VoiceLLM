"""Microphone capture with a pause flag.

Wraps a sounddevice InputStream. Audio frames go onto an internal queue
that the STT node drains. A ``paused`` flag (set via ``mic.pause`` on the
bus) discards frames during TTS so we don't transcribe ourselves.

Ported from MockingAgent/voice_assistant.py:96-125.
"""

from __future__ import annotations

import queue
import sys

import numpy as np
import sounddevice as sd


class MicStream:
    def __init__(
        self,
        *,
        sample_rate: int,
        frame_samples: int,
        max_queue_frames: int | None = None,
        device=None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_frames or 0)
        self.paused = False
        self._stream = sd.InputStream(
            device=device,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            callback=self._cb,
        )

    def _cb(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        if self.paused or frames != self.frame_samples:
            return
        try:
            self.q.put_nowait(indata.copy())
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(indata.copy())
            except queue.Full:
                pass

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        try:
            self._stream.stop()
        finally:
            self._stream.close()

    def drain(self) -> None:
        with self.q.mutex:
            self.q.queue.clear()

    def set_paused(self, paused: bool) -> None:
        if paused == self.paused:
            return
        self.paused = paused
        if not paused:
            # Anything that snuck through during the pause boundary is stale.
            self.drain()

    # Context-manager sugar so the orchestrator can ``with mic: ...`` if it
    # wants short-lived ownership in tests.
    def __enter__(self) -> "MicStream":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
