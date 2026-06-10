"""Tiny UI earcons for wake and follow-up cues."""

from __future__ import annotations

import sys
import time

import numpy as np
import sounddevice as sd

import config as cfg


def _make_chime(freq: float, duration_ms: int) -> np.ndarray:
    n = int(cfg.CHIME_SAMPLE_RATE * duration_ms / 1000)
    t = np.arange(n) / cfg.CHIME_SAMPLE_RATE
    duration_s = duration_ms / 1000
    env = np.minimum(np.minimum(t / 0.01, 1.0), (duration_s - t) / 0.01).clip(0, 1)
    return (cfg.CHIME_VOLUME * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_followup_chime() -> np.ndarray:
    gap = np.zeros(
        int(cfg.CHIME_SAMPLE_RATE * cfg.FOLLOWUP_CHIME_GAP_MS / 1000),
        dtype=np.float32,
    )
    return np.concatenate(
        [
            _make_chime(cfg.FOLLOWUP_CHIME_LOW_FREQ, cfg.FOLLOWUP_CHIME_DURATION_MS),
            gap,
            _make_chime(cfg.FOLLOWUP_CHIME_HIGH_FREQ, cfg.FOLLOWUP_CHIME_DURATION_MS),
        ]
    )


class ChimePlayer:
    def __init__(self) -> None:
        self._wake = _make_chime(cfg.WAKE_CHIME_FREQ, cfg.WAKE_CHIME_DURATION_MS)
        self._followup = _make_followup_chime()

    def enabled(self, kind: str) -> bool:
        if not cfg.CHIMES_ENABLED:
            return False
        if kind == "wake":
            return cfg.WAKE_CHIME_ENABLED
        if kind == "followup":
            return cfg.FOLLOWUP_CHIME_ENABLED
        return False

    def play(self, kind: str, *, output_device=None) -> None:
        if not self.enabled(kind):
            return
        if kind == "wake":
            audio = self._wake
        elif kind == "followup":
            audio = self._followup
        else:
            return

        try:
            sd.play(audio, samplerate=cfg.CHIME_SAMPLE_RATE, device=output_device)
            sd.wait()
            time.sleep(cfg.TTS_TAIL_SLEEP_S)
        except Exception as exc:
            print(f"[chime] playback failed: {exc}", file=sys.stderr, flush=True)
