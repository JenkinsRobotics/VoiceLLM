import numpy as np
from config import AEC_ENABLED

class AECWrapper:
    def __init__(self, sample_rate=16000, frame_ms=10, filter_ms=200):
        self.sr = sample_rate
        self.frame = int(sample_rate * frame_ms / 1000)
        self.impl = None
        self._backend = "none"
        if not AEC_ENABLED:
            return
        try:
            from speexdsp import EchoCanceller  # type: ignore
            filt_len = int(sample_rate * filter_ms / 1000)
            self.impl = EchoCanceller(self.frame, filt_len)
            self._backend = "speexdsp"
        except Exception:
            self.impl = None
            self._backend = "none"

    def process(self, near_int16: np.ndarray, far_int16: np.ndarray | None):
        if self.impl is None:
            return near_int16  # passthrough
        if far_int16 is None:
            far_int16 = np.zeros_like(near_int16)
        out = np.empty_like(near_int16)
        for i in range(0, len(near_int16), self.frame):
            n = near_int16[i:i+self.frame]
            f = far_int16[i:i+self.frame]
            if len(n) < self.frame:
                n = np.pad(n, (0, self.frame-len(n)), 'constant')
                f = np.pad(f, (0, self.frame-len(f)), 'constant')
                y = self.impl.cancel(n, f)[:len(near_int16)-i]
            else:
                y = self.impl.cancel(n, f)
            out[i:i+len(y)] = y
        return out