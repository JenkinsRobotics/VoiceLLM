import webrtcvad
import numpy as np
from collections import deque
from config import (
    SAMPLE_RATE,
    FRAME_MS,
    VAD_AGGRESSIVENESS,
    PRE_ROLL_MS,
    SILENCE_HANGOVER_MS,
    MAX_SPEECH_MS,
)

VAD_FRAME_MS = FRAME_MS
VAD_PRE_SPEECH_MS = PRE_ROLL_MS
VAD_POST_SPEECH_MS = SILENCE_HANGOVER_MS
VAD_MAX_SEGMENT_S = MAX_SPEECH_MS / 1000

class VADSegmenter:
    """
    WebRTC VAD segmentation with pre/post padding and hangover logic.
    Feed PCM16 blocks of any size; it re-chunks to frame_ms internally.
    Call feed(pcm16) repeatedly; poll ready() and pop_segment() when True.
    """
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.frame_len = SAMPLE_RATE * VAD_FRAME_MS // 1000  # samples
        self.buf = np.array([], dtype=np.int16)              # incoming staging
        self.frames = []                                     # list[np.int16]
        self.voicing = []                                    # list[bool]
        self.in_speech = False
        self.pre_ms = VAD_PRE_SPEECH_MS
        self.post_ms = VAD_POST_SPEECH_MS
        self.max_s = VAD_MAX_SEGMENT_S
        self._ready = False
        self._out = None

    def _flush_to_frames(self):
        n = (len(self.buf) // self.frame_len) * self.frame_len
        if n <= 0: return
        chunk = self.buf[:n]
        self.buf = self.buf[n:]
        for i in range(0, len(chunk), self.frame_len):
            fr = chunk[i:i+self.frame_len]
            is_speech = self.vad.is_speech(fr.tobytes(), SAMPLE_RATE)
            self.frames.append(fr)
            self.voicing.append(is_speech)
        # keep buffers bounded
        max_keep = int((SAMPLE_RATE * (self.max_s + self.post_ms/1000)) / self.frame_len)
        if len(self.frames) > max_keep:
            drop = len(self.frames) - max_keep
            self.frames = self.frames[drop:]
            self.voicing = self.voicing[drop:]

    def feed(self, pcm16: np.ndarray):
        # append and chunk into VAD frames
        if pcm16.dtype != np.int16:
            pcm16 = pcm16.astype(np.int16)
        self.buf = np.concatenate([self.buf, pcm16])
        self._flush_to_frames()
        self._detect()

    def _detect(self):
        if not self.frames: return
        # Hangover-based start/stop
        post_frames = self.post_ms // VAD_FRAME_MS
        pre_frames  = self.pre_ms  // VAD_FRAME_MS

        # Track active region
        active_idx = [i for i, v in enumerate(self.voicing) if v]
        if not active_idx: 
            # no speech yet
            return

        first = active_idx[0]
        last  = active_idx[-1]

        # Decide segment:
        # Start: at first speech minus pre padding (bounded)
        start = max(0, first - pre_frames)

        # End condition: enough trailing non-speech after 'last'
        trailing = 0
        for i in range(last+1, len(self.voicing)):
            if self.voicing[i]:
                trailing = 0
                last = i
            else:
                trailing += 1
                if trailing >= post_frames:
                    end = min(len(self.frames), last + 1 + post_frames)
                    self._emit(start, end)
                    # drop consumed frames
                    self.frames = self.frames[end:]
                    self.voicing = self.voicing[end:]
                    self._ready = True
                    return

        # Safety: cap very long segments
        max_frames = int(self.max_s * 1000 / VAD_FRAME_MS)
        if (last - start + 1) > max_frames:
            end = start + max_frames
            self._emit(start, end)
            self.frames = self.frames[end:]
            self.voicing = self.voicing[end:]
            self._ready = True

    def _emit(self, start, end):
        seg = np.concatenate(self.frames[start:end]) if end > start else np.array([], dtype=np.int16)
        self._out = seg

    def ready(self) -> bool:
        return self._ready

    def pop_segment(self) -> np.ndarray:
        self._ready = False
        seg = self._out
        self._out = None
        return seg if seg is not None else np.array([], dtype=np.int16)
