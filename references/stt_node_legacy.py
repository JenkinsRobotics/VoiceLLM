import numpy as np, time, tempfile, wave, os
from pywhispercpp.model import Model
from config import SAMPLE_RATE
from audio.vad import VADSegmenter

class STTNode:
    """
    VAD-gated segmentation:
    - feed() accumulates frames into a segmenter
    - when a segment closes, we transcribe it and publish 'stt.text'
    """
    def __init__(self, bus, model_name="medium.en"):
        self.bus = bus
        self.whisper = Model(model_name)
        self.vad = VADSegmenter()

    def feed(self, pcm16: np.ndarray):
        self.vad.feed(pcm16)
        while self.vad.ready():
            seg = self.vad.pop_segment()
            if seg.size == 0:
                continue
            text = self._transcribe_path(seg)
            text = (text or "").strip()
            if text:
                self.bus.publish("stt.text", text)

    def _transcribe_path(self, pcm16: np.ndarray) -> str:
        # write a temp WAV and pass filename (pywhispercpp path API)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp.name
        tmp.close()
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm16.tobytes())
            out = self.whisper.transcribe(tmp_path)
            if isinstance(out, (list, tuple)):
                return " ".join([s.text if hasattr(s, "text") else str(s) for s in out])
            return str(out)
        finally:
            try: os.unlink(tmp_path)
            except: pass