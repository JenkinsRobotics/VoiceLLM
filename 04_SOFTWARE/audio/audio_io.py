import sounddevice as sd
import numpy as np
from collections import deque
from config import SAMPLE_RATE, BLOCK_SAMPLES, INPUT_DEVICE, AUDIO_DEBUG

def print_devices():
    try:
        devs = sd.query_devices()
        print("\n=== Audio Devices ===")
        for i, d in enumerate(devs):
            ins = d.get("max_input_channels", 0)
            outs = d.get("max_output_channels", 0)
            print(f"[{i:02d}] in={ins} out={outs}  name={d.get('name')}")
        print("=====================\n")
    except Exception as e:
        print("Device listing failed:", e)

class RingAudio:
    def __init__(self, seconds=12):
        n = seconds * SAMPLE_RATE
        self.buf = deque(maxlen=n)
        self.far = deque(maxlen=n)

    def push(self, frames_i16: np.ndarray):
        self.buf.extend(frames_i16.astype(np.int16).tolist())

    def snapshot(self):
        return np.array(self.buf, dtype=np.int16)

    def push_far(self, frames_i16: np.ndarray):
        self.far.extend(frames_i16.astype(np.int16).tolist())

    def snapshot_far(self):
        return np.array(self.far, dtype=np.int16)

def start_input(callback):
    print_devices()
    print(f"Using INPUT_DEVICE={INPUT_DEVICE}  SAMPLE_RATE={SAMPLE_RATE}  BLOCK_SAMPLES={BLOCK_SAMPLES}")
    last_print = [0]

    def _cb(indata, frames, time, status):
        if status:
            print("Audio status:", status)
        pcm = indata.copy().flatten().astype(np.int16)
        if AUDIO_DEBUG:
            import time as _t
            now = _t.time()
            if now - last_print[0] > 1.0:
                # simple RMS meter
                rms = float(np.sqrt((pcm.astype(np.float32)**2).mean()))
                print(f"Mic RMS ~ {rms:.1f}  (frames={frames})")
                last_print[0] = now
        callback(pcm)

    stream = sd.InputStream(
        device=INPUT_DEVICE,  # can be None, index, or name
        channels=1, samplerate=SAMPLE_RATE, dtype='int16',
        blocksize=BLOCK_SAMPLES, callback=_cb
    )
    stream.start()
    return stream