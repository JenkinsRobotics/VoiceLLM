"""Per-turn latency metrics → metrics.csv.

Timestamps are ``time.perf_counter()`` values (see ``now()``); the STT
nodes capture the speech-side ones and pass them in the ``stt.text``
payload, the orchestrator fills in the LLM/TTS side.

Column guide (the perceived-latency number is voice_end→tts_start):
  listen dur            speech onset → last voiced frame
  hangover              last voiced frame → VAD closes the phrase
  stt                   phrase close → accurate transcription done
  stt_done→1st_token    transcription done → first LLM token
  voice_end→tts_start   user stops talking → Eve starts talking
"""

from dataclasses import dataclass
from time import perf_counter
import csv
import os


def now():
    return perf_counter()


@dataclass
class TurnMetrics:
    wake_ts: float = 0.0          # speech onset (stt.text arrival if unknown)
    listen_start_ts: float = 0.0
    last_voice_ts: float = 0.0    # last voiced frame before the hangover
    commit_ts: float = 0.0        # VAD closed the phrase
    stt_done_ts: float = 0.0      # accurate transcription finished
    stt_text: str = ""
    llm_first_token_ts: float = 0.0
    llm_done_ts: float = 0.0
    tts_start_ts: float = 0.0
    tts_end_ts: float = 0.0
    tokens: int = 0

    def as_row(self):
        def d(a, b):
            return round((b - a) * 1000) if a and b else None
        tok_rate = None
        if self.llm_first_token_ts and self.llm_done_ts and self.llm_done_ts > self.llm_first_token_ts:
            tok_rate = round(self.tokens / (self.llm_done_ts - self.llm_first_token_ts), 2)
        return {
            "listen dur(ms)": d(self.listen_start_ts, self.last_voice_ts),
            "hangover(ms)": d(self.last_voice_ts, self.commit_ts),
            "stt(ms)": d(self.commit_ts, self.stt_done_ts),
            "stt_done→1st_token(ms)": d(self.stt_done_ts, self.llm_first_token_ts),
            "voice_end→tts_start(ms)": d(self.last_voice_ts, self.tts_start_ts),
            "tts dur(ms)": d(self.tts_start_ts, self.tts_end_ts),
            "e2e speech→tts_end(ms)": d(self.wake_ts, self.tts_end_ts),
            "tokens": self.tokens,
            "tok/s": tok_rate,
            "stt_text": (self.stt_text or "")[:160],
        }


class MetricsLog:
    def __init__(self, path="metrics.csv"):
        self.path = str(path)
        self.fieldnames = list(TurnMetrics().as_row().keys())
        expected_header = ",".join(self.fieldnames)
        if os.path.exists(self.path):
            with open(self.path, newline="") as f:
                first = f.readline().strip()
            if first != expected_header:
                # Schema changed — rotate the old file rather than appending
                # misaligned rows under a stale header.
                os.replace(self.path, self.path + ".old")
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def write(self, tm: TurnMetrics):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(tm.as_row())
