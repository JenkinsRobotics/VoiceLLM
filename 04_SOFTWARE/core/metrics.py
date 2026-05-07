from dataclasses import dataclass
from time import perf_counter
import csv, os

def now(): return perf_counter()

@dataclass
class TurnMetrics:
    wake_ts: float = 0.0
    listen_start_ts: float = 0.0
    listen_end_ts: float = 0.0
    stt_text: str = ""
    llm_first_token_ts: float = 0.0
    llm_done_ts: float = 0.0
    tts_start_ts: float = 0.0
    tts_end_ts: float = 0.0
    tokens: int = 0

    def as_row(self):
        def d(a,b): return round((b-a)*1000) if a and b else None
        tok_rate = None
        if self.llm_first_token_ts and self.llm_done_ts and self.llm_done_ts > self.llm_first_token_ts:
            tok_rate = round(self.tokens / (self.llm_done_ts - self.llm_first_token_ts), 2)
        return {
            "wake→listen(ms)": d(self.wake_ts, self.listen_start_ts),
            "listen dur(ms)": d(self.listen_start_ts, self.listen_end_ts),
            "stt→1st_token(ms)": d(self.listen_end_ts, self.llm_first_token_ts),
            "1st→tts_start(ms)": d(self.llm_first_token_ts, self.tts_start_ts),
            "tts dur(ms)": d(self.tts_start_ts, self.tts_end_ts),
            "e2e wake→tts_end(ms)": d(self.wake_ts, self.tts_end_ts),
            "tokens": self.tokens,
            "tok/s": tok_rate,
            "stt_text": (self.stt_text or "")[:160]
        }

class MetricsLog:
    def __init__(self, path="metrics.csv"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                import csv
                w = csv.DictWriter(f, fieldnames=list(TurnMetrics().as_row().keys()))
                w.writeheader()
    def write(self, tm: TurnMetrics):
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tm.as_row().keys()))
            w.writerow(tm.as_row())