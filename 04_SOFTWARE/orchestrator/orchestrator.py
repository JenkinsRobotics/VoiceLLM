"""Orchestrator — the bus consumer that wires STT → LLM → TTS together.

States: IDLE → THINKING → RESPONDING → IDLE.

Flow per turn:
  1. STT publishes ``stt.text`` (the committed user command).
  2. Orchestrator: state = THINKING; spawn LLM thread.
  3. LLM publishes ``llm.token`` deltas → forward to TTS.feed_text().
     state = RESPONDING on the first delta.
  4. LLM publishes ``llm.done`` → TTS.flush() to synthesize tail.
  5. TTS publishes ``mic.pause(True)`` while speaking, ``mic.pause(False)``
     and ``tts.done`` when its audio queue empties.
  6. Orchestrator: write metrics, open the STT follow-up window, state = IDLE.
"""

from __future__ import annotations

import threading

from core.bus import Bus
from core.metrics import MetricsLog, TurnMetrics, now
from core.state import State, SysState


class Orchestrator:
    def __init__(self, bus: Bus, llm, tts, stt) -> None:
        self.bus = bus
        self.llm = llm
        self.tts = tts
        self.stt = stt

        self.state = State()
        self.metrics = MetricsLog()
        self.cur: TurnMetrics | None = None
        self._stop = threading.Event()
        self._llm_thread: threading.Thread | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    def run(self) -> None:
        self.stt.start()
        print("\n[ready] VoiceLLM running. Ctrl-C to quit.\n", flush=True)
        try:
            while not self._stop.is_set():
                msg = self.bus.get(timeout=0.3)
                if msg is None:
                    continue
                self._dispatch(msg)
        except KeyboardInterrupt:
            print("\n[bye]", flush=True)
        finally:
            self.stt.stop()

    def shutdown(self) -> None:
        self._stop.set()

    # ── Dispatch ───────────────────────────────────────────────────────

    def _dispatch(self, msg) -> None:
        topic, payload = msg.topic, msg.payload
        if topic == "stt.text":
            self._on_stt_text(payload)
        elif topic == "llm.token":
            self._on_llm_token(payload)
        elif topic == "llm.done":
            self._on_llm_done(payload)
        elif topic == "tts.done":
            self._on_tts_done()
        elif topic == "mic.pause":
            # TTS asks the mic to mute itself while speaking.
            self.stt.set_paused(bool(payload))
        # tts.audio_chunk is published for AEC/similarity-filter subscribers
        # later; we don't need to act on it here.

    # ── Handlers ───────────────────────────────────────────────────────

    def _on_stt_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        # Drop user phrases that arrive while a turn is already in flight.
        # M4 will replace this with a barge-in path (cancel LLM + TTS).
        if self.state.value != SysState.IDLE:
            print(f"[busy — dropped] {text!r}", flush=True)
            return

        self.cur = TurnMetrics()
        self.cur.wake_ts = now()
        self.cur.listen_start_ts = self.cur.wake_ts
        self.cur.listen_end_ts = now()
        self.cur.stt_text = text
        self.state.set(SysState.THINKING)

        print(f"[think]  {text!r}", flush=True)
        self._llm_thread = threading.Thread(
            target=self.llm.ask_stream, args=(text,), daemon=True
        )
        self._llm_thread.start()

    def _on_llm_token(self, delta: str) -> None:
        if self.cur and not self.cur.llm_first_token_ts:
            self.cur.llm_first_token_ts = now()
            if not self.cur.tts_start_ts:
                self.cur.tts_start_ts = self.cur.llm_first_token_ts
        if self.state.value != SysState.RESPONDING:
            self.state.set(SysState.RESPONDING)
        self.tts.feed_text(delta)

    def _on_llm_done(self, reply: str) -> None:
        if self.cur:
            self.cur.llm_done_ts = now()
        # Synthesize whatever tail is still buffered in TTS.
        self.tts.flush()
        if reply:
            print(f"[reply]  {reply!r}", flush=True)

    def _on_tts_done(self) -> None:
        if self.cur:
            self.cur.tts_end_ts = now()
            self.metrics.write(self.cur)
            self.cur = None
        self.state.set(SysState.IDLE)
        # Open a wake-word-free follow-up window for the next utterance.
        self.stt.open_followup()
