"""Fast interface sanity checks that do not load local models."""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import config as cfg
from core.bus import Bus
from core.metrics import MetricsLog
from core.runners.orchestrator import Orchestrator
from core.state import SysState
from plugins.llm_core.backend_base import BackendBase
from plugins.llm_core.node import LLMNode, clean_for_tts


class FakeBackend(BackendBase):
    def __init__(self, chunks: list[str]) -> None:
        super().__init__()
        self.chunks = chunks

    def load(self) -> None:
        pass

    def warm(self) -> None:
        pass

    def stream_chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Iterator[str]:
        self.reset_cancel()
        yield from self.chunks


class FakeTTS:
    def __init__(self, bus: Bus) -> None:
        self.bus = bus
        self.spoken = ""

    def feed_text(self, delta: str) -> None:
        self.spoken += delta

    def flush(self) -> None:
        self.bus.publish("tts.done", None)


class FakeSTT:
    paused = False
    followups = 0

    def set_paused(self, paused: bool) -> None:
        self.paused = paused

    def open_followup(self) -> None:
        self.followups += 1


def pump_until_idle(bus: Bus, orch: Orchestrator, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg = bus.get(timeout=0.05)
        if msg is not None:
            orch._dispatch(msg)
        if orch.state.value == SysState.IDLE and msg is None:
            return
    raise TimeoutError("orchestrator did not return to IDLE")


def main() -> int:
    bus = Bus()
    llm = LLMNode(
        bus,
        FakeBackend(["<ignore>"]),
        cfg.SYSTEM_PROMPT,
        max_tokens=cfg.LLM_MAX_TOKENS,
        temperature=cfg.LLM_TEMPERATURE,
        top_p=cfg.LLM_TOP_P,
        max_history_turns=cfg.MAX_HISTORY_TURNS,
    )
    tts = FakeTTS(bus)
    stt = FakeSTT()
    with TemporaryDirectory() as tmpdir:
        orch = Orchestrator(bus, llm, tts, stt)
        orch.metrics = MetricsLog(Path(tmpdir) / "metrics.csv")

        orch._start_turn("background chatter")
        pump_until_idle(bus, orch)

    assert tts.spoken == ""
    assert len(llm.history_snapshot()) == 1
    assert stt.followups == 1
    assert clean_for_tts("<reply>Hello there.") == "Hello there."
    assert clean_for_tts("<reply>Hello there.</reply>") == "Hello there."
    assert clean_for_tts("<ignore>") == ""

    print("Smoke OK: ignored turns are silent, history is clean, orchestrator idles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
