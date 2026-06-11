"""Fast sanity checks that do not load local models.

Each scenario drives the real Orchestrator/LLMNode/Bus with fake
backend/TTS/STT, including the failure paths a live session hits:
gate tags split across stream deltas, backends that raise, empty
replies, Whisper hallucinations, farewell turns.
"""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import config as cfg
from transport.bus import Bus
from core.metrics import MetricsLog
from agent.orchestrator import Orchestrator
from core.state import SysState
from agent.llm.backend_base import BackendBase
from agent.llm.node import LLMNode, clean_for_tts
from agent.adapters.mlx.backend import _scan_stream_text


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


class ErrorBackend(FakeBackend):
    def stream_chat(self, messages, *, max_tokens, temperature, top_p):
        self.reset_cancel()
        raise RuntimeError("boom")
        yield  # makes this a generator; never reached


class FakeTTS:
    def __init__(self, bus: Bus) -> None:
        self.bus = bus
        self.spoken = ""

    def feed_text(self, delta: str) -> None:
        self.spoken += delta

    def flush(self) -> None:
        self.bus.publish("tts.done", None)


class FakeSTT:
    def __init__(self) -> None:
        self.paused = False
        self.followups = 0

    def set_paused(self, paused: bool) -> None:
        self.paused = paused

    def open_followup(self) -> None:
        self.followups += 1


def make_orch(backend: BackendBase, tmpdir: str, name: str):
    bus = Bus()
    llm = LLMNode(
        bus,
        backend,
        cfg.SYSTEM_PROMPT,
        max_tokens=cfg.LLM_MAX_TOKENS,
        temperature=cfg.LLM_TEMPERATURE,
        top_p=cfg.LLM_TOP_P,
        max_history_turns=cfg.MAX_HISTORY_TURNS,
    )
    tts = FakeTTS(bus)
    stt = FakeSTT()
    orch = Orchestrator(bus, llm, tts, stt)
    orch.metrics = MetricsLog(Path(tmpdir) / f"metrics_{name}.csv")
    orch._eval_log_path = None  # keep tests out of outputs/m3_eval.jsonl
    return bus, llm, tts, stt, orch


def pump_until_idle(bus: Bus, orch: Orchestrator, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg = bus.get(timeout=0.05)
        if msg is not None:
            orch._dispatch(msg)
        if orch.state.value == SysState.IDLE and msg is None:
            return
    raise TimeoutError("orchestrator did not return to IDLE")


def test_ignore_turn(tmpdir: str) -> None:
    bus, llm, tts, stt, orch = make_orch(FakeBackend(["<ignore>"]), tmpdir, "ignore")
    orch._start_turn("background chatter")
    pump_until_idle(bus, orch)
    assert tts.spoken == ""
    assert len(llm.history_snapshot()) == 1, "ignored pair must be discarded"
    assert stt.followups == 0, "ignored turns must not re-arm the followup window"


def test_reply_tag_split_across_deltas(tmpdir: str) -> None:
    bus, llm, tts, stt, orch = make_orch(
        FakeBackend(["<re", "ply>Hel", "lo there."]), tmpdir, "split"
    )
    orch._start_turn("hi eve")
    pump_until_idle(bus, orch)
    assert tts.spoken == "Hello there.", tts.spoken
    assert len(llm.history_snapshot()) == 3
    assert stt.followups == 1


def test_spaced_gate_tag(tmpdir: str) -> None:
    bus, llm, tts, stt, orch = make_orch(FakeBackend(["< Reply > Hi."]), tmpdir, "spaced")
    orch._start_turn("hello")
    pump_until_idle(bus, orch)
    assert tts.spoken.strip() == "Hi.", tts.spoken
    assert "< Reply >" not in tts.spoken


def test_gate_fallback_when_tag_missing(tmpdir: str) -> None:
    text = "This reply has no tag at all, sorry about that."
    bus, llm, tts, stt, orch = make_orch(FakeBackend([text]), tmpdir, "fallback")
    orch._start_turn("hello")
    pump_until_idle(bus, orch)
    assert tts.spoken == text, tts.spoken


def test_backend_error_speaks_and_rolls_back(tmpdir: str) -> None:
    bus, llm, tts, stt, orch = make_orch(ErrorBackend([]), tmpdir, "error")
    orch._start_turn("hello")
    pump_until_idle(bus, orch)
    assert "Sorry" in tts.spoken, "errors must be audible, not silent"
    assert len(llm.history_snapshot()) == 1, "failed user msg must be rolled back"
    assert orch.state.value == SysState.IDLE


def test_empty_reply_discarded(tmpdir: str) -> None:
    bus, llm, tts, stt, orch = make_orch(FakeBackend(["<reply>"]), tmpdir, "empty")
    orch._start_turn("hello")
    pump_until_idle(bus, orch)
    assert tts.spoken == ""
    assert len(llm.history_snapshot()) == 1, "empty pair must not pollute history"


def test_farewell_suppresses_followup(tmpdir: str) -> None:
    bus, llm, tts, stt, orch = make_orch(
        FakeBackend(["<reply>Goodbye! Talk to you later."]), tmpdir, "farewell"
    )
    orch._start_turn("okay goodbye eve")
    pump_until_idle(bus, orch)
    assert "Goodbye" in tts.spoken
    assert stt.followups == 0, "mutual farewell must suppress the followup window"


def test_hallucination_dropped(tmpdir: str) -> None:
    bus, llm, tts, stt, orch = make_orch(FakeBackend(["<reply>no"]), tmpdir, "halluc")
    for artifact in ("[BLANK_AUDIO]", "(clicking)", "♪ music ♪"):
        orch._on_stt_text(artifact)
    assert orch.state.value == SysState.IDLE
    assert len(llm.history_snapshot()) == 1, "hallucinations must not reach the LLM"


def test_stt_timing_payload(tmpdir: str) -> None:
    bus, llm, tts, stt, orch = make_orch(FakeBackend(["<reply>Hi."]), tmpdir, "timing")
    orch._on_stt_text({
        "text": "hi eve",
        "t_speech_start": 10.0,
        "t_last_voice": 11.5,
        "t_commit": 12.1,
        "t_stt_done": 12.4,
    })
    assert orch.cur is not None
    assert orch.cur.wake_ts == 10.0
    assert orch.cur.last_voice_ts == 11.5
    assert orch.cur.commit_ts == 12.1
    assert orch.cur.stt_done_ts == 12.4
    pump_until_idle(bus, orch)


def test_ctx_trim() -> None:
    node = LLMNode(
        Bus(), FakeBackend([]), "sys",
        max_tokens=10, temperature=0.0, top_p=1.0, ctx_limit=200,
    )
    for i in range(6):
        node.history.append({"role": "user", "content": "x" * 400})
        node.history.append({"role": "assistant", "content": "y" * 400})
    node.history.append({"role": "user", "content": "the new question"})
    node._trim_history_for_ctx_locked()
    assert node.history[0]["role"] == "system"
    assert node.history[-1]["content"] == "the new question"
    assert len(node.history) == 2, "oversized pairs must be dropped to fit ctx"


def test_metrics_rotation(tmpdir: str) -> None:
    path = Path(tmpdir) / "metrics_rot.csv"
    path.write_text("old,stale,header\n1,2,3\n")
    MetricsLog(path)
    assert (Path(tmpdir) / "metrics_rot.csv.old").exists()
    assert "stale" not in path.read_text().splitlines()[0]


def test_mlx_stop_marker_holdback() -> None:
    def run(deltas):
        pending, parts, stopped = "", [], False
        for t in deltas:
            emit, pending, stop = _scan_stream_text(pending, t)
            if emit:
                parts.append(emit)
            if stop:
                stopped = True
                break
        if not stopped and pending:
            parts.append(pending)
        return "".join(parts), stopped

    assert run(["Hello", " <end", "_of_turn", ">"]) == ("Hello ", True)
    assert run(["Hi.<end_of_turn>junk"]) == ("Hi.", True)
    assert run(["a < b", " and c"]) == ("a < b and c", False)
    assert run(["tail <e"]) == ("tail <e", False)
    assert run(["done<eo", "s>"]) == ("done", True)


def test_clean_for_tts() -> None:
    assert clean_for_tts("<reply>Hello there.") == "Hello there."
    assert clean_for_tts("<reply>Hello there.</reply>") == "Hello there."
    assert clean_for_tts("<ignore>") == ""
    assert clean_for_tts("< Reply > Hi.") == "Hi."


def main() -> int:
    with TemporaryDirectory() as tmpdir:
        test_ignore_turn(tmpdir)
        test_reply_tag_split_across_deltas(tmpdir)
        test_spaced_gate_tag(tmpdir)
        test_gate_fallback_when_tag_missing(tmpdir)
        test_backend_error_speaks_and_rolls_back(tmpdir)
        test_empty_reply_discarded(tmpdir)
        test_farewell_suppresses_followup(tmpdir)
        test_hallucination_dropped(tmpdir)
        test_stt_timing_payload(tmpdir)
        test_metrics_rotation(tmpdir)
    test_ctx_trim()
    test_mlx_stop_marker_holdback()
    test_clean_for_tts()
    print("Smoke OK: 13 scenarios passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
