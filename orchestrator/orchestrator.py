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

M3 additions:
  • Self-speech filter — drop incoming ``stt.text`` that's too similar to the
    most recent assistant reply (the mic catching the speaker through the air).
  • Pending-turn queue — when an utterance arrives mid-turn, store it (one
    slot, last-write-wins) and fire it as soon as we go idle, provided it's
    not too stale. M4's barge-in path will replace this with cancellation.
  • LLM-gated speech — every LLM reply must begin with ``<ignore>`` or
    ``<reply>``. The orchestrator buffers the first ``LLM_GATE_BUFFER_CHARS``
    of streaming output to decide. ``<ignore>`` = suppress TTS and skip
    straight back to IDLE. ``<reply>`` = forward the tail to TTS as normal.
    See ``_on_llm_token`` and ``_on_llm_done``.
"""

from __future__ import annotations

import json
import re
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

import config as cfg
from core.bus import Bus
from core.metrics import MetricsLog, TurnMetrics, now
from core.state import State, SysState


_RELATED_STOPWORDS = {
    "about", "after", "again", "also", "could", "does", "doing", "have",
    "heard", "just", "like", "only", "really", "said", "should", "that",
    "their", "there", "these", "thing", "think", "those", "what", "when",
    "where", "which", "with", "would", "your",
}
_GATE_END_TAG = re.compile(r"</\s*(?:reply|ignore)\s*>", re.IGNORECASE)
_MAX_GATE_END_TAG_LEN = len("</ignore>")


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

        # M3: pending-turn slot (single-slot, last-write-wins).
        self._pending_text: str | None = None
        self._pending_ts: float = 0.0

        # Active conversation window: after a real reply, brief follow-ups
        # should be treated as addressed even without a wake word.
        self._active_conversation_deadline: float = 0.0
        self._turn_addressed_hint: bool = False
        self._turn_retrying_ignore: bool = False

        # M3: LLM-gate state, reset at the start of each turn.
        self._gate_buffer: str = ""
        self._gate_decided: bool = False
        self._gate_ignore: bool = False
        self._tts_stream_buffer: str = ""

        # M3: optional eval log for offline review.
        self._eval_log_path: Path | None = (
            Path(cfg.M3_EVAL_LOG) if cfg.M3_EVAL_LOG else None
        )
        if self._eval_log_path is not None:
            self._eval_log_path.parent.mkdir(parents=True, exist_ok=True)

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
            if bool(payload) and self.cur and not self.cur.tts_start_ts:
                self.cur.tts_start_ts = now()
            # TTS asks the mic to mute itself while speaking.
            self.stt.set_paused(bool(payload))
        # tts.audio_chunk is published for AEC/similarity-filter subscribers
        # later; we don't need to act on it here.

    # ── Handlers ───────────────────────────────────────────────────────

    def _on_stt_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        is_echo, ratio = self._sounds_like_self(text)
        if is_echo:
            print(f"[self-echo dropped sim={ratio:.2f}] {text!r}", flush=True)
            self._log_eval(text, "dropped_self_echo", similarity=ratio)
            return

        # Busy: queue this utterance; fire it when the current turn ends.
        # Last-write-wins so a quick re-ask supersedes an older pending one.
        if self.state.value != SysState.IDLE:
            self._pending_text = text
            self._pending_ts = time.time()
            print(f"[queued] {text!r}", flush=True)
            self._log_eval(text, "queued_pending")
            return

        addressed_hint = self._is_active_conversation() or self._looks_related(text)
        self._log_eval(text, "accepted", addressed_hint=addressed_hint)
        self._start_turn(text, addressed_hint=addressed_hint)

    def _start_turn(
        self,
        text: str,
        *,
        addressed_hint: bool = False,
        retrying_ignore: bool = False,
    ) -> None:
        self.cur = TurnMetrics()
        self.cur.wake_ts = now()
        self.cur.listen_start_ts = self.cur.wake_ts
        self.cur.listen_end_ts = now()
        self.cur.stt_text = text
        self._turn_addressed_hint = addressed_hint
        self._turn_retrying_ignore = retrying_ignore
        self._gate_buffer = ""
        self._gate_decided = False
        self._gate_ignore = False
        self._tts_stream_buffer = ""
        self.state.set(SysState.THINKING)

        print(f"[think]  {text!r}", flush=True)
        self._llm_thread = threading.Thread(
            target=self.llm.ask_stream,
            args=(text,),
            kwargs={"addressed_hint": addressed_hint},
            daemon=True,
        )
        self._llm_thread.start()

    def _is_active_conversation(self) -> bool:
        return time.time() <= self._active_conversation_deadline

    def _looks_related(self, text: str) -> bool:
        text_terms = self._topic_terms(text)
        if not text_terms:
            return False

        context_terms: set[str] = set()
        for msg in self.llm.history_snapshot()[-6:]:
            if msg.get("role") == "system":
                continue
            context_terms.update(self._topic_terms(msg.get("content", "")))

        overlap = text_terms & context_terms
        is_questionish = "?" in text or any(
            text.lower().strip().startswith(prefix)
            for prefix in ("what ", "why ", "how ", "do ", "does ", "did ", "is ", "are ")
        )
        return len(overlap) >= 2 or (is_questionish and len(overlap) >= 1)

    def _topic_terms(self, text: str) -> set[str]:
        words = re.findall(r"[a-z0-9']+", text.lower())
        return {
            w
            for w in words
            if len(w) > 3 and w not in _RELATED_STOPWORDS
        }

    def _sounds_like_self(self, text: str) -> tuple[bool, float]:
        """Compare text to the most recent assistant reply in LLM history."""
        snap = self.llm.history_snapshot()
        for msg in reversed(snap):
            if msg.get("role") == "assistant" and msg.get("content"):
                ratio = SequenceMatcher(
                    None, text.lower(), msg["content"].lower()
                ).ratio()
                return ratio >= cfg.SELF_SPEECH_SIMILARITY_THRESHOLD, ratio
        return False, 0.0

    def _log_eval(self, text: str, decision: str, **extra) -> None:
        if self._eval_log_path is None:
            return
        record = {"t": time.time(), "decision": decision, "text": text, **extra}
        try:
            with self._eval_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
        except OSError:
            # Best-effort logging — never let it break the conversation loop.
            pass

    def _on_llm_token(self, delta: str) -> None:
        if self.cur and not self.cur.llm_first_token_ts:
            self.cur.llm_first_token_ts = now()
        if self.cur:
            self.cur.tokens += 1

        # Gate phase: buffer until we see <ignore> or <reply>, or hit fallback.
        if not self._gate_decided:
            self._gate_buffer += delta
            tail = self._gate_check()
            if not self._gate_decided:
                return
            if self._gate_ignore:
                return
            # Fall through to forward the post-tag tail (if any) to TTS.
            delta = tail
            if not delta:
                return

        if self._gate_ignore:
            return  # Ignored turn — discard remaining tokens.

        if self.state.value != SysState.RESPONDING:
            self.state.set(SysState.RESPONDING)
        self._feed_tts_delta(delta)

    def _feed_tts_delta(self, delta: str) -> None:
        """Forward streamed text to TTS while stripping late gate end-tags."""
        self._tts_stream_buffer += delta
        self._tts_stream_buffer = _GATE_END_TAG.sub("", self._tts_stream_buffer)

        if len(self._tts_stream_buffer) <= _MAX_GATE_END_TAG_LEN:
            return

        emit = self._tts_stream_buffer[:-_MAX_GATE_END_TAG_LEN]
        self._tts_stream_buffer = self._tts_stream_buffer[-_MAX_GATE_END_TAG_LEN:]
        if emit:
            self.tts.feed_text(emit)

    def _flush_tts_stream_buffer(self) -> None:
        tail = _GATE_END_TAG.sub("", self._tts_stream_buffer)
        self._tts_stream_buffer = ""
        if tail:
            self.tts.feed_text(tail)

    def _gate_check(self) -> str:
        """Inspect ``self._gate_buffer`` and decide the gate. Returns the
        post-tag tail to forward to TTS when ``<reply>`` is found, or "" otherwise."""
        lowered = self._gate_buffer.lower()
        ignore_idx = lowered.find("<ignore>")
        reply_idx = lowered.find("<reply>")

        # Prefer whichever tag appears first.
        if ignore_idx != -1 and (reply_idx == -1 or ignore_idx < reply_idx):
            self._gate_decided = True
            self._gate_ignore = True
            stt_text = self.cur.stt_text if self.cur else ""
            if self._turn_addressed_hint and not self._turn_retrying_ignore:
                print(f"[LLM tried ignore on active follow-up] {stt_text!r}", flush=True)
                self._log_eval(stt_text, "llm_tried_ignore_active_followup")
            else:
                print(f"[ignored by LLM] {stt_text!r}", flush=True)
                self._log_eval(stt_text, "llm_ignored")
            return ""

        if reply_idx != -1:
            self._gate_decided = True
            self._gate_ignore = False
            return self._gate_buffer[reply_idx + len("<reply>"):]

        if len(self._gate_buffer) >= cfg.LLM_GATE_BUFFER_CHARS:
            # LLM forgot the tag protocol — default to reply.
            self._gate_decided = True
            self._gate_ignore = False
            print(f"[gate fallback → reply] {self._gate_buffer!r}", flush=True)
            return self._gate_buffer

        return ""

    def _on_llm_done(self, reply: str) -> None:
        if self.cur:
            self.cur.llm_done_ts = now()

        # Edge case: gate never decided because the LLM produced fewer than
        # LLM_GATE_BUFFER_CHARS total. Treat the buffer as the reply.
        if not self._gate_decided and self._gate_buffer:
            self._gate_decided = True
            self._gate_ignore = False
            self._feed_tts_delta(self._gate_buffer)

        if self._gate_ignore:
            if (
                self._turn_addressed_hint
                and not self._turn_retrying_ignore
                and self.cur
                and self.cur.stt_text
            ):
                text = self.cur.stt_text
                print(f"[active follow-up retry] {text!r}", flush=True)
                self._log_eval(text, "active_followup_retry")
                self.llm.discard_last_turn()
                self._start_turn(
                    text,
                    addressed_hint=True,
                    retrying_ignore=True,
                )
                return
            # TTS never started; do the IDLE handoff ourselves so the next
            # turn (or pending-turn slot) can fire.
            self.llm.discard_last_turn()
            self._on_tts_done()
            return

        # Synthesize whatever tail is still buffered in TTS.
        self._flush_tts_stream_buffer()
        self.tts.flush()
        if reply:
            print(f"[reply]  {reply!r}", flush=True)

    def _on_tts_done(self) -> None:
        was_ignored = self._gate_ignore
        if self.cur:
            self.cur.tts_end_ts = now()
            self.metrics.write(self.cur)
            self.cur = None
        self.state.set(SysState.IDLE)
        if not was_ignored:
            self._active_conversation_deadline = (
                time.time() + cfg.ACTIVE_CONVERSATION_TIMEOUT_S
            )
        # Open a wake-word-free follow-up window for the next utterance.
        # (No-op when REQUIRE_WAKE_WORD = False — STT is already always-on.)
        self.stt.open_followup()

        # M3: drain the pending-turn slot if something arrived mid-turn.
        if self._pending_text is not None:
            text = self._pending_text
            age = time.time() - self._pending_ts
            self._pending_text = None
            if age <= cfg.PENDING_TURN_MAX_AGE_S:
                print(f"[pending → fire age={age:.2f}s]", flush=True)
                self._log_eval(text, "pending_fired", age=age)
                self._on_stt_text(text)
            else:
                print(f"[pending dropped age={age:.2f}s] {text!r}", flush=True)
                self._log_eval(text, "pending_stale", age=age)
