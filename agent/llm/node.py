"""LLM bus adapter.

Owns a BackendBase instance and the conversation history. Streams deltas
to the bus as ``llm.token`` and finishes with ``llm.done``. Cancellable
via ``cancel()`` for barge-in.
"""

from __future__ import annotations

import re
import sys
import threading

from .backend_base import BackendBase


def clean_for_tts(text: str) -> str:
    """Strip markdown / code fences / list bullets so Kokoro doesn't speak them.

    Ported from MockingAgent/voice_assistant.py:265-271.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^[\-\*\d\.\)]+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"</?\s*(?:reply|ignore)\s*>\s*", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _estimate_prompt_tokens(messages: list[dict]) -> int:
    """Rough upper-ish bound: ~4 chars/token of English plus a per-message
    allowance for chat-template wrapping. Only used to keep the prompt
    safely under the backend's context window — precision doesn't matter,
    not blowing up llama.cpp mid-session does."""
    return sum(len(m.get("content", "")) // 4 + 8 for m in messages)


class LLMNode:
    def __init__(
        self,
        bus,
        backend: BackendBase,
        system: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        max_history_turns: int = 8,
        ctx_limit: int = 0,
    ) -> None:
        self.bus = bus
        self.backend = backend
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_history_turns = max_history_turns
        self.ctx_limit = ctx_limit  # backend context window; 0 = no guard
        self.history: list[dict] = [{"role": "system", "content": system}]
        self._lock = threading.Lock()

    def _trim_history_locked(self) -> None:
        """Cap rolling history to max_history_turns user/assistant pairs.
        Caller must hold self._lock. Ported from voice_assistant.py:258-263.
        """
        if self.max_history_turns <= 0:
            return
        keep_msgs = 1 + self.max_history_turns * 2  # system + N pairs
        if len(self.history) > keep_msgs:
            self.history = self.history[:1] + self.history[-self.max_history_turns * 2:]

    def _trim_history_for_ctx_locked(self) -> None:
        """Drop oldest pairs until the estimated prompt fits the context
        window with room for max_tokens of reply. Without this, a long
        chatty session eventually exceeds n_ctx, the backend raises on
        every subsequent turn, and the assistant goes permanently mute.
        Caller must hold self._lock; call after appending the new user msg.
        """
        if self.ctx_limit <= 0:
            return
        budget = self.ctx_limit - self.max_tokens - 128
        while (
            len(self.history) > 2
            and _estimate_prompt_tokens(self.history) > budget
        ):
            dropped = self.history[1:3]
            del self.history[1:3]
            print(
                f"[llm] ctx guard: dropped {len(dropped)} oldest message(s) "
                f"to fit {self.ctx_limit}-token window",
                flush=True,
            )

    def load_and_warm(self) -> None:
        self.backend.load()
        self.backend.warm()

    def cancel(self) -> None:
        self.backend.cancel()

    def reset_history(self) -> None:
        with self._lock:
            self.history = [{"role": "system", "content": self.system}]

    def history_snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.history)

    def discard_last_turn(self) -> None:
        """Remove the most recent user/assistant pair from rolling history."""
        with self._lock:
            if len(self.history) < 3:
                return
            if (
                self.history[-2].get("role") == "user"
                and self.history[-1].get("role") == "assistant"
            ):
                del self.history[-2:]

    def ask_stream(self, user_text: str, *, addressed_hint: bool = False) -> None:
        """Append user_text, stream deltas to the bus, append final reply.

        On backend failure with no usable output, the user message is
        rolled back (so a poisoned/oversized history can't fail every
        subsequent turn) and ``llm.error`` is published before ``llm.done``
        so the orchestrator can speak a fallback instead of going silent.
        """
        with self._lock:
            self.history.append({"role": "user", "content": user_text})
            self._trim_history_for_ctx_locked()
            messages = list(self.history)
            if addressed_hint:
                messages[-1] = {
                    "role": "user",
                    "content": (
                        "This transcript is inside an active conversation or "
                        "appears related to the current topic. Treat it as "
                        "addressed to you unless it is clearly ambient speech "
                        "or your own echoed voice. Respond to the actual user "
                        f"message:\n{user_text}"
                    ),
                }

        parts: list[str] = []
        error: Exception | None = None
        try:
            for delta in self.backend.stream_chat(
                messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            ):
                parts.append(delta)
                self.bus.publish("llm.token", delta)
        except Exception as exc:
            error = exc
            print(f"[llm] generation failed: {exc}", file=sys.stderr, flush=True)

        reply_raw = "".join(parts).strip()
        reply_clean = clean_for_tts(reply_raw)

        if error is not None and not reply_clean:
            # Nothing usable came back — roll back the user message so the
            # same oversized/poisoned prompt isn't resent forever.
            with self._lock:
                if self.history and self.history[-1].get("role") == "user":
                    self.history.pop()
            self.bus.publish("llm.error", str(error))
            self.bus.publish("llm.done", "")
            return

        with self._lock:
            # Store the cleaned reply so future turns aren't poisoned by
            # markdown the model might have emitted.
            self.history.append({"role": "assistant", "content": reply_clean})
            self._trim_history_locked()
        self.bus.publish("llm.done", reply_clean)
