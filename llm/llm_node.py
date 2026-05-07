"""LLM bus adapter.

Owns a BackendBase instance and the conversation history. Streams deltas
to the bus as ``llm.token`` and finishes with ``llm.done``. Cancellable
via ``cancel()`` for barge-in.
"""

from __future__ import annotations

import re
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
    return re.sub(r"\s+", " ", text).strip()


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
    ) -> None:
        self.bus = bus
        self.backend = backend
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.history: list[dict] = [{"role": "system", "content": system}]
        self._lock = threading.Lock()

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

    def ask_stream(self, user_text: str) -> None:
        """Append user_text, stream deltas to the bus, append final reply."""
        with self._lock:
            self.history.append({"role": "user", "content": user_text})
            messages = list(self.history)

        parts: list[str] = []
        try:
            for delta in self.backend.stream_chat(
                messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            ):
                parts.append(delta)
                self.bus.publish("llm.token", delta)
        finally:
            reply_raw = "".join(parts).strip()
            reply_clean = clean_for_tts(reply_raw)
            with self._lock:
                # Store the cleaned reply so future turns aren't poisoned by
                # markdown the model might have emitted.
                self.history.append({"role": "assistant", "content": reply_clean})
            self.bus.publish("llm.done", reply_clean)
