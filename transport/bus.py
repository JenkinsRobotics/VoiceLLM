"""In-process message bus.

One process, many threads. ``publish()`` fans a ``Message`` out to every
subscription whose topic filter matches; each subscription is its own
``queue.Queue`` that the subscriber polls with ``get(timeout)``. The
orchestrator owns the default catch-all subscription via ``Bus.get()``.

INVARIANT: high-rate audio (mic frames) must never be published here.
Raw audio stays on dedicated queues (``MicStream.q``, ``phrase_q``,
``audio_q``); the bus carries control + text + per-sentence TTS reference
audio only. ``publish()`` never blocks — a full subscriber queue drops its
oldest message (with a warning) instead of stalling the publisher.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    topic: str
    payload: Any


class Bus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: list[tuple[set[str] | None, queue.Queue]] = []
        # Default catch-all subscription — preserves the original
        # single-consumer API used by the orchestrator.
        self.q = self.subscribe()

    def subscribe(
        self,
        topics: tuple[str, ...] | list[str] | None = None,
        maxsize: int = 2048,
    ) -> queue.Queue:
        """Register a subscriber. ``topics=None`` receives everything;
        otherwise only the listed topics. Returns the subscriber's queue."""
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.append((set(topics) if topics else None, q))
        return q

    def publish(self, topic: str, payload: Any) -> None:
        msg = Message(topic, payload)
        with self._lock:
            subs = list(self._subs)
        for topic_filter, q in subs:
            if topic_filter is not None and topic not in topic_filter:
                continue
            try:
                q.put_nowait(msg)
            except queue.Full:
                # Never block a publisher: shed the oldest message instead.
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass
                print(f"[bus] subscriber queue full — dropped oldest ({topic})",
                      flush=True)

    def get(self, timeout: float = 0.1) -> Message | None:
        """Poll the default subscription (orchestrator's consumer loop)."""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None
