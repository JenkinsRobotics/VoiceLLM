"""In-process bus — one queue, one delivery thread.

``publish()`` is thread-safe and non-blocking; a full queue raises
:class:`BusOverflowError` instead of hanging the publisher. One bad
subscriber never wedges the bus.
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Any

from .api import Bus, SubscriberFn


class BusOverflowError(RuntimeError):
    """The delivery queue is full — surfaced synchronously so the
    publisher never blocks."""


class InProcBus(Bus):
    def __init__(self, maxsize: int = 2048) -> None:
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)
        self._subs_lock = threading.Lock()
        self._subscribers: dict[str, list[SubscriberFn]] = {}
        self._closed = False
        self._delivery_thread = threading.Thread(
            target=self._delivery_loop, name="bus-delivery", daemon=True,
        )
        self._delivery_thread.start()

    def publish(self, msg: Any) -> None:
        if self._closed:
            return
        try:
            self._q.put_nowait(msg)
        except queue.Full as exc:
            raise BusOverflowError(
                f"bus queue full while publishing {msg.topic}"
            ) from exc

    def subscribe(self, topic: str, callback: SubscriberFn) -> None:
        with self._subs_lock:
            self._subscribers.setdefault(topic, []).append(callback)

    def unsubscribe(self, topic: str, callback: SubscriberFn) -> None:
        with self._subs_lock:
            subs = self._subscribers.get(topic)
            if not subs:
                return
            try:
                subs.remove(callback)
            except ValueError:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._q.put_nowait(None)   # sentinel wakes the thread
        except queue.Full:
            pass
        self._delivery_thread.join(timeout=2.0)
        with self._subs_lock:
            self._subscribers.clear()

    def _delivery_loop(self) -> None:
        while True:
            try:
                msg = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._closed:
                    return
                continue
            if msg is None:
                return
            with self._subs_lock:
                snapshot = list(self._subscribers.get(msg.topic, ()))
            for cb in snapshot:
                try:
                    cb(msg)
                except Exception as exc:  # noqa: BLE001 — never wedge the bus
                    print(
                        f"[bus] subscriber exception on {msg.topic}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr, flush=True,
                    )


__all__ = ["InProcBus", "BusOverflowError"]
