"""ZMQ bus backend — XPUB/XSUB broker topology.

The canonical multiprocess pub/sub shape (Mochi proved it; JROS
ported it): one broker proxies between two well-known endpoints,
every participant — core and subprocess nodes alike — just CONNECTS:

    publishers → CONNECT to XSUB ─→ broker ─→ XPUB ← CONNECT ← subscribers

The chassis runs the broker as a thread inside the core process and
hands the endpoints to subprocess nodes via env vars
(``JAEGER_BUS_XSUB`` / ``JAEGER_BUS_XPUB``). pyzmq is imported
lazily so the chassis works without it when ``[bus] backend="inproc"``.

Wire format: frame 0 = topic bytes (ZMQ prefix-filters on it),
frame 1 = JSON payload decoded through the app's MessageRegistry.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

from .api import Bus, MessageRegistry, SubscriberFn

DEFAULT_XSUB = "tcp://127.0.0.1:7781"
DEFAULT_XPUB = "tcp://127.0.0.1:7782"
ENV_XSUB = "JAEGER_BUS_XSUB"
ENV_XPUB = "JAEGER_BUS_XPUB"


class Broker:
    """XPUB↔XSUB proxy thread. start()/stop(), both idempotent."""

    def __init__(self, *, xsub: str = DEFAULT_XSUB,
                 xpub: str = DEFAULT_XPUB) -> None:
        self.xsub_endpoint = xsub
        self.xpub_endpoint = xpub
        self._sockets: list[Any] = []
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        import zmq
        ctx = zmq.Context.instance()
        xsub = ctx.socket(zmq.XSUB)
        xsub.bind(self.xsub_endpoint)
        xpub = ctx.socket(zmq.XPUB)
        xpub.bind(self.xpub_endpoint)
        self._sockets = [xsub, xpub]

        def proxy() -> None:
            try:
                zmq.proxy(xsub, xpub)
            except zmq.ContextTerminated:
                return
            except zmq.ZMQError:
                if not self._stopped.is_set():
                    print("[broker] proxy exited unexpectedly",
                          file=sys.stderr, flush=True)

        self._thread = threading.Thread(target=proxy, name="bus-broker",
                                        daemon=True)
        self._thread.start()
        time.sleep(0.05)   # let binds settle before publishers connect
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._stopped.set()
        for sock in self._sockets:
            try:
                sock.close(linger=0)   # makes zmq.proxy return
            except Exception:  # noqa: BLE001
                pass
        self._sockets = []
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def env(self) -> dict[str, str]:
        """The env vars a subprocess node needs to find this broker."""
        return {ENV_XSUB: self.xsub_endpoint, ENV_XPUB: self.xpub_endpoint}


class ZmqBus(Bus):
    """Broker-connecting bus: PUB→XSUB, SUB→XPUB, one delivery
    thread fanning out to Python subscribers (same model as
    InProcBus — same caveat: don't block the delivery thread)."""

    def __init__(
        self,
        registry: MessageRegistry,
        *,
        xsub: str | None = None,
        xpub: str | None = None,
        recv_timeout_ms: int = 200,
    ) -> None:
        import zmq
        self._registry = registry
        self._xsub = xsub or os.environ.get(ENV_XSUB, DEFAULT_XSUB)
        self._xpub = xpub or os.environ.get(ENV_XPUB, DEFAULT_XPUB)
        ctx = zmq.Context.instance()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 1000)
        self._pub.connect(self._xsub)
        # ZMQ sockets are NOT thread-safe; nodes + tools publish from
        # many threads. Without this lock, concurrent send_multipart
        # calls interleave frame pairs (topic from one message, payload
        # from another) — found live by the jros-demo smoke run.
        self._pub_lock = threading.Lock()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.RCVHWM, 1000)
        self._sub.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self._sub.connect(self._xpub)
        self._subs_lock = threading.Lock()
        self._subscribers: dict[str, list[SubscriberFn]] = {}
        self._closed = False
        time.sleep(0.05)   # ZMQ late-joiner settle
        self._delivery_thread = threading.Thread(
            target=self._delivery_loop, name="zmq-bus-delivery",
            daemon=True,
        )
        self._delivery_thread.start()

    def publish(self, msg: Any) -> None:
        if self._closed:
            return
        with self._pub_lock:
            self._pub.send_multipart([
                msg.topic.encode("utf-8"), self._registry.encode(msg),
            ])

    def subscribe(self, topic: str, callback: SubscriberFn) -> None:
        import zmq
        with self._subs_lock:
            existing = self._subscribers.setdefault(topic, [])
            if not existing:
                self._sub.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
            existing.append(callback)

    def unsubscribe(self, topic: str, callback: SubscriberFn) -> None:
        import zmq
        with self._subs_lock:
            subs = self._subscribers.get(topic)
            if not subs:
                return
            try:
                subs.remove(callback)
            except ValueError:
                return
            if not subs:
                try:
                    self._sub.setsockopt(
                        zmq.UNSUBSCRIBE, topic.encode("utf-8"))
                except Exception:  # noqa: BLE001
                    pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._delivery_thread.join(timeout=2.0)
        for sock in (self._pub, self._sub):
            try:
                sock.close(linger=0)
            except Exception:  # noqa: BLE001
                pass

    def _delivery_loop(self) -> None:
        import zmq
        while not self._closed:
            try:
                frames = self._sub.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._closed:
                    return
                continue
            if len(frames) < 2:
                continue
            topic = frames[0].decode("utf-8", errors="replace")
            try:
                msg = self._registry.decode(topic, frames[1])
            except Exception as exc:  # noqa: BLE001
                print(f"[zmq-bus] decode error on {topic!r}: {exc}",
                      file=sys.stderr, flush=True)
                continue
            with self._subs_lock:
                snapshot = list(self._subscribers.get(topic, ()))
            for cb in snapshot:
                try:
                    cb(msg)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[zmq-bus] subscriber exception on {topic!r}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr, flush=True,
                    )


__all__ = ["Broker", "ZmqBus", "DEFAULT_XSUB", "DEFAULT_XPUB",
           "ENV_XSUB", "ENV_XPUB"]
