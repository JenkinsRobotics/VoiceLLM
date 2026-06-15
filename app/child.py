"""Subprocess-node entry helper.

A subprocess node module ends with::

    if __name__ == "__main__":
        from app.child import child_main
        child_main(lambda bus, config: MyNode(bus=bus, **config),
                   registry=MESSAGES)

``child_main`` builds the broker-connected ZMQ bus from the env vars
the supervisor injected, instantiates the node with its config slice
(``JAEGER_NODE_CONFIG``, JSON), wires SIGTERM/SIGINT → graceful
``node.stop()``, runs a 1 Hz health-beat thread, and blocks in
``node.run()`` until stopped or killed.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from typing import Any, Callable

from .bus.api import MessageRegistry
from .bus.zmq import ZmqBus
from .health import NodeHealth
from .node import Node

HEALTH_PERIOD_S = 1.0


def child_main(
    node_factory: Callable[[Any, dict[str, Any]], Node],
    *,
    registry: MessageRegistry,
) -> None:
    node_id = os.environ.get("JAEGER_NODE_ID", "node")
    config = json.loads(os.environ.get("JAEGER_NODE_CONFIG", "{}") or "{}")

    bus = ZmqBus(registry)
    node = node_factory(bus, config)

    def on_signal(signum: int, frame: Any) -> None:  # noqa: ARG001
        node.stop()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    stop_beat = threading.Event()

    def beat() -> None:
        while not stop_beat.wait(HEALTH_PERIOD_S):
            try:
                bus.publish(NodeHealth(
                    node=node_id, state=node.state.value,
                    details=node.health(), ts=time.time(),
                ))
            except Exception:  # noqa: BLE001 — heartbeat never dies
                pass

    threading.Thread(target=beat, name=f"{node_id}-health",
                     daemon=True).start()
    try:
        node.run()
    finally:
        stop_beat.set()
        bus.close()


__all__ = ["child_main", "HEALTH_PERIOD_S"]
