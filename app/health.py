"""NodeHealth heartbeats + the liveness cache.

Vocabulary is the Kubernetes split: *liveness* = heartbeats arriving
on ``/sys/node_health`` (and, for subprocess nodes, the process being
alive); *readiness* = whatever the app's capability layer derives
from the cached details. The cache is the HostMonitor idea — one
subscriber holds the latest health per node so surfaces and the
supervisor ask a dict instead of each holding subscriptions.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any

SYS_NODE_HEALTH = "/sys/node_health"


@dataclasses.dataclass
class NodeHealth:
    node: str
    state: str = ""
    details: dict[str, Any] = dataclasses.field(default_factory=dict)
    ts: float = 0.0
    topic: str = SYS_NODE_HEALTH


class HealthCache:
    """Latest NodeHealth per node + age accounting."""

    def __init__(self, bus: Any) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, NodeHealth] = {}
        bus.subscribe(SYS_NODE_HEALTH, self._on_health)

    def _on_health(self, msg: Any) -> None:
        node = getattr(msg, "node", "")
        if not node:
            return
        with self._lock:
            self._latest[node] = msg

    def latest(self, node: str) -> NodeHealth | None:
        with self._lock:
            return self._latest.get(node)

    def age_s(self, node: str) -> float | None:
        """Seconds since the last heartbeat, or None if never seen."""
        h = self.latest(node)
        if h is None or not h.ts:
            return None
        return max(0.0, time.time() - h.ts)

    def snapshot(self) -> dict[str, NodeHealth]:
        with self._lock:
            return dict(self._latest)


__all__ = ["NodeHealth", "HealthCache", "SYS_NODE_HEALTH"]
