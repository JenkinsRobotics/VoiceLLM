"""Subprocess test node — used by the conformance tests to exercise
the SubprocessHandle + child_main path for real (spawn, heartbeat,
command roundtrip, hard crash)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.bus.api import MessageRegistry
from app.health import NodeHealth
from app.logging import LogLine
from app.node import Node


@dataclass
class TestCmd:
    cmd: str = ""
    topic: str = "/test/cmd"


@dataclass
class TestEcho:
    cmd: str = ""
    pid: int = 0
    topic: str = "/test/echo"


MESSAGES = MessageRegistry()
MESSAGES.register_all([TestCmd, TestEcho, NodeHealth, LogLine])


class WorkerNode(Node):
    def __init__(self, *, bus: Any, **_: Any) -> None:
        super().__init__(bus=bus, name="worker", tick_interval_s=0.05)

    def setup(self) -> None:
        self.bus.subscribe("/test/cmd", self._on_cmd)

    def _on_cmd(self, msg: Any) -> None:
        if msg.cmd == "die":
            os._exit(3)          # hard crash — the supervisor's problem
        if msg.cmd == "ping":
            self.bus.publish(TestEcho(cmd="pong", pid=os.getpid()))


if __name__ == "__main__":
    from app.child import child_main
    child_main(lambda bus, config: WorkerNode(bus=bus, **config),
               registry=MESSAGES)
