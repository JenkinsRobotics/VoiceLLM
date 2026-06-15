"""One log-line shape + the bus mirror.

``ts level [app.node] message`` to stderr; operator-relevant lines
also ride ``/sys/log`` so any surface can render them (the CC01
LogBox pattern, generalized).
"""

from __future__ import annotations

import dataclasses
import sys
import time
from typing import Any

SYS_LOG = "/sys/log"


@dataclasses.dataclass
class LogLine:
    source: str
    level: str = "info"
    line: str = ""
    ts: float = 0.0
    topic: str = SYS_LOG


def log(source: str, line: str, *, level: str = "info",
        bus: Any = None) -> None:
    ts = time.time()
    stamp = time.strftime("%H:%M:%S", time.localtime(ts))
    print(f"{stamp} {level:<5} [{source}] {line}",
          file=sys.stderr, flush=True)
    if bus is not None:
        try:
            bus.publish(LogLine(source=source, level=level, line=line,
                                ts=ts))
        except Exception:  # noqa: BLE001 — logging never raises
            pass


__all__ = ["LogLine", "log", "SYS_LOG"]
