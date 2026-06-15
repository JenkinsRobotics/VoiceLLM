"""Node — the universal lifecycle contract.

Four phases: ``setup`` once; ``tick`` repeatedly (default sleeps);
``teardown`` always runs, even after failure; ``health`` queryable any
time. Tick errors are transient (logged, node keeps running); setup
errors are fatal. ``stop()`` is graceful and idempotent.

The state machine maps 1:1 onto the ROS 2 managed-node lifecycle
(spec §node-contract), which is what keeps a future ros2 bridge
mechanical:

    INIT → SETTING_UP → RUNNING → STOPPING → STOPPED
                      ↘ FAILED (setup raised; teardown still ran)
"""

from __future__ import annotations

import abc
import enum
import signal
import sys
import threading
import time
from typing import Any


class NodeState(str, enum.Enum):
    INIT = "init"
    SETTING_UP = "setting_up"
    RUNNING = "running"
    STOPPING = "stopping"
    RESTARTING = "restarting"
    STOPPED = "stopped"
    FAILED = "failed"


class Node(abc.ABC):
    def __init__(
        self,
        *,
        bus: Any,
        name: str | None = None,
        tick_interval_s: float = 0.1,
        install_signal_handlers: bool = False,
    ) -> None:
        self.bus = bus
        self._name = name or self.__class__.__name__
        self._tick_interval_s = tick_interval_s
        self._state = NodeState.INIT
        self._stop_event = threading.Event()
        self._error: BaseException | None = None
        self._t_started_ns: int | None = None
        if install_signal_handlers:
            try:
                signal.signal(signal.SIGTERM, self._on_signal)
                signal.signal(signal.SIGINT, self._on_signal)
            except (ValueError, OSError):
                pass  # only installable on the main thread

    # ── identity / state ─────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> NodeState:
        return self._state

    # ── lifecycle hooks (override) ───────────────────────────────

    def setup(self) -> None:
        """Open resources, subscribe to topics. Raise to abort."""

    def tick(self) -> None:
        """Periodic work. Default sleeps so listen-only nodes don't
        busy-loop."""
        time.sleep(self._tick_interval_s)

    def teardown(self) -> None:
        """Close resources. Runs even when setup/tick failed."""

    def health(self) -> dict[str, Any]:
        uptime_s = 0.0
        if self._t_started_ns is not None:
            uptime_s = (time.time_ns() - self._t_started_ns) / 1e9
        return {
            "name": self.name,
            "state": self._state.value,
            "uptime_s": round(uptime_s, 1),
            "error": (
                None if self._error is None
                else f"{type(self._error).__name__}: {self._error}"
            ),
        }

    # ── run / stop ───────────────────────────────────────────────

    def run(self) -> None:
        """Run the lifecycle to completion. Blocks until stop()."""
        self._t_started_ns = time.time_ns()
        try:
            self._state = NodeState.SETTING_UP
            self._log("setup")
            self.setup()
            self._state = NodeState.RUNNING
            self._log("running")
            while not self._stop_event.is_set():
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001 — tick errors are transient
                    self._log(f"tick error: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 — setup errors are fatal
            self._error = exc
            self._state = NodeState.FAILED
            self._log(f"setup failed: {type(exc).__name__}: {exc}")
        finally:
            try:
                self.teardown()
            except Exception as exc:  # noqa: BLE001
                self._log(f"teardown error: {type(exc).__name__}: {exc}")
                if self._error is None:
                    self._error = exc
            if self._state != NodeState.FAILED:
                self._state = NodeState.STOPPED
                self._log("stopped")

    def stop(self) -> None:
        """Request graceful shutdown. Idempotent."""
        if self._state in (NodeState.STOPPING, NodeState.STOPPED,
                           NodeState.FAILED):
            return
        self._state = NodeState.STOPPING
        self._stop_event.set()

    # ── internals ────────────────────────────────────────────────

    def _on_signal(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        self._log(f"signal {signum}; stopping")
        self.stop()

    def _log(self, msg: str) -> None:
        print(f"[node:{self.name}] {msg}", file=sys.stderr, flush=True)


class FrameNode(Node):
    """Render-loop specialization — the MochiNodeBase shape, formatted.

    Fixed-rate scheduling (deadline-based, so render cost doesn't
    drift the frame rate) with the update/render split Mochi's nodes
    already use:

        update_tick(ts)   advance state — scripts, state machines
        render_tick(ts)   produce + publish one frame

    A node that falls behind resyncs to "now" rather than spiraling.
    """

    def __init__(self, *, bus: Any, fps: float = 30.0,
                 **kwargs: Any) -> None:
        super().__init__(bus=bus,
                         tick_interval_s=1.0 / max(float(fps), 0.5),
                         **kwargs)
        self.fps = float(fps)
        self.frames_rendered = 0
        self._next_deadline: float | None = None

    # override these two, not tick():

    def update_tick(self, ts: float) -> None:
        """Advance state. Default: nothing."""

    def render_tick(self, ts: float) -> None:
        """Produce + publish one frame. Default: nothing."""

    def tick(self) -> None:
        now = time.monotonic()
        if self._next_deadline is None:
            self._next_deadline = now
        self.update_tick(now)
        self.render_tick(now)
        self.frames_rendered += 1
        self._next_deadline += self._tick_interval_s
        sleep_s = self._next_deadline - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            self._next_deadline = time.monotonic()   # fell behind; resync

    def health(self) -> dict[str, Any]:
        base = super().health()
        base["fps_target"] = self.fps
        base["frames_rendered"] = self.frames_rendered
        return base


__all__ = ["Node", "FrameNode", "NodeState"]
