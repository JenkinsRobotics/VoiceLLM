"""Supervisor — ON / OFF / RESTART / DIAGNOSE over every node,
regardless of backend.

One handle contract, two live backends:

  * ``thread``     — a Node instance on a daemon thread (default).
  * ``subprocess`` — ``python -m <module>`` in the app's process
    group, talking over the ZMQ bus; crash isolation per node (the
    humanoid case: a dead limb controller restarts without rebooting
    the robot).
  * ``external``   — declared but not implemented in format 0.1
    (observe-only nodes owned by another host).

Restart policy follows systemd vocabulary per node:
``never | on_failure | always``, exponential backoff between
attempts (1s, 2s, 4s … capped 30s), burst limit (default 5 failures
in 60s → FAILED, stop trying). A watch thread applies policy;
``diagnose()`` returns one screen: state, backend, uptime, restart
history, last error, health age.

Teardown is the no-orphans rule: every child PID is registered on
disk the moment it spawns; ``stop_all()`` walks live handles with
terminate→kill escalation, then :func:`reap_stale` sweeps anything a
previous crashed run left behind (called at next boot, before the
instance slot is taken).
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from .health import HealthCache
from .logging import log
from .manifest import NodeSpec
from .node import Node, NodeState

_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 30.0
_BURST_LIMIT = 5
_BURST_WINDOW_S = 60.0


class NodeHandle:
    """The universal lifecycle surface the supervisor holds."""

    def __init__(self, spec: NodeSpec) -> None:
        self.spec = spec
        self.restarts = 0
        self.failure_times: list[float] = []
        self.last_error: str = ""
        self.started_at: float | None = None
        self.intent_running = False     # operator intent vs observed state

    # backend implements:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def alive(self) -> bool: ...
    def state(self) -> str: ...

    def restart(self) -> None:
        self.stop()
        self.start()
        self.restarts += 1

    def uptime_s(self) -> float:
        if self.started_at is None or not self.alive():
            return 0.0
        return time.monotonic() - self.started_at


class ThreadHandle(NodeHandle):
    """Default backend: Node on a daemon thread. Restart = stop +
    join + a FRESH instance from the factory (never reuse a torn-down
    node object)."""

    def __init__(self, spec: NodeSpec,
                 factory: Callable[[], Node]) -> None:
        super().__init__(spec)
        self._factory = factory
        self._node: Node | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.alive():
            return
        self.intent_running = True
        node = self._factory()
        thread = threading.Thread(target=node.run,
                                  name=f"node-{self.spec.id}", daemon=True)
        thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if node.state in (NodeState.RUNNING, NodeState.FAILED):
                break
            time.sleep(0.01)
        self._node, self._thread = node, thread
        self.started_at = time.monotonic()
        if node.state == NodeState.FAILED:
            self.last_error = str(node.health().get("error") or "setup failed")

    def stop(self) -> None:
        self.intent_running = False
        node, self._node = self._node, None
        thread, self._thread = self._thread, None
        if node is not None:
            node.stop()
        if thread is not None:
            thread.join(timeout=3.0)

    def alive(self) -> bool:
        return (self._node is not None
                and self._node.state == NodeState.RUNNING)

    def state(self) -> str:
        if self._node is None:
            return "off" if not self.intent_running else "starting"
        return self._node.state.value

    def node(self) -> Node | None:
        return self._node


class SubprocessHandle(NodeHandle):
    """Mochi-shaped backend: ``python -m <module>``, same process
    group as the app, env carries broker endpoints + node id +
    config slice. Liveness = process poll; the node's own state rides
    its health heartbeats."""

    def __init__(self, spec: NodeSpec, *, env_extra: dict[str, str],
                 cwd: pathlib.Path, registry_file: pathlib.Path) -> None:
        super().__init__(spec)
        self._env_extra = env_extra
        self._cwd = cwd
        self._registry_file = registry_file
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        if self.alive():
            return
        self.intent_running = True
        env = os.environ.copy()
        env.update(self._env_extra)
        env["JAEGER_NODE_ID"] = self.spec.id
        env["PYTHONPATH"] = (
            f"{self._cwd}{os.pathsep}{env.get('PYTHONPATH', '')}"
        )
        self._proc = subprocess.Popen(
            [sys.executable, "-m", self.spec.module],
            cwd=str(self._cwd), env=env,
        )
        self.started_at = time.monotonic()
        _registry_add(self._registry_file, self._proc.pid, self.spec.id)

    def stop(self) -> None:
        self.intent_running = False
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        _registry_remove(self._registry_file, proc.pid)

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def state(self) -> str:
        if self._proc is None:
            return "off" if not self.intent_running else "starting"
        rc = self._proc.poll()
        return "running" if rc is None else f"exited({rc})"


class Supervisor:
    """Holds every handle; applies restart policy from a watch
    thread; answers ls/status/diagnose."""

    def __init__(self, *, health: HealthCache | None = None,
                 bus: Any = None) -> None:
        self._handles: dict[str, NodeHandle] = {}
        self._lock = threading.Lock()
        self._health = health
        self._bus = bus
        self._stop = threading.Event()
        self._watch_thread: threading.Thread | None = None

    # ── registration / boot ──────────────────────────────────────

    def add(self, handle: NodeHandle) -> None:
        with self._lock:
            self._handles[handle.spec.id] = handle

    def start_all(self) -> None:
        """Start enabled nodes in tier order, then begin watching."""
        for handle in sorted(self._all(), key=lambda h: h.spec.tier):
            if handle.spec.enabled:
                handle.start()
                log("supervisor", f"started node {handle.spec.id} "
                    f"({handle.spec.backend})", bus=self._bus)
        self._stop.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_loop, name="supervisor-watch", daemon=True,
        )
        self._watch_thread.start()

    def stop_all(self) -> None:
        """Reverse tier order; idempotent; never raises."""
        self._stop.set()
        if self._watch_thread is not None:
            self._watch_thread.join(timeout=2.0)
            self._watch_thread = None
        for handle in sorted(self._all(), key=lambda h: -h.spec.tier):
            try:
                handle.stop()
            except Exception as exc:  # noqa: BLE001
                log("supervisor",
                    f"stop {handle.spec.id} error: {exc}", level="warn")

    # ── operator verbs ───────────────────────────────────────────

    def start(self, node_id: str) -> None:
        self._get(node_id).start()
        log("supervisor", f"node {node_id} started", bus=self._bus)

    def stop(self, node_id: str) -> None:
        self._get(node_id).stop()
        log("supervisor", f"node {node_id} stopped", bus=self._bus)

    def restart(self, node_id: str) -> None:
        self._get(node_id).restart()
        log("supervisor", f"node {node_id} restarted", bus=self._bus)

    def ls(self) -> list[dict[str, Any]]:
        rows = []
        for h in sorted(self._all(), key=lambda h: (h.spec.tier, h.spec.id)):
            rows.append({
                "id": h.spec.id,
                "tier": h.spec.tier,
                "backend": h.spec.backend,
                "state": h.state(),
                "uptime_s": round(h.uptime_s(), 1),
                "restarts": h.restarts,
                "restart_policy": h.spec.restart,
                "health_age_s": (
                    self._health.age_s(h.spec.id) if self._health else None
                ),
            })
        return rows

    def diagnose(self, node_id: str) -> dict[str, Any]:
        h = self._get(node_id)
        latest = self._health.latest(node_id) if self._health else None
        recent = [t for t in h.failure_times
                  if time.monotonic() - t < _BURST_WINDOW_S]
        return {
            "id": h.spec.id,
            "state": h.state(),
            "backend": h.spec.backend,
            "intent_running": h.intent_running,
            "uptime_s": round(h.uptime_s(), 1),
            "restarts": h.restarts,
            "restart_policy": h.spec.restart,
            "failures_in_window": len(recent),
            "crash_loop": len(recent) >= _BURST_LIMIT,
            "last_error": h.last_error,
            "health": (latest.details if latest else {}),
            "health_age_s": (
                self._health.age_s(node_id) if self._health else None
            ),
        }

    # ── watch / restart policy ───────────────────────────────────

    def _watch_loop(self) -> None:
        pending: dict[str, float] = {}   # node id → not-before time
        while not self._stop.wait(0.25):
            for h in self._all():
                if not h.intent_running or h.alive():
                    pending.pop(h.spec.id, None) if h.alive() else None
                    continue
                if h.spec.restart == "never":
                    continue
                now = time.monotonic()
                if h.spec.id not in pending:
                    # Newly observed failure — record + schedule.
                    h.failure_times.append(now)
                    h.failure_times = h.failure_times[-_BURST_LIMIT * 2:]
                    recent = [t for t in h.failure_times
                              if now - t < _BURST_WINDOW_S]
                    if len(recent) >= _BURST_LIMIT:
                        h.intent_running = False
                        h.last_error = (
                            f"crash loop: {len(recent)} failures in "
                            f"{int(_BURST_WINDOW_S)}s — giving up"
                        )
                        log("supervisor",
                            f"node {h.spec.id} {h.last_error}",
                            level="error", bus=self._bus)
                        continue
                    delay = min(_BACKOFF_CAP_S,
                                _BACKOFF_BASE_S * (2 ** (len(recent) - 1)))
                    pending[h.spec.id] = now + delay
                    log("supervisor",
                        f"node {h.spec.id} died; restart in {delay:.0f}s "
                        f"(attempt {len(recent)}/{_BURST_LIMIT})",
                        level="warn", bus=self._bus)
                elif now >= pending[h.spec.id]:
                    pending.pop(h.spec.id)
                    h.restart()
                    log("supervisor", f"node {h.spec.id} restarted "
                        f"(restart #{h.restarts})", bus=self._bus)

    # ── internals ────────────────────────────────────────────────

    def _all(self) -> list[NodeHandle]:
        with self._lock:
            return list(self._handles.values())

    def _get(self, node_id: str) -> NodeHandle:
        with self._lock:
            try:
                return self._handles[node_id]
            except KeyError:
                raise KeyError(
                    f"no node {node_id!r}; have {sorted(self._handles)}"
                ) from None


# ── PID registry: the no-orphans net ─────────────────────────────


def _registry_load(path: pathlib.Path) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _registry_write(path: pathlib.Path, entries: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")


def _registry_add(path: pathlib.Path, pid: int, node_id: str) -> None:
    entries = _registry_load(path)
    entries[str(pid)] = node_id
    _registry_write(path, entries)


def _registry_remove(path: pathlib.Path, pid: int) -> None:
    entries = _registry_load(path)
    entries.pop(str(pid), None)
    _registry_write(path, entries)


def reap_stale(path: pathlib.Path) -> list[int]:
    """Kill children a previous (crashed) run left behind. Called at
    boot BEFORE the instance slot is taken. Returns reaped PIDs."""
    entries = _registry_load(path)
    reaped: list[int] = []
    for pid_str, node_id in entries.items():
        pid = int(pid_str)
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            continue   # already gone (or not ours)
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.2)
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            reaped.append(pid)
            log("supervisor",
                f"reaped stale node process {node_id} (pid {pid}) "
                "from a previous run", level="warn")
        except Exception:  # noqa: BLE001 — best-effort sweep
            pass
    _registry_write(path, {})
    return reaped


__all__ = [
    "Supervisor", "NodeHandle", "ThreadHandle", "SubprocessHandle",
    "reap_stale",
]
