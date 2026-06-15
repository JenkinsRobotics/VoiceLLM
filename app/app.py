"""JaegerApp — the chassis. Every app boots the same way:

    manifest → instance slot → config → bus → nodes → surfaces → run
                                                                  ↓
                       teardown (reverse order, signal-safe, atexit)

Desktop-grade guarantees owned here (spec §checklist):
  * single instance — second launch refuses, with the running PID
    named (stale slots from crashed runs are reaped first);
  * no orphans — child PIDs are registered on disk at spawn; boot
    reaps a previous run's leftovers; shutdown walks live handles
    with terminate→kill, then surfaces close, then the bus;
  * one event loop — declared in the manifest (qt | none in format
    0.1; asyncio/tui reserved), owned by the main thread.
"""

from __future__ import annotations

import atexit
import importlib
import os
import pathlib
import signal
import sys
from typing import Any

from .bus.api import Bus, MessageRegistry
from .bus.inproc import InProcBus
from .config import load_config, slice_for
from .health import HealthCache
from .logging import log
from .manifest import AppSpec, NodeSpec, load_manifest
from .supervisor import (
    SubprocessHandle,
    Supervisor,
    ThreadHandle,
    reap_stale,
)
from .surfaces import SurfaceManager


def resolve_ref(ref: str) -> Any:
    """``"pkg.mod:attr"`` → the attribute. Loud on bad refs."""
    mod_path, _, attr = ref.partition(":")
    if not attr:
        raise ValueError(f"ref {ref!r} must be 'module:attr'")
    module = importlib.import_module(mod_path)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise ImportError(f"{mod_path} has no attribute {attr!r}") from None


class SecondInstanceError(RuntimeError):
    """Another instance of this app already holds the slot."""


class JaegerApp:
    def __init__(
        self,
        manifest_path: str | pathlib.Path,
        *,
        registry: MessageRegistry | None = None,
    ) -> None:
        self.root = pathlib.Path(manifest_path)
        if self.root.is_file():
            self.root = self.root.parent
        self.spec: AppSpec = load_manifest(self.root)
        self.registry = registry or MessageRegistry()
        self.config: dict[str, Any] = {}
        self.bus: Bus | None = None
        self.health: HealthCache | None = None
        self.supervisor: Supervisor | None = None
        self.surfaces = SurfaceManager()
        self._broker: Any = None
        self._run_dir = self.root / ".run"
        self._slot_file = self._run_dir / f"{self.spec.name}.pid"
        self._registry_file = self._run_dir / f"{self.spec.name}.children.json"
        self._slot_held = False
        self._shutdown_done = False
        self._qt_app: Any = None

    # ── boot phases ──────────────────────────────────────────────

    def boot(self) -> "JaegerApp":
        """Phases 1-6. Separated from run() so headless tests can
        boot, poke, and shut down without an event loop."""
        self._acquire_instance()
        # Empty config = "" means "this app doesn't use a chassis-loaded
        # config" (some apps route per-instance configs elsewhere).
        # Skip loading without falling into root / "" → root/ →
        # exists() → load_config(dir) → FileNotFoundError.
        if self.spec.config:
            cfg_path = self.root / self.spec.config
            self.config = load_config(cfg_path) if cfg_path.exists() else {}
        else:
            self.config = {}
        self._build_bus()
        self.health = HealthCache(self.bus)
        self._start_nodes()
        self._install_signals()
        atexit.register(self.shutdown)
        log(self.spec.name,
            f"booted: {len([n for n in self.spec.nodes if n.enabled])} "
            f"nodes, bus={self.spec.bus.backend}, "
            f"mode={self.spec.mode}", bus=self.bus)
        return self

    def run(self) -> int:
        """boot() + block on the manifest's event loop + teardown."""
        self.boot()
        try:
            if self.spec.event_loop == "qt":
                return self._run_qt()
            if self.spec.event_loop == "none":
                return 0   # caller drives (tests, scripts)
            raise ValueError(
                f"event_loop {self.spec.event_loop!r} not implemented "
                "in format 0.1 (qt | none)"
            )
        finally:
            if self.spec.event_loop != "none":
                self.shutdown()
        return 0

    def shutdown(self) -> None:
        """Reverse boot order. Idempotent; never raises; the
        windows-die-together + no-orphans moment."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        try:
            self.surfaces.close_all()
        except Exception:  # noqa: BLE001
            pass
        if self.supervisor is not None:
            self.supervisor.stop_all()
        if self._broker is not None:
            try:
                self._broker.stop()
            except Exception:  # noqa: BLE001
                pass
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:  # noqa: BLE001
                pass
        self._release_instance()
        log(self.spec.name, "shutdown complete")

    # ── phase internals ──────────────────────────────────────────

    def _acquire_instance(self) -> None:
        if not self.spec.single_instance:
            return
        self._run_dir.mkdir(parents=True, exist_ok=True)
        if self._slot_file.exists():
            try:
                pid = int(self._slot_file.read_text().strip() or "0")
            except ValueError:
                pid = 0
            if pid and _pid_alive(pid):
                raise SecondInstanceError(
                    f"{self.spec.name} is already running (pid {pid}) — "
                    "quit it first (a second launch never doubles the app)"
                )
            # Stale slot from a crashed run: reap its children first.
            reap_stale(self._registry_file)
        self._slot_file.write_text(str(os.getpid()), encoding="utf-8")
        self._slot_held = True

    def _release_instance(self) -> None:
        if self._slot_held:
            try:
                self._slot_file.unlink(missing_ok=True)
            except OSError:
                pass
            self._slot_held = False

    def _build_bus(self) -> None:
        if self.spec.bus.backend == "zmq":
            from .bus.zmq import (
                DEFAULT_XPUB, DEFAULT_XSUB, Broker, ZmqBus,
            )
            xsub = self.spec.bus.xsub or DEFAULT_XSUB
            xpub = self.spec.bus.xpub or DEFAULT_XPUB
            self._broker = Broker(xsub=xsub, xpub=xpub)
            self._broker.start()
            self.bus = ZmqBus(self.registry, xsub=xsub, xpub=xpub)
        else:
            self.bus = InProcBus()

    def _start_nodes(self) -> None:
        self.supervisor = Supervisor(health=self.health, bus=self.bus)
        for node_spec in self.spec.nodes:
            self.supervisor.add(self._make_handle(node_spec))
        self.supervisor.start_all()

    def _make_handle(self, node_spec: NodeSpec):
        cfg = slice_for(self.config, node_spec.config_key)
        if node_spec.backend == "thread":
            factory_fn = resolve_ref(node_spec.factory)

            def factory(fn=factory_fn, spec=node_spec, cfg=cfg):
                return fn(self.bus, {**spec.args, **cfg})

            return ThreadHandle(node_spec, factory)
        if node_spec.backend == "subprocess":
            import json as _json
            env_extra = dict(self._broker.env()) if self._broker else {}
            env_extra["JAEGER_NODE_CONFIG"] = _json.dumps(
                {**node_spec.args, **cfg})
            return SubprocessHandle(
                node_spec, env_extra=env_extra, cwd=self.root,
                registry_file=self._registry_file,
            )
        raise ValueError(
            f"node {node_spec.id!r}: backend {node_spec.backend!r} "
            "not implemented in format 0.1 (thread | subprocess)"
        )

    def _install_signals(self) -> None:
        def handler(signum: int, frame: Any) -> None:  # noqa: ARG001
            log(self.spec.name, f"signal {signum}; shutting down")
            if self._qt_app is not None:
                self._qt_app.quit()
            else:
                self.shutdown()
                sys.exit(0)

        try:
            signal.signal(signal.SIGTERM, handler)
            signal.signal(signal.SIGINT, handler)
        except (ValueError, OSError):
            pass   # not on the main thread (tests)

    def _run_qt(self) -> int:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        qt_app = QApplication.instance() or QApplication(sys.argv)
        qt_app.setQuitOnLastWindowClosed(self.spec.shell_quits_core)
        self._qt_app = qt_app
        main_obj = self.surfaces.start_all(
            self.spec.surfaces, resolve_ref, self)
        if main_obj is not None and hasattr(main_obj, "show"):
            main_obj.show()
        # Pump so Ctrl-C in the terminal reaches the Python handler.
        pump = QTimer()
        pump.timeout.connect(lambda: None)
        pump.start(200)
        return qt_app.exec()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


__all__ = ["JaegerApp", "SecondInstanceError", "resolve_ref"]
