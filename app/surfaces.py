"""Surfaces — the shell side of the format.

The contract is behavior, not a widget library (spec §shell): each
app picks ONE ui toolkit in its manifest; the chassis provides the
PySide6 reference pieces here, imported lazily so headless apps never
touch Qt.

  * :class:`Surface` — id, main flag, build()/attach()/close().
  * :class:`SurfaceManager` — starts the main surface on the main
    thread inside the chassis's run(); closes everything in
    shutdown() (windows die together, structurally).
  * :class:`BusBridge` — the one sanctioned bus→Qt hop: subscriber
    callbacks run on the bus delivery thread; Qt signal emission is
    the thread-safe crossing into widget land.
"""

from __future__ import annotations

from typing import Any, Callable

from .manifest import SurfaceSpec


class Surface:
    """One operator-facing surface. Subclass or duck-type."""

    def __init__(self, spec: SurfaceSpec) -> None:
        self.spec = spec

    def build(self, ctx: Any) -> Any:
        """Create the runnable (window/tray/TUI). ``ctx`` is the
        JaegerApp — bus, supervisor, health, config all hang off it."""
        raise NotImplementedError

    def close(self) -> None:
        """Idempotent."""


class SurfaceManager:
    def __init__(self) -> None:
        self._surfaces: list[tuple[SurfaceSpec, Any]] = []

    def start_all(self, specs: list[SurfaceSpec],
                  resolve: Callable[[str], Any], ctx: Any) -> Any:
        """Build every enabled surface; return the main one (the
        chassis blocks on its event loop). Non-main surfaces are
        in-shell children (windows/tray) — never processes."""
        main_obj = None
        for spec in specs:
            if not spec.enabled:
                continue
            factory = resolve(spec.factory)
            obj = factory(ctx, spec)
            self._surfaces.append((spec, obj))
            if spec.main:
                main_obj = obj
        return main_obj

    def close_all(self) -> None:
        for _spec, obj in reversed(self._surfaces):
            try:
                close = getattr(obj, "close", None)
                if callable(close):
                    close()
            except Exception:  # noqa: BLE001 — teardown never raises
                pass
        self._surfaces.clear()


def make_bus_bridge(bus: Any, topics: list[str]) -> Any:
    """PySide6 BusBridge: one QObject with a ``message`` signal,
    emitted from bus callbacks (queued across threads by Qt). Lazy Qt
    import — only ui="pyside6" apps call this.

    Teardown-safe: during quit, nodes publish their final states
    after Qt has deleted the bridge's C++ object — the callback
    swallows that window's RuntimeError instead of spraying the log
    (and ``close()`` unsubscribes outright for tidy owners)."""
    from PySide6.QtCore import QObject, Signal

    class BusBridge(QObject):
        message = Signal(object)

        def __init__(self) -> None:
            super().__init__()
            self._pairs = [(t, self._make_cb()) for t in topics]
            for topic, cb in self._pairs:
                bus.subscribe(topic, cb)

        def _make_cb(self):
            def cb(msg: Any) -> None:
                try:
                    self.message.emit(msg)
                except RuntimeError:
                    pass   # Qt object already deleted (app quitting)
            return cb

        def close(self) -> None:
            for topic, cb in self._pairs:
                bus.unsubscribe(topic, cb)
            self._pairs = []

    return BusBridge()


__all__ = ["Surface", "SurfaceManager", "make_bus_bridge"]
