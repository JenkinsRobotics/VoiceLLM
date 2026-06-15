"""jaeger.toml — the app manifest: what this app IS MADE OF.

Structure lives here (nodes, surfaces, bus, event loop, instance
policy); behavior lives in config.yaml (§config). The manifest is not
runtime-overridable — a different shape is a different manifest
(``jaeger.sim.toml`` is the sanctioned simulation profile).

Validation refuses loudly with the offending field named: unknown
keys, bad enums, missing factories, more than one main surface, and a
``requires_framework`` newer than this chassis copy's stamp.
"""

from __future__ import annotations

import dataclasses
import pathlib
import tomllib
from typing import Any

_MODES = ("fused", "split")
_EVENT_LOOPS = ("qt", "asyncio", "tui", "none")
_UIS = ("pyside6", "swift", "tui", "none")
_BACKENDS = ("thread", "subprocess", "external")
_RESTARTS = ("never", "on_failure", "always")
_BUS_BACKENDS = ("inproc", "zmq")


@dataclasses.dataclass
class BusSpec:
    backend: str = "inproc"
    xsub: str = ""
    xpub: str = ""


@dataclasses.dataclass
class NodeSpec:
    id: str
    tier: int = 3
    backend: str = "thread"
    factory: str = ""          # thread: "pkg.mod:fn" returning Node(s)
    module: str = ""           # subprocess: spawn `python -m module`
    restart: str = "on_failure"
    config_key: str = ""
    enabled: bool = True
    args: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SurfaceSpec:
    id: str
    main: bool = False
    factory: str = ""          # "pkg.mod:fn" returning a Surface
    enabled: bool = True


@dataclasses.dataclass
class AppSpec:
    name: str
    version: str = "0.0.0"
    requires_framework: str = ""
    mode: str = "fused"
    event_loop: str = "none"
    ui: str = "none"
    single_instance: bool = True
    autostart: bool = False
    shell_quits_core: bool = True
    config: str = "config.yaml"
    bus: BusSpec = dataclasses.field(default_factory=BusSpec)
    nodes: list[NodeSpec] = dataclasses.field(default_factory=list)
    surfaces: list[SurfaceSpec] = dataclasses.field(default_factory=list)


def _refuse(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"jaeger.toml: {msg}")


def _check_keys(table: dict, allowed: set[str], where: str) -> None:
    unknown = set(table) - allowed
    _refuse(not unknown, f"unknown keys in {where}: {sorted(unknown)}")


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.strip().lstrip(">=").split(".")
                 if p.isdigit())


def load_manifest(path: str | pathlib.Path) -> AppSpec:
    """Parse + validate ``jaeger.toml``. ``path`` may be the file or
    its directory."""
    from . import FRAMEWORK_FORMAT

    p = pathlib.Path(path)
    if p.is_dir():
        p = p / "jaeger.toml"
    if not p.is_file():
        raise FileNotFoundError(f"no manifest at {p}")
    raw = tomllib.loads(p.read_text(encoding="utf-8"))

    _check_keys(raw, {"app", "bus", "node", "surface"}, "top level")
    app_raw = raw.get("app") or {}
    _check_keys(app_raw, {
        "name", "version", "requires_framework", "mode", "event_loop",
        "ui", "single_instance", "autostart", "shell_quits_core", "config",
    }, "[app]")
    _refuse(bool(app_raw.get("name")), "[app] name is required")

    spec = AppSpec(
        name=str(app_raw["name"]),
        version=str(app_raw.get("version", "0.0.0")),
        requires_framework=str(app_raw.get("requires_framework", "")),
        mode=str(app_raw.get("mode", "fused")),
        event_loop=str(app_raw.get("event_loop", "none")),
        ui=str(app_raw.get("ui", "none")),
        single_instance=bool(app_raw.get("single_instance", True)),
        autostart=bool(app_raw.get("autostart", False)),
        shell_quits_core=bool(app_raw.get("shell_quits_core", True)),
        config=str(app_raw.get("config", "config.yaml")),
    )
    _refuse(spec.mode in _MODES, f"[app] mode {spec.mode!r} not in {_MODES}")
    _refuse(spec.event_loop in _EVENT_LOOPS,
            f"[app] event_loop {spec.event_loop!r} not in {_EVENT_LOOPS}")
    _refuse(spec.ui in _UIS, f"[app] ui {spec.ui!r} not in {_UIS}")

    req = spec.requires_framework.strip()
    if req:
        _refuse(req.startswith(">="),
                f"requires_framework supports only '>=X.Y', got {req!r}")
        if _version_tuple(req) > _version_tuple(FRAMEWORK_FORMAT):
            raise RuntimeError(
                f"app needs format {req}, this chassis copy is "
                f"{FRAMEWORK_FORMAT} — refusing to boot"
            )

    bus_raw = raw.get("bus") or {}
    _check_keys(bus_raw, {"backend", "xsub", "xpub"}, "[bus]")
    spec.bus = BusSpec(
        backend=str(bus_raw.get("backend", "inproc")),
        xsub=str(bus_raw.get("xsub", "")),
        xpub=str(bus_raw.get("xpub", "")),
    )
    _refuse(spec.bus.backend in _BUS_BACKENDS,
            f"[bus] backend {spec.bus.backend!r} not in {_BUS_BACKENDS}")

    for n_raw in raw.get("node") or []:
        _check_keys(n_raw, {
            "id", "tier", "backend", "factory", "module", "restart",
            "config_key", "enabled", "args",
        }, "[[node]]")
        node = NodeSpec(
            id=str(n_raw.get("id", "")),
            tier=int(n_raw.get("tier", 3)),
            backend=str(n_raw.get("backend", "thread")),
            factory=str(n_raw.get("factory", "")),
            module=str(n_raw.get("module", "")),
            restart=str(n_raw.get("restart", "on_failure")),
            config_key=str(n_raw.get("config_key", "")),
            enabled=bool(n_raw.get("enabled", True)),
            args=dict(n_raw.get("args") or {}),
        )
        _refuse(bool(node.id), "[[node]] id is required")
        _refuse(node.backend in _BACKENDS,
                f"node {node.id!r}: backend {node.backend!r} not in {_BACKENDS}")
        _refuse(node.restart in _RESTARTS,
                f"node {node.id!r}: restart {node.restart!r} not in {_RESTARTS}")
        if node.backend == "thread":
            _refuse(bool(node.factory),
                    f"node {node.id!r}: thread backend needs `factory`")
        if node.backend == "subprocess":
            _refuse(bool(node.module),
                    f"node {node.id!r}: subprocess backend needs `module`")
        spec.nodes.append(node)
    ids = [n.id for n in spec.nodes]
    _refuse(len(ids) == len(set(ids)), f"duplicate node ids: {ids}")

    for s_raw in raw.get("surface") or []:
        _check_keys(s_raw, {"id", "main", "factory", "enabled"},
                    "[[surface]]")
        surface = SurfaceSpec(
            id=str(s_raw.get("id", "")),
            main=bool(s_raw.get("main", False)),
            factory=str(s_raw.get("factory", "")),
            enabled=bool(s_raw.get("enabled", True)),
        )
        _refuse(bool(surface.id), "[[surface]] id is required")
        _refuse(bool(surface.factory),
                f"surface {surface.id!r}: `factory` is required")
        spec.surfaces.append(surface)
    mains = [s for s in spec.surfaces if s.main and s.enabled]
    _refuse(len(mains) <= 1,
            f"at most one main surface; got {[s.id for s in mains]}")
    if spec.event_loop == "qt":
        _refuse(len(mains) == 1,
                "event_loop='qt' requires exactly one main surface")

    # Subprocess nodes can't ride the in-process bus.
    if any(n.backend == "subprocess" and n.enabled for n in spec.nodes):
        _refuse(spec.bus.backend == "zmq",
                "subprocess nodes require [bus] backend='zmq'")
    return spec


__all__ = ["AppSpec", "BusSpec", "NodeSpec", "SurfaceSpec", "load_manifest"]
