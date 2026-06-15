"""Conformance tests for the Jaeger app format (format 0.1).

THIS FILE IS PART OF WHAT YOU COPY: it runs against whichever chassis
copy sits next to it, in that app's own venv. Boot order, node
lifecycle, supervisor verbs + restart policy, manifest/config
refusals, single-instance, no-orphans teardown — the spec'd contract,
executable.
"""

from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import sys
import time
from typing import Any

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from app import FRAMEWORK_FORMAT, JaegerApp  # noqa: E402
from app.app import SecondInstanceError, resolve_ref  # noqa: E402
from app.bus.api import MessageRegistry, RawMessage  # noqa: E402
from app.bus.inproc import InProcBus  # noqa: E402
from app.config import load_config, slice_for  # noqa: E402
from app.health import HealthCache, NodeHealth  # noqa: E402
from app.manifest import load_manifest  # noqa: E402
from app.node import FrameNode, Node, NodeState  # noqa: E402
from app import supervisor as supervisor_mod  # noqa: E402
from app.supervisor import (  # noqa: E402
    SubprocessHandle,
    Supervisor,
    ThreadHandle,
    reap_stale,
)
from app.manifest import NodeSpec  # noqa: E402

REPO = pathlib.Path(__file__).parent.parent


def _wait_for(cond, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


# ── messages used by the in-file test nodes ────────────────────────


@dataclasses.dataclass
class Ping:
    n: int = 0
    topic: str = "/test/ping"


class TickerNode(Node):
    def __init__(self, *, bus: Any, **_: Any) -> None:
        super().__init__(bus=bus, name="ticker", tick_interval_s=0.01)
        self.ticks = 0

    def tick(self) -> None:
        self.ticks += 1
        time.sleep(0.01)


def make_ticker(bus: Any, config: dict[str, Any]) -> Node:
    return TickerNode(bus=bus, **config)


class DyingNode(Node):
    """Runs ~50 ms then stops itself — the restart-policy fixture."""

    def __init__(self, *, bus: Any, **_: Any) -> None:
        super().__init__(bus=bus, name="dying", tick_interval_s=0.01)

    def tick(self) -> None:
        time.sleep(0.05)
        self.stop()


def make_dying(bus: Any, config: dict[str, Any]) -> Node:
    return DyingNode(bus=bus, **config)


# ── manifest ───────────────────────────────────────────────────────


_GOOD_MANIFEST = """
[app]
name = "conftest-app"
version = "0.0.1"
requires_framework = ">=0.1"
mode = "fused"
event_loop = "none"
single_instance = true

[bus]
backend = "inproc"

[[node]]
id = "ticker"
tier = 3
backend = "thread"
factory = "tests.test_app_format:make_ticker"
restart = "never"
config_key = "ticker"
"""


def _write_app(tmp_path: pathlib.Path, manifest: str = _GOOD_MANIFEST,
               config: str = "ticker:\n  tick_interval_s: 0.01\n") -> pathlib.Path:
    (tmp_path / "jaeger.toml").write_text(manifest, encoding="utf-8")
    (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
    return tmp_path


def test_manifest_happy_path(tmp_path):
    spec = load_manifest(_write_app(tmp_path))
    assert spec.name == "conftest-app"
    assert spec.bus.backend == "inproc"
    assert spec.nodes[0].id == "ticker"
    assert spec.nodes[0].restart == "never"


def test_manifest_refuses_unknown_keys(tmp_path):
    bad = _GOOD_MANIFEST.replace('mode = "fused"', 'modee = "fused"')
    with pytest.raises(ValueError, match="modee"):
        load_manifest(_write_app(tmp_path, bad))


def test_manifest_refuses_bad_enums(tmp_path):
    bad = _GOOD_MANIFEST.replace('backend = "thread"',
                                 'backend = "hovercraft"')
    with pytest.raises(ValueError, match="hovercraft"):
        load_manifest(_write_app(tmp_path, bad))


def test_manifest_refuses_thread_without_factory(tmp_path):
    bad = _GOOD_MANIFEST.replace(
        'factory = "tests.test_app_format:make_ticker"', 'factory = ""')
    with pytest.raises(ValueError, match="needs `factory`"):
        load_manifest(_write_app(tmp_path, bad))


def test_manifest_refuses_subprocess_on_inproc_bus(tmp_path):
    bad = _GOOD_MANIFEST.replace(
        'backend = "thread"\nfactory = "tests.test_app_format:make_ticker"',
        'backend = "subprocess"\nmodule = "tests._worker_node"',
    )
    with pytest.raises(ValueError, match="zmq"):
        load_manifest(_write_app(tmp_path, bad))


def test_manifest_refuses_future_format(tmp_path):
    bad = _GOOD_MANIFEST.replace('">=0.1"', '">=99.0"')
    with pytest.raises(RuntimeError, match="refusing to boot"):
        load_manifest(_write_app(tmp_path, bad))
    assert FRAMEWORK_FORMAT == "0.1"


def test_manifest_refuses_two_main_surfaces(tmp_path):
    bad = _GOOD_MANIFEST + (
        '\n[[surface]]\nid = "a"\nmain = true\nfactory = "x:y"\n'
        '\n[[surface]]\nid = "b"\nmain = true\nfactory = "x:y"\n'
    )
    with pytest.raises(ValueError, match="at most one main"):
        load_manifest(_write_app(tmp_path, bad))


def test_manifest_qt_requires_a_main_surface(tmp_path):
    bad = _GOOD_MANIFEST.replace('event_loop = "none"',
                                 'event_loop = "qt"')
    with pytest.raises(ValueError, match="main surface"):
        load_manifest(_write_app(tmp_path, bad))


# ── config ─────────────────────────────────────────────────────────


def test_config_refuses_non_mapping(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


def test_config_slices(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("ticker:\n  rate: 5\n", encoding="utf-8")
    cfg = load_config(p)
    assert slice_for(cfg, "ticker") == {"rate": 5}
    assert slice_for(cfg, "absent") == {}
    assert slice_for(cfg, "") == {}


# ── message registry ───────────────────────────────────────────────


def test_registry_roundtrip_and_fallback():
    reg = MessageRegistry()
    reg.register(Ping)
    wire = reg.encode(Ping(n=7))
    back = reg.decode("/test/ping", wire)
    assert isinstance(back, Ping) and back.n == 7
    raw = reg.decode("/not/registered", b'{"x": 1, "topic": "/not/registered"}')
    assert isinstance(raw, RawMessage) and raw.data == {"x": 1}


def test_registry_refuses_topicless_dataclass():
    @dataclasses.dataclass
    class NoTopic:
        x: int = 0

    with pytest.raises(ValueError, match="topic"):
        MessageRegistry().register(NoTopic)


# ── bus (inproc) ───────────────────────────────────────────────────


def test_inproc_bus_routes_and_isolates_bad_subscribers():
    bus = InProcBus()
    try:
        got: list[Any] = []
        bus.subscribe("/test/ping", lambda m: 1 / 0)
        bus.subscribe("/test/ping", got.append)
        bus.publish(Ping(n=1))
        assert _wait_for(lambda: got)
        assert got[0].n == 1
    finally:
        bus.close()
        bus.close()   # idempotent


# ── node lifecycle ─────────────────────────────────────────────────


def test_node_lifecycle_and_fatal_setup():
    bus = InProcBus()
    try:
        node = TickerNode(bus=bus)
        import threading
        t = threading.Thread(target=node.run, daemon=True)
        t.start()
        assert _wait_for(lambda: node.ticks >= 3)
        assert node.state == NodeState.RUNNING
        node.stop()
        t.join(timeout=2.0)
        assert node.state == NodeState.STOPPED

        class Doomed(Node):
            def setup(self) -> None:
                raise RuntimeError("no device")

        torn = []
        doomed = Doomed(bus=bus)
        doomed.teardown = lambda: torn.append(1)  # type: ignore[method-assign]
        doomed.run()
        assert doomed.state == NodeState.FAILED
        assert torn == [1]
        assert "no device" in doomed.health()["error"]
    finally:
        bus.close()


def test_frame_node_update_render_split_and_pacing():
    class Blinker(FrameNode):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.updates = 0
            self.renders = 0

        def update_tick(self, ts: float) -> None:
            self.updates += 1

        def render_tick(self, ts: float) -> None:
            self.renders += 1

    bus = InProcBus()
    try:
        node = Blinker(bus=bus, fps=50, name="blinker")
        import threading
        t = threading.Thread(target=node.run, daemon=True)
        start = time.monotonic()
        t.start()
        assert _wait_for(lambda: node.frames_rendered >= 10)
        elapsed = time.monotonic() - start
        node.stop()
        t.join(timeout=2.0)
        assert node.updates == node.renders == node.frames_rendered
        # 10 frames at 50 fps ≈ 0.2 s — generous bounds, but it must
        # be PACED, not a busy loop.
        assert 0.1 < elapsed < 2.0
        assert node.health()["fps_target"] == 50.0
    finally:
        bus.close()


# ── supervisor: thread backend + restart policy ────────────────────


def test_thread_handle_restart_makes_fresh_instance():
    bus = InProcBus()
    try:
        spec = NodeSpec(id="ticker", factory="x:y")
        handle = ThreadHandle(spec, lambda: TickerNode(bus=bus))
        handle.start()
        assert _wait_for(handle.alive)
        first = handle.node()
        handle.restart()
        assert _wait_for(handle.alive)
        assert handle.node() is not first
        assert handle.restarts == 1
        handle.stop()
        assert not handle.alive()
        assert handle.state() == "off"
    finally:
        bus.close()


def test_supervisor_on_failure_restarts_with_backoff(monkeypatch):
    monkeypatch.setattr(supervisor_mod, "_BACKOFF_BASE_S", 0.05)
    monkeypatch.setattr(supervisor_mod, "_BACKOFF_CAP_S", 0.1)
    bus = InProcBus()
    sup = Supervisor(bus=bus)
    try:
        spec = NodeSpec(id="dying", factory="x:y", restart="on_failure")
        handle = ThreadHandle(spec, lambda: DyingNode(bus=bus))
        sup.add(handle)
        sup.start_all()
        assert _wait_for(lambda: handle.restarts >= 2, timeout_s=5.0)
    finally:
        sup.stop_all()
        bus.close()


def test_supervisor_burst_limit_gives_up(monkeypatch):
    monkeypatch.setattr(supervisor_mod, "_BACKOFF_BASE_S", 0.02)
    monkeypatch.setattr(supervisor_mod, "_BACKOFF_CAP_S", 0.02)
    monkeypatch.setattr(supervisor_mod, "_BURST_LIMIT", 3)
    bus = InProcBus()
    sup = Supervisor(bus=bus)
    try:
        spec = NodeSpec(id="dying", factory="x:y", restart="on_failure")
        handle = ThreadHandle(spec, lambda: DyingNode(bus=bus))
        sup.add(handle)
        sup.start_all()
        assert _wait_for(lambda: not handle.intent_running, timeout_s=5.0)
        diag = sup.diagnose("dying")
        assert diag["crash_loop"] is True
        assert "giving up" in diag["last_error"]
    finally:
        sup.stop_all()
        bus.close()


def test_supervisor_never_policy_stays_down():
    bus = InProcBus()
    sup = Supervisor(bus=bus)
    try:
        spec = NodeSpec(id="dying", factory="x:y", restart="never")
        handle = ThreadHandle(spec, lambda: DyingNode(bus=bus))
        sup.add(handle)
        sup.start_all()
        assert _wait_for(lambda: not handle.alive())
        time.sleep(0.3)
        assert handle.restarts == 0
    finally:
        sup.stop_all()
        bus.close()


# ── the chassis, headless ──────────────────────────────────────────


def test_app_boots_runs_verbs_and_shuts_down(tmp_path):
    app = JaegerApp(_write_app(tmp_path)).boot()
    try:
        rows = app.supervisor.ls()
        assert rows[0]["id"] == "ticker"
        assert _wait_for(
            lambda: app.supervisor.ls()[0]["state"] == "running")
        app.supervisor.stop("ticker")
        assert app.supervisor.ls()[0]["state"] == "off"
        app.supervisor.start("ticker")
        assert _wait_for(
            lambda: app.supervisor.ls()[0]["state"] == "running")
        app.supervisor.restart("ticker")
        diag = app.supervisor.diagnose("ticker")
        assert diag["restarts"] == 1 and diag["crash_loop"] is False
    finally:
        app.shutdown()
        app.shutdown()   # idempotent
    assert not (tmp_path / ".run" / "conftest-app.pid").exists()


def test_second_instance_refused(tmp_path):
    root = _write_app(tmp_path)
    app1 = JaegerApp(root).boot()
    try:
        with pytest.raises(SecondInstanceError, match="already running"):
            JaegerApp(root).boot()
    finally:
        app1.shutdown()
    # Slot released — a third boot succeeds.
    app3 = JaegerApp(root).boot()
    app3.shutdown()


def test_resolve_ref_contract():
    assert resolve_ref("tests.test_app_format:make_ticker") is make_ticker
    with pytest.raises(ValueError, match="module:attr"):
        resolve_ref("no_colon")
    with pytest.raises(ImportError, match="no attribute"):
        resolve_ref("tests.test_app_format:missing")


# ── no-orphans: reap_stale ─────────────────────────────────────────


def test_reap_stale_kills_leftover_children(tmp_path):
    child = subprocess.Popen([sys.executable, "-c",
                              "import time; time.sleep(60)"])
    registry = tmp_path / "children.json"
    registry.write_text(f'{{"{child.pid}": "leftover"}}', encoding="utf-8")
    try:
        reaped = reap_stale(registry)
        assert child.pid in reaped
        assert _wait_for(lambda: child.poll() is not None)
    finally:
        if child.poll() is None:
            child.kill()


# ── subprocess backend + zmq bus, end to end ───────────────────────


@pytest.fixture
def zmq_stack():
    pytest.importorskip("zmq")
    from app.bus.zmq import Broker, ZmqBus
    from tests._worker_node import MESSAGES

    broker = Broker(xsub="tcp://127.0.0.1:7791", xpub="tcp://127.0.0.1:7792")
    broker.start()
    bus = ZmqBus(MESSAGES, xsub=broker.xsub_endpoint,
                 xpub=broker.xpub_endpoint)
    yield broker, bus
    bus.close()
    broker.stop()


def test_subprocess_node_roundtrip_crash_and_supervised_restart(
        zmq_stack, tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor_mod, "_BACKOFF_BASE_S", 0.1)
    broker, bus = zmq_stack
    from tests._worker_node import TestCmd

    health = HealthCache(bus)
    sup = Supervisor(health=health, bus=bus)
    spec = NodeSpec(id="worker", backend="subprocess",
                    module="tests._worker_node", restart="on_failure")
    handle = SubprocessHandle(
        spec, env_extra=broker.env(), cwd=REPO,
        registry_file=tmp_path / "children.json",
    )
    sup.add(handle)
    sup.start_all()
    try:
        # Liveness: heartbeats arrive over the wire.
        assert _wait_for(lambda: health.latest("worker") is not None,
                         timeout_s=8.0), "no heartbeat from child"
        # Roundtrip: command in, echo out.
        echoes: list[Any] = []
        bus.subscribe("/test/echo", echoes.append)
        deadline = time.monotonic() + 8.0
        while not echoes and time.monotonic() < deadline:
            bus.publish(TestCmd(cmd="ping"))
            time.sleep(0.3)
        assert echoes and echoes[0].cmd == "pong"
        first_pid = echoes[0].pid

        # Crash it — the supervisor's restart policy takes over.
        bus.publish(TestCmd(cmd="die"))
        assert _wait_for(lambda: handle.restarts >= 1, timeout_s=10.0), \
            "supervisor never restarted the crashed child"
        echoes.clear()
        deadline = time.monotonic() + 8.0
        while not echoes and time.monotonic() < deadline:
            bus.publish(TestCmd(cmd="ping"))
            time.sleep(0.3)
        assert echoes, "restarted child never answered"
        assert echoes[0].pid != first_pid   # genuinely a new process
    finally:
        sup.stop_all()
        assert handle.state().startswith(("off", "exited"))
        # Registry is clean — no orphans on disk.
        leftovers = [p for p in
                     (tmp_path / "children.json",) if p.exists()
                     and p.read_text() not in ("{}", "")]
        assert not leftovers


def test_node_health_message_shape():
    h = NodeHealth(node="x", state="running", details={"a": 1}, ts=1.0)
    assert h.topic == "/sys/node_health"
    assert dataclasses.asdict(h)["details"] == {"a": 1}
