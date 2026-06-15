"""The Jaeger app-format reference chassis.

This directory IS the thing you copy (docs/JAEGER_APP_FORMAT.md §1):
every Jenkins Robotics app carries its own copy of these modules in
its own repo and owns it outright — the Microsoft-Office model, one
design system, many independent codebases. Nothing imports this
across repositories.

Copy-ability rules (enforced by tests/test_app_format.py):
  * modules import only the stdlib, pyyaml, optional pyzmq, and each
    other — RELATIVELY. Paste this directory anywhere and it works.
  * every copy carries the FRAMEWORK_FORMAT stamp below; an app's
    jaeger.toml declares ``requires_framework`` against it.

Layout:
  app.py         the chassis — manifest → config → bus → nodes →
                 surfaces → run → teardown
  node.py        Node + NodeState (the universal lifecycle contract)
  supervisor.py  NodeHandle + thread/subprocess backends, restart
                 policy (never|on_failure|always + backoff), diagnose
  manifest.py    jaeger.toml loader + validator
  config.py      config.yaml loader (refuse loudly)
  bus/           Bus ABC + in-process and ZMQ-broker backends
  surfaces.py    Surface contract + SurfaceManager + bus→Qt bridge
  health.py      NodeHealth heartbeats + the liveness cache
  logging.py     one log-line shape + the /sys/log bus mirror
  child.py       subprocess-node entry helper
"""

from __future__ import annotations

from .app import JaegerApp
from .bus.api import Bus, MessageRegistry, RawMessage
from .bus.inproc import InProcBus
from .config import load_config
from .health import HealthCache, NodeHealth
from .logging import LogLine, log
from .manifest import AppSpec, BusSpec, NodeSpec, SurfaceSpec, load_manifest
from .node import FrameNode, Node, NodeState
from .supervisor import NodeHandle, Supervisor

# The copy's format stamp — what jaeger.toml's `requires_framework`
# checks against (manifest.py reads it lazily; no import cycle).
FRAMEWORK_FORMAT = "0.1"

__all__ = [
    "FRAMEWORK_FORMAT",
    "JaegerApp",
    "Node", "FrameNode", "NodeState",
    "Supervisor", "NodeHandle",
    "Bus", "InProcBus", "MessageRegistry", "RawMessage",
    "AppSpec", "NodeSpec", "SurfaceSpec", "BusSpec", "load_manifest",
    "load_config",
    "NodeHealth", "HealthCache",
    "LogLine", "log",
]
