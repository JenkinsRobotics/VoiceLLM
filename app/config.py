"""config.yaml — behavior knobs. Structure lives in the manifest.

The loader refuses loudly (non-mapping top level, missing file when
the manifest names one); per-node tunables live under each node's
``config_key`` and are handed to factories/subprocesses as plain
dicts. Schema validation beyond shape is the app's business — it
knows its knobs.
"""

from __future__ import annotations

import pathlib
from typing import Any

import yaml


def load_config(path: str | pathlib.Path) -> dict[str, Any]:
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no config at {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    return raw


def slice_for(config: dict[str, Any], config_key: str) -> dict[str, Any]:
    """A node's slice of the config. Missing key = empty dict (a node
    with no tunables is normal); a non-mapping slice refuses."""
    if not config_key:
        return {}
    section = config.get(config_key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"config[{config_key!r}] must be a mapping")
    return section


__all__ = ["load_config", "slice_for"]
