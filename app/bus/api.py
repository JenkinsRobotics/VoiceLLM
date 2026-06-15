"""Bus interface + message registry.

A message is any dataclass instance with a ``topic: str`` field.
``/act/*`` topics are things the operator/app wants done; ``/sense/*``
are things nodes report; ``/sys/*`` belong to the chassis (health,
log). Apps define their own message dataclasses and register them so
the ZMQ backend can decode wire frames back into typed objects — the
in-process backend passes objects through untouched and never needs
the registry.
"""

from __future__ import annotations

import abc
import dataclasses
import json
from typing import Any, Callable

SubscriberFn = Callable[[Any], None]


@dataclasses.dataclass
class RawMessage:
    """Fallback for wire messages whose topic isn't registered —
    delivered rather than dropped, so a missing registration is
    visible in the consumer, not silent."""
    topic: str
    data: dict[str, Any] = dataclasses.field(default_factory=dict)


class MessageRegistry:
    """topic string → dataclass, for wire decode (ZMQ backend)."""

    def __init__(self) -> None:
        self._by_topic: dict[str, type] = {}

    def register(self, cls: type) -> type:
        """Register a message dataclass by its ``topic`` default.
        Usable as a decorator. Refuses classes without one."""
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")
        topic = ""
        for f in dataclasses.fields(cls):
            if f.name == "topic":
                topic = f.default if isinstance(f.default, str) else ""
        if not topic:
            raise ValueError(
                f"{cls.__name__} needs a `topic: str = \"/...\"` field"
            )
        self._by_topic[topic] = cls
        return cls

    def register_all(self, classes: list[type]) -> None:
        for cls in classes:
            self.register(cls)

    def encode(self, msg: Any) -> bytes:
        return json.dumps(dataclasses.asdict(msg)).encode("utf-8")

    def decode(self, topic: str, payload: bytes) -> Any:
        data = json.loads(payload.decode("utf-8"))
        cls = self._by_topic.get(topic)
        if cls is None:
            data.pop("topic", None)
            return RawMessage(topic=topic, data=data)
        return cls(**data)


class Bus(abc.ABC):
    """publish / subscribe / unsubscribe / close — that's the whole
    contract. Subscribers run on the bus's delivery thread: hand off
    anything slow (a REQ round-trip, a frame decode) to your own
    thread or queue, or you back-pressure every other topic."""

    @abc.abstractmethod
    def publish(self, msg: Any) -> None: ...

    @abc.abstractmethod
    def subscribe(self, topic: str, callback: SubscriberFn) -> None: ...

    @abc.abstractmethod
    def unsubscribe(self, topic: str, callback: SubscriberFn) -> None: ...

    @abc.abstractmethod
    def close(self) -> None:
        """Idempotent."""


__all__ = ["Bus", "MessageRegistry", "RawMessage", "SubscriberFn"]
