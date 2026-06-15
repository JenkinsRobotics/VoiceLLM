"""Bus backends — one interface, picked by the manifest ``[bus]``.

  * ``inproc`` — queue + delivery thread; fused single-process apps.
  * ``zmq``    — XPUB/XSUB broker topology; required the moment any
                 node runs as a subprocess (cross-process pub/sub).
"""

from .api import Bus, MessageRegistry, RawMessage, SubscriberFn
from .inproc import InProcBus

__all__ = [
    "Bus", "MessageRegistry", "RawMessage", "SubscriberFn", "InProcBus",
]
