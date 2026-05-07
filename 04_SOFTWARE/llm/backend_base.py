"""Abstract base class for swappable LLM backends.

Both MLX and llama.cpp implementations conform to this so the rest of
VoiceLLM doesn't care which is loaded.
"""

from __future__ import annotations

import abc
import threading
from typing import Iterator


class BackendBase(abc.ABC):
    """Common surface area: load → warm → stream_chat → cancel."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()

    @abc.abstractmethod
    def load(self) -> None:
        """Load weights into memory. Blocking."""

    @abc.abstractmethod
    def warm(self) -> None:
        """One-token generation to pay graph compile / KV alloc up front."""

    @abc.abstractmethod
    def stream_chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Iterator[str]:
        """Yield text deltas for an in-progress reply.

        Implementations must check ``self.stop_event`` between yields so
        ``cancel()`` can interrupt mid-generation for barge-in.
        """

    def cancel(self) -> None:
        """Signal the in-flight stream_chat to stop after the current token."""
        self.stop_event.set()

    def reset_cancel(self) -> None:
        """Call at the top of each stream_chat to clear the prior signal."""
        self.stop_event.clear()
