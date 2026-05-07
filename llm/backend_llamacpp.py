"""llama-cpp-python backend (Metal-accelerated GGUF inference).

Mirrors MockingAgent/ollamacpp/chat_llama.py: load, warm with one token,
then stream deltas via the OpenAI-style chat completion API.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

from .backend_base import BackendBase


class LlamaCppBackend(BackendBase):
    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
    ) -> None:
        super().__init__()
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.llm = None

    def load(self) -> None:
        from llama_cpp import Llama

        path = Path(self.model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        kwargs: dict = {
            "model_path": str(path),
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": False,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads

        print(f"[llama.cpp] Loading {path.name}...", flush=True)
        t0 = time.perf_counter()
        self.llm = Llama(**kwargs)
        print(f"[llama.cpp] Loaded in {time.perf_counter() - t0:.1f}s.", flush=True)

    def warm(self) -> None:
        t0 = time.perf_counter()
        self.llm.create_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.0,
        )
        print(f"[llama.cpp] Warm-up in {time.perf_counter() - t0:.1f}s.", flush=True)

    def stream_chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Iterator[str]:
        self.reset_cancel()
        completion = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        for chunk in completion:
            if self.stop_event.is_set():
                break
            delta = chunk["choices"][0].get("delta", {}).get("content", "")
            if delta:
                yield delta
