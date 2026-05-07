"""mlx-lm backend (Apple Silicon MLX-accelerated inference).

Mirrors MockingAgent/ollamacpp/chat_mlx.py: load, register Gemma 4's EOT
token, warm with one token, then stream deltas.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

from .backend_base import BackendBase


class MLXBackend(BackendBase):
    def __init__(self, model_path: str | Path) -> None:
        super().__init__()
        self.model_path = str(model_path)
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        from mlx_lm import load

        candidate = Path(self.model_path).expanduser()
        model_id = str(candidate) if candidate.exists() else self.model_path

        print(f"[mlx-lm] Loading {model_id}...", flush=True)
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_id)
        print(f"[mlx-lm] Loaded in {time.perf_counter() - t0:.1f}s.", flush=True)

        # Gemma 4's tokenizer separates <eos> from <end_of_turn>; mlx-lm
        # only stops on eos_token_id by default, so add the EOT too.
        eot = getattr(self.tokenizer, "eot_token", None)
        if eot and eot != self.tokenizer.eos_token:
            self.tokenizer.add_eos_token(eot)

    def warm(self) -> None:
        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        t0 = time.perf_counter()
        generate(self.model, self.tokenizer, prompt=prompt, max_tokens=1, verbose=False)
        print(f"[mlx-lm] Warm-up in {time.perf_counter() - t0:.1f}s.", flush=True)

    def _make_sampler(self, temperature: float, top_p: float):
        try:
            from mlx_lm.sample_utils import make_sampler
            return make_sampler(temp=temperature, top_p=top_p)
        except Exception:
            return None

    def stream_chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Iterator[str]:
        from mlx_lm import stream_generate

        self.reset_cancel()
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        kwargs: dict = {"max_tokens": max_tokens}
        sampler = self._make_sampler(temperature, top_p)
        if sampler is not None:
            kwargs["sampler"] = sampler

        for resp in stream_generate(self.model, self.tokenizer, prompt=prompt, **kwargs):
            if self.stop_event.is_set():
                break
            text = resp.text
            if text:
                yield text
