"""mlx-lm backend (Apple Silicon MLX-accelerated inference).

Mirrors MockingAgent/ollamacpp/chat_mlx.py: load, register Gemma 4's EOT
token, warm with one token, then stream deltas.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

from agent.llm.backend_base import BackendBase

# Gemma's chat template ends each assistant turn with <end_of_turn>.
# mlx-lm only stops on the tokenizer's eos_token_id by default, which is
# <eos> (session end), not <end_of_turn>. Without an in-stream check the
# model runs to max_tokens and starts looping. The TokenizerWrapper API
# for adding additional EOS varies by version, so we watch the streamed
# text and stop when we see a marker.
STOP_MARKERS = ("<end_of_turn>", "<|im_end|>", "<eos>")
_MAX_HOLDBACK = max(len(m) for m in STOP_MARKERS) - 1


def _scan_stream_text(pending: str, text: str) -> tuple[str, str, bool]:
    """Scan ``pending + text`` for stop markers. Returns
    ``(emit, new_pending, stopped)``.

    Any trailing run that is a prefix of a marker is held back in
    ``new_pending`` rather than emitted — otherwise a marker straddling
    two deltas leaks its already-emitted head ("<end_of") into TTS, which
    Kokoro reads aloud.
    """
    out = pending + text
    idxs = [i for i in (out.find(m) for m in STOP_MARKERS) if i != -1]
    if idxs:
        return out[:min(idxs)], "", True
    for k in range(min(len(out), _MAX_HOLDBACK), 0, -1):
        suffix = out[-k:]
        if any(m.startswith(suffix) for m in STOP_MARKERS):
            return out[:-k], suffix, False
    return out, "", False


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
        except Exception as exc:
            # Don't silently diverge from the llama.cpp backend's sampling.
            print(
                f"[mlx-lm] sampler unavailable ({exc}) — temperature/top_p "
                "ignored, using mlx-lm defaults",
                flush=True,
            )
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

        pending = ""
        for resp in stream_generate(self.model, self.tokenizer, prompt=prompt, **kwargs):
            if self.stop_event.is_set():
                return
            text = resp.text
            if not text:
                continue

            emit, pending, stopped = _scan_stream_text(pending, text)
            if emit:
                yield emit
            if stopped:
                return

        # Stream ended with a held-back partial that never became a marker.
        if pending:
            yield pending
