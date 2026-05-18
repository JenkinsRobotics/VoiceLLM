# LLM backends — MLX and llama.cpp behind one interface

We support two local backends. Both run **Gemma 4 26B-A4B (4-bit)** so we can
A/B them with the same model behavior. Backend selection is a config flag.

## Reference implementations

- `MockingAgent/ollamacpp/chat_llama.py` — llama-cpp-python, GGUF.
- `MockingAgent/ollamacpp/chat_mlx.py` — mlx-lm, MLX 4-bit.
- `MockingAgent/ollamacpp/bench.py` — full 2x2 matrix benchmark, this is what
  validated Gemma 4 26B-A4B as our default.

The chat scripts already implement the right "load + warm + stream" pattern
that voice needs. We're going to extract that into a `BackendBase` ABC.

## Models on disk (verified)

```
/Users/jonathanjenkins/.lmstudio/models/
├── lmstudio-community/gemma-4-26B-A4B-it-GGUF/
│   ├── gemma-4-26B-A4B-it-Q4_K_M.gguf          # llama.cpp default
│   └── mmproj-gemma-4-26B-A4B-it-BF16.gguf     # vision adapter (unused for voice)
└── mlx-community/gemma-4-26b-a4b-4bit/
    ├── config.json
    ├── model-0000{1,2,3}-of-00003.safetensors
    └── tokenizer{,_config}.json                # MLX default
```

`config.py` references these via constants:

```python
LLM_BACKEND = "llamacpp"  # or "mlx" (post-fix; see "MLX EOT" below)
GGUF_PATH   = "/Users/.../gemma-4-26B-A4B-it-Q4_K_M.gguf"
MLX_PATH    = "/Users/.../mlx-community/gemma-4-26b-a4b-4bit"
```

## The interface

```python
# plugins/llm_core/backend_base.py
class BackendBase(abc.ABC):
    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def warm(self) -> None:
        """One-token generation to pay graph compile / KV alloc tax up front."""

    @abc.abstractmethod
    def stream_chat(
        self, messages: list[dict], *, max_tokens: int, temperature: float, top_p: float
    ) -> Iterator[str]:
        """Yield text deltas. Must be cancellable via stop_event for barge-in."""

    @abc.abstractmethod
    def cancel(self) -> None: ...
```

`plugins/llm_core/node.py` stays thin:

```python
class LLMNode:
    def __init__(self, bus, backend: BackendBase, system: str):
        self.bus, self.backend, self.system = bus, backend, system
        self.history = [{"role": "system", "content": system}]
        bus.subscribe("llm.request", self.on_request)
        bus.subscribe("tts.cancel", lambda _: backend.cancel())

    def on_request(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})
        reply_parts = []
        for delta in self.backend.stream_chat(self.history, ...):
            self.bus.publish("llm.token", delta)
            reply_parts.append(delta)
        self.history.append({"role": "assistant", "content": "".join(reply_parts)})
        self.bus.publish("llm.done", None)
```

(Note: `core/bus.py` currently exposes `get()` polling, not `subscribe()`. We
either add a tiny dispatcher or stick with one consumer thread per node — see
`01_architecture.md`.)

## Why both backends

| | llama.cpp (GGUF, default) | MLX (mlx-lm) |
|---|---|---|
| First-token latency | Slightly higher on M-series | Lower on M-series |
| Decode tok/s | Solid; mature kernels | Generally higher on Apple Silicon |
| Memory | Higher for same nominal quantization | Lower (unified-memory native) |
| Ecosystem | Cross-platform; GGUF is the de-facto local format | macOS-only |
| Tokenizer / stop tokens | Chat completion handles Gemma's `<end_of_turn>` natively. Just works. | mlx-lm's `TokenizerWrapper` API for additional stop tokens varies by version; we use an in-stream marker detector instead (see "MLX EOT" below). |
| Stream API | `create_chat_completion(stream=True)` yielding deltas | `stream_generate(...)` yielding `.text` |

**llama.cpp is the default** because chat completion handles Gemma's stop
tokens correctly out of the box. We keep MLX because:
1. it's faster on Apple Silicon when it works,
2. it's the path for any future MLX-only models (e.g. dense Gemma at 8-bit),
3. the in-stream stop fix means MLX is now also safe to enable.

## MLX EOT (the fix that took us a while)

The original [backend_mlx.py](../plugins/mlx_llm/backend.py) load tried this:

```python
eot = getattr(self.tokenizer, "eot_token", None)
if eot and eot != self.tokenizer.eos_token:
    self.tokenizer.add_eos_token(eot)
```

The mlx-lm `TokenizerWrapper` has no attribute called `eot_token` —
that's a made-up name. `getattr` returned `None`, the `if` was always
false, and Gemma's `<end_of_turn>` never got registered as a stop. The
model ran to `LLM_MAX_TOKENS` every turn and started looping the same
sentence. Symptom: 4-paragraph rambling replies even on simple prompts.

The fix in [backend.py:stream_chat()](../plugins/mlx_llm/backend.py) is
backend-API-agnostic — we watch the streamed text:

```python
STOP_MARKERS = ("<end_of_turn>", "<|im_end|>", "<eos>")
buffered = ""
for resp in stream_generate(self.model, self.tokenizer, prompt=prompt, **kwargs):
    text = resp.text
    if not text:
        continue
    buffered = (buffered + text)[-32:]
    # detect a marker straddling a token boundary, yield the head, break
    ...
```

Robust to mlx-lm version changes; no tokenizer-wrapper guessing.

## Sampling defaults for voice

Voice replies should be short and conversational, not essay-length. The
SYSTEM_PROMPT also carries the `<ignore>`/`<reply>` gate protocol — see
[01_architecture.md](01_architecture.md#llm-gate) and
[05_barge_in_and_self_speech.md](05_barge_in_and_self_speech.md).

```python
LLM_MAX_TOKENS  = 220        # ~30 seconds of speech is plenty
LLM_TEMPERATURE = 0.6        # MockingAgent uses 0.7; lower for tighter answers
LLM_TOP_P       = 0.9
MAX_HISTORY_TURNS = 8        # cap rolling user/assistant pairs
```

`clean_for_tts()` from `voice_assistant.py:265-271` (strips markdown/code
fences/list bullets) ports over verbatim into [plugins/llm_core/node.py](../plugins/llm_core/node.py).

## History trimming

[LLMNode](../plugins/llm_core/node.py) caps conversation history at
`cfg.MAX_HISTORY_TURNS` user/assistant pairs (system prompt always
preserved). Without this, a long session grows the prompt monotonically
and per-turn latency drifts up. Ported from
[voice_assistant.py:258-263](../../MockingAgent/voice_assistant.py#L258-L263).

## Cancellation (for barge-in)

Both backends must respect a per-call stop signal. Easiest pattern:

- llama.cpp: `Llama.create_chat_completion(stream=True)` is a generator; we
  break the for-loop and call `llm.reset()` if we want to clear KV.
- mlx-lm: `stream_generate(...)` is a generator too; we break the for-loop.

`backend.cancel()` flips a `threading.Event`; the generator loop checks it
between yields and raises `StopIteration`. Concretely:

```python
for delta in backend.stream_chat(...):
    if stop_event.is_set(): break
    bus.publish("llm.token", delta)
```

## Open questions for LLM

- ~~Where does conversation history live?~~ **Resolved:** in the LLM node.
  Exposed via `node.history_snapshot()` for self-speech filter + metrics.
- ~~History length cap?~~ **Resolved:** `MAX_HISTORY_TURNS = 8` user/assistant
  pairs (see above).
- Function/tool calling? Out of scope for v1; revisit if we want timers,
  weather, music, etc.
- Repetition penalty for MLX? mlx-lm's `make_sampler` accepts
  `repetition_penalty` in newer versions. Worth wiring once we have a
  baseline run on patched MLX to compare.
