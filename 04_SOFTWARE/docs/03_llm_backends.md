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

`config.py` will reference these via constants:

```python
LLM_BACKEND = "mlx"   # or "llamacpp"
GGUF_PATH   = "/Users/.../gemma-4-26B-A4B-it-Q4_K_M.gguf"
MLX_PATH    = "/Users/.../mlx-community/gemma-4-26b-a4b-4bit"
```

## The interface

```python
# llm/backend_base.py
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

`llm/llm_node.py` becomes thin:

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

| | MLX (mlx-lm) | llama.cpp (GGUF) |
|---|---|---|
| First-token latency | Lower on M-series | Higher; depends on Metal kernel cache |
| Decode tok/s | Generally higher on M-series | Slightly lower for MoE, comparable for dense |
| Memory | Lower for 4-bit weight-only | Higher for same nominal quantization |
| Ecosystem | macOS-only | Cross-platform; GGUF is the de-facto local format |
| Tokenizer config | Need to handle Gemma 4 EOT separately (see chat_mlx.py:52-54) | Built into GGUF |
| Stream API | `stream_generate(...)` yielding `.text` | `create_chat_completion(stream=True)` yielding deltas |

For voice we want lowest TTFT → MLX is the default. We keep llama.cpp because:
1. it's the format MockingAgent's `voice_assistant.py` already runs against,
2. some Gemma 4 quants only ship as GGUF,
3. it's the portability fallback if MLX changes break us.

## Sampling defaults for voice

Voice replies should be short and conversational, not essay-length:

```python
SYSTEM_PROMPT = (
    "You are a voice assistant having a real-time spoken conversation. "
    "Reply in 1–3 short sentences of natural conversational English. "
    "No markdown, no lists, no code blocks, no emoji. If unsure, say so briefly."
)
LLM_MAX_TOKENS  = 220        # ~30 seconds of speech is plenty
LLM_TEMPERATURE = 0.6        # MockingAgent uses 0.7; lower for tighter answers
LLM_TOP_P       = 0.9
```

`clean_for_tts()` from `voice_assistant.py:265-271` (strips markdown/code
fences/list bullets) ports over verbatim.

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

- Do we keep the conversation history in the LLM node or in the orchestrator?
  (Lean: in LLM node — only it knows the chat template format.)
- History length cap? Voice goes long. Cap at the last N user/assistant pairs
  plus a "summary so far" injection? Defer until we see real usage.
- Function/tool calling? Out of scope for v1; revisit if we want timers,
  weather, music, etc.
