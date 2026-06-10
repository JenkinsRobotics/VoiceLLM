"""LLM backend adapters — one per model runtime.

Each adapter implements BackendBase and is selected at boot
via config.LLM_BACKEND.
"""

from .llama_cpp.backend import LlamaCppBackend  # noqa: F401
from .mlx.backend import MLXBackend  # noqa: F401
