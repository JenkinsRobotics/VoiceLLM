"""VoiceLLM entrypoint — wire the bus, build nodes, run the orchestrator."""

from __future__ import annotations

import sys

import config as cfg
from transport.bus import Bus
from agent.orchestrator import Orchestrator
from nodes.tts.node import KokoroNode
from agent.adapters.llama_cpp.backend import LlamaCppBackend
from agent.llm.backend_base import BackendBase
from agent.llm.node import LLMNode
from agent.adapters.mlx.backend import MLXBackend
from nodes.stt.continuous import STTContinuousNode
from nodes.stt.two_pass import STTTwoPassNode


def make_backend() -> BackendBase:
    if cfg.LLM_BACKEND == "mlx":
        return MLXBackend(cfg.MLX_PATH)
    if cfg.LLM_BACKEND == "llamacpp":
        return LlamaCppBackend(
            cfg.GGUF_PATH,
            n_ctx=cfg.LLM_CTX,
            n_gpu_layers=cfg.LLM_GPU_LAYERS,
        )
    raise ValueError(f"Unknown LLM_BACKEND: {cfg.LLM_BACKEND!r}")


def make_stt(bus: Bus):
    if cfg.STT_MODE == "two_pass":
        return STTTwoPassNode(
            bus,
            fast_model_name=cfg.STT_FAST_MODEL,
            accurate_model_name=cfg.STT_ACCURATE_MODEL,
            require_wake_word=cfg.REQUIRE_WAKE_WORD,
            wake_phrases=cfg.WAKE_PHRASES,
            wake_match_threshold=cfg.WAKE_MATCH_THRESHOLD,
            followup_window_s=cfg.FOLLOWUP_WINDOW_S,
            sample_rate=cfg.SAMPLE_RATE,
            frame_samples=cfg.FRAME_SAMPLES,
            frame_ms=cfg.FRAME_MS,
            vad_aggressiveness=cfg.VAD_AGGRESSIVENESS,
            pre_roll_ms=cfg.PRE_ROLL_MS,
            post_padding_ms=cfg.POST_PADDING_MS,
            silence_hangover_ms=cfg.SILENCE_HANGOVER_MS,
            min_speech_ms=cfg.MIN_SPEECH_MS,
            max_speech_ms=cfg.MAX_SPEECH_MS,
            short_phrase_max_ms=getattr(cfg, "SHORT_PHRASE_MAX_MS", 0),
            short_phrase_hangover_ms=getattr(
                cfg, "SHORT_PHRASE_HANGOVER_MS", 0
            ),
            input_device=cfg.INPUT_DEVICE,
        )
    if cfg.STT_MODE == "continuous":
        return STTContinuousNode(
            bus,
            model_name=cfg.STT_CONTINUOUS_MODEL,
            require_wake_word=cfg.REQUIRE_WAKE_WORD,
            wake_phrases=cfg.WAKE_PHRASES,
            wake_match_threshold=cfg.WAKE_MATCH_THRESHOLD,
            followup_window_s=cfg.FOLLOWUP_WINDOW_S,
            sample_rate=cfg.SAMPLE_RATE,
            block_ms=cfg.STT_BLOCK_MS,
            phrase_timeout_s=cfg.STT_PHRASE_TIMEOUT_S,
            max_phrase_s=cfg.STT_MAX_PHRASE_S,
            transcribe_every_s=cfg.STT_TRANSCRIBE_EVERY_S,
            min_transcribe_s=cfg.STT_MIN_TRANSCRIBE_S,
            energy_threshold=cfg.STT_ENERGY_THRESHOLD,
            post_padding_ms=cfg.POST_PADDING_MS,
            duplicate_similarity=cfg.STT_DUPLICATE_SIMILARITY,
            input_device=cfg.INPUT_DEVICE,
        )
    raise NotImplementedError(f"STT_MODE={cfg.STT_MODE!r} not implemented yet")


def main() -> int:
    bus = Bus()

    # Backend + LLM node first — the load+warm pause is several seconds and
    # we don't want the mic running while we wait.
    backend = make_backend()
    llm = LLMNode(
        bus,
        backend,
        cfg.SYSTEM_PROMPT,
        max_tokens=cfg.LLM_MAX_TOKENS,
        temperature=cfg.LLM_TEMPERATURE,
        top_p=cfg.LLM_TOP_P,
        max_history_turns=cfg.MAX_HISTORY_TURNS,
        ctx_limit=cfg.LLM_CTX,
    )
    llm.load_and_warm()

    # TTS warm-up is also slow (~3-5 s) — pay it now.
    tts = KokoroNode(
        bus,
        voice=cfg.KOKORO_VOICE,
        lang=cfg.KOKORO_LANG,
        sr=cfg.TTS_SAMPLE_RATE,
        min_chars=cfg.TTS_MIN_CHARS,
        tail_sleep_s=cfg.TTS_TAIL_SLEEP_S,
        output_device=cfg.OUTPUT_DEVICE,
    )

    # STT last — once it starts, the mic is open.
    stt = make_stt(bus)

    # Pause the mic synchronously when TTS starts playing — the mic.pause
    # bus message only lands after the orchestrator's next dispatch, by
    # which time the first TTS samples are already in the air.
    tts.pause_mic = stt.set_paused

    if cfg.REQUIRE_WAKE_WORD:
        print(f"[ready] say one of: {', '.join(cfg.WAKE_PHRASES)}", flush=True)

    Orchestrator(bus, llm, tts, stt).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
