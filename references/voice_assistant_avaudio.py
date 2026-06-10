#!/usr/bin/env python3
"""Google-Home-style local voice assistant — AVAUDIO BRIDGE TEST.

Same pipeline as ``voice_assistant.py`` but with the audio I/O layer
routed through ``jaeger_os.plugins.avaudio_io`` (PyObjC AVAudioEngine)
instead of sounddevice/PortAudio.

Purpose: isolate the bridge from the full ``voice_loop.py`` stack so
we can see whether a *simple linear demo* hits the same CoreAudio
2003329396 ('what') error.

  - If this demo WORKS → ``voice_loop.py`` has an integration race
    (load order, multiple engines fighting, daemon attach state,
    etc.).  The bridge itself is fine.
  - If this demo ALSO FAILS → the bridge has a deeper conflict with
    pywhispercpp / ggml-metal init (Metal context contention) and
    needs more work before it can replace sounddevice anywhere.

Run::

    cd /Users/.../JROS
    source dev_scripts/dev_env.sh   # only needed if testing daemon flow
    PYTHONPATH=. .venv/bin/python voice_assistant_avaudio.py
"""

from __future__ import annotations

import collections
import queue
import re
import sys
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
# NOTE: ``sounddevice`` is intentionally NOT imported here — the whole
# point of this test file is to route I/O through avaudio_io instead.
# References to "sd." in docstrings below refer to the original
# voice_assistant.py we forked from.
import webrtcvad


# ── config ─────────────────────────────────────────────────────────────
# Resolve the LLM model path relative to this script so the demo runs
# on any clone of the JROS repo without hand-editing.  Picks up the
# Gemma weights that ship under ``jaeger_os/models/``.  Falls back to
# the operator's LM Studio cache if the in-repo path is missing.
_REPO_ROOT = Path(__file__).resolve().parent
_IN_REPO_MODEL = _REPO_ROOT / "jaeger_os" / "models" / "gemma-4-26B-A4B-it-Q4_K_M.gguf"
_LMSTUDIO_MODEL = Path(
    "/Users/jonathanjenkins/.lmstudio/models/lmstudio-community/"
    "gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf"
)
LLM_MODEL_PATH = _IN_REPO_MODEL if _IN_REPO_MODEL.exists() else _LMSTUDIO_MODEL

# Two-pass STT: fast model runs every phrase to detect wake words. Accurate
# model only re-transcribes when wake matches or we're in follow-up mode,
# so we don't pay its cost on background noise.
STT_FAST = "base.en"
STT_ACCURATE = "medium.en"

KOKORO_VOICE = "af_heart"
KOKORO_LANG = "a"

# Whisper often mishears "jaeger" — covering common phonetic transcriptions
# so any of yeager/yager/jager/jaeger triggers the wake.
_WAKE_PREFIXES = ("ok", "okay", "hey")
_ASSISTANT_NAMES = ("jaeger", "yeager", "yager", "jager")
WAKE_PHRASES = tuple(f"{p} {n}" for p in _WAKE_PREFIXES for n in _ASSISTANT_NAMES)
WAKE_MATCH_THRESHOLD = 0.78
FOLLOWUP_WINDOW_S = 10.0      # listen this long after a reply, no wake needed

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 480 samples
VAD_AGGRESSIVENESS = 2

PRE_ROLL_MS = 240             # capture speech onset
POST_PADDING_MS = 250         # capture trailing word — fixes "what time is it" → "time is in"
SILENCE_HANGOVER_MS = 700     # match the working command listener
MIN_SPEECH_MS = 400
MAX_SPEECH_MS = 8000

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Answer in 1–2 short sentences "
    "in plain conversational English. No markdown, no code blocks, no "
    "emojis, no lists. If you don't know, say so briefly."
)

# Cap rolling chat history so prompt size (and latency) stays bounded.
MAX_HISTORY_TURNS = 8


# ── short chime so the user knows we're listening ──────────────────────
def make_beep(freq: float = 880.0, duration_ms: int = 110,
              sr: int = 24000, amp: float = 0.25) -> np.ndarray:
    n = int(sr * duration_ms / 1000)
    t = np.arange(n) / sr
    # short fade-in/out to avoid clicks
    env = np.minimum(np.minimum(t / 0.01, 1.0), (duration_ms / 1000 - t) / 0.01).clip(0, 1)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


BEEP = make_beep()
DOUBLE_BEEP = np.concatenate([
    make_beep(freq=660, duration_ms=80),
    np.zeros(int(24000 * 0.05), dtype=np.float32),
    make_beep(freq=880, duration_ms=80),
])


# ── audio capture ──────────────────────────────────────────────────────
class MicStream:
    """AVAudioEngine input tap → frame queue, with a pause flag for TTS.

    Drop-in for the original sounddevice-based MicStream — same API
    (``q``, ``paused``, ``__enter__`` / ``__exit__``, ``drain``) so the
    rest of the demo doesn't have to change.
    """

    def __init__(self) -> None:
        from jaeger_os.plugins.avaudio_io import InputStream as _AVInputStream

        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.paused = False
        # voice_processing=True flips on AVAudioEngine's built-in
        # AEC + noise suppression + AGC — Apple's pre-canned voice
        # pipeline (what FaceTime uses).  Without it, the speakers
        # playing the assistant's reply get captured by the mic and
        # Whisper transcribes the agent talking to itself as a fresh
        # "user" command in the follow-up window.  AEC subtracts the
        # speaker output from the mic input in real time so we hear
        # only what the operator says, not what we just said.
        #
        # This is the speexdsp replacement called out in the 0.3.0
        # pivot plan — pure Python + Apple AEC, no native lib needed.
        self._stream = _AVInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=self._cb,
            voice_processing=True,
        )

    def _cb(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        if self.paused or frames != FRAME_SAMPLES:
            return
        self.q.put(indata.copy())

    def __enter__(self) -> "MicStream":
        print("[mic] starting AVAudioEngine input...", flush=True)
        self._stream.start()
        print("[mic] AVAudioEngine input ready", flush=True)
        return self

    def __exit__(self, *exc) -> None:
        self._stream.stop()
        self._stream.close()

    def drain(self) -> None:
        with self.q.mutex:
            self.q.queue.clear()


# ── VAD worker thread ──────────────────────────────────────────────────
class VadWorker(threading.Thread):
    """Reads audio blocks, runs VAD, accumulates phrases, fast-transcribes,
    then pushes (audio_float32, fast_transcript) onto phrase_queue.
    """

    def __init__(self, mic: MicStream, fast_model,
                 phrase_queue: queue.Queue, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.mic = mic
        self.fast_model = fast_model
        self.phrase_queue = phrase_queue
        self.stop_event = stop_event
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        self.silence_blocks_to_end = max(1, SILENCE_HANGOVER_MS // FRAME_MS)
        self.min_speech_blocks = max(1, MIN_SPEECH_MS // FRAME_MS)
        self.max_speech_blocks = max(self.min_speech_blocks, MAX_SPEECH_MS // FRAME_MS)
        self.pre_roll_blocks = max(0, PRE_ROLL_MS // FRAME_MS)
        self.post_pad_samples = int(SAMPLE_RATE * POST_PADDING_MS / 1000)

        # Exposed so the main loop can avoid expiring the follow-up window
        # while the user is still mid-sentence.  We publish two flags:
        #   in_utterance  — RAW signal, True from the first VAD-positive
        #                   frame until SILENCE_HANGOVER_MS of silence.
        #                   Used by the main loop's expiry guard so
        #                   speech that starts right at the deadline
        #                   gets a chance to finish even if it hasn't
        #                   yet crossed the MIN_SPEECH_MS confidence
        #                   threshold.
        #   in_speech     — Confidence-gated: only True once we have
        #                   sustained voice past MIN_SPEECH_MS.
        self.in_utterance = False
        self.in_speech = False

    def _is_speech(self, chunk: np.ndarray) -> bool:
        pcm = (chunk[:, 0] * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        return self.vad.is_speech(pcm, SAMPLE_RATE)

    def _finalize(self, chunks: list[np.ndarray]) -> None:
        audio = np.concatenate(chunks, axis=0).astype(np.float32).reshape(-1)
        audio = np.concatenate([audio, np.zeros(self.post_pad_samples, dtype=np.float32)])
        try:
            segments = self.fast_model.transcribe(audio, language="en")
            text = " ".join(s.text for s in segments).strip()
        except Exception as exc:
            print(f"[stt-fast] {exc}", file=sys.stderr)
            text = ""
        if text:
            self.phrase_queue.put((audio, text))

    def run(self) -> None:
        pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=self.pre_roll_blocks)
        speech: list[np.ndarray] = []
        speech_blocks = 0
        silent_blocks = 0
        in_speech = False

        while not self.stop_event.is_set():
            try:
                chunk = self.mic.q.get(timeout=0.3)
            except queue.Empty:
                continue

            is_speech = self._is_speech(chunk)

            if is_speech:
                if not in_speech:
                    speech = list(pre_roll)
                    speech_blocks = len(speech)
                    silent_blocks = 0
                    in_speech = True
                speech.append(chunk)
                speech_blocks += 1
                silent_blocks = 0
            elif in_speech:
                speech.append(chunk)
                silent_blocks += 1
            else:
                pre_roll.append(chunk)

            # Publish both signals.  See __init__ for the contract.
            self.in_utterance = in_speech and silent_blocks < self.silence_blocks_to_end
            self.in_speech = self.in_utterance and speech_blocks >= self.min_speech_blocks

            phrase_done = in_speech and speech_blocks >= self.min_speech_blocks and (
                silent_blocks >= self.silence_blocks_to_end
                or speech_blocks >= self.max_speech_blocks
            )
            if phrase_done:
                self._finalize(speech)
                speech = []
                speech_blocks = 0
                silent_blocks = 0
                in_speech = False
                self.in_speech = False
                self.in_utterance = False
                pre_roll.clear()


# ── whisper noise filter ───────────────────────────────────────────────
# Whisper, when forced to transcribe silence or non-speech audio, emits
# bracketed / parenthetical markers (BLANK_AUDIO, no_speech, beep,
# beeping, music, applause, laughter, sigh, breathing, sniff, …) which
# we must drop in the follow-up window — otherwise clicks, breathing,
# lip smacks, and the assistant's own playback tail burn turns.
#
# The check is wrapper-tolerant: we strip ``[...]`` / ``(...)`` and
# compare the INNER token to an allowlist.  That keeps real wrapped
# user responses like "(yes)" or "[no]" flowing through to the LLM
# instead of getting silently dropped as "noise".
_NON_SPEECH_MARKERS = {
    "blank_audio", "no_speech", "beep", "beeping",
    "music", "applause", "laughter", "sigh", "sniff",
    "breathing", "background noise", "silence",
}
_WRAPPED_RE = re.compile(r"^\s*[\[\(]([^\]\)]{1,40})[\]\)]\s*[.!,?]*\s*$")


def is_non_speech_marker(text: str) -> bool:
    """True if Whisper transcribed silence / noise rather than real speech.

    Strips bracket/paren wrappers (if present) and matches the inner
    token against the known-marker allowlist.  Free text and wrapped
    real words like "(yes)" / "[no]" are treated as legitimate
    commands.
    """
    s = (text or "").strip()
    if not s:
        return True
    m = _WRAPPED_RE.match(s)
    if m:
        inner = m.group(1).lower().strip(".!?, ")
        return inner in _NON_SPEECH_MARKERS
    lowered = s.lower().strip(".!?, ")
    return lowered in _NON_SPEECH_MARKERS


# ── wake-word logic ────────────────────────────────────────────────────
def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def find_wake(text: str) -> tuple[bool, str]:
    """Return (matched, remainder_after_wake_phrase).

    Two-pass: exact token-window match first so word boundaries are
    enforced ("book yager …" must NOT hijack "ok yager"), then a
    fuzzy fallback for phonetic drift (Whisper sometimes renders
    "okay" as "ok" etc.).
    """
    norm = normalize(text)
    tokens = norm.split()
    # Pass 1: exact token-window match — strict word boundaries.
    for phrase in WAKE_PHRASES:
        n = len(phrase.split())
        for i in range(0, max(0, len(tokens) - n + 1)):
            window = " ".join(tokens[i:i + n])
            if window == phrase:
                return True, " ".join(tokens[i + n:]).strip()
    # Pass 2: fuzzy fallback for slight phonetic drift.
    for phrase in WAKE_PHRASES:
        n = len(phrase.split())
        for i in range(0, max(0, len(tokens) - n + 1)):
            window = " ".join(tokens[i:i + n])
            if SequenceMatcher(None, window, phrase).ratio() >= WAKE_MATCH_THRESHOLD:
                return True, " ".join(tokens[i + n:]).strip()
    return False, ""


# ── LLM ────────────────────────────────────────────────────────────────
def load_llm():
    from llama_cpp import Llama
    if not LLM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Gemma model not found at {LLM_MODEL_PATH}. "
            "Expected the in-repo weights under jaeger_os/models/. "
            "Run the JROS setup wizard or copy the GGUF there."
        )
    print(f"[llm] loading {LLM_MODEL_PATH.name}...", flush=True)
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(LLM_MODEL_PATH),
        n_ctx=4096, n_gpu_layers=-1, verbose=False,
    )
    print(f"[llm] loaded in {time.perf_counter()-t0:.1f}s, warming up...", flush=True)
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}], max_tokens=1, temperature=0.0,
    )
    print("[llm] ready", flush=True)
    return llm


def trim_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    # history[0] is system; the rest is user/assistant pairs appended by think().
    # Slicing from the tail in pairs keeps the boundary on a user message.
    if len(history) <= 1 + max_turns * 2:
        return history
    return history[:1] + history[-max_turns * 2:]


def think(llm, history: list[dict], user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    out = llm.create_chat_completion(
        messages=history, max_tokens=200, temperature=0.7, top_p=0.95,
    )
    reply = out["choices"][0]["message"]["content"].strip()
    history.append({"role": "assistant", "content": reply})
    return clean_for_tts(reply)


def clean_for_tts(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^[\-\*\d\.\)]+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# ── TTS ────────────────────────────────────────────────────────────────
def load_tts():
    from kokoro import KPipeline
    print("[tts] loading Kokoro...", flush=True)
    t0 = time.perf_counter()
    pipe = KPipeline(lang_code=KOKORO_LANG)
    list(pipe("Ready.", voice=KOKORO_VOICE))     # warm-up
    print(f"[tts] ready ({time.perf_counter()-t0:.1f}s)", flush=True)
    return pipe


def drain_phrase_queue(q: queue.Queue) -> None:
    """Discard phrases the VAD finalized while we were speaking — otherwise the
    follow-up window would treat stale buffered speech as a fresh command."""
    with q.mutex:
        q.queue.clear()


class SessionPlayer:
    """Persistent AVAudioEngine + AVAudioPlayerNode for the whole session.

    Industry-standard pattern — one engine, opened at session start,
    closed at session shutdown.  Audio is delivered through the
    player node's internal schedule queue: ``enqueue(buffer)`` is
    non-blocking (it just hands the buffer to the player), and the
    player plays them back-to-back as the audio hardware asks for
    samples.  No engine churn between utterances, no per-call
    teardown, no power-cycle clicks at the device.

    This is how AVAudioEngine is *meant* to be used.  Apple's sample
    code, FaceTime, Voice Memos all follow this shape.

    API mirrors ``voice_assistant_persistent.PersistentPlayer``:

        player = SessionPlayer(samplerate=24000)
        player.start()
        ...
        player.enqueue(chunk)              # non-blocking
        player.enqueue(next_chunk)         # streams behind the first
        player.wait_until_drained()        # block until everything plays
        ...
        player.close()                     # only at app shutdown

    ``play_blocking(audio)`` is the convenience wrapper for the common
    "play this whole buffer, wait for it" pattern.
    """

    def __init__(self, *, samplerate: int = 24000, channels: int = 1) -> None:
        import AVFoundation  # type: ignore
        self._av = AVFoundation
        self._samplerate = float(samplerate)
        self._channels = int(channels)
        self._engine = None
        self._player = None
        self._format = None
        self._running = False
        # Drain tracking via per-buffer completion handlers.  Counting
        # callbacks beats reading ``sampleTime()`` because the player
        # node's sample clock is absolute (it ticks from the moment
        # ``play()`` is called), so comparing it to a per-utterance
        # scheduled-sample total returns "drained" instantly on the
        # second utterance once the absolute clock has raced ahead.
        # AVAudioEngine fires the completion block when the buffer has
        # actually been consumed by the player — exactly the signal
        # we need.
        self._scheduled_count = 0
        self._drained_count = 0
        self._drain_lock = threading.Lock()
        self._drain_event = threading.Event()
        self._drain_event.set()           # nothing scheduled → drained
        # Keep PyObjC block wrappers alive until they fire.  Without
        # this list the Python callables get GC'd between schedule
        # and completion and AVAudioEngine fires into freed memory.
        self._pending_callbacks: list = []

    def start(self) -> None:
        if self._running:
            return
        av = self._av
        engine = av.AVAudioEngine.alloc().init()
        player = av.AVAudioPlayerNode.alloc().init()
        fmt = av.AVAudioFormat.alloc(
        ).initWithCommonFormat_sampleRate_channels_interleaved_(
            av.AVAudioPCMFormatFloat32,
            self._samplerate,
            self._channels,
            False,
        )
        engine.attachNode_(player)
        engine.connect_to_format_(player, engine.mainMixerNode(), fmt)
        success, err = engine.startAndReturnError_(None)
        if not success:
            raise RuntimeError(
                f"SessionPlayer engine start failed: "
                f"{err.localizedDescription() if err else 'unknown'}"
            )
        player.play()  # node starts in "playing" state, drains buffers as they arrive
        self._engine = engine
        self._player = player
        self._format = fmt
        self._running = True

    def enqueue(self, audio: np.ndarray) -> None:
        """Append a buffer to the player node's schedule queue.
        Non-blocking — caller can keep synthesizing more chunks while
        the player is still draining earlier ones.  This is what makes
        streaming TTS sound seamless (no audible gaps between chunks)."""
        if not self._running:
            self.start()
        if audio is None or len(audio) == 0:
            return
        av = self._av
        n = int(len(audio))
        pcm = av.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
            self._format, n,
        )
        pcm.setFrameLength_(n)
        floats = pcm.floatChannelData()
        ch0 = floats[0]
        # Slice-assign — write into the AVAudioPCMBuffer's mutable
        # float storage (same pattern InputStream uses for reading).
        ch0[0:n] = np.asarray(audio, dtype=np.float32).tolist()

        # Increment scheduled count + clear drain event BEFORE
        # scheduling.  If we did this after, the completion handler
        # for a previous buffer could fire and (wrongly) set the
        # event while we're mid-enqueue.
        with self._drain_lock:
            self._scheduled_count += 1
            self._drain_event.clear()

        def _on_done() -> None:
            with self._drain_lock:
                self._drained_count += 1
                # Drop our own callback ref so the closure (and the
                # AVAudioPCMBuffer it implicitly retains via PyObjC
                # block capture) can be released.
                try:
                    self._pending_callbacks.remove(_on_done)
                except ValueError:
                    pass
                if self._drained_count >= self._scheduled_count:
                    self._drain_event.set()

        # Hold a strong reference to the Python callable for the life
        # of the scheduled buffer — PyObjC's block bridge does NOT
        # keep Python objects alive on its own.
        self._pending_callbacks.append(_on_done)
        # 2-arg variant.  The 4-arg ``…atTime:options:completionHandler:``
        # crashes PyObjC signature inference on the block argument.
        # The 2-arg form's handler fires when the buffer has been
        # consumed by the player node — exactly the drain signal we
        # need.
        try:
            self._player.scheduleBuffer_completionHandler_(pcm, _on_done)
        except Exception:
            # Roll back the counter + callback ref + event state.  If a
            # schedule raises after we've incremented ``_scheduled_count``
            # and the handler will never fire, every future
            # ``wait_until_drained`` would time out (drained can never
            # catch scheduled).  Re-balance so the player can recover
            # and continue working for the next utterance.
            with self._drain_lock:
                self._scheduled_count -= 1
                try:
                    self._pending_callbacks.remove(_on_done)
                except ValueError:
                    pass
                if self._drained_count >= self._scheduled_count:
                    self._drain_event.set()
            raise

    def mark_end(self) -> None:
        """No-op for the AVAudioEngine path — each scheduled buffer
        fires its own completion handler when the player node consumes
        it, and ``wait_until_drained`` blocks on a threading.Event
        those handlers signal.  Kept on the API for symmetry with
        ``PersistentPlayer.mark_end``."""

    def wait_until_drained(self, timeout: float = 60.0) -> bool:
        """Block until every buffer ever ``enqueue``-d has fired its
        completion handler — i.e. the player node has finished
        consuming them.  Returns False on timeout.

        Counting per-buffer completions is the canonical Apple pattern
        for "wait for playback".  Reading ``sampleTime()`` instead
        gets the *absolute* sample clock of the player node (which
        starts ticking when ``play()`` was called at session start),
        so on the second utterance it's already huge and the drain
        check would return instantly.

        On timeout, callers SHOULD call ``reset()`` — otherwise a
        dropped completion handler poisons future drains because
        ``_drained_count`` can never catch ``_scheduled_count``.
        """
        if self._player is None:
            return True
        return self._drain_event.wait(timeout=timeout)

    def reset(self) -> None:
        """Recover the drain accounting after a wedged
        ``wait_until_drained``.  Drops retained completion-handler
        refs and rebalances ``_drained_count``/``_drain_event`` so
        subsequent enqueues work normally.

        The audio engine + player node are NOT cycled — they stay
        alive for the rest of the session.  This is the light-touch
        recovery path; tearing down and rebuilding the engine is
        exactly what we built SessionPlayer to avoid.
        """
        with self._drain_lock:
            self._pending_callbacks.clear()
            self._drained_count = self._scheduled_count
            self._drain_event.set()

    def play_blocking(self, audio: np.ndarray) -> None:
        """Convenience: ``enqueue`` + ``mark_end`` + ``wait_until_drained``.
        Mirrors ``PersistentPlayer.play_blocking`` so the call sites
        in this file are identical across the two backends.
        ``mark_end`` is a no-op here but called for surface parity."""
        self.enqueue(audio)
        self.mark_end()
        self.wait_until_drained()

    def close(self) -> None:
        if not self._running:
            return
        try:
            if self._player is not None:
                self._player.stop()
            if self._engine is not None:
                self._engine.stop()
        except Exception:
            pass
        self._running = False
        self._engine = None
        self._player = None
        self._format = None


def play_audio_with_mic_paused(
    mic: MicStream, player: SessionPlayer, audio: np.ndarray,
) -> None:
    """Play through the persistent SessionPlayer; mic capture is
    suppressed so we don't transcribe ourselves."""
    mic.paused = True
    try:
        player.play_blocking(audio)
        time.sleep(0.12)              # let the speaker drain
    finally:
        mic.drain()
        mic.paused = False


def speak(pipe, mic: MicStream, player: SessionPlayer, text: str) -> bool:
    """Stream Kokoro chunks through the persistent SessionPlayer.

    Each synthesized chunk is ``enqueue``-d on the player node AS
    IT'S PRODUCED — the audio thread drains them back-to-back without
    cycling the device between chunks.  Kokoro can be synthesizing
    chunk N+1 while the player is still draining chunk N (true
    streaming).  When the pipeline finishes producing,
    ``wait_until_drained`` blocks here until the player has actually
    finished rendering.

    Returns True if audio was actually played, False if Kokoro yielded
    no usable chunks for this reply.  Callers should treat False as
    "the user got no feedback" and skip the follow-up open.

    Mic stays paused across synth + playback so we never re-capture
    our own voice.
    """
    if not text:
        return False
    mic.paused = True
    queued = False
    try:
        for r in pipe(text, voice=KOKORO_VOICE):
            if r.audio is None:
                continue
            chunk = np.asarray(r.audio, dtype=np.float32)
            player.enqueue(chunk)
            queued = True
        if not queued:
            print(f"[tts] WARN: Kokoro produced no audio for {text!r}",
                  file=sys.stderr, flush=True)
            return False
        player.mark_end()
        if not player.wait_until_drained():
            # Dropped completion handler — recover the accounting so
            # the next utterance isn't poisoned by stale state.
            print("[av-session] WARN: drain timeout — resetting player accounting",
                  file=sys.stderr, flush=True)
            player.reset()
        time.sleep(0.12)              # let the speaker drain
        return True
    finally:
        mic.drain()
        mic.paused = False


# ── main loop ──────────────────────────────────────────────────────────
def warm_stt(model, label: str) -> None:
    """Prime pywhispercpp once so the first real phrase avoids setup latency."""
    warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    print(f"[{label}] warming up...", flush=True)
    t0 = time.perf_counter()
    try:
        list(model.transcribe(warm_audio, language="en"))
    except Exception as exc:
        # Some Whisper builds dislike pure silence. Startup should continue;
        # the model is still loaded and ready for normal speech.
        print(f"[{label}] warm-up skipped: {exc}", file=sys.stderr, flush=True)
    else:
        print(f"[{label}] primed ({time.perf_counter()-t0:.1f}s)", flush=True)


def main() -> int:
    from pywhispercpp.model import Model as STTModel

    print(f"[stt-fast] loading {STT_FAST}...", flush=True)
    t0 = time.perf_counter()
    fast_stt = STTModel(
        STT_FAST, print_realtime=False, print_progress=False,
        single_segment=True, no_context=True,
    )
    print(f"[stt-fast] ready ({time.perf_counter()-t0:.1f}s)", flush=True)
    warm_stt(fast_stt, "stt-fast")

    print(f"[stt-accurate] loading {STT_ACCURATE}...", flush=True)
    t0 = time.perf_counter()
    accurate_stt = STTModel(
        STT_ACCURATE, print_realtime=False, print_progress=False,
        single_segment=True, no_context=True,
    )
    print(f"[stt-accurate] ready ({time.perf_counter()-t0:.1f}s)", flush=True)
    warm_stt(accurate_stt, "stt-accurate")

    def transcribe_accurate(audio: np.ndarray) -> str:
        segments = accurate_stt.transcribe(audio, language="en")
        return " ".join(s.text for s in segments).strip()

    llm = load_llm()
    tts = load_tts()
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    phrase_queue: queue.Queue[tuple[np.ndarray, str]] = queue.Queue()
    stop_event = threading.Event()

    print(f"\n[ready] say one of: {', '.join(WAKE_PHRASES)} — Ctrl-C to quit.\n")

    state = "WAKE"           # "WAKE" or "FOLLOWUP"
    followup_deadline = 0.0

    # 0.3.0 persistent-pipeline pattern: ONE SessionPlayer opens here
    # and lives for the whole session.  Closed in the outer finally
    # so a clean exit releases the AVAudioEngine resources before the
    # process tears down.  Symmetric with ``voice_assistant_persistent``.
    player = SessionPlayer(samplerate=24000)
    player.start()
    print("[output] persistent AVAudioEngine session open — stays alive for the session.",
          flush=True)

    with MicStream() as mic:
        worker = VadWorker(mic, fast_stt, phrase_queue, stop_event)
        worker.start()
        try:
            while True:
                try:
                    audio, fast_text = phrase_queue.get(timeout=0.3)
                except queue.Empty:
                    # No phrase pending — ONLY now is it safe to expire
                    # the follow-up window.  Checking before ``get`` lost
                    # races where VadWorker finalized a phrase straddling
                    # the deadline and we'd then misclassify it as
                    # WAKE-mode below.  Triple-guard so we also wait out
                    # any in-progress utterance AND any final phrase
                    # that hasn't been popped off the queue yet.
                    if (
                        state == "FOLLOWUP"
                        and time.time() > followup_deadline
                        and not worker.in_utterance
                        and phrase_queue.empty()
                    ):
                        print("[follow-up window expired — say wake word again]")
                        state = "WAKE"
                    continue

                print(f"[heard]  {fast_text!r}")

                # Decide whether to act
                if state == "FOLLOWUP":
                    # In follow-up window any utterance counts as a command —
                    # EXCEPT Whisper's non-speech markers ([BLANK_AUDIO],
                    # (beep), (music), etc.) which Whisper emits when forced
                    # to transcribe silence or clicks.  Otherwise the agent
                    # burns turns replying to its own playback tail.
                    command = transcribe_accurate(audio).strip() or fast_text
                    if is_non_speech_marker(command):
                        print(f"[follow-up skipped — non-speech: {command!r}]")
                        continue
                    print(f"[follow-up command]  {command!r}")
                else:
                    matched, remainder = find_wake(fast_text)
                    if not matched:
                        continue

                    # Re-transcribe the same audio with the accurate model
                    accurate_text = transcribe_accurate(audio)
                    a_matched, a_remainder = find_wake(accurate_text)
                    if a_matched and (a_remainder or not remainder):
                        remainder = a_remainder
                        print(f"[heard*] {accurate_text!r}")

                    if remainder:
                        command = remainder
                    else:
                        # Wake-only utterance: chime, then wait for the command.
                        play_audio_with_mic_paused(mic, player, BEEP)
                        drain_phrase_queue(phrase_queue)
                        try:
                            cmd_audio, cmd_fast = phrase_queue.get(timeout=6.0)
                        except queue.Empty:
                            print("[no command — back to wake]")
                            continue
                        print(f"[heard]  {cmd_fast!r}")
                        command = transcribe_accurate(cmd_audio).strip() or cmd_fast

                if not command:
                    continue

                # Think
                print(f"[think]  {command!r}")
                t0 = time.perf_counter()
                reply = think(llm, history, command)
                history = trim_history(history)
                print(f"[reply]  {reply!r}  ({time.perf_counter()-t0:.2f}s)")

                # Speak (mic paused inside)
                spoke = speak(tts, mic, player, reply)
                drain_phrase_queue(phrase_queue)
                if not spoke:
                    # Kokoro produced no audio — the operator got no
                    # feedback, so opening a follow-up window they
                    # can't see would just confuse the flow.  Back to
                    # WAKE; they'll re-trigger when ready.
                    print("[follow-up skipped — no TTS audio]")
                    state = "WAKE"
                    continue

                # Open follow-up window
                play_audio_with_mic_paused(mic, player, DOUBLE_BEEP)
                drain_phrase_queue(phrase_queue)
                state = "FOLLOWUP"
                followup_deadline = time.time() + FOLLOWUP_WINDOW_S
                print(f"[follow-up open for {FOLLOWUP_WINDOW_S:.0f}s — keep talking]")

        except KeyboardInterrupt:
            print("\n[bye]")
            return 0
        finally:
            stop_event.set()
            worker.join(timeout=2)
            # Release the persistent AVAudioEngine — symmetric with
            # ``voice_assistant_persistent.main``'s ``player.close()``.
            player.close()


if __name__ == "__main__":
    raise SystemExit(main())
