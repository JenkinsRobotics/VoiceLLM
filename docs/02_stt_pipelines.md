# STT pipelines — pywhispercpp strategies

This is the most important file in this folder. Continuous, human-style hearing
is the part of the system most likely to need iteration, so we are *not* hiding
it behind one implementation. The four strategies below come straight from
`MockingAgent/PywisperCpp/pywhispercpp_examples/llm_listener/`. We will port
each one to a node form (publishes `stt.text` on the bus) and pick a default
empirically.

The reference files:

- `MockingAgent/PywisperCpp/pywhispercpp_examples/llm_listener/always_listening_phrase_buffer_pipeline.py`
- `MockingAgent/PywisperCpp/pywhispercpp_examples/llm_listener/always_listening_word_cursor_pipeline.py`
- `MockingAgent/PywisperCpp/pywhispercpp_examples/llm_listener/always_listening_hybrid_phrase_word_pipeline.py`
- `MockingAgent/PywisperCpp/pywhispercpp_examples/llm_listener/human_style_overlapping_memory_pipeline.py`
- `MockingAgent/voice_assistant.py` (VAD-segmented two-pass, the proven baseline)

## Strategy comparison

| Pipeline | How it builds the transcript | When it commits | Strengths | Weaknesses |
|---|---|---|---|---|
| **VAD-segmented (baseline)** | WebRTC VAD finds speech start/end, transcribe the slice once. | When VAD closes a phrase. | Cheap, low-latency for clean rooms. | Cuts off trailing words (we already saw "what time is it" → "time is in"); misses overlapping speech; binary on/off feel. |
| **Phrase buffer** | Continuously append blocks to a phrase buffer, retranscribe live every ~1.2 s. Commit when buffer goes quiet. | Activity quiet `PHRASE_TIMEOUT_SECONDS` *or* `MAX_PHRASE_SECONDS` reached. | Doesn't lose trailing words; live partials available. | Re-transcribes the same audio many times → CPU heavy; whole phrase replaced on each pass. |
| **Word cursor** | Rolling 18 s memory, transcribe last 7 s every 1.2 s, commit only the *new* words past a karaoke cursor. | Whenever new words appear past the cursor. | Steady karaoke-style commit, no big block rewrites. | Cursor advancement can drift; needs overlap-matching tuning. |
| **Hybrid phrase/word** | Phrase buffer for capture, word cursor for committing inside the buffer. | Commits new words during the phrase; phrase resets on quiet. | Best of both — fewer rewrites, phrase boundaries still respected. | Most code, most knobs. |
| **Human-style overlapping memory** | 28 s rolling memory, retranscribe overlapping windows, advance a long word timeline. | Continuous word advance with stable-repeat heuristics. | Closest to "always hearing" feel; survives noisy rooms best. | Heaviest CPU; latency to "this is committed" is the highest. |

## Default for VoiceLLM v1

**Shipped status:**
- ✅ `stt/stt_two_pass.py` — VAD two-pass, default for `STT_MODE = "two_pass"`.
- ✅ `stt/stt_continuous.py` — hybrid phrase/word, opt-in via `STT_MODE = "continuous"` (M3.5).

The default stays `"two_pass"` because it's the proven baseline and the
M3 quick path runs cleanly on top of it (orchestrator self-speech filter +
LLM gate handle the noise rejection that the wake word used to do). Flip
to `"continuous"` when you want lower-latency phrase commit on short
utterances; revert via the same single line if anything regresses.

Other strategies (phrase buffer, word cursor, human-style overlapping
memory) remain unported. They can each become a `stt/*.py` node with the
same interface (`start`/`stop`/`set_paused`/`open_followup`,
publishes `stt.text`) if a future need surfaces.

## Open trial: ggml-large-v3-turbo for accurate pass

`voice_assistant.py` uses `base.en` for fast pass and `medium.en` for accurate.
On Apple Silicon, `large-v3-turbo` is roughly the same speed as `medium` and
materially more accurate for natural speech. Worth a head-to-head once the
pipeline is wired.

## Latency budget (target)

For the "ChatGPT Voice" feel we want, end-to-end (user stops talking →
first audible TTS phoneme) under ~1500 ms. Rough breakdown:

| Stage | Budget | Notes |
|---|---|---|
| VAD close → STT result | ≤300 ms | base.en single-segment on M-series |
| STT result → LLM first token (TTFT) | ≤500 ms | Gemma 4 26B-A4B 4-bit MLX on warm cache |
| LLM first sentence → Kokoro audio out | ≤700 ms | first sentence is short; Kokoro warm |
| Total | ≤1500 ms | |

If we miss this, suspects in order: STT model size, KV cache cold, Kokoro
voice cold, sentence-boundary delay in TTS.

## Open questions for STT

- Do we publish `stt.partial` (live transcript) at all, or only commit
  events? Partial events are great for a UI but pointless for the LLM path.
  Currently no; revisit if/when a GUI ships.
- ~~How do we suppress committing the assistant's own audio?~~
  **Resolved:** mic-pause during TTS (Layer A) + similarity filter
  (Layer C) + LLM gate (Layer D). See
  [05_barge_in_and_self_speech.md](05_barge_in_and_self_speech.md).
- ~~Wake-word "engaged mode" toggle?~~ **Resolved:**
  `REQUIRE_WAKE_WORD = False` is the M3 default; flip to `True` to
  restore the Google-Home-style flow. The LLM gate replaces the
  "engaged mode" idea — the LLM itself decides whether each phrase is
  directed at it.
