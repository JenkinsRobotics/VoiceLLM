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

Start with **hybrid phrase/word** as the primary `stt_continuous.py` because:
- it preserves trailing words (the bug that bit `voice_assistant.py`),
- karaoke commits are friendly to a streaming LLM (we can fire `llm.request`
  on phrase close, then update if revised before the LLM reads it),
- it's the strategy already validated as "best of both" in the demo notes.

Keep the **VAD two-pass** strategy from `voice_assistant.py` available as
`stt_two_pass.py` — it's still the right answer when we want a wake word.
A `config.STT_MODE` flag picks which gets instantiated.

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

- Do we publish `stt.partial` (live live transcript) at all, or only commit
  events? Partial events are great for a UI but pointless for the LLM path.
- How do we suppress committing the assistant's own audio? Mic-pause during
  TTS handles 90% of it; AEC handles overlap; do we also need a soft
  similarity filter against the most recent reply?
- Wake-word "engaged mode" toggle: should we have a soft hotword that only
  switches us *into* a window where any utterance is a turn (Alexa-style)?
  Or do we just trust VAD + barge-in and commit every confident phrase?
  Likely controlled by `config.REQUIRE_WAKE_WORD`.
