# VoiceLLM (Eve) — review brief for an external reviewer

**Date:** 2026-06-10
**Branch:** master (the working branch — `7f1d3dd` rename to Eve)
**Author of the code under review:** Claude, working with the operator
**Audience:** an LLM-driven code-review tool, or a senior reviewer
**Goal of this document:** give you enough context to do a
thorough, opinionated review of where VoiceLLM is + where it
should go next. Copy-paste this entire file into the review tool's
input.

---

## 0. Prime the pattern search — lessons from a prior review

We just had Mochi reviewed.  Five real bugs surfaced that we'd
been missing.  Watch for the same families here:

| Family | Mochi instance | VoiceLLM has the same risk because… |
|---|---|---|
| **Worker-thread + main-thread socket lifecycle** | `HealthService.stop()` closed the ZMQ SUB socket from the main thread while the worker was in `recv_multipart` | VoiceLLM uses an in-process `queue.Queue`, NOT ZMQ.  Race surface different — look for slow consumer / dropped audio / queue overflow behaviour instead. |
| **Layout / state rebuild every tick** | Nodes-table tore down + rebuilt rows ~2x/sec, destroying click targets mid-press | No Qt UI here, but the orchestrator + STT continuous loop might mutate shared state while readers iterate.  Lock discipline? |
| **Doc ahead of code** | Three convention docs described phantom tools / unwired fields / unimplemented syntax | `docs/00-08` plus `docs/STATUS.md`, `docs/06_milestones.md`, `docs/07_open_questions.md` — call out every "described but not implemented" gap. |
| **Smoke test asserting fake conditions** | Sprite handler "verified at t=0" — production `t` is never 0 | `smoke_test.py` exists but uses `FakeBackend` — does it cover real bus/queue behaviour, or just interface shape? |
| **Schema without validator** | `schema:` version field in every yaml, zero validators | No yaml schemas here, but `config.py` exposes ~40 tunables.  Is there any guard against invalid combos? |

---

## 1. What VoiceLLM is

A **fully local voice assistant** named **Eve** (Adam-and-Eve
naming).  Always-on microphone → STT → LLM (Gemma via Ollama or
mlx) → TTS (Kokoro) → speaker.  Designed to be the operator's
daily voice companion on a Mac Studio, with everything running
offline (no cloud, no API keys).

Three operator goals stack:

1. **Daily-use voice tool** for the operator on their workstation
2. **Testbed for JROS's voice pipeline** — lessons here feed back
   into the larger Jenkins Robot Operating System
3. **Open reference for fast-feeling local voice agents** — the
   operator regularly demos the fact that this responds faster
   than JROS while running entirely offline

Part of the **Jenkins Robotics** stack:
[Home](https://jenkinsrobotics.github.io) ·
[Discord](https://discord.gg/sAnE5pRVyT) ·
[YouTube](https://www.youtube.com/@Jenkins_Robotics) ·
[GitHub](https://github.com/jonathanjenkins).

## 2. Why we're building it

| Audience | What they want |
|---|---|
| **Operator (Jonathan)** | "I want to talk to my computer and it talks back, fast, offline, without sending audio to anyone's server." |
| **JROS** | A proven voice loop that can drop into a larger embodied agent.  Architectural pattern + tuning knobs both feed back. |
| **Other developers** | A small, readable, working reference for "local voice agent" without the Jaeger / pipecat / Vocode framework overhead. |

The operator has reported this **feels faster** than JROS's voice
loop.  That's an outcome worth understanding + propagating.

## 3. What's shipped (current state)

### Directory tree

```
VoiceLLM/  (28 py files, single process)
├── main.py                    ← entrypoint — wires bus + nodes + agent
├── config.py                  ← ~40 tunables (LLM choice, STT timing,
│                                 farewell, chimes, prompts)
├── smoke_test.py              ← interface-shape tests w/ FakeBackend
│
├── agent/                     ← cognitive layer
│   ├── orchestrator.py        (453 lines — biggest single file)
│   ├── llm/
│   │   ├── node.py            (LLM node — runs the backend, parses
│   │   ├── backend_base.py     <ignore>/<reply> gate, splits chunks)
│   │   └── __init__.py
│   └── adapters/              ← multi-backend LLM
│       ├── llama_cpp/
│       │   └── backend.py     (llama.cpp via llama-cpp-python)
│       └── mlx/
│           └── backend.py     (Apple MLX — needed hand-rolled stop check
│                                 because mlx-lm doesn't honor Gemma's
│                                 <end_of_turn>)
│
├── nodes/                     ← peripheral I/O
│   ├── audio_session/         (mic pipeline + wakeword + chimes + AEC)
│   │   ├── audio_io.py
│   │   ├── mic_stream.py
│   │   ├── vad.py             (WebRTC VAD)
│   │   ├── aec.py             (acoustic-echo cancellation)
│   │   ├── wakeword.py
│   │   └── chimes.py          (followup-window beep markers)
│   ├── stt/
│   │   ├── continuous.py      (continuous-stream STT)
│   │   └── two_pass.py        (VAD-gated two-pass STT — has the
│   │                            short-phrase adaptive hangover)
│   └── tts/
│       └── node.py            (Kokoro KPipeline streaming TTS)
│
├── transport/
│   └── bus.py                 ← **18 lines.** queue.Queue + Message
│                                 dataclass.  In-process pub/sub.
├── core/
│   ├── metrics.py             (MetricsLog → metrics.csv)
│   └── state.py               (SysState — shared session state)
│
└── docs/  (00-08 + STATUS + README)
    ├── 00_overview.md
    ├── 01_architecture.md
    ├── 02_stt_pipelines.md
    ├── 03_llm_backends.md
    ├── 04_tts_kokoro.md
    ├── 05_barge_in_and_self_speech.md
    ├── 06_milestones.md
    ├── 07_open_questions.md
    ├── 08_vocabulary_contract.md
    ├── README.md
    └── STATUS.md
```

### Critical implementation details

- **Bus = `queue.Queue(maxsize=2048)`**.  Single shared queue.
  Topic is just a string field on `Message`.  Subscribers poll.
  Designed for single-process; will NOT scale to multi-process /
  multi-machine (that's intentional for now).
- **LLM gate**: every reply must start with `<ignore>` or
  `<reply>` so the orchestrator knows whether the input was
  actually directed at Eve.  Always-on mic = lots of ambient
  speech to filter.
- **Two-pass STT** + **adaptive short-phrase hangover** —
  silence_hangover defaults to 600ms, but if the spoken phrase
  is shorter than `SHORT_PHRASE_MAX_MS` (1500ms), commit after
  only `SHORT_PHRASE_HANGOVER_MS` (350ms).  Operator feedback
  flagged this — short queries felt sluggish before the adaptive
  branch.
- **Follow-up window** — after Eve replies, mic stays armed for
  `FOLLOWUP_WINDOW_S` (15s) so user can chain a follow-up without
  re-saying the wake word.  Chimes mark the open/close of the
  window.
- **Farewell detection** — if user AND assistant BOTH say a farewell
  ("bye", "goodbye", "see you later", etc., 17 regex patterns in
  `config.py:FAREWELL_PHRASES`; the mirror requirement stops a stray
  "good night" mid-story from closing the loop), the orchestrator sets
  `_end_of_conversation = True` and skips the follow-up window.
  Conversation ends cleanly.
  *(Correction 2026-06-10: this brief originally said "user OR
  assistant" and "16 patterns" — the code has always required both
  sides, and there are 17.)*

### Recent commits

```
7f1d3dd  voice: rename assistant from Jaeger → Eve
b019678  voice: faster commit + farewell-detection skips followup
bb2b330  reorg: VoiceLLM docs — clean up markdown links the first sweep missed
52c39c2  reorg: align VoiceLLM with JROS 0.5 structure (agent + nodes + transport + core)
6c999a0  2.0  Updated Files strucutre. Project completed
6e38452  1.6 code improvments
9841c31  1.5 Working Voice LLM
```

Last ~5 commits are recent Claude sessions.  Earlier commits
(1.x, 2.0) are the operator's pre-Claude history.

## 4. Architecture / design

| Layer | What's there | Notes |
|---|---|---|
| `agent/` | Orchestrator + LLM node + backend adapters | Cognitive layer.  Orchestrator owns the turn lifecycle. |
| `nodes/` | Audio session + STT + TTS | Peripheral I/O.  Threads, microphones, speakers, real-time. |
| `transport/` | In-process Bus | Plain `queue.Queue`, NOT ZMQ.  Conscious simplification vs Mochi/JROS. |
| `core/` | Metrics + SysState | Shared infra. |

The 3-layer separation matches JROS + Mochi.  But the
**`transport/bus.py` is 18 lines** versus Mochi's ZMQ broker —
order-of-magnitude simpler.  Worth understanding if that
simplification holds up under load, or if it's just hiding
problems that haven't bitten yet.

## 5. The turn lifecycle (what runs when)

```
                                 [follow-up window: open] (15s)
                                            │
                                            ▼
  ┌─ MIC ─→ VAD ─→ STT.two_pass ──→ (final transcript) ──┐
  │  audio_session.mic_stream                              │
  │  audio_session.vad                                     │
  │  audio_session.aec                                     │
  └──────────                                              │
                                                          │
   adaptive: SILENCE_HANGOVER_MS=600 normally,             │
             SHORT_PHRASE_HANGOVER_MS=350 if phrase < 1.5s │
                                                          ▼
                          orchestrator.handle_stt_text()
                              │
                              ├── is farewell? → set _end_of_conv
                              │
                              ▼
                          llm_node.generate(text)
                              │
                              ▼
                       <ignore>… → log + drop
                       <reply>… → strip tag + clean for tts + speak
                              │
                              ▼
                          tts_node.speak(reply)
                              │
                              ▼ (audible)
                              │
                          orchestrator._on_tts_done()
                              │
              end_of_conv? ─── yes → close session
                              │
                              no
                              │
                              ▼
                       chime + open_followup() ─── (back to top)
```

## 6. What's NOT shipped yet — feature gap

Operator hasn't prioritized these; this is my read.  Reviewer
should sanity-check.

### Tier 1 — voice UX polish

| # | Feature |
|---|---|
| 1.1 | Barge-in / interruption (user speaks while Eve is talking → cut TTS, re-listen) |
| 1.2 | Self-speech echo cancellation rigor under load |
| 1.3 | Wake-word retraining / better tolerance for "Eve" vs "Eaves" vs "Yves" |
| 1.4 | Long-conversation context window pruning beyond `MAX_HISTORY_TURNS=8` |

### Tier 2 — multi-persona / multi-language

| # | Feature |
|---|---|
| 2.1 | Persona switching (Eve, Adam, named characters) |
| 2.2 | Multi-language STT + TTS |
| 2.3 | Voice cloning / per-persona TTS voice |

### Tier 3 — JROS bridge

| # | Feature |
|---|---|
| 3.1 | Run as a JROS voice node (bus protocol shim) |
| 3.2 | Share LLM session w/ JROS agent (avoid two model loads) |
| 3.3 | Coordinated wake-word w/ JROS animation node |

### Tier 4 — observability

| # | Feature |
|---|---|
| 4.1 | Real-time latency dashboard (`metrics.csv` exists but no viewer) |
| 4.2 | Per-turn waveform + transcript replay |
| 4.3 | Eve self-monitoring / health endpoint |

## 7. What to develop FIRST

Operator hasn't said.  My read (subject to reviewer + operator
override):

### Priority A — barge-in / interruption

Currently if Eve starts a long reply and the user wants to redirect,
they have to wait for the TTS to finish.  Real-time
conversation UX requires the user can interrupt.  This is the
single biggest "feels old" issue if you compare to commercial
voice assistants.

### Priority B — `metrics.csv` viewer

The codebase already records per-turn metrics (STT time, LLM
time, TTS time, end-to-end).  No viewer exists.  A small
visualization page (could be terminal-rendered or a tiny Flask
endpoint) would tell us where latency goes.

### Priority C — multi-persona pluggable system prompt

`SYSTEM_PROMPT` is hard-coded in `config.py`.  Adding a
`personas/eve.md` + `personas/adam.md` + a `--persona` flag would
let operator switch characters without code changes.

### Priority D — JROS bridge

When you're ready to fold this back into JROS as a voice node.
Probably blocked on JROS voice node's bus protocol stabilizing.

## 8. Specific questions for the reviewer

### A. Concurrency + bus

1. **Bus is `queue.Queue(maxsize=2048)`** ([transport/bus.py:11](VoiceLLM/transport/bus.py:11)).
   What's the **drop policy**?  Today `put` blocks if full.  Under
   bursty audio frames + slow LLM + slow TTS, will the queue
   block the producer (mic thread)?  Should there be a
   per-publisher drop policy?
2. **Single shared queue means all nodes share one ordering**.
   Is that the right model, or should there be per-topic queues?
   Specifically: audio frames + STT events + LLM tokens + TTS
   chunks all serialize through one queue right now.
3. **`smoke_test.py` uses FakeBackend** — does it actually cover
   the queue / threading behaviour, or does it just verify the
   call shape?  When was the last time it failed for a real bug?

### B. Orchestrator structure

4. **`agent/orchestrator.py` is 453 lines.**  Should it split?
   Candidates: turn lifecycle, farewell detection, follow-up
   window, end-of-conversation logic, metrics emission.  Or is
   keeping the turn state machine in one file actually right?
5. **`_end_of_conversation` is a bool flag mutated from
   multiple sites**.  Race-free?  Should it be on `SysState`
   instead?
6. **Farewell detection uses 16 regex patterns** ([orchestrator.py:410+](VoiceLLM/agent/orchestrator.py:410)).
   False positive rate seems low in operator's daily use but
   you should sanity-check.  Is there a "should this farewell
   close the conversation?" hook the operator could override?

### C. STT pipeline

7. **Adaptive short-phrase hangover** ([two_pass.py:53+](VoiceLLM/nodes/stt/two_pass.py:53)).
   The "if phrase is shorter than 1500ms, commit after 350ms of
   silence" branch — does it handle phrases that ALMOST hit the
   threshold cleanly?  Any timing edge cases?
8. **Two-pass STT vs Continuous STT** — both exist (`two_pass.py`
   + `continuous.py`).  Which is the production path?  Is the
   other dead code?
9. **WebRTC VAD aggressiveness** — operator has tuned
   `SILENCE_HANGOVER_MS=600` but the VAD aggressiveness setting
   might dominate.  Worth checking.

### D. LLM gate + persona

10. **`<ignore>` / `<reply>` gate** — what happens if the model
    forgets the tag entirely?  Does the orchestrator default to
    one or the other?  What about a malformed tag like
    `<reply>` followed by no text?
11. **`MAX_HISTORY_TURNS=8`** — is the cap applied per-turn?
    Mid-session?  When the system prompt + truncated history +
    new user message exceeds `LLM_CTX=4096`, what gives?
12. **Multi-backend (llama.cpp + mlx)** — mlx needed a
    hand-rolled stop check.  Are there other latent divergences
    between the two adapters?  Test coverage?

### E. Operator's "feels faster than JROS" claim

13. Operator regularly says VoiceLLM feels faster than JROS for
    the same voice task.  Where would you LOOK to verify or
    falsify that — `metrics.csv`?  Specific code paths?  Are
    there architectural choices here that JROS could adopt?

### F. Forward direction

14. **Where would YOU start** if you were taking over?  Barge-in?
    Metrics viewer?  Persona system?  Something I missed?
15. **What's the biggest risk** in the current code that would
    bite in 3 months?
16. **What looks over-engineered** that should die now?

## 9. Practical context the reviewer needs

- Python 3.12.6 (per requirements.txt setup comment)
- Single process, multi-threaded
- macOS-first, Apple Silicon optimized (MLX backend uses Metal,
  llama.cpp with `LLM_GPU_LAYERS=-1`)
- Operator runs gemma-4-26B-A4B at Q4_K_M via llama.cpp
- TTS via Kokoro pip package (KPipeline API)
- STT via pywhispercpp (Metal acceleration on Apple Silicon)
- VAD via webrtcvad

To run + test:

```bash
cd /path/to/VoiceLLM
source venv/bin/activate
python main.py            # the whole agent

python smoke_test.py      # interface-shape tests (no model load)
```

To explore:

```bash
git log --oneline -20
head -200 config.py        # every tunable in one place
cat docs/00_overview.md
cat docs/STATUS.md
```

## 10. Output I want from the reviewer

In order of value:

1. **Direct answers to Section 8 questions** — specific, actionable.
   Bullet-point each.  Section 0 patterns first if found.
2. **A prioritised list of issues found** — severity × effort.
   Bugs first, then over-engineering / smell, then doc-vs-code
   drift.
3. **A concrete "if I were you, I'd do X next" recommendation**
   for the next 5-10 commits.  Not a year-long roadmap.
4. **One brutally honest piece of feedback** about something
   that's wrong / over-engineered / missing — even if it doesn't
   fit a numbered category.

Don't pad with "what's good" — focus on what to fix + what's next.

---

End of brief.  Branch is `master`; latest commit at writing is
`7f1d3dd`.  Have at it.
