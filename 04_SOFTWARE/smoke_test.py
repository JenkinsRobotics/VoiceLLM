# Quick interface sanity check
from config import SAMPLE_RATE, WAKEWORDS
from core.bus import Bus
from stt.stt_node import STTNode
from llm.llm_node import LLMNode
from tts.kokoro_node import KokoroNode
from orchestrator.orchestrator import Orchestrator

print("Config OK:", SAMPLE_RATE, WAKEWORDS)
bus = Bus()
stt = STTNode(bus)
llm = LLMNode(bus)
tts = KokoroNode(bus)
orch = Orchestrator(bus, llm, tts)
print("Orchestrator wired.")