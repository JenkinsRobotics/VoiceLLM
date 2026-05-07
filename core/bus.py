import queue
from dataclasses import dataclass
from typing import Any

@dataclass
class Message:
    topic: str
    payload: Any

class Bus:
    def __init__(self):
        self.q = queue.Queue(maxsize=2048)
    def publish(self, topic, payload):
        self.q.put(Message(topic, payload))
    def get(self, timeout=0.1):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None