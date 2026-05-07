from enum import Enum, auto

class SysState(Enum):
    IDLE = auto()
    THINKING = auto()
    RESPONDING = auto()

class State:
    def __init__(self):
        self.value = SysState.IDLE
    def set(self, v):
        self.value = v