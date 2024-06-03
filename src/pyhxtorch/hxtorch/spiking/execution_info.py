from dataclasses import dataclass
import pygrenade_vx as grenade


@dataclass
class ExecutionInfo:
    time: grenade.signal_flow.ExecutionTimeInfo \
        = grenade.signal_flow.ExecutionTimeInfo()
    health: grenade.signal_flow.ExecutionHealthInfo \
        = grenade.signal_flow.ExecutionHealthInfo()
