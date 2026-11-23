"""Multi-agent People Analytics team: a coordinator delegating to specialist agents."""
from .agents import SPECIALISTS, HRAgentTeam, RunResult
from .offline import run_offline

__all__ = ["HRAgentTeam", "RunResult", "SPECIALISTS", "run_offline"]
