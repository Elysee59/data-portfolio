"""Multi-agent People Analytics team: a coordinator delegating to specialist agents."""
from .agents import SPECIALISTS, HRAgentTeam, RunResult, estimate_cost
from .offline import run_offline

__all__ = ["HRAgentTeam", "RunResult", "SPECIALISTS", "estimate_cost", "run_offline"]
