"""Tests for the tools, the offline team and the agent loop (with a fake Claude client, no API call)."""
import json
import threading
from types import SimpleNamespace as NS

import pytest

from hr_agents import HRAgentTeam, estimate_cost, run_offline
from hr_agents.offline import route
from hr_agents.tools import TOOLS, call_tool, pay_gap_analysis


# ------------------------------------------------------------------ tools
@pytest.mark.parametrize("name", list(TOOLS))
def test_every_tool_returns_json(name):
    json.dumps(call_tool(name, {}))


def test_tool_errors_are_returned_not_raised():
    assert "error" in call_tool("turnover_breakdown", {"dimension": "employee_id"})
    assert "error" in call_tool("workforce_overview", {"year": 1999})
    assert "error" in call_tool("does_not_exist", {})


def test_adjusted_pay_gap_recovers_the_simulated_gap():
    # generate_data.py builds in a 4 % gap for women at equal level/department
    lo, hi = pay_gap_analysis()["adjusted_gap_95ci_pct"]
    assert lo < 4 < hi


def test_tools_never_return_employee_ids():
    for name in TOOLS:
        assert "E1" not in json.dumps(call_tool(name, {}))


# ------------------------------------------------------------ offline mode
def test_routing():
    assert route("Is there a gender pay gap?") == ["pay_equity"]
    assert route("Quels services ont le plus de départs à risque ?") == ["retention_risk"]
    assert set(route("Give me an HR health check")) == {"turnover", "pay_equity", "retention_risk"}


def test_offline_memo_uses_real_figures():
    r = run_offline("full health check")
    assert "17.8 %" in r.memo and "Offline mode" in r.memo
    assert [e["type"] for e in r.events][-1] == "memo"


# ------------------------------------------------------- LLM loop, faked
def text(t):
    return NS(type="text", text=t)


def tool_use(i, name, inp):
    return NS(type="tool_use", id=i, name=name, input=inp)


def resp(blocks, stop):
    return NS(content=blocks, stop_reason=stop, usage=NS(input_tokens=10, output_tokens=5))


class FakeClient:
    """Plays the coordinator and the specialists according to their system prompt."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()
        self.messages = NS(create=self.create)

    def create(self, *, model, max_tokens, system, tools, messages):
        with self.lock:
            self.calls.append({"system": system, "tools": [t["name"] for t in tools], "messages": messages,
                               "model": model, "max_tokens": max_tokens})
        last = messages[-1]["content"]
        is_coordinator = "Head of People Analytics for" in system
        if is_coordinator:
            if isinstance(last, str):  # first turn: delegate to two specialists at once
                return resp([tool_use("d1", "delegate", {"specialist": "turnover", "task": "turnover 2025"}),
                             tool_use("d2", "delegate", {"specialist": "pay_equity", "task": "pay gap"})], "tool_use")
            reports = " | ".join(r["content"] for r in last)
            return resp([text(f"MEMO based on: {reports}")], "end_turn")
        if isinstance(last, str):  # specialist first turn: call its first tool
            return resp([tool_use("t1", tools[0]["name"], {})], "tool_use")
        result = json.loads(last[0]["content"])
        return resp([text(f"report:{sorted(result)[0]}")], "end_turn")


def test_coordinator_delegates_in_parallel_and_writes_memo():
    events = []
    client = FakeClient()
    result = HRAgentTeam(client=client, model="fake", on_event=events.append).run("How are we doing?")
    types = [e["type"] for e in events]
    assert types.count("delegate") == 2 and types.count("specialist_done") == 2
    assert types[-1] == "memo" and result.memo.startswith("MEMO based on: report:")
    # specialists only see their own tools
    spec_tools = [c["tools"] for c in client.calls if "Head of People Analytics for" not in c["system"]]
    assert ["pay_gap_analysis"] in spec_tools
    assert all("delegate" not in t for t in spec_tools)
    assert result.input_tokens == 10 * len(client.calls)


def test_unknown_specialist_is_reported_as_error():
    class BadClient(FakeClient):
        def create(self, **kw):
            last = kw["messages"][-1]["content"]
            if isinstance(last, str):
                return resp([tool_use("d1", "delegate", {"specialist": "ceo", "task": "x"})], "tool_use")
            assert last[0]["is_error"] is True
            return resp([text("done")], "end_turn")

    assert HRAgentTeam(client=BadClient(), model="fake").run("q").memo == "done"


def test_specialists_use_the_cheaper_model_and_cost_is_computed():
    client = FakeClient()
    result = HRAgentTeam(client=client, model="claude-sonnet-5-5", specialist_model="claude-haiku-5-5").run("q")
    coord = [c for c in client.calls if "Head of People Analytics for" in c["system"]]
    spec = [c for c in client.calls if "Head of People Analytics for" not in c["system"]]
    assert {c["model"] for c in coord} == {"claude-sonnet-5-5"}
    assert {c["model"] for c in spec} == {"claude-haiku-5-5"}
    assert all(c["max_tokens"] <= 700 for c in spec)
    assert result.usage_by_model["claude-haiku-5-5"] == (10 * len(spec), 5 * len(spec))
    expected = (10 * len(coord) * 2 + 5 * len(coord) * 10 + 10 * len(spec) * 0.10 + 5 * len(spec) * 0.50) / 1e6
    assert result.cost_usd == pytest.approx(expected)


def test_cost_unknown_model_is_none():
    assert estimate_cost({"some-future-model": (1000, 100)}) is None
    assert estimate_cost({"claude-sonnet-5-5": (6421, 3032)}) == pytest.approx(0.043162)
