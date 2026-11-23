"""A coordinator agent that delegates to specialist sub-agents (Claude Messages API, tool use).

    question ─► Coordinator ──delegate()──┬─► Turnover specialist ───── data tools
                     ▲                    ├─► Pay-equity specialist ── data tools
                     │                    └─► Retention-risk specialist ─ data tools
                     └──── specialist reports (in parallel) ◄─┘
                ─► memo for the HR director

Each specialist is its own agent loop with its own system prompt, its own
subset of tools and its own context. When the coordinator delegates to
several specialists in one turn, they run in parallel threads.

Everything that happens is reported through an ``on_event`` callback, so a
UI can stream the trace (who is working, which tool was called, results).
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .tools import call_tool, tool_schemas

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
# Specialists do simple work (pick a tool, read figures, summarise): a smaller, much cheaper model is enough.
DEFAULT_SPECIALIST_MODEL = os.environ.get("ANTHROPIC_SPECIALIST_MODEL", "claude-haiku-5-5")
MAX_TURNS = 8  # safety net per agent loop
MAX_TOKENS_SPECIALIST = 700   # short reports: output tokens are the most expensive
MAX_TOKENS_COORDINATOR = 1200

# USD per million tokens (input, output), from platform.claude.com/docs/en/about-claude/pricing (Nov 2026).
# Update when prices change. Haiku 5.5 base tier shown (prompts up to 100k tokens).
PRICES_PER_MTOK = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.0, 20.0)}


def estimate_cost(usage_by_model: dict[str, tuple[int, int]]) -> float | None:
    """Cost in USD, or None if a model has no known price."""
    total = 0.0
    for model, (tin, tout) in usage_by_model.items():
        if model not in PRICES_PER_MTOK:
            return None
        pin, pout = PRICES_PER_MTOK[model]
        total += tin * pin / 1e6 + tout * pout / 1e6
    return total


@dataclass(frozen=True)
class Specialist:
    key: str
    name: str
    focus: str
    tools: list[str]

    def system_prompt(self) -> str:
        return (PROMPTS / "specialist.md").read_text(encoding="utf-8").format(name=self.name, focus=self.focus)


SPECIALISTS = {
    s.key: s for s in [
        Specialist("turnover", "Turnover & Absence",
                   "You analyse headcount, hires, leavers, turnover and absenteeism.",
                   ["workforce_overview", "turnover_breakdown", "absence_overview"]),
        Specialist("pay_equity", "Pay Equity",
                   "You analyse the gender pay gap, raw and adjusted, and judge whether it is statistically meaningful.",
                   ["pay_gap_analysis"]),
        Specialist("retention_risk", "Retention Risk",
                   "You analyse the attrition risk model: its quality, its drivers and which segments are most at risk.",
                   ["attrition_drivers", "risk_by_segment"]),
    ]
}

DELEGATE_TOOL = {
    "name": "delegate",
    "description": "Assign a task to one specialist analyst and get back their report. Call it several times in the "
                   "same turn to run specialists in parallel.",
    "input_schema": {
        "type": "object",
        "properties": {
            "specialist": {"type": "string", "enum": list(SPECIALISTS)},
            "task": {"type": "string", "description": "Precise assignment: what to find, year, breakdown."},
        },
        "required": ["specialist", "task"],
    },
}

Event = dict[str, Any]


@dataclass
class RunResult:
    memo: str
    events: list[Event] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    usage_by_model: dict[str, tuple[int, int]] = field(default_factory=dict)
    cost_usd: float | None = None


class _Usage:
    """Thread-safe token counter, per model (specialists run in parallel threads)."""

    def __init__(self):
        self.by_model: dict[str, list[int]] = {}
        self._lock = threading.Lock()

    def add(self, model: str, response):
        u = getattr(response, "usage", None)
        if u is not None:
            with self._lock:
                row = self.by_model.setdefault(model, [0, 0])
                row[0] += getattr(u, "input_tokens", 0) or 0
                row[1] += getattr(u, "output_tokens", 0) or 0

    @property
    def input(self) -> int:
        return sum(v[0] for v in self.by_model.values())

    @property
    def output(self) -> int:
        return sum(v[1] for v in self.by_model.values())


def _text(response) -> str:
    return "\n".join(b.text for b in response.content if b.type == "text").strip()


def _assistant_content(response) -> list[dict]:
    """Convert response blocks back into message params for the next turn."""
    out = []
    for b in response.content:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


class HRAgentTeam:
    def __init__(self, client=None, model: str | None = None, on_event: Callable[[Event], None] | None = None,
                 thread_setup: Callable[[], None] | None = None, specialist_model: str | None = None):
        """``model`` drives the coordinator, ``specialist_model`` the specialists.

        ``thread_setup`` runs at the start of each specialist thread (e.g. to attach a UI context).
        """
        if client is None:
            import anthropic  # lazy: the rest of the package works without the SDK
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or DEFAULT_MODEL
        self.specialist_model = specialist_model or DEFAULT_SPECIALIST_MODEL
        self.on_event = on_event or (lambda e: None)
        self.thread_setup = thread_setup
        self.events: list[Event] = []
        self.usage = _Usage()

    # ------------------------------------------------------------------ events
    def _emit(self, **event):
        self.events.append(event)
        self.on_event(event)

    # -------------------------------------------------------------- specialists
    def run_specialist(self, key: str, task: str) -> str:
        spec = SPECIALISTS[key]
        self._emit(type="specialist_start", agent=key, name=spec.name, task=task)
        messages: list[dict] = [{"role": "user", "content": task}]
        for _ in range(MAX_TURNS):
            response = self.client.messages.create(
                model=self.specialist_model, max_tokens=MAX_TOKENS_SPECIALIST, system=spec.system_prompt(),
                tools=tool_schemas(spec.tools), messages=messages)
            self.usage.add(self.specialist_model, response)
            if response.stop_reason != "tool_use":
                report = _text(response)
                self._emit(type="specialist_done", agent=key, name=spec.name, report=report)
                return report
            messages.append({"role": "assistant", "content": _assistant_content(response)})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                self._emit(type="tool_call", agent=key, tool=block.name, input=block.input)
                result = call_tool(block.name, block.input)
                self._emit(type="tool_result", agent=key, tool=block.name, result=result)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(result, default=str, separators=(",", ":")),
                                **({"is_error": True} if "error" in result else {})})
            messages.append({"role": "user", "content": results})
        report = "Stopped: too many tool calls without a final answer."
        self._emit(type="specialist_done", agent=key, name=spec.name, report=report)
        return report

    # -------------------------------------------------------------- coordinator
    def run(self, question: str) -> RunResult:
        self._emit(type="question", text=question)
        system = (PROMPTS / "coordinator.md").read_text(encoding="utf-8")
        messages: list[dict] = [{"role": "user", "content": question}]
        memo = ""
        for _ in range(MAX_TURNS):
            response = self.client.messages.create(
                model=self.model, max_tokens=MAX_TOKENS_COORDINATOR, system=system, tools=[DELEGATE_TOOL],
                messages=messages)
            self.usage.add(self.model, response)
            if response.stop_reason != "tool_use":
                memo = _text(response)
                break
            messages.append({"role": "assistant", "content": _assistant_content(response)})
            calls = [b for b in response.content if b.type == "tool_use"]
            for b in calls:
                self._emit(type="delegate", agent=b.input.get("specialist"), task=b.input.get("task"))

            def work(block):
                if self.thread_setup:
                    self.thread_setup()
                key = block.input.get("specialist")
                if block.name != "delegate" or key not in SPECIALISTS:
                    return block.id, f"Error: unknown specialist {key!r}", True
                return block.id, self.run_specialist(key, block.input.get("task", "")), False

            with ThreadPoolExecutor(max_workers=len(SPECIALISTS)) as pool:  # specialists work in parallel
                outcomes = list(pool.map(work, calls))
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tid, "content": text, **({"is_error": True} if err else {})}
                for tid, text, err in outcomes]})
        else:
            memo = "The coordinator did not finish within the turn limit."
        self._emit(type="memo", text=memo)
        by_model = {m: (v[0], v[1]) for m, v in self.usage.by_model.items()}
        return RunResult(memo=memo, events=self.events, input_tokens=self.usage.input,
                         output_tokens=self.usage.output, usage_by_model=by_model, cost_usd=estimate_cost(by_model))
