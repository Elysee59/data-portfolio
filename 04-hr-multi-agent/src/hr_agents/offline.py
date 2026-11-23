"""Offline mode: the same team and tools, without an LLM.

Used when no API key is configured (public demo, tests, CI). A keyword router
picks the specialists, each specialist runs its tools directly, and a fixed
template turns the tool results into a memo. It emits the same events as the
LLM team, so the UI is identical; the memo is clearly labelled as generated
without AI.
"""
from __future__ import annotations

import re
from typing import Callable

from .agents import SPECIALISTS, Event, RunResult
from .tools import call_tool

ROUTES = {
    "pay_equity": r"pay|salar|gap|equal|écart|egalit|égalit|rémun|remun",
    "retention_risk": r"risk|risque|leave|départ|depart|retention|rétention|attrition|quit|flight",
    "turnover": r"turnover|leaver|headcount|effectif|absen|hire|embauche|rotation|sortie",
}

PLANS = {  # tool calls each specialist makes in offline mode
    "turnover": [("workforce_overview", {"year": 2025}), ("turnover_breakdown", {"dimension": "department", "year": 2025}),
                 ("absence_overview", {"dimension": "department"})],
    "pay_equity": [("pay_gap_analysis", {})],
    "retention_risk": [("attrition_drivers", {}), ("risk_by_segment", {"dimension": "department"})],
}

LABELS = {"engagement_score": "engagement", "avg_monthly_overtime_h": "overtime",
          "salary_to_market_ratio": "pay vs. market", "department": "department", "job_level": "job level"}


def route(question: str) -> list[str]:
    q = question.lower()
    keys = [k for k, pattern in ROUTES.items() if re.search(pattern, q)]
    return keys or list(SPECIALISTS)


def _report(key: str, r: dict) -> tuple[str, list[str], list[str]]:
    """Specialist report text, memo findings and actions from tool results."""
    if key == "turnover":
        o, t, a = r["workforce_overview"], r["turnover_breakdown"], r["absence_overview"]
        top, low = t["rows"][0], t["rows"][-1]
        resign = t["leave_reasons_pct"].get("Resignation", 0)
        findings = [
            f"Turnover was **{o['turnover_rate_pct']} %** in {o['year']} ({o['leavers']} leavers, {o['hires']} hires, "
            f"headcount {o['headcount_end_of_year']:,} at year end).",
            f"**{top['department']}** loses the most people ({top['leaving_rate_pct']} % of staff left) vs "
            f"{low['department']} ({low['leaving_rate_pct']} %); {resign:.0f} % of departures are resignations.",
        ]
        actions = [f"Run stay interviews and an exit-survey review in {top['department']}, where most avoidable "
                   f"departures (resignations) happen."]
        a_top = a["rows"][0]
        report = "\n".join(findings + [f"Highest absence: {a_top['department']} ({a_top['avg_absence_days']} days on average)."])
        return report, findings, actions
    if key == "pay_equity":
        p = r["pay_gap_analysis"]
        worst = max(p["raw_gap_by_level"], key=lambda x: x["raw_gap_pct"])
        lo, hi = p["adjusted_gap_95ci_pct"]
        findings = [
            f"Raw median pay gap reaches **{worst['raw_gap_pct']} %** at {worst['job_level']} level.",
            f"Adjusted for level, department, country and age, women earn **{p['adjusted_gap_pct']} % less** than "
            f"comparable men (95 % CI {lo} to {hi} %): "
            + ("the gap is statistically significant." if lo > 0 or hi < 0 else "not statistically significant."),
        ]
        actions = ["Launch a salary review for women paid below the adjusted benchmark of their role, with a budget "
                   "sized on the adjusted gap."]
        return "\n".join(findings), findings, actions
    m, s = r["attrition_drivers"], r["risk_by_segment"]
    drivers = ", ".join(LABELS.get(d["feature"], d["feature"]) for d in m["drivers_ranked"][:3])
    top = s["rows"][0]
    findings = [
        f"The risk model (ROC AUC {m['roc_auc']}) puts **{m['share_of_leavers_in_top20pct_risk']} %** of leavers in "
        f"its top 20 % risk group; main drivers: {drivers}.",
        f"Highest predicted risk: **{top['department']}** ({top['avg_risk_pct']:.1f} % average 12-month risk, "
        f"{top['high_risk_share_pct']:.0f} % of staff above 30 %).",
    ]
    actions = [f"Act on the levers the model highlights ({drivers}) in {top['department']} first; review progress "
               f"quarterly with HR business partners."]
    return "\n".join(findings), findings, actions


def run_offline(question: str, on_event: Callable[[Event], None] | None = None) -> RunResult:
    events: list[Event] = []

    def emit(**e):
        events.append(e)
        (on_event or (lambda _: None))(e)

    emit(type="question", text=question)
    keys = route(question)
    all_findings, all_actions = [], []
    for key in keys:
        task = f"Offline plan for {SPECIALISTS[key].name}: run the standard analysis."
        emit(type="delegate", agent=key, task=task)
        emit(type="specialist_start", agent=key, name=SPECIALISTS[key].name, task=task)
        results = {}
        for tool, args in PLANS[key]:
            emit(type="tool_call", agent=key, tool=tool, input=args)
            results[tool] = call_tool(tool, args)
            emit(type="tool_result", agent=key, tool=tool, result=results[tool])
        report, findings, actions = _report(key, results)
        emit(type="specialist_done", agent=key, name=SPECIALISTS[key].name, report=report)
        all_findings += findings
        all_actions += actions

    memo = "\n".join([
        "_Offline mode: this memo was assembled from the specialists' tool results with a fixed template, "
        "without an AI model. Add an Anthropic API key to get a memo written by the coordinator agent._",
        "",
        "**Key findings**",
        *[f"- {f}" for f in all_findings],
        "",
        "**Recommended actions**",
        *[f"- {a}" for a in all_actions],
        "",
        "**Caveats**",
        "- Synthetic data. Aggregated analysis only: no automated decision about any individual employee (GDPR).",
    ])
    emit(type="memo", text=memo)
    return RunResult(memo=memo, events=events)
