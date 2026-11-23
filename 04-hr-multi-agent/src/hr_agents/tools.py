"""Data tools the HR agents can call.

Every tool reads the synthetic HR dataset of project 01 and returns a small,
JSON-serialisable dict. Tools only ever return **aggregates** (by department,
level, country...), never data about an identifiable employee: an agent
cannot be asked "who will leave?", only "which groups are at risk?".
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "01-hr-analytics" / "data" / "employees.csv"
sys.path.insert(0, str(REPO / "01-hr-analytics" / "src"))

DIMENSIONS = ["department", "country", "job_level", "gender"]
LEVELS = ["Associate", "Professional", "Manager", "Director"]
MIN_GROUP = 10  # groups smaller than this are suppressed (k-anonymity style)


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    return pd.read_csv(DATA, parse_dates=["hire_date", "leave_date"])


def _year_bounds(year: int):
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


def _present(df, year):
    start, end = _year_bounds(year)
    return df[(df.hire_date <= end) & (df.leave_date.isna() | (df.leave_date >= start))]


def _check_year(year: int) -> int:
    year = int(year)
    if year not in (2023, 2024, 2025):
        raise ValueError("year must be 2023, 2024 or 2025")
    return year


# --------------------------------------------------------------------------- tools
def workforce_overview(year: int = 2025) -> dict:
    """Headcount, hires, leavers and turnover rate for one year."""
    year = _check_year(year)
    df = load()
    start, end = _year_bounds(year)
    months = pd.date_range(start, end, freq="MS") + pd.offsets.MonthEnd(0)
    hc = [int(((df.hire_date <= m) & (df.leave_date.isna() | (df.leave_date > m))).sum()) for m in months]
    leavers = int(df.leave_date.between(start, end).sum())
    return {
        "year": year,
        "headcount_end_of_year": hc[-1],
        "average_headcount": round(float(np.mean(hc)), 1),
        "hires": int(df.hire_date.between(start, end).sum()),
        "leavers": leavers,
        "turnover_rate_pct": round(100 * leavers / float(np.mean(hc)), 1),
        "definition": "turnover = leavers during the year / average month-end headcount",
    }


def turnover_breakdown(dimension: str = "department", year: int = 2025) -> dict:
    """Share of employees present during the year who left, by a dimension, plus leave reasons."""
    year = _check_year(year)
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    pres = _present(load(), year)
    start, end = _year_bounds(year)
    left = pres.leave_date.between(start, end)
    g = pres.assign(left=left).groupby(dimension).agg(employees=("left", "size"), leavers=("left", "sum"))
    g = g[g.employees >= MIN_GROUP]
    g["leaving_rate_pct"] = (100 * g.leavers / g.employees).round(1)
    reasons = pres[left].leave_reason.value_counts(normalize=True).mul(100).round(1)
    return {
        "year": year,
        "dimension": dimension,
        "rows": g.sort_values("leaving_rate_pct", ascending=False).reset_index().to_dict("records"),
        "leave_reasons_pct": reasons.to_dict(),
    }


def pay_gap_analysis(year_end_active_only: bool = True) -> dict:
    """Median pay by level and gender (raw gap) and a gap adjusted for level, department, country and age."""
    df = load()
    act = df[df.leave_date.isna()] if year_end_active_only else df
    med = act.groupby(["job_level", "gender"]).annual_salary_eur.median().unstack().reindex(LEVELS)
    n = act.groupby(["job_level", "gender"]).size().unstack().reindex(LEVELS)
    raw = [{"job_level": lvl, "women": int(n.loc[lvl, "F"]), "men": int(n.loc[lvl, "M"]),
            "median_women_eur": float(med.loc[lvl, "F"]), "median_men_eur": float(med.loc[lvl, "M"]),
            "raw_gap_pct": round(float(100 * (med.loc[lvl, "M"] - med.loc[lvl, "F"]) / med.loc[lvl, "M"]), 1)}
           for lvl in LEVELS]
    # Adjusted gap: OLS of log(salary) on gender + controls, via numpy least squares
    X = pd.get_dummies(act[["job_level", "department", "country"]], drop_first=True).astype(float)
    X["female"] = (act.gender == "F").astype(float)
    X["age"] = act.age.astype(float)
    X.insert(0, "const", 1.0)
    y = np.log(act.annual_salary_eur.values)
    beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
    resid = y - X.values @ beta
    dof = len(y) - X.shape[1]
    cov = (resid @ resid / dof) * np.linalg.inv(X.values.T @ X.values)
    i = list(X.columns).index("female")
    coef, se = beta[i], float(np.sqrt(cov[i, i]))
    return {
        "population": "active employees" if year_end_active_only else "all employees",
        "raw_gap_by_level": raw,
        "adjusted_gap_pct": round(float(-100 * (np.exp(coef) - 1)), 1),
        "adjusted_gap_95ci_pct": [round(float(-100 * (np.exp(coef + 1.96 * se) - 1)), 1),
                                  round(float(-100 * (np.exp(coef - 1.96 * se) - 1)), 1)],
        "method": "OLS on log salary with controls for job level, department, country and age; "
                  "positive = women paid less than comparable men",
    }


@lru_cache(maxsize=1)
def _model():
    from analysis import attrition_model  # project 01
    return attrition_model(load())


def attrition_drivers() -> dict:
    """Quality and main drivers of the attrition risk model (trained on a 1 Jan 2025 snapshot)."""
    scores, best, importance, base_rate, capture, _ = _model()
    return {
        "model": best,
        "roc_auc": round(scores[best]["ROC AUC"], 3),
        "base_leaving_rate_pct": round(100 * base_rate, 1),
        "share_of_leavers_in_top20pct_risk": round(100 * capture, 1),
        "drivers_ranked": [{"feature": f, "importance": round(float(v), 4)}
                           for f, v in importance.sort_values(ascending=False).items() if v > 0.001],
    }


def risk_by_segment(dimension: str = "department") -> dict:
    """Average predicted 12-month leaving risk of current employees, by segment (aggregated only)."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    *_, fitted = _model()
    df = load()
    snap = pd.Timestamp("2026-01-01")
    act = df[df.leave_date.isna()].copy()
    act["tenure_years"] = (snap - act.hire_date).dt.days / 365.25
    act["risk"] = fitted.predict_proba(act)[:, 1]
    g = act.groupby(dimension).agg(employees=("risk", "size"), avg_risk_pct=("risk", "mean"),
                                   high_risk_share_pct=("risk", lambda r: (r > 0.30).mean()),
                                   avg_engagement=("engagement_score", "mean"),
                                   avg_salary_to_market=("salary_to_market_ratio", "mean"),
                                   avg_overtime_h=("avg_monthly_overtime_h", "mean"))
    g = g[g.employees >= MIN_GROUP]
    g[["avg_risk_pct", "high_risk_share_pct"]] *= 100
    return {"dimension": dimension, "high_risk_threshold": "predicted risk above 30 %",
            "rows": g.round(2).sort_values("avg_risk_pct", ascending=False).reset_index().to_dict("records")}


def absence_overview(dimension: str = "department") -> dict:
    """Average absence days over the last 12 months and share above 15 days, by segment."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    act = load()[load().leave_date.isna()]
    g = act.groupby(dimension).agg(employees=("absence_days_last_12m", "size"),
                                   avg_absence_days=("absence_days_last_12m", "mean"),
                                   share_above_15_days_pct=("absence_days_last_12m", lambda s: 100 * (s > 15).mean()))
    g = g[g.employees >= MIN_GROUP]
    return {"dimension": dimension, "rows": g.round(1).sort_values("avg_absence_days", ascending=False).reset_index().to_dict("records")}


# --------------------------------------------------------------------- registry
_DIM = {"type": "string", "enum": DIMENSIONS}
_YEAR = {"type": "integer", "enum": [2023, 2024, 2025]}

TOOLS: dict[str, tuple[Callable[..., dict], dict]] = {
    "workforce_overview": (workforce_overview, {
        "description": "Headcount, hires, leavers and turnover rate for a year.",
        "input_schema": {"type": "object", "properties": {"year": _YEAR}}}),
    "turnover_breakdown": (turnover_breakdown, {
        "description": "Share of employees who left during a year, broken down by department, country, job_level "
                       "or gender, with the distribution of leave reasons.",
        "input_schema": {"type": "object", "properties": {"dimension": _DIM, "year": _YEAR}}}),
    "pay_gap_analysis": (pay_gap_analysis, {
        "description": "Gender pay gap: raw median gap by job level and an adjusted gap from a regression "
                       "controlling for level, department, country and age, with a 95 % confidence interval.",
        "input_schema": {"type": "object", "properties": {"year_end_active_only": {"type": "boolean"}}}}),
    "attrition_drivers": (attrition_drivers, {
        "description": "Quality (ROC AUC) and ranked drivers of the attrition risk model.",
        "input_schema": {"type": "object", "properties": {}}}),
    "risk_by_segment": (risk_by_segment, {
        "description": "Average predicted 12-month leaving risk of current employees by segment, with average "
                       "engagement, pay vs market and overtime. Aggregates only, never individuals.",
        "input_schema": {"type": "object", "properties": {"dimension": _DIM}}}),
    "absence_overview": (absence_overview, {
        "description": "Average absence days in the last 12 months and share of employees above 15 days, by segment.",
        "input_schema": {"type": "object", "properties": {"dimension": _DIM}}}),
}


def tool_schemas(names: list[str]) -> list[dict]:
    return [{"name": n, **TOOLS[n][1]} for n in names]


def call_tool(name: str, args: dict[str, Any]) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    try:
        return TOOLS[name][0](**(args or {}))
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}
