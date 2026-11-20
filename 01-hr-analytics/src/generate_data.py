"""Generate a synthetic HR dataset (fictional company, no real person).

The data mimics what an HRIS such as Workday exports: one row per employee
with hiring / leaving dates, organisation, pay and absence information.
Attrition is driven by plausible factors (pay vs. market, engagement,
overtime, tenure) so that the analysis and the model have signal to find.

Usage: python src/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_EMPLOYEES = 4000
START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2025-12-31")

DEPARTMENTS = {
    # name: (share of headcount, median salary k€, base monthly leave risk)
    "Retail Operations": (0.38, 29, 0.020),
    "Supply Chain": (0.20, 34, 0.014),
    "Digital & Data": (0.12, 52, 0.013),
    "Finance": (0.08, 48, 0.008),
    "HR": (0.06, 42, 0.008),
    "Marketing": (0.08, 44, 0.011),
    "Product Design": (0.08, 47, 0.009),
}
COUNTRIES = {"France": 0.55, "Spain": 0.15, "Germany": 0.12, "Belgium": 0.10, "Italy": 0.08}
LEVELS = {"Associate": 0.55, "Professional": 0.30, "Manager": 0.12, "Director": 0.03}
LEVEL_PAY = {"Associate": 0.85, "Professional": 1.05, "Manager": 1.45, "Director": 2.2}


def main(out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = N_EMPLOYEES

    dept_names = list(DEPARTMENTS)
    dept = rng.choice(dept_names, size=n, p=[DEPARTMENTS[d][0] for d in dept_names])
    country = rng.choice(list(COUNTRIES), size=n, p=list(COUNTRIES.values()))
    level = rng.choice(list(LEVELS), size=n, p=list(LEVELS.values()))
    gender = rng.choice(["F", "M"], size=n, p=[0.48, 0.52])
    age = np.clip(rng.normal(36, 9, n).round(), 20, 64).astype(int)

    # Hire dates: some employees were already there before the window starts.
    days_span = (END - START).days
    hire_offset = rng.integers(-4 * 365, days_span - 30, n)
    hire_date = START + pd.to_timedelta(hire_offset, unit="D")

    median_pay = np.array([DEPARTMENTS[d][1] for d in dept]) * np.array([LEVEL_PAY[l] for l in level])
    # Built-in, modest gender pay gap so the equal-pay analysis has something to show.
    gap = np.where(gender == "F", 0.96, 1.0)
    salary = (median_pay * gap * rng.lognormal(0, 0.12, n) * 1000).round(-2)
    market_ratio = np.clip(rng.normal(1.0, 0.10, n), 0.7, 1.3)  # salary vs. market benchmark

    engagement = np.clip(rng.normal(3.6, 0.7, n), 1, 5).round(1)  # survey score /5
    overtime_h = np.clip(rng.gamma(2, 3, n), 0, 40).round(1)      # avg monthly overtime
    remote_days = rng.choice([0, 1, 2, 3], size=n, p=[0.45, 0.2, 0.25, 0.1])
    remote_days = np.where(dept == "Retail Operations", 0, remote_days)
    perf = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.04, 0.16, 0.5, 0.22, 0.08])
    absence_days = np.clip(rng.poisson(6 + 3 * (engagement < 3), n) + rng.integers(0, 4, n), 0, 60)

    # Monthly leaving hazard
    base = np.array([DEPARTMENTS[d][2] for d in dept])
    hazard = base
    hazard = hazard * np.exp(-3.5 * (market_ratio - 1.0))
    hazard = hazard * np.exp(-0.9 * (engagement - 3.6))
    hazard = hazard * np.exp(0.06 * (overtime_h - 6))
    hazard = hazard * np.where(age < 28, 1.7, 1.0)
    hazard = hazard * np.where(level == "Director", 0.5, 1.0)
    hazard = hazard * np.where(perf >= 4, 1.15, 1.0)  # high performers get poached

    observe_from = np.maximum(hire_date, START)
    months_observable = ((END - observe_from).days / 30.44).astype(int)
    months_to_leave = rng.geometric(np.clip(hazard, 1e-4, 0.5))
    leaves = months_to_leave <= months_observable
    leave_date = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    leave_date[leaves] = observe_from[leaves] + pd.to_timedelta(
        months_to_leave[leaves] * 30.44 - rng.integers(0, 28, leaves.sum()), unit="D"
    )
    leave_reason = np.where(
        leaves,
        rng.choice(["Resignation", "End of contract", "Dismissal", "Retirement"], size=n, p=[0.72, 0.14, 0.08, 0.06]),
        None,
    )

    df = pd.DataFrame(
        {
            "employee_id": [f"E{100000 + i}" for i in range(n)],
            "department": dept,
            "country": country,
            "job_level": level,
            "gender": gender,
            "age": age,
            "hire_date": hire_date.normalize(),
            "leave_date": leave_date.dt.normalize(),
            "leave_reason": leave_reason,
            "annual_salary_eur": salary,
            "salary_to_market_ratio": market_ratio.round(2),
            "engagement_score": engagement,
            "avg_monthly_overtime_h": overtime_h,
            "remote_days_per_week": remote_days,
            "performance_rating": perf,
            "absence_days_last_12m": absence_days,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "employees.csv", index=False)
    return df


if __name__ == "__main__":
    data = main(Path(__file__).resolve().parents[1] / "data")
    print(f"Wrote {len(data)} employees, {data['leave_date'].notna().sum()} leavers")
