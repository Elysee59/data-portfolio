"""HR analytics: workforce KPIs, equal-pay check and attrition risk model.

Reads data/employees.csv (see generate_data.py), writes charts to figures/
and a KPI summary to results.md.

Usage: python src/analysis.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
COLORS = {"main": "#2563EB", "accent": "#F97316", "grey": "#9CA3AF"}

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "employees.csv", parse_dates=["hire_date", "leave_date"])
    return df


def monthly_headcount(df: pd.DataFrame) -> pd.DataFrame:
    months = pd.date_range("2022-01-31", "2025-12-31", freq="ME")
    rows = []
    for m in months:
        start = m - pd.offsets.MonthBegin(1)
        active = (df.hire_date <= m) & (df.leave_date.isna() | (df.leave_date > m))
        leavers = df.leave_date.between(start, m)
        hires = df.hire_date.between(start, m)
        rows.append({"month": m, "headcount": active.sum(), "hires": hires.sum(), "leavers": leavers.sum()})
    out = pd.DataFrame(rows)
    out["turnover_rate_12m"] = out.leavers.rolling(12).sum() / out.headcount.rolling(12).mean()
    return out


def turnover_by(df: pd.DataFrame, col: str, year: int = 2025) -> pd.Series:
    start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    present = (df.hire_date <= end) & (df.leave_date.isna() | (df.leave_date >= start))
    sub = df[present]
    left = sub.leave_date.between(start, end)
    return (left.groupby(sub[col]).mean()).sort_values(ascending=False)


def pay_gap(df: pd.DataFrame) -> pd.DataFrame:
    active = df[df.leave_date.isna()]
    g = active.groupby(["job_level", "gender"]).annual_salary_eur.median().unstack()
    g["gap_pct"] = (g["M"] - g["F"]) / g["M"] * 100
    order = ["Associate", "Professional", "Manager", "Director"]
    return g.loc[order]


def attrition_model(df: pd.DataFrame):
    """Predict who left in 2025 among employees active on 2025-01-01."""
    snap = pd.Timestamp("2025-01-01")
    pop = df[(df.hire_date < snap) & (df.leave_date.isna() | (df.leave_date >= snap))].copy()
    pop["tenure_years"] = (snap - pop.hire_date).dt.days / 365.25
    y = pop.leave_date.notna().astype(int)

    num = ["age", "tenure_years", "salary_to_market_ratio", "engagement_score", "avg_monthly_overtime_h",
           "remote_days_per_week", "performance_rating", "absence_days_last_12m"]
    cat = ["department", "country", "job_level", "gender"]
    X = pop[num + cat]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    models = {
        "Logistic regression": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Gradient boosting": Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0))]),
    }
    scores = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        scores[name] = {"ROC AUC": roc_auc_score(y_te, p), "PR AUC": average_precision_score(y_te, p)}
    best = max(scores, key=lambda k: scores[k]["ROC AUC"])
    imp = permutation_importance(models[best], X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=0)
    importance = pd.Series(imp.importances_mean, index=X.columns).sort_values()
    # Lift: how many leavers fall in the top 20 % risk scores?
    p = models[best].predict_proba(X_te)[:, 1]
    top = p >= np.quantile(p, 0.8)
    capture = y_te[top].sum() / y_te.sum()
    return scores, best, importance, y.mean(), capture


def main() -> None:
    FIG.mkdir(exist_ok=True)
    df = load()

    hc = monthly_headcount(df)
    fig, ax1 = plt.subplots(figsize=(8, 3.6))
    ax1.plot(hc.month, hc.headcount, color=COLORS["main"], lw=2, label="Headcount")
    ax1.set_ylabel("Headcount")
    ax2 = ax1.twinx()
    ax2.plot(hc.month, hc.turnover_rate_12m * 100, color=COLORS["accent"], lw=2, label="Rolling 12-month turnover (%)")
    ax2.set_ylabel("Turnover (%)")
    ax2.spines["top"].set_visible(False)
    ax1.set_title("Headcount and rolling 12-month turnover")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "headcount_turnover.png")
    plt.close(fig)

    t_dept = turnover_by(df, "department")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(t_dept.index[::-1], t_dept.values[::-1] * 100, color=COLORS["main"])
    for i, v in enumerate(t_dept.values[::-1] * 100):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    ax.set_xlabel("2025 turnover rate (%)")
    ax.set_title("Turnover by department, 2025")
    fig.tight_layout()
    fig.savefig(FIG / "turnover_by_department.png")
    plt.close(fig)

    gap = pay_gap(df)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(gap))
    ax.bar(x - 0.2, gap["F"] / 1000, 0.4, label="Women", color=COLORS["accent"])
    ax.bar(x + 0.2, gap["M"] / 1000, 0.4, label="Men", color=COLORS["main"])
    for i, g in enumerate(gap.gap_pct):
        ax.text(i, max(gap.iloc[i][["F", "M"]]) / 1000 + 2, f"gap {g:.1f}%", ha="center")
    ax.set_xticks(x, gap.index)
    ax.set_ylabel("Median salary (k€)")
    ax.set_title("Median salary by level and gender (active employees)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "pay_gap.png")
    plt.close(fig)

    scores, best, importance, base_rate, capture = attrition_model(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(importance.index, importance.values, color=COLORS["main"])
    ax.set_xlabel("Drop in ROC AUC when the feature is shuffled")
    ax.set_title(f"What drives attrition? ({best})")
    fig.tight_layout()
    fig.savefig(FIG / "attrition_drivers.png")
    plt.close(fig)

    last = hc.iloc[-1]
    lines = [
        "# Results (synthetic data)",
        "",
        f"- Headcount on 2025-12-31: **{int(last.headcount)}**",
        f"- 2025 turnover rate: **{last.turnover_rate_12m:.1%}**",
        f"- Highest-turnover department: **{t_dept.index[0]}** ({t_dept.iloc[0]:.1%}); lowest: {t_dept.index[-1]} ({t_dept.iloc[-1]:.1%})",
        f"- Median pay gap (men vs. women, same level): {', '.join(f'{lvl} {g:.1f}%' for lvl, g in gap.gap_pct.items())}",
        "",
        "## Attrition model (who leaves in 2025, from a 1 January snapshot)",
        "",
        f"Base leaving rate: {base_rate:.1%}",
        "",
        "| Model | ROC AUC | PR AUC |",
        "|---|---|---|",
        *[f"| {k} | {v['ROC AUC']:.3f} | {v['PR AUC']:.3f} |" for k, v in scores.items()],
        "",
        f"The 20 % of employees with the highest risk score account for **{capture:.0%}** of actual leavers (test set).",
        "",
        "Top drivers: " + ", ".join(importance.sort_values(ascending=False).index[:4]),
    ]
    (ROOT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
