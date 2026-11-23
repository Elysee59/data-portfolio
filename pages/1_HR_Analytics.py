import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_utils import BLUE, GREY, ORANGE, ROOT, hero, insight, setup
from analysis import attrition_model, monthly_headcount

setup("HR Analytics", "👥")
hero("👥 HR Analytics", "Turnover, equal pay and attrition risk for a 4,000-employee retail group (synthetic data).",
     ["Workday-style HRIS export", "SQL / Databricks", "Power BI DAX", "scikit-learn"])


@st.cache_data
def load() -> pd.DataFrame:
    return pd.read_csv(ROOT / "01-hr-analytics" / "data" / "employees.csv", parse_dates=["hire_date", "leave_date"])


@st.cache_resource
def model(df: pd.DataFrame):
    return attrition_model(df)


df_all = load()

# ------------------------------------------------------------------ filters
with st.container(border=True):
    f1, f2, f3 = st.columns([2, 2, 1])
    depts = f1.multiselect("Department", sorted(df_all.department.unique()), placeholder="All departments")
    countries = f2.multiselect("Country", sorted(df_all.country.unique()), placeholder="All countries")
    year = f3.selectbox("Year", [2025, 2024, 2023])

df = df_all
if depts:
    df = df[df.department.isin(depts)]
if countries:
    df = df[df.country.isin(countries)]
if df.empty:
    st.warning("No employees match these filters.")
    st.stop()

start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
present = df[(df.hire_date <= end) & (df.leave_date.isna() | (df.leave_date >= start))]
active_end = present[present.leave_date.isna() | (present.leave_date > end)]
leavers = present.leave_date.between(start, end)
prev = df[(df.hire_date <= start - pd.Timedelta(days=1)) &
          (df.leave_date.isna() | (df.leave_date >= start - pd.DateOffset(years=1)))]
prev_turnover = prev.leave_date.between(start - pd.DateOffset(years=1), start - pd.Timedelta(days=1)).mean()

hc_all = monthly_headcount(df)


def turnover(y: int) -> float:
    """Leavers during the year / average month-end headcount (standard HR definition)."""
    h = hc_all[hc_all.month.dt.year == y]
    return h.leavers.sum() / h.headcount.mean()


k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Headcount (31 Dec)", f"{len(active_end):,}")
k2.metric("Turnover rate", f"{turnover(year):.1%}",
          f"{(turnover(year) - turnover(year - 1)) * 100:+.1f} pts vs {year - 1}" if year > 2022 else None,
          delta_color="inverse")
k3.metric("Hires", f"{df.hire_date.between(start, end).sum():,}")
k4.metric("Avg absence (days)", f"{active_end.absence_days_last_12m.mean():.1f}")
k5.metric("Women", f"{(active_end.gender == 'F').mean():.0%}")
st.caption("Turnover = leavers during the year ÷ average month-end headcount.")

tab1, tab2, tab3 = st.tabs(["📊 Workforce overview", "⚖️ Equal pay", "🎯 Attrition risk model"])

# ------------------------------------------------------------------ overview
with tab1:
    hc = hc_all
    fig = go.Figure()
    fig.add_bar(x=hc.month, y=hc.hires, name="Hires", marker_color="rgba(37,99,235,0.35)", yaxis="y2")
    fig.add_bar(x=hc.month, y=-hc.leavers, name="Leavers", marker_color="rgba(249,115,22,0.45)", yaxis="y2")
    fig.add_scatter(x=hc.month, y=hc.headcount, name="Headcount", line=dict(color=BLUE, width=3))
    fig.update_layout(title="Headcount, hires and leavers per month", barmode="relative", height=380,
                      yaxis=dict(title="Headcount"),
                      yaxis2=dict(title="Hires / leavers", overlaying="y", side="right", showgrid=False,
                                  tickformat="d"))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    by_dept = (present.assign(left=leavers).groupby("department").left.mean().sort_values() * 100).round(1)
    fig = px.bar(by_dept, orientation="h", text=by_dept.map(lambda v: f"{v:.1f}%"),
                 title=f"Share of employees who left, by department ({year})",
                 labels={"value": "% of employees present during the year", "department": ""})
    fig.update_traces(marker_color=BLUE, textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, height=360, xaxis_range=[0, by_dept.max() * 1.2])
    c1.plotly_chart(fig, use_container_width=True)

    reasons = present[leavers].leave_reason.value_counts()
    fig = px.pie(values=reasons.values, names=reasons.index, hole=0.55, title=f"Why people left in {year}")
    fig.update_layout(height=360)
    c2.plotly_chart(fig, use_container_width=True)

    worst = by_dept.index[-1]
    insight(f"<b>{worst}</b> loses the largest share of its people in {year} ({by_dept.iloc[-1]:.1f} %). "
            f"Resignations are the main reason for leaving, so most departures are avoidable.")

# ------------------------------------------------------------------ equal pay
with tab2:
    order = ["Associate", "Professional", "Manager", "Director"]
    med = active_end.groupby(["job_level", "gender"]).annual_salary_eur.median().unstack().reindex(order)
    med["gap"] = (med["M"] - med["F"]) / med["M"] * 100
    c1, c2 = st.columns([3, 2])
    fig = go.Figure()
    fig.add_bar(x=med.index, y=med["F"], name="Women", marker_color=ORANGE)
    fig.add_bar(x=med.index, y=med["M"], name="Men", marker_color=BLUE)
    fig.update_layout(title="Median annual salary by level (€)", barmode="group", height=380)
    c1.plotly_chart(fig, use_container_width=True)

    fig = px.bar(med.reset_index(), x="job_level", y="gap", text=med.gap.map(lambda v: f"{v:.1f}%").values,
                 title="Raw pay gap, men vs. women (%)", labels={"job_level": "", "gap": "Gap (%)"})
    fig.update_traces(marker_color=[ORANGE if g > 5 else GREY for g in med.gap], textposition="outside", cliponaxis=False)
    fig.add_hline(y=5, line_dash="dot", annotation_text="5 % alert threshold")
    fig.update_layout(height=380)
    c2.plotly_chart(fig, use_container_width=True)

    counts = active_end.groupby(["job_level", "gender"]).size().unstack().reindex(order)
    st.dataframe(pd.DataFrame({"Women": counts["F"], "Men": counts["M"],
                               "Median women (€)": med["F"].round(-2), "Median men (€)": med["M"].round(-2),
                               "Gap (%)": med.gap.round(1)}), use_container_width=True)
    insight("Raw gaps are not adjusted for department, country or tenure, and some groups are small. "
            "A proper equal-pay audit would use a regression controlling for these factors before drawing conclusions.")

# ------------------------------------------------------------------ attrition model
with tab3:
    scores, best, importance, base_rate, capture, fitted = model(df_all)
    st.caption("Model trained on all employees active on 1 January 2025 (filters above do not apply here). "
               "Target: did the employee leave during 2025?")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model", "Logistic reg." if best == "Logistic regression" else best)
    m2.metric("ROC AUC (test set)", f"{scores[best]['ROC AUC']:.2f}")
    m3.metric("Leavers caught in top 20 % risk", f"{capture:.0%}", f"vs 20 % with no model")

    c1, c2 = st.columns([1, 1])
    labels = {"engagement_score": "Engagement score", "salary_to_market_ratio": "Salary vs. market",
              "avg_monthly_overtime_h": "Overtime", "department": "Department", "job_level": "Job level",
              "remote_days_per_week": "Remote days", "country": "Country", "performance_rating": "Performance",
              "tenure_years": "Tenure", "age": "Age", "gender": "Gender", "absence_days_last_12m": "Absence days"}
    imp = importance.rename(index=labels).sort_values()
    imp = imp[imp > 0.001]
    fig = px.bar(imp, orientation="h", title="What drives attrition? (permutation importance)",
                 labels={"value": "Drop in ROC AUC when shuffled", "index": ""})
    fig.update_traces(marker_color=BLUE)
    fig.update_layout(showlegend=False, height=420)
    c1.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### 🔮 What-if simulator")
        st.caption("Change an employee profile and see the predicted probability of leaving within a year.")
        s1, s2 = st.columns(2)
        dept = s1.selectbox("Department ", sorted(df_all.department.unique()), index=4)
        level = s2.selectbox("Job level", ["Associate", "Professional", "Manager", "Director"])
        engagement = st.slider("Engagement score (/5)", 1.0, 5.0, 3.6, 0.1)
        market = st.slider("Salary vs. market (1.0 = at market)", 0.70, 1.30, 1.00, 0.01)
        overtime = st.slider("Average monthly overtime (hours)", 0, 40, 6)
        age = st.slider("Age", 20, 64, 32)
        profile = pd.DataFrame([{
            "age": age, "tenure_years": 3.0, "salary_to_market_ratio": market, "engagement_score": engagement,
            "avg_monthly_overtime_h": overtime, "remote_days_per_week": 1, "performance_rating": 3,
            "absence_days_last_12m": 6, "department": dept, "country": "France", "job_level": level, "gender": "F",
        }])
        p = float(fitted.predict_proba(profile)[0, 1])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=p * 100, number={"suffix": " %", "font": {"size": 40}},
            delta={"reference": base_rate * 100, "suffix": " pts", "increasing": {"color": "#DC2626"},
                   "decreasing": {"color": "#16A34A"}},
            gauge={"axis": {"range": [0, 60]}, "bar": {"color": BLUE},
                   "threshold": {"line": {"color": ORANGE, "width": 3}, "value": base_rate * 100}},
            title={"text": "Probability of leaving in the next 12 months", "font": {"size": 15}}))
        gauge.update_layout(height=320, margin=dict(t=70, b=20, l=30, r=30))
        st.plotly_chart(gauge, use_container_width=True)
        st.caption(f"Orange line = company average ({base_rate:.1%}).")

    insight("Engagement, pay versus market and overtime are the strongest levers, and all three can be acted on. "
            "Recommended use: a quarterly list for HR business partners with the reasons behind each risk, "
            "never an automated individual decision (GDPR).")
