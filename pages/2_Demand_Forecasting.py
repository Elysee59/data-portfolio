import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_utils import BLUE, GREY, ORANGE, ROOT, hero, insight, setup

setup("Demand Forecasting", "📈")
hero("📈 Demand Forecasting", "Weekly sales forecasts 4 weeks ahead for 12 stores × 4 product families of a sports retailer (synthetic data).",
     ["Rolling backtest", "No data leakage", "Gradient boosting (Poisson)", "WAPE & bias"])

DATA = ROOT / "02-sales-forecasting" / "data"


@st.cache_data
def load():
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["date"])
    sales["week"] = sales.date.dt.to_period("W-SUN").dt.start_time
    full_weeks = sales.groupby("week").date.nunique().loc[lambda s: s == 7].index  # drop partial weeks
    weekly = sales[sales.week.isin(full_weeks)].groupby(["store_id", "family", "week"], as_index=False).units_sold.sum()
    preds = pd.read_csv(DATA / "backtest_predictions.csv", parse_dates=["cutoff", "week"])
    return weekly, preds


def wape(y, p):
    return np.abs(y - p).sum() / np.abs(y).sum()


weekly, preds = load()

MODELS = {"pred_gbm": "Gradient boosting", "pred_seasonal": "Same week last year (current practice)",
          "pred_naive": "Last week (naive)"}

# ------------------------------------------------------------------ headline
k = {name: (wape(preds.units, preds[col]), (preds[col].sum() - preds.units.sum()) / preds.units.sum())
     for col, name in MODELS.items()}
gbm, seas = k["Gradient boosting"], k["Same week last year (current practice)"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast error (WAPE) - model", f"{gbm[0]:.1%}", f"{(gbm[0] - seas[0]) * 100:+.1f} pts vs current practice",
          delta_color="inverse")
c2.metric("Error reduction", f"{1 - gbm[0] / seas[0]:.0%}")
c3.metric("Bias - model", f"{gbm[1]:+.1%}")
c4.metric("Bias - current practice", f"{seas[1]:+.1%}", "systematic under-ordering", delta_color="off")

tab1, tab2, tab3 = st.tabs(["🔎 Explore a forecast", "🗺️ Error map", "🧠 Method"])

# ------------------------------------------------------------------ explorer
with tab1:
    # Default to a typical series (median improvement), not a cherry-picked or worst one
    per = preds.groupby(["store_id", "family"]).apply(
        lambda d: wape(d.units, d.pred_seasonal) - wape(d.units, d.pred_gbm), include_groups=False)
    d_store, d_family = (per - per.median()).abs().idxmin()
    stores, families = sorted(preds.store_id.unique()), sorted(preds.family.unique())
    f1, f2, f3 = st.columns([1, 1, 2])
    store = f1.selectbox("Store", stores, index=stores.index(d_store))
    family = f2.selectbox("Product family", families, index=families.index(d_family))
    cutoffs = sorted(preds.cutoff.unique())
    cut = f3.select_slider("Forecast made on (cut-off week)", options=cutoffs, value=cutoffs[-1],
                           format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))

    hist = weekly[(weekly.store_id == store) & (weekly.family == family)]
    hist = hist[hist.week >= pd.Timestamp(cut) - pd.Timedelta(weeks=30)]
    fc = preds[(preds.store_id == store) & (preds.family == family) & (preds.cutoff == cut)]

    fig = go.Figure()
    fig.add_scatter(x=hist.week, y=hist.units_sold, name="Actual", line=dict(color="#334155", width=2))
    fig.add_scatter(x=fc.week, y=fc.pred_seasonal, name="Same week last year", mode="lines+markers",
                    line=dict(color=GREY, dash="dash"))
    fig.add_scatter(x=fc.week, y=fc.pred_gbm, name="Model forecast", mode="lines+markers",
                    line=dict(color=BLUE, width=3), marker=dict(size=8))
    fig.add_vline(x=pd.Timestamp(cut), line_dash="dot", line_color=ORANGE)
    fig.add_annotation(x=pd.Timestamp(cut), y=1, yref="paper", text="forecast date", showarrow=False,
                       font=dict(color=ORANGE), xanchor="right")
    fig.add_vrect(x0=pd.Timestamp(cut), x1=fc.week.max() + pd.Timedelta(days=3), fillcolor=BLUE, opacity=0.06,
                  line_width=0)
    fig.update_layout(title=f"{family} · store {store}: weekly units", height=440, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    sel = preds[(preds.store_id == store) & (preds.family == family)]
    s1, s2 = st.columns(2)
    s1.metric(f"Model error on this series (all cut-offs)", f"{wape(sel.units, sel.pred_gbm):.1%}")
    s2.metric("Current practice error on this series", f"{wape(sel.units, sel.pred_seasonal):.1%}")
    table = fc[["week", "units", "pred_gbm", "pred_seasonal"]].rename(columns={
        "week": "Week", "units": "Actual", "pred_gbm": "Model", "pred_seasonal": "Same week last year"})
    table["Week"] = table["Week"].dt.strftime("%d %b %Y")
    st.dataframe(table.round(0), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ error map
with tab2:
    m = preds.groupby(["store_id", "family"]).apply(
        lambda d: pd.Series({"model": wape(d.units, d.pred_gbm), "current": wape(d.units, d.pred_seasonal)}),
        include_groups=False).reset_index()
    m["gain"] = (m.current - m.model) * 100
    c1, c2 = st.columns(2)
    heat = m.pivot(index="family", columns="store_id", values="model") * 100
    fig = px.imshow(heat.round(1), text_auto=True, color_continuous_scale="Blues", aspect="auto",
                    title="Model error (WAPE %) by store and family", labels=dict(color="WAPE %", x="", y=""))
    fig.update_layout(height=360)
    c1.plotly_chart(fig, use_container_width=True)
    gain = m.pivot(index="family", columns="store_id", values="gain")
    fig = px.imshow(gain.round(1), text_auto=True, color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                    aspect="auto", title="Error reduction vs. current practice (points)", labels=dict(color="pts", x="", y=""))
    fig.update_layout(height=360)
    c2.plotly_chart(fig, use_container_width=True)

    fam = m.groupby("family")[["model", "current"]].mean() * 100
    fig = go.Figure()
    fig.add_bar(x=fam.index, y=fam.current, name="Same week last year", marker_color=GREY)
    fig.add_bar(x=fam.index, y=fam.model, name="Model", marker_color=BLUE)
    fig.update_layout(title="Average error by product family (WAPE %)", barmode="group", height=340)
    st.plotly_chart(fig, use_container_width=True)
    insight("The model beats the current practice on most store × family combinations and on every family on average. "
            "Cycling and Water sports remain the hardest (strong seasonality, frequent promotions): "
            "they are the first candidates for safety-stock rules.")

# ------------------------------------------------------------------ method
with tab3:
    st.markdown("""
**Business setting.** Orders are placed 4 weeks in advance, at the store × product-family level. Today, store managers
copy the sales of the same week last year.

**Approach**
1. Daily sales aggregated to weekly units per store and family.
2. Features available **4 weeks before** the target week only: lagged sales, 8- and 26-week rolling means,
   sales 52 weeks earlier, calendar, planned promotions and public holidays. No information from the future leaks in.
3. **Rolling-origin backtest**: 12 successive cut-off dates, each time training on the past and forecasting the next
   4 weeks, exactly as the model would be used in production.
4. Metrics: **WAPE** (weighted absolute percentage error, the retail standard) and **bias**
   (systematic over- or under-forecasting, which drives over-stock or stock-outs).
""")
    res = pd.DataFrame({name: {"WAPE": f"{v[0]:.1%}", "Bias": f"{v[1]:+.1%}"} for name, v in k.items()}).T
    st.dataframe(res, use_container_width=True)
    st.markdown("**Next steps on real data:** weather and local events, product-level forecasts, turning forecasts "
                "into order quantities with a service-level target, and a scheduled pipeline (Databricks job) with "
                "error monitoring.")
