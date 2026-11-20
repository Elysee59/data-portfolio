"""Weekly demand forecasting per store and product family.

Compares two baselines with a gradient-boosting model on a rolling-origin
backtest (12 weekly cut-offs), forecasting 4 weeks ahead each time.
Promotions are assumed known in advance (they are planned by marketing).

Usage: python src/forecast.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
HORIZON = 4          # weeks ahead
N_CUTOFFS = 12       # backtest origins
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week"] = df.date.dt.to_period("W-SUN").dt.start_time
    w = (df.groupby(["store_id", "family", "week"])
           .agg(units=("units_sold", "sum"), promo_days=("promo", "sum"), holidays=("holiday", "sum"))
           .reset_index())
    # Drop incomplete first/last weeks
    counts = df.groupby("week").date.nunique()
    full = counts[counts == 7].index
    return w[w.week.isin(full)].sort_values(["store_id", "family", "week"]).reset_index(drop=True)


def add_features(w: pd.DataFrame) -> pd.DataFrame:
    """Features use only information available HORIZON weeks before the target week."""
    w = w.copy()
    g = w.groupby(["store_id", "family"]).units
    for lag in (HORIZON, HORIZON + 1, HORIZON + 2, 52):
        w[f"lag_{lag}"] = g.shift(lag)
    w["roll_mean_8"] = g.transform(lambda s: s.shift(HORIZON).rolling(8).mean())
    w["roll_mean_26"] = g.transform(lambda s: s.shift(HORIZON).rolling(26).mean())
    w["week_of_year"] = w.week.dt.isocalendar().week.astype(int)
    w["month"] = w.week.dt.month
    w["store_code"] = w.store_id.str[1:].astype(int)
    w["family_code"] = w.family.astype("category").cat.codes
    return w


FEATURES = ["lag_4", "lag_5", "lag_6", "lag_52", "roll_mean_8", "roll_mean_26", "week_of_year", "month",
            "store_code", "family_code", "promo_days", "holidays"]


def wape(y, p):
    return np.abs(y - p).sum() / np.abs(y).sum()


def backtest(w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weeks = np.sort(w.week.unique())
    cutoffs = weeks[-(N_CUTOFFS + HORIZON):-HORIZON]  # each cut-off has HORIZON full weeks after it
    preds = []
    for cut in cutoffs:
        train = w[(w.week <= cut) & w[FEATURES].notna().all(axis=1)]
        test_weeks = weeks[(weeks > cut)][:HORIZON]
        test = w[w.week.isin(test_weeks)].copy()
        model = HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.05,
                                              max_leaf_nodes=31, random_state=0)
        model.fit(train[FEATURES], train.units)
        test["pred_gbm"] = model.predict(test[FEATURES])
        last = w[w.week == cut].set_index(["store_id", "family"]).units
        test["pred_naive"] = test.set_index(["store_id", "family"]).index.map(last).values
        test["pred_seasonal"] = test["lag_52"]
        test["cutoff"] = cut
        preds.append(test)
    p = pd.concat(preds)
    scores = pd.DataFrame({
        name: {"WAPE": wape(p.units, p[col]), "Bias": (p[col].sum() - p.units.sum()) / p.units.sum()}
        for name, col in [("Last week (naive)", "pred_naive"), ("Same week last year", "pred_seasonal"),
                          ("Gradient boosting", "pred_gbm")]
    }).T
    return p, scores


def main() -> None:
    FIG.mkdir(exist_ok=True)
    sales = pd.read_csv(ROOT / "data" / "sales.csv", parse_dates=["date"])
    w = add_features(weekly(sales))
    preds, scores = backtest(w)

    # Chart 1: total weekly units, actual vs. forecasts on the backtest window (the 4th week after each cut-off)
    first = preds[preds.week == preds.cutoff + pd.Timedelta(weeks=HORIZON)]
    tot = first.groupby("week")[["units", "pred_gbm", "pred_seasonal"]].sum()
    hist = w[w.week >= tot.index.min() - pd.Timedelta(weeks=40)].groupby("week").units.sum()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(hist.index, hist.values, color="#111827", lw=1.6, label="Actual")
    ax.plot(tot.index, tot.pred_gbm, color="#2563EB", lw=2, marker="o", ms=3, label="Gradient boosting (4 weeks ahead)")
    ax.plot(tot.index, tot.pred_seasonal, color="#9CA3AF", lw=1.5, ls="--", label="Same week last year")
    ax.set_title("Total weekly units: actual vs. 4-week-ahead forecasts")
    ax.set_ylabel("Units")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "forecast_vs_actual.png")
    plt.close(fig)

    # Chart 2: error by family
    fam = preds.groupby("family").apply(
        lambda d: pd.Series({"Gradient boosting": wape(d.units, d.pred_gbm), "Same week last year": wape(d.units, d.pred_seasonal)}),
        include_groups=False)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(fam))
    ax.bar(x - 0.2, fam["Same week last year"] * 100, 0.4, color="#9CA3AF", label="Same week last year")
    ax.bar(x + 0.2, fam["Gradient boosting"] * 100, 0.4, color="#2563EB", label="Gradient boosting")
    ax.set_xticks(x, fam.index)
    ax.set_ylabel("WAPE (%) - lower is better")
    ax.set_title("Forecast error by product family")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "error_by_family.png")
    plt.close(fig)

    lines = ["# Backtest results (synthetic data)", "",
             f"{N_CUTOFFS} weekly cut-offs, horizon {HORIZON} weeks, 12 stores x 4 families.", "",
             "| Model | WAPE | Bias |", "|---|---|---|",
             *[f"| {k} | {v.WAPE:.1%} | {v.Bias:+.1%} |" for k, v in scores.iterrows()], "",
             "WAPE by family (gradient boosting): " + ", ".join(f"{k} {v:.1%}" for k, v in fam["Gradient boosting"].items())]
    (ROOT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
