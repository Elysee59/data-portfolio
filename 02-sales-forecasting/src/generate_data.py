"""Generate synthetic daily sales for a fictional sports retailer.

12 stores x 4 product families, 2023-01-01 to 2025-12-31, with trend,
weekly and yearly seasonality, promotions, French public holidays and noise.

Usage: python src/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
DATES = pd.date_range("2023-01-01", "2025-12-31", freq="D")
FAMILIES = {
    # family: (base daily units, yearly peak day-of-year, yearly amplitude)
    "Running": (40, 100, 0.25),       # spring peak
    "Cycling": (25, 160, 0.45),       # early summer
    "Water sports": (18, 195, 0.80),  # strong summer peak
    "Winter sports": (15, 10, 0.90),  # winter peak
}
HOLIDAYS = pd.to_datetime([
    f"{y}-{md}" for y in (2023, 2024, 2025)
    for md in ("01-01", "05-01", "05-08", "07-14", "08-15", "11-01", "11-11", "12-25")
] + ["2023-04-10", "2024-04-01", "2025-04-21", "2023-05-18", "2024-05-09", "2025-05-29"])


def main(out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    t = np.arange(len(DATES))
    doy = DATES.dayofyear.values
    dow = DATES.dayofweek.values
    weekly = np.array([0.85, 0.80, 0.90, 0.92, 1.05, 1.45, 0.55])[dow]  # Sunday mostly closed
    holiday = np.isin(DATES, HOLIDAYS)

    for store in range(1, 13):
        store_size = rng.uniform(0.6, 1.6)
        store_growth = rng.normal(0.08, 0.05)  # yearly growth
        for fam, (base, peak, amp) in FAMILIES.items():
            yearly = 1 + amp * np.cos(2 * np.pi * (doy - peak) / 365.25)
            trend = (1 + store_growth) ** (t / 365.25)
            # Promotions: ~6 % of days, in short bursts
            promo = np.zeros(len(DATES), dtype=int)
            for start in rng.choice(len(DATES) - 7, size=14, replace=False):
                promo[start:start + rng.integers(3, 8)] = 1
            mu = base * store_size * yearly * weekly * trend * np.where(promo == 1, 1.35, 1.0)
            mu = np.where(holiday, mu * 0.3, mu)
            units = rng.negative_binomial(n=8, p=8 / (8 + mu))
            rows.append(pd.DataFrame({
                "date": DATES, "store_id": f"S{store:02d}", "family": fam,
                "promo": promo, "holiday": holiday.astype(int), "units_sold": units,
            }))
    df = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sales.csv", index=False)
    return df


if __name__ == "__main__":
    d = main(Path(__file__).resolve().parents[1] / "data")
    print(f"Wrote {len(d):,} rows")
