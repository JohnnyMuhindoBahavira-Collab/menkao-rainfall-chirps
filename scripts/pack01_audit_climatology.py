#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from menkao_analysis.common import ensure_dir, load_daily_data, annual_core_indices, rainfall_class_distribution

INPUT_XLSX = REPO_ROOT / "data" / "raw" / "FINAL1981to2025_CHIRPS_Daily_Mean_Min_Max.xlsx"
OUT = ensure_dir(REPO_ROOT / "outputs" / "pack01_audit_climatology")
DATA = ensure_dir(OUT / "data")
FIG = ensure_dir(OUT / "figures")


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()


df = load_daily_data(INPUT_XLSX)
daily_enriched = df.rename(columns={"date":"Date","p_mean_mm":"P_mean_mm","p_min_mm":"P_min_mm","p_max_mm":"P_max_mm","amp_mm":"Range_mm"})
daily_enriched.to_csv(DATA / "daily_enriched_1981_2025.csv", index=False)

qa = pd.DataFrame([{
    "n_rows": len(df),
    "start_date": df["date"].min().date().isoformat(),
    "end_date": df["date"].max().date().isoformat(),
    "duplicate_dates": int(df["date"].duplicated().sum()),
    "missing_p_mean": int(df["p_mean_mm"].isna().sum()),
    "coherence_violations": int(((df["p_min_mm"] > df["p_mean_mm"]) | (df["p_mean_mm"] > df["p_max_mm"])).sum()),
}])
qa.to_csv(DATA / "qa_summary.csv", index=False)

annual = annual_core_indices(df)
annual.to_csv(DATA / "annual_summary.csv", index=False)

monthly = df.groupby("month", as_index=False).agg(
    mean_daily_mm=("p_mean_mm", "mean"),
    median_daily_mm=("p_mean_mm", "median"),
    wet_day_percent=("wet", lambda s: 100 * s.mean()),
    mean_range_mm=("amp_mm", "mean"),
)
monthly.to_csv(DATA / "monthly_climatology.csv", index=False)

summary = pd.DataFrame([
    {"metric":"annual_total_mean_mm","value":annual["total_mm"].mean()},
    {"metric":"annual_total_median_mm","value":annual["total_mm"].median()},
    {"metric":"wet_day_percent","value":100*df["wet"].mean()},
    {"metric":"mean_daily_range_mm","value":df["amp_mm"].mean()},
    {"metric":"mean_wet_day_range_mm","value":df.loc[df["wet"],"amp_mm"].mean()},
    {"metric":"max_1day_pmean_mm","value":df["p_mean_mm"].max()},
    {"metric":"max_1day_pmax_mm","value":df["p_max_mm"].max()},
])
summary.to_csv(DATA / "summary_statistics.csv", index=False)

class_dist = rainfall_class_distribution(df)
class_dist.to_csv(DATA / "rainfall_class_distribution.csv", index=False)

annual.nlargest(10, "total_mm")[["year","total_mm"]].to_csv(DATA / "top10_wettest_years.csv", index=False)
annual.nsmallest(10, "total_mm")[["year","total_mm"]].to_csv(DATA / "top10_driest_years.csv", index=False)

fig = annual[["year","total_mm"]]
fig.to_csv(FIG / "fig01_annual_total_mm.csv", index=False)
plt.figure(figsize=(9,4))
plt.plot(fig["year"], fig["total_mm"], marker="o", linewidth=1)
plt.xlabel("Year"); plt.ylabel("Annual total rainfall (mm)"); plt.title("Annual total rainfall")
savefig("fig01_annual_total_mm")

fig = monthly[["month","mean_daily_mm"]]
fig.to_csv(FIG / "fig02_monthly_mean_daily_mm.csv", index=False)
plt.figure(figsize=(8,4))
plt.plot(fig["month"], fig["mean_daily_mm"], marker="o")
plt.xlabel("Month"); plt.ylabel("Mean daily rainfall (mm)"); plt.title("Monthly mean daily rainfall")
savefig("fig02_monthly_mean_daily_mm")

fig = monthly[["month","wet_day_percent"]]
fig.to_csv(FIG / "fig03_monthly_wet_day_pct.csv", index=False)
plt.figure(figsize=(8,4))
plt.bar(fig["month"], fig["wet_day_percent"])
plt.xlabel("Month"); plt.ylabel("Wet-day share (%)"); plt.title("Monthly wet-day percentage")
savefig("fig03_monthly_wet_day_pct")

fig = annual[["year","rx1_mean_mm"]]
fig.to_csv(FIG / "fig04_annual_max_1day_mm.csv", index=False)
plt.figure(figsize=(9,4))
plt.plot(fig["year"], fig["rx1_mean_mm"], marker="o", linewidth=1)
plt.xlabel("Year"); plt.ylabel("Rx1day area mean (mm)"); plt.title("Annual maximum 1-day area-mean rainfall")
savefig("fig04_annual_max_1day_mm")

fig = monthly[["month","mean_range_mm"]]
fig.to_csv(FIG / "fig05_monthly_mean_range_mm.csv", index=False)
plt.figure(figsize=(8,4))
plt.plot(fig["month"], fig["mean_range_mm"], marker="o")
plt.xlabel("Month"); plt.ylabel("Mean daily spatial range (mm)"); plt.title("Monthly mean spatial range")
savefig("fig05_monthly_mean_range_mm")

fig = annual[["year","mean_range_mm"]]
fig.to_csv(FIG / "fig06_annual_mean_range_mm.csv", index=False)
plt.figure(figsize=(9,4))
plt.plot(fig["year"], fig["mean_range_mm"], marker="o", linewidth=1)
plt.xlabel("Year"); plt.ylabel("Mean spatial range (mm)"); plt.title("Annual mean spatial range")
savefig("fig06_annual_mean_range_mm")

fig = class_dist.copy()
fig.to_csv(FIG / "fig07_rainfall_class_distribution.csv", index=False)
plt.figure(figsize=(8,4))
plt.bar(fig["class_mm"], fig["days"])
plt.xlabel("Class (mm)"); plt.ylabel("Days"); plt.title("Rainfall class distribution")
savefig("fig07_rainfall_class_distribution")

print(f"Pack 01 outputs written to {OUT}")
