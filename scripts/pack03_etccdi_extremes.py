#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from menkao_analysis.common import ensure_dir, load_daily_data, etccdi_annual, mk_test

INPUT_XLSX = REPO_ROOT / "data" / "raw" / "FINAL1981to2025_CHIRPS_Daily_Mean_Min_Max.xlsx"
OUT = ensure_dir(REPO_ROOT / "outputs" / "pack03_etccdi_extremes")
DATA = ensure_dir(OUT / "data")
FIG = ensure_dir(OUT / "figures")


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()


df = load_daily_data(INPUT_XLSX)
annual, thresholds = etccdi_annual(df)
annual.to_csv(DATA / "annual_etccdi_indices.csv", index=False)
thresholds.to_csv(DATA / "etccdi_reference_thresholds.csv", index=False)

trend_rows = []
for col in annual.columns:
    if col == "year":
        continue
    mk = mk_test(annual[col])
    trend_rows.append({
        "metric": col,
        "sen_slope_per_year": mk["sen_slope"],
        "sen_slope_per_decade": mk["sen_slope"] * 10 if pd.notna(mk["sen_slope"]) else np.nan,
        "tau": mk["tau"],
        "p_value": mk["p_value"],
        "trend": mk["trend"],
    })
pd.DataFrame(trend_rows).to_csv(DATA / "annual_etccdi_trend_tests.csv", index=False)

monthly_clim = df.groupby("month", as_index=False).agg(
    mean_r20_days=("p_mean_mm", lambda s: (s >= 20).sum() / len(df["year"].unique())),
    mean_rx1_mm=("p_mean_mm", "max"),
    mean_p95_excess_mm=("p_mean_mm", lambda s: s[s > thresholds.loc[0, "p95_wet_mm"]].sum() / len(df["year"].unique())),
)
monthly_clim.to_csv(DATA / "monthly_extremes_climatology.csv", index=False)

ym = df.groupby(["year", "month"], as_index=False).agg(
    R20mm_days=("p_mean_mm", lambda s: (s >= 20).sum()),
    R95pTOT_mm=("p_mean_mm", lambda s: s[s > thresholds.loc[0, "p95_wet_mm"]].sum()),
    Rx1day_mm=("p_mean_mm", "max"),
)
ym.to_csv(DATA / "year_month_extreme_counts.csv", index=False)

top_years = annual[["year","PRCPTOT_mm","Rx1day_mm","Rx5day_mm","R20mm_days","CDD_days","CWD_days","R95pTOT_mm","MeanRangeWet_mm"]]
top_years.to_csv(DATA / "top_years_selected_indices.csv", index=False)

events = df.nlargest(10, "p_mean_mm")[["date","p_mean_mm","p_max_mm","p_min_mm","amp_mm"]]
events.to_csv(DATA / "top10_daily_extreme_events.csv", index=False)

for metric, fname, ylabel in [
    ("PRCPTOT_mm", "fig01_prcptot_annual_trend", "mm"),
    ("Rx1day_mm", "fig02_rx1day_annual_trend", "mm"),
    ("Rx3day_mm", "fig03_rx3day_annual_trend", "mm"),
    ("R20mm_days", "fig04_r20mm_days_annual_trend", "days"),
    ("CDD_days", "fig05_cdd_days_annual_trend", "days"),
    ("CWD_days", "fig06_cwd_days_annual_trend", "days"),
    ("R95pTOT_mm", "fig07_r95ptot_annual_trend", "mm"),
    ("MeanRangeWet_mm", "fig08_meanrangewet_annual_trend", "mm"),
]:
    fig = annual[["year", metric]].copy()
    fig.to_csv(FIG / f"{fname}.csv", index=False)
    plt.figure(figsize=(9, 4))
    plt.plot(fig["year"], fig[metric], marker="o", linewidth=1)
    plt.xlabel("Year"); plt.ylabel(ylabel); plt.title(metric)
    savefig(fname)

monthly_counts = ym.groupby("month", as_index=False).agg(
    mean_r20_days=("R20mm_days", "mean"),
    mean_r95ptot_mm=("R95pTOT_mm", "mean"),
    mean_rx1_mm=("Rx1day_mm", "mean"),
)
monthly_counts.to_csv(FIG / "fig09_monthly_extreme_counts.csv", index=False)
plt.figure(figsize=(8,4))
plt.plot(monthly_counts["month"], monthly_counts["mean_r20_days"], marker="o", label="R20mm days")
plt.plot(monthly_counts["month"], monthly_counts["mean_rx1_mm"], marker="o", label="Rx1day")
plt.xlabel("Month"); plt.ylabel("Mean value"); plt.title("Monthly extreme counts and intensity"); plt.legend()
savefig("fig09_monthly_extreme_counts")

monthly_rx1 = ym.groupby("month", as_index=False)["Rx1day_mm"].mean()
monthly_rx1.to_csv(FIG / "fig10_monthly_rx1_climatology.csv", index=False)
plt.figure(figsize=(8,4))
plt.bar(monthly_rx1["month"], monthly_rx1["Rx1day_mm"])
plt.xlabel("Month"); plt.ylabel("Mean monthly Rx1day (mm)"); plt.title("Monthly Rx1day climatology")
savefig("fig10_monthly_rx1_climatology")

heat_r20 = ym.pivot(index="year", columns="month", values="R20mm_days")
heat_r20.to_csv(FIG / "fig11_heatmap_r20_counts.csv")
plt.figure(figsize=(10,6))
plt.imshow(heat_r20.values, aspect="auto", interpolation="nearest")
plt.colorbar(label="R20mm days")
plt.xlabel("Month"); plt.ylabel("Year index"); plt.title("Heatmap of monthly R20mm counts")
savefig("fig11_heatmap_r20_counts")

heat_r95 = ym.pivot(index="year", columns="month", values="R95pTOT_mm")
heat_r95.to_csv(FIG / "fig12_heatmap_r95ptot.csv")
plt.figure(figsize=(10,6))
plt.imshow(heat_r95.values, aspect="auto", interpolation="nearest")
plt.colorbar(label="R95pTOT (mm)")
plt.xlabel("Month"); plt.ylabel("Year index"); plt.title("Heatmap of monthly R95pTOT")
savefig("fig12_heatmap_r95ptot")

print(f"Pack 03 outputs written to {OUT}")
