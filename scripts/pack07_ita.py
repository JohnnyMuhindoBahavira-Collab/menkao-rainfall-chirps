#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from menkao_analysis.common import ensure_dir, load_daily_data, annual_core_indices, ita_summary

INPUT_XLSX = REPO_ROOT / "data" / "raw" / "FINAL1981to2025_CHIRPS_Daily_Mean_Min_Max.xlsx"
OUT = ensure_dir(REPO_ROOT / "outputs" / "pack07_ita")
DATA = ensure_dir(OUT / "data")
FIG = ensure_dir(OUT / "figures")


def ita_pairs(series):
    s = pd.Series(series).dropna().to_numpy(dtype=float)
    if len(s) % 2 == 1:
        mid = len(s)//2
        s = np.concatenate([s[:mid], s[mid+1:]])
    half = len(s)//2
    return np.sort(s[:half]), np.sort(s[half:])


def save_scatter(series, fname, title):
    first, second = ita_pairs(series)
    d = pd.DataFrame({"first_half_sorted": first, "second_half_sorted": second})
    d.to_csv(FIG / f"{fname}.csv", index=False)
    plt.figure(figsize=(5,5))
    lim = [0, max(d.max())*1.05]
    plt.plot(lim, lim, linestyle="--")
    plt.scatter(d["first_half_sorted"], d["second_half_sorted"])
    plt.xlabel("First half"); plt.ylabel("Second half"); plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG / f"{fname}.png", dpi=200, bbox_inches="tight")
    plt.close()


df = load_daily_data(INPUT_XLSX)
annual = annual_core_indices(df)
wet = df[df["wet"]].copy()
annual_q90 = wet.groupby("year")["p_mean_mm"].quantile(0.90).reset_index(name="q90_wet_mm")
annual_amp = wet.groupby("year")["amp_mm"].mean().reset_index(name="wet_amp_mm")
annual = annual.merge(annual_q90, on="year").merge(annual_amp, on="year")

annual_series = annual[["year","r20mm_days","q90_wet_mm","wet_amp_mm","cwd_days"]]
annual_series.to_csv(DATA / "annual_series.csv", index=False)

annual_summary_rows = []
for col in ["r20mm_days","q90_wet_mm","wet_amp_mm","cwd_days"]:
    annual_summary_rows.append({"metric": col, **ita_summary(annual[col])})
pd.DataFrame(annual_summary_rows).to_csv(DATA / "ita_annual_summary.csv", index=False)

monthly = df.groupby(["year","month"], as_index=False).agg(
    total_mm=("p_mean_mm", "sum"),
    r20mm_days=("p_mean_mm", lambda s: (s >= 20).sum()),
    amp_mm=("amp_mm", "mean"),
)
monthly_rows = []
for month in range(1, 13):
    g = monthly[monthly["month"] == month]
    for col in ["total_mm","r20mm_days","amp_mm"]:
        monthly_rows.append({"month": month, "metric": col, **ita_summary(g[col])})
monthly_summary = pd.DataFrame(monthly_rows)
monthly_summary.to_csv(DATA / "ita_monthly_summary.csv", index=False)

save_scatter(annual["r20mm_days"], "fig01_ita_r20_annual", "ITA - annual R20mm days")
save_scatter(annual["q90_wet_mm"], "fig02_ita_q90_wet_annual", "ITA - annual wet-day Q90")
save_scatter(annual["wet_amp_mm"], "fig03_ita_wet_amp_annual", "ITA - annual wet-day spatial amplitude")
save_scatter(annual["cwd_days"], "fig04_ita_cwd_annual", "ITA - annual CWD")

for metric, fname in [("total_mm","fig05_monthly_ita_total_change"),("r20mm_days","fig06_monthly_ita_r20_change"),("amp_mm","fig07_monthly_ita_amp_change")]:
    g = monthly_summary[monthly_summary["metric"] == metric][["month","mean_relative_change_pct"]]
    g.to_csv(FIG / f"{fname}.csv", index=False)
    plt.figure(figsize=(8,4))
    plt.bar(g["month"], g["mean_relative_change_pct"])
    plt.xlabel("Month"); plt.ylabel("Relative change (%)"); plt.title(f"Monthly ITA change: {metric}")
    plt.tight_layout()
    plt.savefig(FIG / f"{fname}.png", dpi=200, bbox_inches="tight")
    plt.close()

annual_sum = pd.read_csv(DATA / "ita_annual_summary.csv")
annual_sum.to_csv(FIG / "fig08_annual_ita_summary.csv", index=False)
plt.figure(figsize=(8,4))
plt.bar(annual_sum["metric"], annual_sum["mean_relative_change_pct"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Relative change (%)"); plt.title("Annual ITA summary")
plt.tight_layout()
plt.savefig(FIG / "fig08_annual_ita_summary.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"Pack 07 outputs written to {OUT}")
