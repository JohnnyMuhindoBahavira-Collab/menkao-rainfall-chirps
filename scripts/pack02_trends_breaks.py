#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from menkao_analysis.common import (
    ensure_dir, load_daily_data, annual_core_indices, monthly_totals,
    mk_test, sen_per_decade, pettitt_test, ita_summary
)

INPUT_XLSX = REPO_ROOT / "data" / "raw" / "FINAL1981to2025_CHIRPS_Daily_Mean_Min_Max.xlsx"
OUT = ensure_dir(REPO_ROOT / "outputs" / "pack02_trends_breaks")
DATA = ensure_dir(OUT / "data")
FIG = ensure_dir(OUT / "figures")


def add_trendline(years, values):
    coef = np.polyfit(years, values, 1)
    return np.polyval(coef, years)


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()


df = load_daily_data(INPUT_XLSX)
annual = annual_core_indices(df)
annual.to_csv(DATA / "annual_core_indices.csv", index=False)

monthly = monthly_totals(df)
monthly["date"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
monthly.to_csv(DATA / "monthly_core_series.csv", index=False)

annual_metrics = ["total_mm", "wet_days", "dry_days", "rx1_mean_mm", "sdii_mm_per_wetday", "r20mm_days", "cwd_days", "cdd_days", "mean_range_mm", "wet_mean_range_mm"]
trend_rows = []
break_rows = []
ita_rows = []
for col in annual_metrics:
    mk = mk_test(annual[col])
    pt = pettitt_test(annual[col])
    ita = ita_summary(annual[col])
    trend_rows.append({
        "metric": col,
        "sen_slope_per_year": mk["sen_slope"],
        "sen_slope_per_decade": mk["sen_slope"] * 10 if pd.notna(mk["sen_slope"]) else np.nan,
        "tau": mk["tau"],
        "p_value": mk["p_value"],
        "trend": mk["trend"],
    })
    break_rows.append({
        "metric": col,
        "break_year": int(annual.loc[pt["break_index"], "year"]) if pd.notna(pt["break_index"]) else np.nan,
        "pettitt_k": pt["k_stat"],
        "pettitt_p_value": pt["p_value"],
    })
    ita_rows.append({
        "metric": col,
        **ita
    })
pd.DataFrame(trend_rows).to_csv(DATA / "annual_trend_tests.csv", index=False)
pd.DataFrame(break_rows).to_csv(DATA / "change_point_tests.csv", index=False)
pd.DataFrame(ita_rows).to_csv(DATA / "ita_summary.csv", index=False)

monthly_total_trends = []
monthly_range_trends = []
for m in range(1, 13):
    g = monthly[monthly["month"] == m]
    mk1 = mk_test(g["total_mm"])
    mk2 = mk_test(g["mean_range_mm"])
    monthly_total_trends.append({"month": m, "sen_slope_per_decade": mk1["sen_slope"] * 10, "tau": mk1["tau"], "p_value": mk1["p_value"]})
    monthly_range_trends.append({"month": m, "sen_slope_per_decade": mk2["sen_slope"] * 10, "tau": mk2["tau"], "p_value": mk2["p_value"]})
pd.DataFrame(monthly_total_trends).to_csv(DATA / "monthly_total_trends.csv", index=False)
pd.DataFrame(monthly_range_trends).to_csv(DATA / "monthly_range_trends.csv", index=False)

season_map = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
monthly["season"] = monthly["month"].map(season_map)
seasonal = monthly.groupby(["year", "season"], as_index=False)["total_mm"].sum()
season_rows = []
for season, g in seasonal.groupby("season"):
    mk = mk_test(g["total_mm"])
    season_rows.append({"season": season, "sen_slope_per_decade": mk["sen_slope"] * 10, "tau": mk["tau"], "p_value": mk["p_value"], "trend": mk["trend"]})
pd.DataFrame(season_rows).to_csv(DATA / "seasonal_mk_summary.csv", index=False)

for metric, fname, ylabel, title in [
    ("total_mm", "fig_total_mm_annual_trend", "Annual rainfall total (mm)", "Annual total rainfall"),
    ("r20mm_days", "fig_r20mm_days_annual_trend", "Days", "Annual number of days >= 20 mm"),
    ("cwd_days", "fig_cwd_days_annual_trend", "Days", "Annual maximum consecutive wet days"),
    ("mean_range_mm", "fig_mean_range_mm_annual_trend", "Range (mm)", "Annual mean spatial range"),
    ("wet_mean_range_mm", "fig_wet_mean_range_mm_annual_trend", "Range (mm)", "Annual wet-day mean spatial range"),
    ("sdii_mm_per_wetday", "fig_sdii_mm_annual_trend", "mm/wet day", "Annual SDII"),
]:
    fig = annual[["year", metric]].copy()
    fig["linear_fit"] = add_trendline(fig["year"], fig[metric])
    fig.to_csv(FIG / f"{fname}.csv", index=False)
    plt.figure(figsize=(9, 4))
    plt.plot(fig["year"], fig[metric], marker="o", linewidth=1, label="Observed")
    plt.plot(fig["year"], fig["linear_fit"], linestyle="--", label="Linear fit")
    plt.xlabel("Year"); plt.ylabel(ylabel); plt.title(title); plt.legend()
    savefig(fname)

for src, fname, title, ylabel in [
    (pd.DataFrame(monthly_total_trends), "fig_monthly_total_trend_bars", "Monthly trend in rainfall total", "Sen slope per decade (mm)"),
    (pd.DataFrame(monthly_range_trends), "fig_monthly_range_trend_bars", "Monthly trend in spatial range", "Sen slope per decade (mm)"),
]:
    src.to_csv(FIG / f"{fname}.csv", index=False)
    plt.figure(figsize=(8, 4))
    plt.bar(src["month"], src["sen_slope_per_decade"])
    plt.xlabel("Month"); plt.ylabel(ylabel); plt.title(title)
    savefig(fname)

for metric, fname in [("total_mm","fig_ita_total_mm"),("r20mm_days","fig_ita_r20mm_days"),("mean_range_mm","fig_ita_mean_range_mm"),("cwd_days","fig_ita_cwd_days")]:
    s = annual[metric].dropna().to_numpy(dtype=float)
    if len(s) % 2 == 1:
        s = np.concatenate([s[:len(s)//2], s[len(s)//2+1:]])
    h = len(s)//2
    d = pd.DataFrame({"first_half_sorted": np.sort(s[:h]), "second_half_sorted": np.sort(s[h:])})
    d.to_csv(FIG / f"{fname}.csv", index=False)
    plt.figure(figsize=(5,5))
    lim = [0, max(d.max())*1.05]
    plt.plot(lim, lim, linestyle="--")
    plt.scatter(d["first_half_sorted"], d["second_half_sorted"])
    plt.xlabel("First half"); plt.ylabel("Second half"); plt.title(f"ITA - {metric}")
    savefig(fname)

print(f"Pack 02 outputs written to {OUT}")
