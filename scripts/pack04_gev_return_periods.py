#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from menkao_analysis.common import (
    ensure_dir, load_daily_data, annual_core_indices, fit_stationary_gev,
    fit_nonstationary_gev, gev_return_levels_stationary, gev_return_levels_nonstationary
)

INPUT_XLSX = REPO_ROOT / "data" / "raw" / "FINAL1981to2025_CHIRPS_Daily_Mean_Min_Max.xlsx"
OUT = ensure_dir(REPO_ROOT / "outputs" / "pack04_gev_return_periods")
DATA = ensure_dir(OUT / "data")
FIG = ensure_dir(OUT / "figures")
RETURN_PERIODS = np.array([2, 5, 10, 20, 50, 100], dtype=float)


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()


df = load_daily_data(INPUT_XLSX)
annual = annual_core_indices(df)

annual_max = annual[["year","rx1_mean_mm","rx5_mean_mm","rx1_max_mm"]].rename(columns={
    "rx1_mean_mm":"rx1_mean",
    "rx5_mean_mm":"rx5_mean",
    "rx1_max_mm":"rx1_max",
})
annual_max.to_csv(DATA / "annual_maxima_series.csv", index=False)

series_map = {
    "rx1_mean": annual_max["rx1_mean"],
    "rx5_mean": annual_max["rx5_mean"],
    "rx1_max": annual_max["rx1_max"],
}
stat_rows, nonstat_rows, model_rows = [], [], []
stationary_return_tables = []
nonstationary_return_tables = []

for name, s in series_map.items():
    st = fit_stationary_gev(s)
    ns = fit_nonstationary_gev(s, annual_max["year"])
    stat_rows.append({"series": name, **st})
    nonstat_rows.append({"series": name, **ns})
    model_rows.append({
        "series": name,
        "stationary_aic": st["aic"],
        "nonstationary_aic": ns["aic"],
        "preferred_model_aic": "nonstationary" if ns["aic"] < st["aic"] else "stationary",
    })
    rl_s = gev_return_levels_stationary(st, RETURN_PERIODS)
    rl_s.insert(0, "series", name)
    stationary_return_tables.append(rl_s)
    rl_ns = gev_return_levels_nonstationary(ns, RETURN_PERIODS, [1981, 2025])
    rl_ns.insert(0, "series", name)
    nonstationary_return_tables.append(rl_ns)

pd.DataFrame(stat_rows).to_csv(DATA / "gev_stationary_summary.csv", index=False)
pd.DataFrame(nonstat_rows).to_csv(DATA / "gev_nonstationary_summary.csv", index=False)
pd.DataFrame(model_rows).to_csv(DATA / "model_comparison.csv", index=False)
pd.concat(stationary_return_tables, ignore_index=True).to_csv(DATA / "return_levels_stationary.csv", index=False)
pd.concat(nonstationary_return_tables, ignore_index=True).to_csv(DATA / "return_levels_nonstationary_1981_2025.csv", index=False)

peak_months = (
    df.loc[df.groupby("year")["p_mean_mm"].idxmax(), ["year", "month"]]
    .rename(columns={"month": "peak_month"})
    .reset_index(drop=True)
)
peak_month_counts = peak_months["peak_month"].value_counts().sort_index().rename_axis("month").reset_index(name="count")
peak_month_counts.to_csv(DATA / "annual_peak_month_counts.csv", index=False)

for col, fname, title in [
    ("rx1_mean", "fig01_rx1_mean_annual_series", "Area-mean Rx1day annual series"),
    ("rx5_mean", "fig02_rx5_mean_annual_series", "Area-mean Rx5day annual series"),
    ("rx1_max", "fig03_rx1_max_annual_series", "Spatial-maximum Rx1day annual series"),
]:
    fig = annual_max[["year", col]].copy()
    fig.to_csv(FIG / f"{fname}.csv", index=False)
    plt.figure(figsize=(9, 4))
    plt.plot(fig["year"], fig[col], marker="o", linewidth=1)
    plt.xlabel("Year"); plt.ylabel("mm"); plt.title(title)
    savefig(fname)

for series, fname in [("rx1_mean","fig04_return_levels_rx1_mean"),("rx5_mean","fig05_return_levels_rx5_mean"),("rx1_max","fig06_return_levels_rx1_max")]:
    sdata = annual_max[series].sort_values().reset_index(drop=True)
    n = len(sdata)
    empirical = pd.DataFrame({
        "return_period_years": (n + 1) / (np.arange(1, n + 1)[::-1]),
        "empirical_return_level_mm": sdata.values,
    }).sort_values("return_period_years")
    model = pd.read_csv(DATA / "return_levels_stationary.csv")
    model = model[model["series"] == series][["return_period_years","return_level_mm"]]
    empirical.to_csv(FIG / f"{fname}_empirical.csv", index=False)
    model.to_csv(FIG / f"{fname}_model.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.scatter(empirical["return_period_years"], empirical["empirical_return_level_mm"], label="Empirical")
    plt.plot(model["return_period_years"], model["return_level_mm"], label="GEV stationary")
    plt.xscale("log")
    plt.xlabel("Return period (years)"); plt.ylabel("Return level (mm)"); plt.title(series); plt.legend()
    savefig(fname)

ns_change = pd.read_csv(DATA / "return_levels_nonstationary_1981_2025.csv")
wide = ns_change.pivot_table(index=["series","return_period_years"], columns="year", values="return_level_mm").reset_index()
wide["percent_change_1981_2025"] = 100 * (wide[2025] - wide[1981]) / wide[1981]
wide.to_csv(FIG / "fig07_nonstationary_returnlevel_change.csv", index=False)
plt.figure(figsize=(8, 4))
for series, g in wide.groupby("series"):
    plt.plot(g["return_period_years"], g["percent_change_1981_2025"], marker="o", label=series)
plt.xscale("log")
plt.xlabel("Return period (years)"); plt.ylabel("Change 1981-2025 (%)"); plt.title("Non-stationary return level change"); plt.legend()
savefig("fig07_nonstationary_returnlevel_change")

peak_month_counts.to_csv(FIG / "fig08_peak_months.csv", index=False)
plt.figure(figsize=(8,4))
plt.bar(peak_month_counts["month"], peak_month_counts["count"])
plt.xlabel("Month"); plt.ylabel("Count"); plt.title("Peak month of annual maximum rainfall")
savefig("fig08_peak_months")

print(f"Pack 04 outputs written to {OUT}")
