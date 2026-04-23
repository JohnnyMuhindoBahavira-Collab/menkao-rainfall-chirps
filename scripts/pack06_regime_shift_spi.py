#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from menkao_analysis.common import (
    ensure_dir, load_daily_data, etccdi_annual, pettitt_test, compute_spi, spi_category
)

INPUT_XLSX = REPO_ROOT / "data" / "raw" / "FINAL1981to2025_CHIRPS_Daily_Mean_Min_Max.xlsx"
OUT = ensure_dir(REPO_ROOT / "outputs" / "pack06_regime_shift_spi")
DATA = ensure_dir(OUT / "data")
FIG = ensure_dir(OUT / "figures")
SPI_SCALES = [1, 3, 6, 12]


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()


df = load_daily_data(INPUT_XLSX)
annual, thresholds = etccdi_annual(df)

wet = df[df["wet"]].copy()
annual_q = wet.groupby("year").agg(
    q90_wet_mm=("p_mean_mm", lambda s: s.quantile(0.90)),
    q95_wet_mm=("p_mean_mm", lambda s: s.quantile(0.95)),
    q99_wet_mm=("p_mean_mm", lambda s: s.quantile(0.99)),
    wet_amp_mean_mm=("amp_mm", "mean"),
).reset_index()

annual_ext = annual[["year","R20mm_days","R95pTOT_mm","CWD_days","CDD_days","MeanRangeWet_mm"]]
annual_series = annual_q.merge(annual_ext, on="year")
annual_series.to_csv(DATA / "annual_extreme_and_quantile_series.csv", index=False)

break_rows = []
for col in annual_series.columns:
    if col == "year":
        continue
    pt = pettitt_test(annual_series[col])
    break_year = int(annual_series.loc[pt["break_index"], "year"]) if pd.notna(pt["break_index"]) else np.nan
    pre = annual_series.loc[annual_series["year"] <= break_year, col].mean() if pd.notna(break_year) else np.nan
    post = annual_series.loc[annual_series["year"] > break_year, col].mean() if pd.notna(break_year) else np.nan
    pct = 100 * (post - pre) / pre if pd.notna(pre) and pre != 0 else np.nan
    break_rows.append({
        "metric": col,
        "break_year": break_year,
        "pettitt_k": pt["k_stat"],
        "pettitt_p_value": pt["p_value"],
        "pre_mean": pre,
        "post_mean": post,
        "percent_change_post_vs_pre": pct,
    })
break_df = pd.DataFrame(break_rows)
break_df.to_csv(DATA / "annual_breaks_all_metrics.csv", index=False)
break_df[break_df["metric"].str.contains("q")].to_csv(DATA / "annual_quantile_breaks.csv", index=False)
break_df[~break_df["metric"].str.contains("q")].to_csv(DATA / "annual_extreme_breaks.csv", index=False)

prepost_rows = []
for _, row in break_df.iterrows():
    by = row["break_year"]
    if pd.isna(by):
        continue
    metric = row["metric"]
    s = annual_series[["year", metric]].copy()
    s["period"] = np.where(s["year"] <= by, "pre_break", "post_break")
    s["metric"] = metric
    prepost_rows.append(s)
pd.concat(prepost_rows, ignore_index=True).to_csv(DATA / "pre_post_break_series.csv", index=False)

monthly = df.set_index("date").resample("MS")["p_mean_mm"].sum().to_frame("monthly_total_mm")
for scale in SPI_SCALES:
    monthly[f"spi_{scale}"] = compute_spi(monthly["monthly_total_mm"], scale)
monthly_reset = monthly.reset_index()
monthly_reset.to_csv(DATA / "monthly_spi_series.csv", index=False)

cat_rows = []
for scale in SPI_SCALES:
    cats = monthly_reset[f"spi_{scale}"].apply(spi_category)
    counts = cats.value_counts().rename_axis("category").reset_index(name="months")
    counts["scale"] = scale
    cat_rows.append(counts)
pd.concat(cat_rows, ignore_index=True).to_csv(DATA / "annual_spi_counts.csv", index=False)

event_rows = []
for scale in SPI_SCALES:
    s = monthly_reset[f"spi_{scale}"]
    cats = s.apply(spi_category)
    wet = cats.isin(["Moderately wet","Severely wet","Extremely wet"]).to_numpy()
    dry = cats.isin(["Moderately dry","Severely dry","Extremely dry"]).to_numpy()
    for label, mask in [("wet", wet), ("dry", dry)]:
        run = 0
        start_idx = None
        for i, val in enumerate(mask):
            if val:
                if run == 0:
                    start_idx = i
                run += 1
            elif run > 0:
                event_rows.append({
                    "scale": scale, "event_type": label,
                    "start_date": monthly_reset.loc[start_idx, "date"],
                    "end_date": monthly_reset.loc[i-1, "date"],
                    "duration_months": run,
                })
                run = 0
        if run > 0:
            event_rows.append({
                "scale": scale, "event_type": label,
                "start_date": monthly_reset.loc[start_idx, "date"],
                "end_date": monthly_reset.loc[len(mask)-1, "date"],
                "duration_months": run,
            })
events = pd.DataFrame(event_rows)
events.to_csv(DATA / "spi_event_catalog.csv", index=False)
events.sort_values("duration_months", ascending=False).groupby(["scale","event_type"]).head(3).to_csv(DATA / "spi_longest_events.csv", index=False)

period_summary_rows = []
for scale in [3, 6, 12]:
    s = monthly_reset[["date", f"spi_{scale}"]].copy()
    s["period"] = np.where(s["date"].dt.year <= 2002, "1981_2002", "2003_2025")
    cats = s[f"spi_{scale}"].apply(spi_category)
    for period, g in s.groupby("period"):
        subcats = g[f"spi_{scale}"].apply(spi_category)
        wet_mod = int(subcats.eq("Moderately wet").sum())
        dry_mod = int(subcats.eq("Moderately dry").sum())
        period_summary_rows.append({"scale": scale, "period": period, "moderately_wet_months": wet_mod, "moderately_dry_months": dry_mod})
period_summary = pd.DataFrame(period_summary_rows)
period_summary.to_csv(DATA / "spi_period_summary.csv", index=False)

break_df.to_csv(FIG / "fig01_break_summary.csv", index=False)
plt.figure(figsize=(9,4))
plot = break_df.sort_values("percent_change_post_vs_pre")
plt.barh(plot["metric"], plot["percent_change_post_vs_pre"])
plt.xlabel("Post-vs-pre change (%)"); plt.ylabel("Metric"); plt.title("Break-related percent changes")
savefig("fig01_break_summary")

qbreak = break_df[break_df["metric"].str.contains("q") | break_df["metric"].str.contains("wet_amp")]
qbreak.to_csv(FIG / "fig02_quantile_breaks.csv", index=False)
plt.figure(figsize=(8,4))
plt.bar(qbreak["metric"], qbreak["percent_change_post_vs_pre"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Change (%)"); plt.title("Quantile and heterogeneity break shifts")
savefig("fig02_quantile_breaks")

ebreak = break_df[~(break_df["metric"].str.contains("q") | break_df["metric"].str.contains("wet_amp"))]
ebreak.to_csv(FIG / "fig03_extreme_breaks.csv", index=False)
plt.figure(figsize=(8,4))
plt.bar(ebreak["metric"], ebreak["percent_change_post_vs_pre"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Change (%)"); plt.title("Extreme-index break shifts")
savefig("fig03_extreme_breaks")

monthly_reset.to_csv(FIG / "fig04_spi_series.csv", index=False)
plt.figure(figsize=(10,4))
for scale in [3, 12]:
    plt.plot(monthly_reset["date"], monthly_reset[f"spi_{scale}"], label=f"SPI-{scale}")
plt.axhline(1, linestyle="--"); plt.axhline(-1, linestyle="--")
plt.ylabel("SPI"); plt.title("SPI time series"); plt.legend()
savefig("fig04_spi_series")

spi_counts = pd.read_csv(DATA / "annual_spi_counts.csv")
spi_counts.to_csv(FIG / "fig05_spi_counts.csv", index=False)
pivot = spi_counts.pivot(index="category", columns="scale", values="months").fillna(0)
pivot.plot(kind="bar", figsize=(9,5))
plt.ylabel("Months"); plt.title("SPI category counts by scale")
savefig("fig05_spi_counts")

longest = pd.read_csv(DATA / "spi_longest_events.csv")
longest.to_csv(FIG / "fig06_longest_events.csv", index=False)
plt.figure(figsize=(8,4))
for (scale, etype), g in longest.groupby(["scale","event_type"]):
    plt.scatter([f"{etype}-SPI{scale}"] * len(g), g["duration_months"])
plt.xticks(rotation=45, ha="right"); plt.ylabel("Duration (months)"); plt.title("Longest SPI events")
savefig("fig06_longest_events")

period_summary.to_csv(FIG / "fig07_period_summary.csv", index=False)
plt.figure(figsize=(8,4))
for scale, g in period_summary.groupby("scale"):
    change = g.set_index("period")
    vals = change["moderately_wet_months"].values
    plt.plot(g["period"], g["moderately_wet_months"], marker="o", label=f"SPI-{scale} wet")
plt.ylabel("Months"); plt.title("Moderately wet SPI months by period"); plt.legend()
savefig("fig07_period_summary")

print(f"Pack 06 outputs written to {OUT}")
