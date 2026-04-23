#!/usr/bin/env python3
import os
from pathlib import Path
import sys
os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[1] / ".mplconfig")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from menkao_analysis.common import ensure_dir, robust_score

OUT = ensure_dir(REPO_ROOT / "outputs" / "pack08_comparative_synthesis")
DATA = ensure_dir(OUT / "data")
FIG = ensure_dir(OUT / "figures")

PACK2 = REPO_ROOT / "outputs" / "pack02_trends_breaks" / "data"
PACK3 = REPO_ROOT / "outputs" / "pack03_etccdi_extremes" / "data"
PACK4 = REPO_ROOT / "outputs" / "pack04_gev_return_periods" / "data"
PACK5 = REPO_ROOT / "outputs" / "pack05_quantile_regression" / "data"
PACK6 = REPO_ROOT / "outputs" / "pack06_regime_shift_spi" / "data"
PACK7 = REPO_ROOT / "outputs" / "pack07_ita" / "data"


def savefig(name: str):
    plt.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()


p2_trends = pd.read_csv(PACK2 / "annual_trend_tests.csv")
p2_breaks = pd.read_csv(PACK2 / "change_point_tests.csv")
p3_trends = pd.read_csv(PACK3 / "annual_etccdi_trend_tests.csv")
p4_models = pd.read_csv(PACK4 / "model_comparison.csv")
p5_qr = pd.read_csv(PACK5 / "quantile_regression_summary.csv")
p6_breaks = pd.read_csv(PACK6 / "annual_breaks_all_metrics.csv")
p6_spi = pd.read_csv(PACK6 / "spi_period_summary.csv")
p7_ita = pd.read_csv(PACK7 / "ita_annual_summary.csv")

metric_map = [
    ("Annual total rainfall", "total_mm", "PRCPTOT_mm", None, None, None),
    ("Heavy rainfall frequency", "r20mm_days", "R20mm_days", None, None, "r20mm_days"),
    ("Wet spell duration", "cwd_days", "CWD_days", None, None, "cwd_days"),
    ("Dry spell duration", "cdd_days", "CDD_days", None, None, None),
    ("Wet-day heterogeneity", "wet_mean_range_mm", "MeanRangeWet_mm", None, None, "wet_amp_mm"),
    ("Wet-day Q90", None, None, ("p_mean_mm", 0.90), None, "q90_wet_mm"),
    ("Wet-day Q95", None, None, ("p_mean_mm", 0.95), None, None),
    ("Wet-day Q99", None, None, ("p_mean_mm", 0.99), None, None),
    ("Rx1day area mean", "rx1_mean_mm", "Rx1day_mm", None, "rx1_mean", None),
    ("Rx5day area mean", "rx5_mean_mm", "Rx5day_mm", None, "rx5_mean", None),
    ("Rx1day spatial maximum", "rx1_max_mm", None, None, "rx1_max", None),
    ("R95pTOT", None, "R95pTOT_mm", None, None, None),
]

rows = []
for label, p2_metric, p3_metric, qr_key, gev_key, ita_metric in metric_map:
    row = {"indicator": label}
    if p2_metric:
        s = p2_trends[p2_trends["metric"] == p2_metric]
        b = p2_breaks[p2_breaks["metric"] == p2_metric]
        if not s.empty:
            row["sen_slope_per_decade"] = s.iloc[0]["sen_slope_per_decade"]
            row["mk_p_value"] = s.iloc[0]["p_value"]
            row["mk_trend"] = s.iloc[0]["trend"]
            row["mk_sig"] = bool(s.iloc[0]["p_value"] < 0.05)
        if not b.empty:
            row["break_year"] = b.iloc[0]["break_year"]
            row["pettitt_p_value"] = b.iloc[0]["pettitt_p_value"]
            row["pettitt_sig"] = bool(b.iloc[0]["pettitt_p_value"] < 0.05)
    if p3_metric:
        s = p3_trends[p3_trends["metric"] == p3_metric]
        b = p6_breaks[p6_breaks["metric"] == p3_metric]
        if not s.empty:
            row["sen_slope_per_decade"] = s.iloc[0]["sen_slope_per_decade"]
            row["mk_p_value"] = s.iloc[0]["p_value"]
            row["mk_trend"] = s.iloc[0]["trend"]
            row["mk_sig"] = bool(s.iloc[0]["p_value"] < 0.05)
        if not b.empty:
            row["break_year"] = b.iloc[0]["break_year"]
            row["pettitt_p_value"] = b.iloc[0]["pettitt_p_value"]
            row["pettitt_sig"] = bool(b.iloc[0]["pettitt_p_value"] < 0.05)
            row["post_pre_change_pct"] = b.iloc[0]["percent_change_post_vs_pre"]
    if qr_key:
        var, tau = qr_key
        q = p5_qr[(p5_qr["variable"] == var) & (p5_qr["tau"].round(2) == round(tau,2))]
        if not q.empty:
            row["quantile_slope_per_decade"] = q.iloc[0]["slope_per_decade"]
            row["quantile_p_value"] = q.iloc[0]["p_year"]
            row["quantile_sig"] = bool(q.iloc[0]["p_year"] < 0.05)
    if gev_key:
        g = p4_models[p4_models["series"] == gev_key]
        if not g.empty:
            row["preferred_gev_model"] = g.iloc[0]["preferred_model_aic"]
            row["gev_nonstationary"] = bool(g.iloc[0]["preferred_model_aic"] == "nonstationary")
    if ita_metric:
        i = p7_ita[p7_ita["metric"] == ita_metric]
        if not i.empty:
            row["ita_change_pct"] = i.iloc[0]["mean_relative_change_pct"]
            row["ita_strong"] = bool(abs(i.iloc[0]["mean_relative_change_pct"]) >= 5 or i.iloc[0]["pct_above_1to1"] in [0,100])
    rows.append(row)

master = pd.DataFrame(rows)
master["robustness_score"] = master.apply(robust_score, axis=1)
master["robustness_class"] = pd.cut(master["robustness_score"], bins=[-1,1,2,3,5], labels=["weak","moderate","robust","very robust"])
master.to_csv(DATA / "master_evidence_table.csv", index=False)

master[master["indicator"].str.contains("Q9|R95|Heavy")].to_csv(DATA / "high_quantile_synthesis.csv", index=False)
master[master["indicator"].str.contains("Rx1|Rx5")].to_csv(DATA / "gev_synthesis.csv", index=False)
p6_spi.to_csv(DATA / "spi_synthesis.csv", index=False)

p2_month_total = pd.read_csv(PACK2 / "monthly_total_trends.csv")
p2_month_range = pd.read_csv(PACK2 / "monthly_range_trends.csv")
p7_month = pd.read_csv(PACK7 / "ita_monthly_summary.csv")
monthly_hotspots = []
for m in range(1, 13):
    monthly_hotspots.append({
        "month": m,
        "total_trend_per_decade": p2_month_total.loc[p2_month_total["month"] == m, "sen_slope_per_decade"].iloc[0],
        "range_trend_per_decade": p2_month_range.loc[p2_month_range["month"] == m, "sen_slope_per_decade"].iloc[0],
        "ita_total_change_pct": p7_month[(p7_month["month"] == m) & (p7_month["metric"] == "total_mm")]["mean_relative_change_pct"].iloc[0],
        "ita_r20_change_pct": p7_month[(p7_month["month"] == m) & (p7_month["metric"] == "r20mm_days")]["mean_relative_change_pct"].iloc[0],
        "ita_amp_change_pct": p7_month[(p7_month["month"] == m) & (p7_month["metric"] == "amp_mm")]["mean_relative_change_pct"].iloc[0],
    })
monthly_hotspots = pd.DataFrame(monthly_hotspots)
monthly_hotspots.to_csv(DATA / "monthly_hotspots_synthesis.csv", index=False)

pd.DataFrame([
    {"theme":"Annual totals","interpretation":"Weak to uncertain trend in annual total rainfall"},
    {"theme":"Heavy rainfall","interpretation":"Frequency and upper quantiles of heavy rainfall increase"},
    {"theme":"Wet spells","interpretation":"Consecutive wet spells shorten"},
    {"theme":"Dryness","interpretation":"No simple drying narrative from annual totals and SPI alone"},
    {"theme":"Spatial heterogeneity","interpretation":"Intra-area rainfall heterogeneity increases"},
]).to_csv(DATA / "thematic_synthesis.csv", index=False)

pd.DataFrame([
    {"message":"The strongest signal is intensification of heavy rainfall rather than a clear rise in annual totals."},
    {"message":"Wet spells shorten while high wet-day quantiles increase."},
    {"message":"Intra-area spatial rainfall heterogeneity strengthens over time."},
    {"message":"December emerges as the main monthly hotspot of change."},
]).to_csv(DATA / "article_ready_messages.csv", index=False)

pd.DataFrame([
    {"output":"master_evidence_table.csv","source":"pack02, pack03, pack04, pack05, pack06, pack07"},
    {"output":"monthly_hotspots_synthesis.csv","source":"pack02, pack07"},
    {"output":"spi_synthesis.csv","source":"pack06"},
]).to_csv(DATA / "traceability_sources.csv", index=False)

# figures
heat = master[["indicator","robustness_score"]].copy()
heat.to_csv(FIG / "fig01_evidence_heatmap.csv", index=False)
plt.figure(figsize=(7,5))
plt.imshow(heat[["robustness_score"]].values, aspect="auto", interpolation="nearest")
plt.yticks(range(len(heat)), heat["indicator"], fontsize=8)
plt.xticks([0], ["Robustness"])
plt.colorbar(label="Score")
plt.title("Evidence heatmap")
savefig("fig01_evidence_heatmap")

scores = master[["indicator","robustness_score"]].copy().sort_values("robustness_score", ascending=False)
scores.to_csv(FIG / "fig02_robustness_scores.csv", index=False)
plt.figure(figsize=(7,5))
plt.barh(scores["indicator"], scores["robustness_score"])
plt.xlabel("Robustness score")
plt.title("Comparative robustness")
savefig("fig02_robustness_scores")

timeline = master[["indicator","break_year"]].dropna().sort_values("break_year")
timeline.to_csv(FIG / "fig03_break_timeline.csv", index=False)
plt.figure(figsize=(7,4))
plt.barh(timeline["indicator"], timeline["break_year"])
plt.xlabel("Break year")
plt.title("Break timeline")
savefig("fig03_break_timeline")

tails = p5_qr[(p5_qr["variable"] == "p_mean_mm") & (p5_qr["tau"] >= 0.9)][["tau","slope_per_decade"]]
tails.to_csv(FIG / "fig04_quantile_tail_slopes.csv", index=False)
plt.figure(figsize=(6,4))
plt.plot(tails["tau"], tails["slope_per_decade"], marker="o")
plt.xlabel("Quantile")
plt.ylabel("Slope per decade (mm)")
plt.title("Upper-tail quantile slopes")
savefig("fig04_quantile_tail_slopes")

spi_shift = p6_spi.copy()
spi_shift["shift"] = spi_shift["moderately_wet_months"] - spi_shift["moderately_dry_months"]
spi_shift.to_csv(FIG / "fig05_spi_shift_months_covered.csv", index=False)
plt.figure(figsize=(7,4))
for scale, g in spi_shift.groupby("scale"):
    plt.plot(g["period"], g["shift"], marker="o", label=f"SPI-{scale}")
plt.ylabel("Wet - dry months")
plt.title("SPI balance shift by period")
plt.legend()
savefig("fig05_spi_shift_months_covered")

monthly_hotspots.to_csv(FIG / "fig06_monthly_hotspots.csv", index=False)
plt.figure(figsize=(7,4))
plt.plot(monthly_hotspots["month"], monthly_hotspots["ita_total_change_pct"], marker="o", label="ITA total")
plt.plot(monthly_hotspots["month"], monthly_hotspots["ita_r20_change_pct"], marker="o", label="ITA R20")
plt.xlabel("Month")
plt.ylabel("Change (%)")
plt.title("Monthly hotspots")
plt.legend()
savefig("fig06_monthly_hotspots")

print(f"Pack 08 outputs written to {OUT}")
