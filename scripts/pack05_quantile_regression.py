#!/usr/bin/env python3
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from menkao_analysis.common import ensure_dir, load_daily_data

warnings.filterwarnings("ignore")

INPUT_XLSX = REPO_ROOT / "data" / "raw" / "FINAL1981to2025_CHIRPS_Daily_Mean_Min_Max.xlsx"
OUT = ensure_dir(REPO_ROOT / "outputs" / "pack05_quantile_regression")
DATA = ensure_dir(OUT / "data")
FIG = ensure_dir(OUT / "figures")

TAUS = [0.50, 0.75, 0.90, 0.95, 0.99]
MONTHLY_TAUS = [0.95, 0.99]
PRED_YEARS = np.arange(1981, 2026)


def fit_qr(df: pd.DataFrame, value_col: str, taus=TAUS):
    x = pd.DataFrame({"const": 1.0, "year": df["year"].astype(float)})
    y = df[value_col].astype(float)
    rows = []
    preds = []
    for tau in taus:
        model = sm.QuantReg(y, x).fit(q=tau)
        rows.append({
            "variable": value_col,
            "tau": tau,
            "intercept": model.params["const"],
            "slope_per_year": model.params["year"],
            "slope_per_decade": model.params["year"] * 10,
            "p_year": model.pvalues["year"] if "year" in model.pvalues else np.nan,
            "prsquared": getattr(model, "prsquared", np.nan),
        })
        px = pd.DataFrame({"const": 1.0, "year": PRED_YEARS.astype(float)})
        py = model.predict(px)
        preds.append(pd.DataFrame({"variable": value_col, "tau": tau, "year": PRED_YEARS, "predicted": py}))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()


df = load_daily_data(INPUT_XLSX)
wet = df[df["wet"]].copy()
wet["range_mm"] = wet["amp_mm"]

summary_rows = []
preds = []
for var in ["p_mean_mm", "p_max_mm", "range_mm"]:
    s, p = fit_qr(wet, var)
    summary_rows.append(s)
    preds.append(p)
summary = pd.concat(summary_rows, ignore_index=True)
pred = pd.concat(preds, ignore_index=True)
summary.to_csv(DATA / "quantile_regression_summary.csv", index=False)
pred.to_csv(DATA / "quantile_regression_predictions.csv", index=False)

annual_emp = wet.groupby("year").agg(
    pmean_q50=("p_mean_mm", lambda s: s.quantile(0.50)),
    pmean_q90=("p_mean_mm", lambda s: s.quantile(0.90)),
    pmean_q95=("p_mean_mm", lambda s: s.quantile(0.95)),
    pmean_q99=("p_mean_mm", lambda s: s.quantile(0.99)),
    pmax_q99=("p_max_mm", lambda s: s.quantile(0.99)),
    range_q99=("range_mm", lambda s: s.quantile(0.99)),
).reset_index()
annual_emp.to_csv(DATA / "annual_empirical_quantiles.csv", index=False)

annual_trends = []
for col in annual_emp.columns:
    if col == "year":
        continue
    x = annual_emp["year"].values.astype(float)
    y = annual_emp[col].values.astype(float)
    coef = np.polyfit(x, y, 1)
    annual_trends.append({"metric": col, "slope_per_year": coef[0], "slope_per_decade": coef[0] * 10})
pd.DataFrame(annual_trends).to_csv(DATA / "annual_quantile_trends.csv", index=False)

period_collections = []
for label, years in [("1981_2002", wet["year"].between(1981, 2002)), ("2003_2025", wet["year"].between(2003, 2025))]:
    g = wet[years]
    period_collections.append(pd.DataFrame({
        "period": [label],
        "pmean_q90": [g["p_mean_mm"].quantile(0.90)],
        "pmean_q95": [g["p_mean_mm"].quantile(0.95)],
        "pmean_q99": [g["p_mean_mm"].quantile(0.99)],
        "pmax_q99": [g["p_max_mm"].quantile(0.99)],
        "range_q99": [g["range_mm"].quantile(0.99)],
    }))
period_df = pd.concat(period_collections, ignore_index=True)
comp = period_df.set_index("period").T
comp["percent_change"] = 100 * (comp["2003_2025"] - comp["1981_2002"]) / comp["1981_2002"]
comp.reset_index(names="metric").to_csv(DATA / "period_quantile_comparison.csv", index=False)

monthly_rows = []
for month, g in wet.groupby("month"):
    for tau in MONTHLY_TAUS:
        monthly_rows.append({"month": month, "tau": tau, "pmean_quantile_mm": g["p_mean_mm"].quantile(tau)})
monthly_df = pd.DataFrame(monthly_rows)
monthly_df.to_csv(DATA / "monthly_quantile_regression_pmean.csv", index=False)

for var, fname, title in [("p_mean_mm","fig01_qr_pmean","Quantile regression: area-mean rainfall"),
                          ("p_max_mm","fig02_qr_pmax","Quantile regression: spatial maximum rainfall"),
                          ("range_mm","fig03_qr_range","Quantile regression: spatial range")]:
    obs = wet.groupby("year", as_index=False)[var].median().rename(columns={var:"median_obs"})
    plot_df = pred[pred["variable"] == var]
    plot_df.to_csv(FIG / f"{fname}.csv", index=False)
    plt.figure(figsize=(9,4))
    plt.plot(obs["year"], obs["median_obs"], color="black", linewidth=1, label="Annual median")
    for tau in [0.50, 0.90, 0.95, 0.99]:
        g = plot_df[plot_df["tau"] == tau]
        plt.plot(g["year"], g["predicted"], label=f"Q{int(tau*100)}")
    plt.xlabel("Year"); plt.ylabel("mm"); plt.title(title); plt.legend(ncol=5, fontsize=8)
    savefig(fname)

slopes = summary[["variable","tau","slope_per_decade","p_year"]]
slopes.to_csv(FIG / "fig04_slope_by_quantile.csv", index=False)
plt.figure(figsize=(8,4))
for var, g in slopes.groupby("variable"):
    plt.plot(g["tau"], g["slope_per_decade"], marker="o", label=var)
plt.xlabel("Quantile"); plt.ylabel("Slope per decade (mm)"); plt.title("Quantile regression slopes"); plt.legend()
savefig("fig04_slope_by_quantile")

period_df.to_csv(FIG / "fig05_period_change_pmean.csv", index=False)
plt.figure(figsize=(7,4))
base = comp.loc[["pmean_q90","pmean_q95","pmean_q99"], "percent_change"]
plt.bar(base.index, base.values)
plt.ylabel("Percent change (%)"); plt.title("Change in high quantiles between periods")
savefig("fig05_period_change_pmean")

comp.reset_index(names="metric").to_csv(FIG / "fig06_period_change_pct.csv", index=False)
plt.figure(figsize=(7,4))
plt.bar(comp.index, comp["percent_change"])
plt.ylabel("Percent change (%)"); plt.title("Period change in selected quantiles")
savefig("fig06_period_change_pct")

for tau, fname in [(0.95, "fig07_monthly_q95"), (0.99, "fig08_monthly_q99")]:
    g = monthly_df[monthly_df["tau"] == tau]
    g.to_csv(FIG / f"{fname}.csv", index=False)
    plt.figure(figsize=(8,4))
    plt.plot(g["month"], g["pmean_quantile_mm"], marker="o")
    plt.xlabel("Month"); plt.ylabel("mm"); plt.title(f"Monthly Q{int(tau*100)} of wet-day rainfall")
    savefig(fname)

annual_emp.to_csv(FIG / "fig09_annual_quantiles.csv", index=False)
plt.figure(figsize=(9,4))
for col in ["pmean_q90","pmean_q95","pmean_q99"]:
    plt.plot(annual_emp["year"], annual_emp[col], label=col)
plt.xlabel("Year"); plt.ylabel("mm"); plt.title("Annual empirical wet-day quantiles"); plt.legend()
savefig("fig09_annual_quantiles")

print(f"Pack 05 outputs written to {OUT}")
