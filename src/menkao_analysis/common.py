
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist
from scipy.stats import genextreme, kendalltau, norm, theilslopes

WET_DAY_THRESHOLD_MM = 1.0
REFERENCE_START = 1981
REFERENCE_END = 2010


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_daily_data(input_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(input_xlsx)
    rename_map = {"Date": "date", "P_mean_mm": "p_mean_mm", "P_min_mm": "p_min_mm", "P_max_mm": "p_max_mm"}
    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["amp_mm"] = df["p_max_mm"] - df["p_min_mm"]
    df["wet"] = df["p_mean_mm"] >= WET_DAY_THRESHOLD_MM
    df["dry"] = ~df["wet"]
    return df


def max_consecutive_true(values: Iterable[bool]) -> int:
    max_run = 0
    run = 0
    for v in values:
        if bool(v):
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return int(max_run)


def annual_core_indices(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in df.groupby("year"):
        wet = g[g["wet"]]
        rows.append({
            "year": int(year),
            "total_mm": float(g["p_mean_mm"].sum()),
            "wet_days": int(g["wet"].sum()),
            "dry_days": int(g["dry"].sum()),
            "rx1_mean_mm": float(g["p_mean_mm"].max()),
            "rx1_max_mm": float(g["p_max_mm"].max()),
            "sdii_mm_per_wetday": float(wet["p_mean_mm"].mean()) if not wet.empty else np.nan,
            "r10mm_days": int((g["p_mean_mm"] >= 10).sum()),
            "r20mm_days": int((g["p_mean_mm"] >= 20).sum()),
            "r30mm_days": int((g["p_mean_mm"] >= 30).sum()),
            "r50mm_days": int((g["p_mean_mm"] >= 50).sum()),
            "cdd_days": max_consecutive_true(g["dry"]),
            "cwd_days": max_consecutive_true(g["wet"]),
            "mean_range_mm": float(g["amp_mm"].mean()),
            "wet_mean_range_mm": float(wet["amp_mm"].mean()) if not wet.empty else np.nan,
        })
    out = pd.DataFrame(rows)
    out["rx3_mean_mm"] = rolling_max_by_year(df, "p_mean_mm", 3)
    out["rx5_mean_mm"] = rolling_max_by_year(df, "p_mean_mm", 5)
    return out


def rolling_max_by_year(df: pd.DataFrame, value_col: str, window: int) -> pd.Series:
    vals = []
    years = []
    for year, g in df.groupby("year"):
        s = g[value_col].rolling(window=window, min_periods=window).sum().max()
        vals.append(float(s) if pd.notna(s) else np.nan)
        years.append(int(year))
    return pd.Series(vals, index=range(len(vals)))


def rainfall_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-0.001, 1, 5, 10, 20, 50, np.inf]
    labels = ["0-1", "1-5", "5-10", "10-20", "20-50", ">50"]
    cat = pd.cut(df["p_mean_mm"], bins=bins, labels=labels)
    counts = cat.value_counts().reindex(labels, fill_value=0)
    out = pd.DataFrame({"class_mm": labels, "days": counts.values})
    out["percent"] = 100 * out["days"] / out["days"].sum()
    return out


def mk_test(x: Iterable[float]) -> dict:
    arr = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(arr)
    if n < 3:
        return {"n": n, "tau": np.nan, "s": np.nan, "z": np.nan, "p_value": np.nan, "trend": "insufficient", "sen_slope": np.nan}
    s = 0
    ties = {}
    for k in range(n - 1):
        diff = arr[k + 1:] - arr[k]
        s += np.sign(diff).sum()
    _, counts = np.unique(arr, return_counts=True)
    tie_sum = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    tau, _ = kendalltau(np.arange(n), arr)
    sen = theilslopes(arr, np.arange(n))[0]
    trend = "increasing" if (p < 0.05 and sen > 0) else "decreasing" if (p < 0.05 and sen < 0) else "no trend"
    return {"n": n, "tau": float(tau), "s": float(s), "z": float(z), "p_value": float(p), "trend": trend, "sen_slope": float(sen)}


def sen_per_decade(series: Iterable[float]) -> float:
    arr = pd.Series(series).dropna().to_numpy(dtype=float)
    if len(arr) < 3:
        return np.nan
    return float(theilslopes(arr, np.arange(len(arr)))[0] * 10.0)


def pettitt_test(x: Iterable[float]) -> dict:
    arr = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(arr)
    if n < 3:
        return {"break_index": np.nan, "break_position": np.nan, "k_stat": np.nan, "p_value": np.nan}
    U = np.zeros(n)
    for t in range(n):
        s = 0
        for i in range(t):
            s += np.sign(arr[t+1:] - arr[i]).sum()
        U[t] = s
    K = np.max(np.abs(U))
    k = int(np.argmax(np.abs(U)))
    p = 2 * np.exp((-6 * K**2) / (n**3 + n**2))
    return {"break_index": k, "break_position": k, "k_stat": float(K), "p_value": float(min(p, 1.0))}


def ita_summary(series: Iterable[float]) -> dict:
    s = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(s)
    if n < 4:
        return {"n_pairs": 0, "pct_above_1to1": np.nan, "pct_below_1to1": np.nan, "mean_abs_change": np.nan, "mean_relative_change_pct": np.nan}
    if n % 2 == 1:
        mid = n // 2
        s = np.concatenate([s[:mid], s[mid + 1:]])
    half = len(s) // 2
    first = np.sort(s[:half])
    second = np.sort(s[half:])
    diff = second - first
    rel = np.full(len(first), np.nan, dtype=float)
    mask = first != 0
    rel[mask] = diff[mask] / first[mask]
    mean_rel = float(np.nanmean(rel) * 100) if np.isfinite(rel).any() else np.nan
    med_rel = float(np.nanmedian(rel) * 100) if np.isfinite(rel).any() else np.nan
    return {
        "n_pairs": int(len(first)),
        "pct_above_1to1": float(np.mean(second > first) * 100),
        "pct_below_1to1": float(np.mean(second < first) * 100),
        "mean_abs_change": float(np.mean(diff)),
        "mean_relative_change_pct": mean_rel,
        "median_relative_change_pct": med_rel,
    }


def monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["year", "month"], as_index=False).agg(
        total_mm=("p_mean_mm", "sum"),
        wet_days=("wet", "sum"),
        mean_daily_mm=("p_mean_mm", "mean"),
        mean_range_mm=("amp_mm", "mean"),
    )
    return out


def reference_thresholds(df: pd.DataFrame) -> dict:
    ref = df[(df["year"] >= REFERENCE_START) & (df["year"] <= REFERENCE_END) & (df["wet"])]
    return {
        "p95_wet": float(ref["p_mean_mm"].quantile(0.95)),
        "p99_wet": float(ref["p_mean_mm"].quantile(0.99)),
    }


def etccdi_annual(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = reference_thresholds(df)
    p95 = thresholds["p95_wet"]
    p99 = thresholds["p99_wet"]
    rows = []
    for year, g in df.groupby("year"):
        wet = g[g["wet"]]
        rows.append({
            "year": int(year),
            "PRCPTOT_mm": float(wet["p_mean_mm"].sum()),
            "WetDays_n": int(len(wet)),
            "SDII_mm_per_wetday": float(wet["p_mean_mm"].mean()) if not wet.empty else np.nan,
            "Rx1day_mm": float(g["p_mean_mm"].max()),
            "Rx3day_mm": float(g["p_mean_mm"].rolling(3, min_periods=3).sum().max()),
            "Rx5day_mm": float(g["p_mean_mm"].rolling(5, min_periods=5).sum().max()),
            "R10mm_days": int((g["p_mean_mm"] >= 10).sum()),
            "R20mm_days": int((g["p_mean_mm"] >= 20).sum()),
            "R30mm_days": int((g["p_mean_mm"] >= 30).sum()),
            "R50mm_days": int((g["p_mean_mm"] >= 50).sum()),
            "CDD_days": max_consecutive_true(g["dry"]),
            "CWD_days": max_consecutive_true(g["wet"]),
            "R95pTOT_mm": float(g.loc[g["p_mean_mm"] > p95, "p_mean_mm"].sum()),
            "R99pTOT_mm": float(g.loc[g["p_mean_mm"] > p99, "p_mean_mm"].sum()),
            "MeanRangeWet_mm": float(wet["amp_mm"].mean()) if not wet.empty else np.nan,
        })
    return pd.DataFrame(rows), pd.DataFrame([{"reference_period": f"{REFERENCE_START}-{REFERENCE_END}", "p95_wet_mm": p95, "p99_wet_mm": p99}])


def fit_stationary_gev(series: Iterable[float]) -> dict:
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    c, loc, scale = genextreme.fit(x)
    ll = np.sum(genextreme.logpdf(x, c, loc=loc, scale=scale))
    k = 3
    n = len(x)
    return {"shape": float(c), "loc": float(loc), "scale": float(scale), "loglik": float(ll), "aic": float(2 * k - 2 * ll), "bic": float(k * np.log(n) - 2 * ll)}


def fit_nonstationary_gev(series: Iterable[float], years: Iterable[float]) -> dict:
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    years = pd.Series(years).dropna().to_numpy(dtype=float)
    t = years - years.min()
    init = fit_stationary_gev(x)
    x_mean = x.mean()

    def nll(params):
        shape, beta0, beta1, log_scale = params
        scale = np.exp(log_scale)
        loc = beta0 + beta1 * t
        if scale <= 0 or np.any(~np.isfinite(loc)):
            return 1e12
        lp = genextreme.logpdf(x, shape, loc=loc, scale=scale)
        if np.any(~np.isfinite(lp)):
            return 1e12
        return -np.sum(lp)

    x0 = np.array([init["shape"], x_mean, 0.0, np.log(init["scale"])])
    bounds = [(-1.0, 1.0), (None, None), (None, None), (-10, 10)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    shape, beta0, beta1, log_scale = res.x
    scale = float(np.exp(log_scale))
    ll = -float(res.fun)
    k = 4
    n = len(x)
    return {
        "shape": float(shape),
        "beta0": float(beta0),
        "beta1": float(beta1),
        "scale": scale,
        "loglik": ll,
        "aic": float(2 * k - 2 * ll),
        "bic": float(k * np.log(n) - 2 * ll),
        "converged": bool(res.success),
    }


def gev_return_levels_stationary(model: dict, return_periods: Iterable[float]) -> pd.DataFrame:
    rows = []
    for rp in return_periods:
        p = 1 - 1 / rp
        rl = genextreme.ppf(p, model["shape"], loc=model["loc"], scale=model["scale"])
        rows.append({"return_period_years": float(rp), "return_level_mm": float(rl)})
    return pd.DataFrame(rows)


def gev_return_levels_nonstationary(model: dict, return_periods: Iterable[float], years: Iterable[int]) -> pd.DataFrame:
    rows = []
    start = min(years)
    for year in years:
        t = year - start
        loc = model["beta0"] + model["beta1"] * t
        for rp in return_periods:
            p = 1 - 1 / rp
            rl = genextreme.ppf(p, model["shape"], loc=loc, scale=model["scale"])
            rows.append({"year": int(year), "return_period_years": float(rp), "return_level_mm": float(rl)})
    return pd.DataFrame(rows)


def compute_spi(monthly_series: pd.Series, scale: int) -> pd.Series:
    roll = monthly_series.rolling(scale, min_periods=scale).sum()
    spi = pd.Series(index=roll.index, dtype=float)
    for month in range(1, 13):
        idx = roll.index.month == month
        x = roll[idx].dropna()
        if len(x) < 8:
            spi.loc[idx] = np.nan
            continue
        zeros = (x <= 0).sum()
        q = zeros / len(x)
        xp = x[x > 0]
        if len(xp) < 5:
            spi.loc[idx] = np.nan
            continue
        a, loc, scale_param = gamma_dist.fit(xp, floc=0)
        vals = roll[idx]
        cdf = pd.Series(index=vals.index, dtype=float)
        positive = vals > 0
        cdf.loc[positive] = q + (1 - q) * gamma_dist.cdf(vals.loc[positive], a, loc=0, scale=scale_param)
        cdf.loc[~positive] = q
        cdf = cdf.clip(1e-6, 1 - 1e-6)
        spi.loc[idx] = norm.ppf(cdf)
    return spi


def spi_category(x: float) -> str:
    if pd.isna(x):
        return "Missing"
    if x >= 2:
        return "Extremely wet"
    if x >= 1.5:
        return "Severely wet"
    if x >= 1:
        return "Moderately wet"
    if x > -1:
        return "Near normal"
    if x > -1.5:
        return "Moderately dry"
    if x > -2:
        return "Severely dry"
    return "Extremely dry"


def robust_score(row: pd.Series) -> int:
    score = 0
    for col in ["mk_sig", "pettitt_sig", "ita_strong", "quantile_sig", "gev_nonstationary"]:
        score += int(bool(row.get(col, False)))
    return score
