#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
scripts = [
    "pack01_audit_climatology.py",
    "pack02_trends_breaks.py",
    "pack03_etccdi_extremes.py",
    "pack04_gev_return_periods.py",
    "pack05_quantile_regression.py",
    "pack06_regime_shift_spi.py",
    "pack07_ita.py",
    "pack08_comparative_synthesis.py",
]
for name in scripts:
    print(f"Running {name} ...")
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / name)], check=True)
print("All analytical packs completed successfully.")
