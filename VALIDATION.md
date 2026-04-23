# Validation

This repository structure was checked after refactoring to repository-relative paths.

## Validation performed

The analytical scripts were executed individually from the repository root using the included input files in `data/raw/`.

Validated scripts:

- `scripts/pack01_audit_climatology.py`
- `scripts/pack02_trends_breaks.py`
- `scripts/pack03_etccdi_extremes.py`
- `scripts/pack04_gev_return_periods.py`
- `scripts/pack05_quantile_regression.py`
- `scripts/pack06_regime_shift_spi.py`
- `scripts/pack07_ita.py`
- `scripts/pack08_comparative_synthesis.py`

## Notes

- All scripts resolve paths relative to the repository root.
- The repository ships with an empty `outputs/` directory by design.
- Running `python scripts/run_all.py` regenerates all analytical outputs.
