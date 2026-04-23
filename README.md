# Menkao rainfall analysis from CHIRPS daily statistics (1981-2025)

This repository contains the reproducible analytical workflow used to study long-term changes in rainfall regime, extremes, dry-wet structure, and intra-area spatial heterogeneity over the Menkao region, Kinshasa, Democratic Republic of the Congo, from daily CHIRPS-derived rainfall statistics.

## Authors

1. **Johnny MUHINDO BAHAVIRA**  
   Section Bâtiment et Travaux Publics, Institut National du Bâtiment et des Travaux Publics (INBTP), Kinshasa, Democratic Republic of the Congo  
   Email: johnny.muhindo@gmail.com

2. **Espérant KALUME KABENGELE**  
   Section Génie Rural, Institut National du Bâtiment et des Travaux Publics (INBTP), Kinshasa, Democratic Republic of the Congo

3. **Papy KABADI LELO ODIMBA**  
   Section Génie Rural, Institut National du Bâtiment et des Travaux Publics (INBTP), Kinshasa, Democratic Republic of the Congo

## Repository URL

https://github.com/JohnnyMuhindoBahavira-Collab/menkao-rainfall-chirps

## What this repository reproduces

The scripts reproduce the analytical workflow corresponding to eight analytical packs:

1. Data audit and descriptive climatology
2. Trends and break diagnostics
3. ETCCDI-style rainfall extremes
4. Generalised Extreme Value modelling and return periods
5. Quantile regression
6. Regime shifts and Standardised Precipitation Index analysis
7. Innovative Trend Analysis
8. Comparative synthesis across methods


## Repository structure

```text
.
├── data/
│   └── raw/
├── outputs/
├── scripts/
├── src/
│   └── menkao_analysis/
├── .github/workflows/
├── AUTHORS.md
├── CITATION.cff
├── environment.yml
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Input data

Place the input files in `data/raw/`:

- `FINAL1981to2025_CHIRPS_Daily_Mean_Min_Max.xlsx`
- `FINALCodeChirpsMENKAO.txt`

These files are already included in this package.

## Quick start

### Option 1: pip

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_all.py
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate menkao-rainfall-analysis
python scripts/run_all.py
```

## Running a single analytical block

```bash
python scripts/pack03_etccdi_extremes.py
```

## Outputs

Each script writes its own outputs to `outputs/<pack_name>/`, including:

- CSV tables
- PNG figures
- CSV source files for each figure

## Reproducibility notes

- All paths are repository-relative.
- The scripts do not depend on `/mnt/data/...` or any local notebook-specific paths.
- The workflow is designed to be executable from the repository root.


## Data provenance note

The raw rainfall table is derived from CHIRPS daily precipitation data extracted in Google Earth Engine over the Menkao study area.
