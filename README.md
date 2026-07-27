# CLEAR-CARD

## Study Objective

Quantify longitudinal citation impact, publication share, and journal-metric associations of original research articles (ORA) versus meta-analysis, review, and guideline/statement/statistics articles (NORA) in JACC, Circulation, and European Heart Journal (2008–2022); JAMA Cardiology is supplemental.

## Folder structure

```text
data/                 Frozen de-identified inputs and sample.xlsx
notebooks/            Thin manuscript reproduction notebook
scripts/              Command-line entry points
sql/                  Optional SQL Server schema and extraction query
src/clear_card/       Cohorts, statistics, tables, figures, and experiments
tests/                Reproducibility checks
outputs/              Generated figures and tables (git-ignored)
```

## Environment

Use Python 3.11 with Arial installed. SQL Server and Microsoft ODBC Driver 17 are optional and needed only to re-extract source data; run `python -m pip install -e ".[db]"`, copy `.env.example` to `.env`, and enter a read-only database account. `data/sample.xlsx` illustrates the input schema, while exact results use the committed frozen snapshots in `data/processed/`.

## How to run

```powershell
git clone https://github.com/ts06068/clear-card.git
cd clear-card
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m clear_card validate
python scripts/reproduce.py --target all
```

Results are written to `outputs/figures/` and `outputs/tables/`. For an optional DB check, run `python -m clear_card db-check --database scopus` after configuring `.env`.
