# CLEAR-CARD

## Study Objective

Compare citation impact, publication patterns, and journal-metric associations of original research (ORA) versus meta-analysis, review, and guideline/statement/statistics articles (NORA) across JACC, Circulation, and European Heart Journal (2008–2022); JAMA Cardiology is supplemental.

## Folder structure

```text
src/clear_card/       Analysis modules
scripts/              CLI entry points
notebooks/            Interactive wrapper
data/                 Frozen inputs and sample.xlsx
sql/                  Optional database files
tests/                Reproducibility tests
.github/workflows/    CI
outputs/              Generated files (git-ignored)
```

## Code structure

- **`src/clear_card/cli.py`**: Parses commands and selects the requested workflow.
- **`src/clear_card/pipeline.py`**: Coordinates validation, figures, and supplemental tables.
- **`src/clear_card/data.py` and `validation.py`**: Load the frozen cohort and verify its schema.
- **`src/clear_card/experiments/`**: Implements one module for each numbered manuscript figure.
- **`src/clear_card/statistics.py`, `figures.py`, and `tables.py`**: Provide shared analysis, plotting, and table functions.

## Environment

- Python 3.11 or 3.12 with Arial installed.
- Install packages with `python -m pip install -r requirements.txt`.
- `data/processed/` reproduces the manuscript; `sample.xlsx` shows the schema.
- Optional SQL access: ODBC Driver 17, `python -m pip install -e ".[db]"`, and read-only `.env` credentials.

## How to run

```powershell
python -m clear_card validate
python -m pytest
python scripts/reproduce.py --target all
```

Results are written to `outputs/figures/` and `outputs/tables/`.

Optional database access:

```powershell
python -m clear_card db-check --database scopus
python scripts/export_citation_edges.py --database scopus --output data/private/scopus_citation_edges.csv.gz
```
