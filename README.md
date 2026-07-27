# CLEAR-CARD

## Study Objective

Compare citation impact, publication trends, and journal-metric associations between original research (ORA) and meta-analysis, review, guideline, statement, and statistics articles (NORA) in JACC, Circulation, and European Heart Journal (2008–2022). JAMA Cardiology is supplemental.

## Folder structure

```text
src/clear_card/       Core analysis package
scripts/              Command-line wrappers
notebooks/            Manuscript reproduction notebook
data/                 Frozen inputs and schema example
sql/                  Optional database queries
tests/                Input and workflow checks
.github/workflows/    CI
outputs/              Generated artifacts (git-ignored)
```

## Code structure

- **`src/clear_card/cli.py`** parses commands and selects a workflow.
- **`src/clear_card/pipeline.py`** controls execution order.
- **`src/clear_card/data.py`** loads frozen inputs; **`validation.py`** rejects checksum, schema, and benchmark drift.
- **`src/clear_card/experiments/`** keeps one module per numbered figure.
- **`src/clear_card/statistics.py`**, **`figures.py`**, and **`tables.py`** own shared calculations and rendering.

## Environment

- Use Python 3.11 or 3.12. Install Arial for manuscript-matched figures.
- Install with `python -m pip install -r requirements.txt`.
- Full reproduction uses `data/processed/`; `data/sample.xlsx` shows the schema only.
- SQL export is optional. It needs Microsoft ODBC Driver 17 for SQL Server, `python -m pip install -e ".[db]"`, and read-only `.env` credentials.

## How to run

```powershell
python -m clear_card validate
python -m pytest
python scripts/reproduce.py --target all
```

Output goes to `outputs/figures/` and `outputs/tables/`.

Optional database check and export:

```powershell
python -m clear_card db-check --database scopus
python scripts/export_citation_edges.py --database scopus --output data/private/scopus_citation_edges.csv.gz
```
