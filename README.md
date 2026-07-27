# CLEAR-CARD

## Study Objective

Compare citation impact, publication trends, and journal-metric associations between original research (ORA) and meta-analysis, review, guideline, statement, and statistics articles (NORA) in JACC, Circulation, and European Heart Journal (2008–2022). JAMA Cardiology is supplemental.

## Folder structure

```text
.
├── .github/                                   # GitHub automation
│   └── workflows/                             # CI workflows
│       └── ci.yml                            # Cross-platform lint, tests, and build
├── data/                                      # Reviewed release data
│   ├── expected/                              # Reproducibility fixtures
│   │   └── manuscript_benchmarks.json        # Expected cohort counts
│   ├── processed/                             # Frozen manuscript inputs
│   │   ├── article_classifications.csv.gz    # Frozen article categories
│   │   ├── icite_validation.csv.gz           # iCite validation snapshot
│   │   ├── impact_factor_simulation.csv      # Impact-factor simulation inputs
│   │   ├── manuscript_analysis.csv.gz        # Primary analysis cohort
│   │   ├── scopus_validation.csv.gz          # Scopus validation snapshot
│   │   ├── supplemental_category_counts.csv  # Frozen taxonomy counts
│   │   ├── wos_publication_counts.csv        # Annual Web of Science counts
│   │   └── wos_scopus_validation.csv.gz      # Matched WoS–Scopus citations
│   ├── journal_metrics.csv                    # Annual journal metrics
│   ├── manifest.json                          # Provenance and SHA-256 checksums
│   ├── README.md                              # Data inventory and release policy
│   └── sample.xlsx                            # Schema example; not a reproduction input
├── notebooks/                                 # Interactive entry points
│   └── reproduce_manuscript.ipynb             # Validate inputs and run the pipeline
├── scripts/                                   # Command-line wrappers
│   ├── export_citation_edges.py               # Export SQL citation edges
│   └── reproduce.py                            # Reproduction CLI shortcut
├── sql/                                       # Optional SQL Server support
│   ├── extract_citation_edges.sql             # Citation-edge extraction query
│   └── schema.sql                              # Minimal extraction schema
├── src/                                       # Installable source tree
│   └── clear_card/                            # Manuscript reproduction package
│       ├── experiments/                       # Figure orchestration
│       │   ├── __init__.py                    # Figure package declaration
│       │   ├── common.py                      # Input loading and metric alignment
│       │   ├── fitness_validation.py          # Long-term fitness validation figures
│       │   ├── main_figures.py                # Main manuscript figures
│       │   ├── supplemental_figures.py        # Supplemental analysis figures
│       │   └── validation_figures.py          # Source-validation figures
│       ├── plotting/                          # Reusable manuscript plots
│       │   ├── __init__.py                    # Internal plotting package
│       │   ├── composition.py                 # Publication and citation composition
│       │   ├── correlations.py                # Journal-metric correlation panels
│       │   ├── distributions.py               # Manuscript histograms
│       │   ├── journal_trends.py              # Annual trend plots
│       │   ├── style.py                       # Figure configuration and saving
│       │   └── validation.py                  # Count and citation concordance
│       ├── __init__.py                        # Package version metadata
│       ├── __main__.py                        # Module CLI entry point
│       ├── cli.py                             # Command parsing and dispatch
│       ├── comparisons.py                     # External validation datasets
│       ├── config.py                          # Study constants and windows
│       ├── correlation.py                     # Correlations and bootstrap comparisons
│       ├── data.py                            # Input loading and cohort selection
│       ├── database.py                        # SQL connection and citation extraction
│       ├── figures.py                         # Public plotting API
│       ├── modeling.py                        # Fitness-validation ROC models
│       ├── pipeline.py                        # Figures and table orchestration
│       ├── settings.py                        # Paths and database credentials
│       ├── statistics.py                      # Public statistical API
│       ├── summaries.py                       # Group summaries and formatting
│       ├── tables.py                          # Eighteen supplemental tables
│       ├── taxonomy.py                        # Supplemental taxonomy counts
│       └── validation.py                      # Input and benchmark verification
├── tests/                                     # Fast reproducibility checks
│   ├── test_manuscript_definitions.py         # Study definitions and cohort rules
│   └── test_reproducibility.py                # Hashes, benchmarks, and taxonomy
├── outputs/                                   # Generated figures and tables; ignored
├── .env.example                               # Read-only database template
├── .gitattributes                             # Text rules and dataset byte safety
├── .gitignore                                 # Secrets, raw data, and outputs
├── pyproject.toml                             # Package, dependencies, and test settings
├── requirements.txt                           # Reproducibility environment installer
└── README.md                                  # Setup and execution guide
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
