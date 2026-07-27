from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from clear_card.config import ORA, STUDY

OUTCOME_COLUMNS = {
    "delta_5_year_citations": "citation_within_10_year_subsequent_5_years",
    "citations_at_year_10": "citation_annual_10_year",
}
PREDICTOR_COLUMNS = {
    "5-year citations": "citation_within_5_year",
    "fitness": "lambda",
}


def prepare_fitness_validation(analysis: pd.DataFrame) -> pd.DataFrame:
    """Build the original-research cohort with complete 10-year follow-up."""

    cohort = analysis.loc[
        analysis["manuscript_category"].eq(ORA)
        & analysis["pub_year"].le(STUDY.validation_end_year)
        & analysis["lambda"].notna()
    ].copy()
    cohort[OUTCOME_COLUMNS["delta_5_year_citations"]] = (
        cohort["citation_within_10_year"] - cohort["citation_within_5_year"]
    )
    cohort[OUTCOME_COLUMNS["citations_at_year_10"]] = (
        cohort["citation_within_10_year"] - cohort["citation_within_9_year"]
    )
    return cohort


def estimate_roc_curves(
    cohort: pd.DataFrame,
    *,
    outcome_column: str,
    quantiles: Sequence[float] = (0.90, 0.75, 0.50),
    train_years: tuple[int, int] = (2008, 2011),
    test_years: tuple[int, int] = (2012, 2014),
    random_seed: int = STUDY.random_seed,
) -> pd.DataFrame:
    """Estimate the manuscript's held-out ROC curves for both predictors."""

    rows: list[pd.DataFrame] = []
    # The fixed chronological split prevents later publications from informing training.
    train_mask = cohort["pub_year"].between(*train_years)
    test_mask = cohort["pub_year"].between(*test_years)
    for quantile in quantiles:
        threshold = cohort[outcome_column].quantile(quantile)
        for predictor_label, predictor_column in PREDICTOR_COLUMNS.items():
            train = cohort.loc[train_mask, [predictor_column, outcome_column]].dropna()
            test = cohort.loc[test_mask, [predictor_column, outcome_column]].dropna()
            y_train = train[outcome_column].ge(threshold).astype(int)
            y_test = test[outcome_column].ge(threshold).astype(int)
            model = make_pipeline(
                FunctionTransformer(np.log1p, validate=False),
                StandardScaler(),
                LogisticRegression(
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_seed,
                ),
            )
            model.fit(train[[predictor_column]], y_train)
            score = model.predict_proba(test[[predictor_column]])[:, 1]
            false_positive_rate, true_positive_rate, _ = roc_curve(y_test, score)
            rows.append(
                pd.DataFrame(
                    {
                        "quantile": quantile,
                        "predictor": predictor_label,
                        "false_positive_rate": false_positive_rate,
                        "true_positive_rate": true_positive_rate,
                        "auc": roc_auc_score(y_test, score),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)
