"""Classical baselines: linear SVM, RBF SVM, random forest, gradient boosting.

These are not throwaway baselines. On short-to-medium free text with a strong
vocabulary signal, a well-tuned linear SVM over TF-IDF is a genuinely hard
target — and it trains in seconds on CPU, needs no GPU, and ships as a few MB
of coefficients. Any neural model in this project has to justify itself
*against these numbers*, not against a strawman.

Notes on the modelling choices:

* `class_weight="balanced"` throughout. The corpus is dominated by credit
  reporting; without reweighting the minority routes (prepaid cards, money
  transfer) collapse to near-zero recall.
* Tree ensembles and the RBF SVM run on LSA-reduced features. Fitting them on
  200k sparse dimensions is either impossibly slow or badly overfit; the SVD
  projection is the standard remedy and is applied inside each pipeline so
  nothing leaks from test into the fitted vocabulary.
* The RBF SVM is trained on a subsample. Kernel SVMs are O(n^2)-ish in the
  number of samples; fitting on the full corpus is not practical and is not
  how anyone would deploy it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC, LinearSVC

from . import config as cfg
from . import evaluate as ev
from .features import build_lsa, build_word_tfidf

log = logging.getLogger(__name__)


@dataclass
class ClassicalSpec:
    """A named pipeline plus how much data it can realistically be fitted on."""

    name: str
    pipeline: Pipeline
    max_train: int | None = None
    notes: str = ""


def build_specs(seed: int = cfg.RANDOM_SEED) -> list[ClassicalSpec]:
    """The baseline suite, in increasing order of training cost."""
    return [
        ClassicalSpec(
            name="logreg-tfidf",
            pipeline=make_pipeline(
                build_word_tfidf(),
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
            notes="reference point for the linear family",
        ),
        ClassicalSpec(
            name="linear-svm-tfidf",
            pipeline=make_pipeline(
                build_word_tfidf(),
                LinearSVC(C=0.5, class_weight="balanced", random_state=seed),
            ),
            notes="strongest cheap baseline; the bar the neural models must clear",
        ),
        ClassicalSpec(
            name="random-forest-lsa",
            pipeline=make_pipeline(
                build_lsa(n_components=300, seed=seed),
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
            max_train=200_000,
            notes="300 LSA components; sparse TF-IDF is impractical for trees",
        ),
        ClassicalSpec(
            name="hist-gbdt-lsa",
            pipeline=make_pipeline(
                build_lsa(n_components=300, seed=seed),
                HistGradientBoostingClassifier(
                    max_iter=400,
                    learning_rate=0.1,
                    early_stopping=True,
                    validation_fraction=0.1,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
            max_train=200_000,
            notes="gradient boosted trees on LSA features",
        ),
        ClassicalSpec(
            name="rbf-svm-lsa",
            pipeline=make_pipeline(
                build_lsa(n_components=200, seed=seed),
                SVC(
                    C=4.0,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    cache_size=1000,
                    random_state=seed,
                ),
            ),
            max_train=40_000,
            notes="subsampled: kernel SVM scales poorly in n",
        ),
    ]


def _subsample(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame
    log.info("Subsampling %s -> %s rows for tractability", f"{len(frame):,}", f"{limit:,}")
    return frame.sample(n=limit, random_state=seed)


def train_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    ledger: ev.ResultsLedger | None = None,
    specs: list[ClassicalSpec] | None = None,
    split_strategy: str = "temporal",
    latency_samples: int = 200,
    seed: int = cfg.RANDOM_SEED,
) -> tuple[ev.ResultsLedger, dict[str, Pipeline]]:
    """Fit every baseline, score it on `test`, and record the results.

    Returns the ledger and the fitted pipelines (kept so the notebooks can do
    error analysis without refitting).
    """
    ledger = ledger or ev.ResultsLedger()
    specs = specs or build_specs(seed=seed)
    fitted: dict[str, Pipeline] = {}

    bench_texts = test["text"].head(latency_samples).tolist()

    for spec in specs:
        subset = _subsample(train, spec.max_train, seed)
        log.info("Fitting %s on %s rows", spec.name, f"{len(subset):,}")
        spec.pipeline.fit(subset["text"], subset["label"])

        predictions = spec.pipeline.predict(test["text"])
        latency = ev.benchmark_latency(
            lambda t: spec.pipeline.predict([t]), bench_texts, repeats=1
        )

        note = spec.notes
        if spec.max_train and len(train) > spec.max_train:
            note = f"{note}; trained on {len(subset):,} of {len(train):,}"

        ledger.add(
            ev.score(
                name=spec.name,
                family="classical",
                y_true=test["label"].to_numpy(),
                y_pred=predictions,
                latency_ms_per_item=latency,
                split_strategy=split_strategy,
                notes=note,
            )
        )
        fitted[spec.name] = spec.pipeline

    return ledger, fitted


def top_features_per_class(
    pipeline: Pipeline, top_k: int = 15
) -> pd.DataFrame:
    """Highest-weighted n-grams per class for a fitted linear TF-IDF pipeline.

    This is the cheapest useful interpretability check in the project: if the
    top features for `mortgage` are 'escrow', 'loan modification' and
    'foreclosure', the model has learned the domain. If they are artefacts of
    redaction or boilerplate, it has not.
    """
    vectoriser = pipeline[0]
    classifier = pipeline[-1]
    if not hasattr(classifier, "coef_"):
        raise TypeError(f"{type(classifier).__name__} exposes no linear coefficients")

    names = np.asarray(vectoriser.get_feature_names_out())
    rows: list[dict[str, object]] = []
    for index, label in enumerate(classifier.classes_):
        weights = classifier.coef_[index]
        for rank, position in enumerate(np.argsort(weights)[::-1][:top_k], start=1):
            rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "feature": names[position],
                    "weight": float(weights[position]),
                }
            )
    return pd.DataFrame(rows)
