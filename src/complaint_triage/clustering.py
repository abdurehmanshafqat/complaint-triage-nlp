"""Unsupervised structure: what themes exist in the narratives?

This is not decoration. Before building a router it is worth asking whether the
CFPB's own product labels actually match how complaints cluster in text space.
Where they diverge is where the classifier will struggle later, and the answer
feeds directly into the error analysis.

Two complementary views:

* **K-means over LSA** — hard partitions, cheap, good for "how many distinct
  topics are there" via inertia and silhouette.
* **NMF over TF-IDF** — soft, parts-based topics with directly readable term
  lists, which is what you want when explaining findings to a non-ML teammate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, silhouette_score

from . import config as cfg
from .features import build_lsa, build_word_tfidf


@dataclass
class ClusterSweep:
    """Diagnostics for one candidate value of k."""

    k: int
    inertia: float
    silhouette: float
    adjusted_rand_vs_labels: float


def sweep_kmeans(
    texts: pd.Series,
    labels: pd.Series,
    k_values: tuple[int, ...] = (4, 6, 8, 10, 12, 16),
    n_components: int = 200,
    sample: int = 40_000,
    seed: int = cfg.RANDOM_SEED,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit K-means for several k and report how well each matches the taxonomy.

    `adjusted_rand_vs_labels` is the interesting column: a value near zero means
    the text clusters have little to do with the official product labels, which
    is a warning that the labels encode business structure the text does not.
    """
    if len(texts) > sample:
        index = texts.sample(n=sample, random_state=seed).index
        texts, labels = texts.loc[index], labels.loc[index]

    lsa = build_lsa(n_components=n_components, seed=seed)
    embedded = lsa.fit_transform(texts)

    rows: list[ClusterSweep] = []
    for k in k_values:
        model = MiniBatchKMeans(
            n_clusters=k, random_state=seed, n_init=5, batch_size=2048
        )
        assignments = model.fit_predict(embedded)
        rows.append(
            ClusterSweep(
                k=k,
                inertia=float(model.inertia_),
                silhouette=float(
                    silhouette_score(embedded, assignments, sample_size=8000, random_state=seed)
                ),
                adjusted_rand_vs_labels=float(adjusted_rand_score(labels, assignments)),
            )
        )
    return pd.DataFrame([vars(r) for r in rows]), embedded


def nmf_topics(
    texts: pd.Series,
    n_topics: int = 10,
    top_terms: int = 12,
    max_features: int = 50_000,
    seed: int = cfg.RANDOM_SEED,
) -> pd.DataFrame:
    """Fit NMF and return the top terms per topic as a tidy frame."""
    vectoriser = build_word_tfidf(max_features=max_features, ngram_range=(1, 2))
    matrix = vectoriser.fit_transform(texts)
    model = NMF(
        n_components=n_topics, random_state=seed, init="nndsvda", max_iter=400
    )
    model.fit(matrix)

    names = np.asarray(vectoriser.get_feature_names_out())
    rows: list[dict[str, object]] = []
    for topic_index, weights in enumerate(model.components_):
        for rank, position in enumerate(np.argsort(weights)[::-1][:top_terms], start=1):
            rows.append(
                {
                    "topic": topic_index,
                    "rank": rank,
                    "term": names[position],
                    "weight": float(weights[position]),
                }
            )
    return pd.DataFrame(rows)


def cluster_label_crosstab(
    labels: pd.Series, assignments: np.ndarray, normalise: bool = True
) -> pd.DataFrame:
    """Where do the official labels land across discovered clusters?"""
    table = pd.crosstab(labels, assignments)
    if normalise:
        table = table.div(table.sum(axis=1), axis=0)
    return table
