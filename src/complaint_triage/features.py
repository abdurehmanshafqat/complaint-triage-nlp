"""Text featurisation for the classical baselines.

Two representations are built here:

* **Sparse TF-IDF** — what linear models want. Word and character n-grams are
  both useful on this corpus: word n-grams capture product vocabulary
  ("escrow", "APR"), character n-grams absorb the misspellings and mangled
  formatting common in free-text complaint forms.
* **Dense LSA** — TF-IDF reduced with truncated SVD. Tree ensembles and an RBF
  SVM are impractical on 200k-dimensional sparse input; projecting to a few
  hundred dense components makes them trainable without discarding much signal.
"""
from __future__ import annotations

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer

from . import config as cfg


def build_word_tfidf(
    max_features: int = 200_000,
    min_df: int = 3,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Word-level TF-IDF with sublinear term frequency."""
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
    )


def build_char_tfidf(
    max_features: int = 150_000,
    min_df: int = 3,
    ngram_range: tuple[int, int] = (3, 5),
) -> TfidfVectorizer:
    """Character n-gram TF-IDF, robust to the typos in hand-typed narratives."""
    return TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        sublinear_tf=True,
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
    )


def build_lsa(
    n_components: int = 300,
    max_features: int = 200_000,
    seed: int = cfg.RANDOM_SEED,
) -> Pipeline:
    """TF-IDF -> truncated SVD -> L2 normalise.

    Normalising after SVD matters: the raw component scales differ by an order
    of magnitude, which distorts the RBF kernel's distance metric.
    """
    return make_pipeline(
        build_word_tfidf(max_features=max_features),
        TruncatedSVD(n_components=n_components, random_state=seed),
        Normalizer(copy=False),
    )
