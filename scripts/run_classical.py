#!/usr/bin/env python3
"""Fit and score the classical baselines, writing results to reports/.

These are the numbers every later model is judged against. Run:

    python scripts/run_classical.py                      # temporal split
    python scripts/run_classical.py --dataset random     # for the leakage table
    python scripts/run_classical.py --limit-train 50000  # quick pass
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complaint_triage import classical, config as cfg, data, evaluate as ev  # noqa: E402

log = logging.getLogger("run_classical")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="temporal",
                        help="dataset variant stem, e.g. temporal, random, random_withdupes")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="cap training rows, for a fast pass")
    parser.add_argument("--only", nargs="*", default=None,
                        help="fit only these named baselines")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )

    path = cfg.PROCESSED_DIR / f"complaints_{args.dataset}.parquet"
    frame = data.load_processed(path=path)
    labels = ev.labels_in(frame)

    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    if args.limit_train:
        train = train.sample(
            n=min(len(train), args.limit_train), random_state=cfg.RANDOM_SEED
        )

    log.info(
        "dataset=%s  train=%s  test=%s  classes=%d",
        args.dataset, f"{len(train):,}", f"{len(test):,}", len(labels),
    )

    specs = classical.build_specs()
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            print(f"no baseline matches {sorted(wanted)}", file=sys.stderr)
            return 2

    ledger = ev.ResultsLedger.load()
    started = time.perf_counter()
    ledger, fitted = classical.train_and_score(
        train, test, ledger=ledger, specs=specs, split_strategy=args.dataset
    )
    log.info("fitted %d baselines in %.1fs", len(specs), time.perf_counter() - started)

    print()
    print(ledger.to_markdown())
    ledger.save()

    # The linear model's top features are the cheapest check that the model
    # learned the domain rather than an artefact of redaction or boilerplate.
    if "linear-svm-tfidf" in fitted:
        features = classical.top_features_per_class(fitted["linear-svm-tfidf"], top_k=12)
        out = cfg.REPORT_DIR / f"top_features_{args.dataset}.json"
        out.write_text(
            json.dumps(
                {
                    label: group.sort_values("rank")["feature"].tolist()
                    for label, group in features.groupby("label")
                },
                indent=2,
            )
        )
        print(f"\ntop features -> {out}")

        best = ledger.to_frame().iloc[0]
        print(f"\nbest so far: {best['model']} macro-F1={best['macro_f1']}")

    # One report per model, named after the model.
    #
    # An earlier version wrote a single `classification_report_<dataset>.txt`
    # and fell back to "whichever model was fitted first" when the linear SVM
    # was not part of the run. A later partial run therefore overwrote the
    # headline model's report with a different model's numbers, under a
    # filename that still implied the headline model. Nothing errored; the
    # file was simply wrong. Naming each report after its model removes the
    # possibility.
    for name, pipeline in fitted.items():
        report = cfg.REPORT_DIR / f"classification_report_{args.dataset}_{name}.txt"
        report.write_text(
            f"{name} on {args.dataset} split, {len(train):,} training rows\n\n"
            + ev.report_text(test["label"], pipeline.predict(test["text"]), labels)
        )
        print(f"per-class report -> {report.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
