# Complaint triage

Routing consumer-finance complaints to the right team, from a TF-IDF baseline
through to a distilled model small enough to serve on CPU.

> **Status: in progress.** The data pipeline, the label taxonomy and the
> classical baselines are complete and measured. The neural models,
> distillation and quantisation stages are implemented but not yet trained —
> every number below is one this repository actually produced, and anything not
> yet measured is absent rather than estimated.

## The problem

A consumer files a complaint about a financial product in free text. Before
anyone can act on it, it has to reach the right desk — mortgage servicing,
debt collection, credit reporting, and so on. Done by hand this is slow and
inconsistent; done badly it breaches response clocks.

The obvious modern answer is to call a hosted LLM. Two reasons not to, and they
are the reasons this project exists:

1. **The text carries PII.** Complaint narratives contain names, account
   details and medical circumstances. Sending them to a third-party API is a
   problem a model running in-house does not have.
2. **Cost and latency at volume.** The corpus here is 1.7 million complaints.
   Per-ticket API pricing against a queue that size is real money, and a
   routing decision wants to be milliseconds, not a round trip.

So the target is a model that is accurate enough to automate the clear cases,
honest enough to escalate the unclear ones, and small enough to run on a CPU
beside the queue. A companion repository,
[complaint-triage-service](https://github.com/abdurehmanshafqat/complaint-triage-service),
is the Java service that loads the exported model and applies that routing
policy.

## Data provenance

This is worth reading before the results, because the obvious source does not
work.

The CFPB's [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
is the natural home for this task. **It no longer publishes the complaint
narratives.** Both the bulk export (`files.consumerfinance.gov/ccdb/complaints.csv.zip`,
370 MB, verified) and the search API return 15 columns, none of which is the
free text. The project therefore trains on a
[CC0-licensed snapshot](https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints)
captured while the narratives were still public: 1,689,573 rows, identical
column names.

Being a frozen snapshot is an advantage here — results stay reproducible
instead of drifting under a live feed. The CFPB states that it does not publish
personal information in this database
([data use](https://www.consumerfinance.gov/complaint/data-use/)), and that the
database is not a statistical sample of consumer experience, so nothing here
should be read as a measure of the market.

Nothing large is committed. `python scripts/download_data.py` fetches the three
parquet shards (~880 MB), and `python scripts/prefetch.py` additionally caches
the transformer weights so every later stage runs offline.

## The label taxonomy

The CFPB renamed and regrouped its `Product` values several times since 2011.
Training on the raw strings splits one real routing destination across two or
three labels, which depresses accuracy for reasons that have nothing to do with
the text. Every historical name is mapped onto a stable taxonomy, and every
merge is deliberate:

| canonical label | merged from |
| --- | --- |
| `credit_reporting` | "Credit reporting"; "Credit reporting, credit repair services, or other personal consumer reports"; "Credit reporting or other personal consumer reports" |
| `debt_collection` | "Debt collection"; "Debt or credit management" |
| `card` | "Credit card"; "Prepaid card"; "Credit card or prepaid card" |
| `bank_account` | "Bank account or service"; "Checking or savings account" |
| `mortgage` | "Mortgage" |
| `student_loan` | "Student loan" |
| `vehicle_or_consumer_loan` | "Consumer Loan"; "Vehicle loan or lease" |
| `payday_or_personal_loan` | "Payday loan"; "Payday loan, title loan, or personal loan"; "Payday loan, title loan, personal loan, or advance loan" |
| `money_transfer_or_crypto` | "Money transfers"; "Money transfer, virtual currency, or money service" |

`card` is a lossy merge and is marked as such: once the CFPB combined credit and
prepaid cards into one value, the earlier split cannot be recovered.

## Cleaning

| stage | rows |
| --- | ---: |
| read from source | 1,689,573 |
| dropped: no narrative | 0 |
| dropped: under 40 characters | -6,336 |
| dropped: unmapped product | 0 |
| dropped: duplicate narrative | -344,451 |
| dropped: class under 1,000 examples | -291 |
| **kept** | **335,486** |

Nine classes, 234,842 train / 50,322 validation / 50,322 test, filed between
2015-03-19 and 2024-02-09. Median narrative length 929 characters.

### Two findings that changed the project

**20.4% of narratives are duplicates.** 344,451 of 1,689,573. Boilerplate filed
in bulk — credit-repair mills especially — is far more common than expected. Had
those been left in, a large share of the test set would have been text the model
memorised during training. Exact duplicates are dropped; which copy survives
does not matter, only that no narrative appears on both sides of a split.

**One class is unlearnable and was excluded.** `other_financial_service`, the
CFPB's catch-all, holds 291 narratives in the entire corpus, and none landed in
the temporal test split. Keeping it does not produce a model that routes to it;
it produces a class with near-zero F1 that drags macro-F1 down by roughly a
tenth for a destination nobody could serve. A support floor excludes it, and the
cleaning report names what it removed rather than doing it silently.

### Splitting

The default split is **temporal**: validate and test on complaints filed after
everything the model trained on. That is the deployment setting — a routing queue
sees future complaints — and it is strictly harder than a random split, because
product mix and company behaviour drift. Three dataset variants are built
(`temporal`, `random`, `random_withdupes`) so the cost of a careless split can be
quantified rather than asserted.

## Results

Macro-F1 is the headline metric, not accuracy. Credit-reporting complaints
dominate the corpus, so a model that routes everything to one queue scores well
on accuracy and is useless.

| model | macro-F1 | accuracy | ms/item | train rows |
| --- | ---: | ---: | ---: | ---: |
| **linear SVM + TF-IDF** | **0.793** | 0.798 | **0.43** | 30,000 |
| logistic regression + TF-IDF | 0.787 | 0.792 | 0.44 | 30,000 |
| RBF SVM + LSA | 0.765 | 0.774 | 35.9 | 25,000 |
| histogram GBDT + LSA | 0.756 | 0.766 | 37.4 | 80,000 |
| random forest + LSA | 0.726 | 0.746 | 53.9 | 80,000 |

Single-item latency on CPU, measured one complaint at a time because that is how
a queue is served, not in batches.

### The cheapest model wins, by a wide margin

The linear SVM over sparse TF-IDF is both the **most accurate** and roughly
**85–125× faster** than every tree or kernel alternative. It is also the only one
trained on more data than it was given — it needed no subsampling to stay
tractable.

That is not an accident of tuning, and it is worth being precise about the cause.
Tree ensembles and kernel SVMs cannot work on 200,000 sparse dimensions, so they
run on an LSA projection to 256 dense components. That projection is where the
accuracy goes: complaint routing turns on the presence of specific, relatively
rare terms — "escrow", "deferment", "charge-off", "chargeback" — and compressing
a 100,000-term vocabulary into 256 directions blurs exactly the distinctions the
task depends on. A linear model on the uncompressed sparse matrix keeps them.

The practical consequence is that this project's bar is high and cheap: any
neural model has to beat 0.793 macro-F1 at 0.43 ms and ~8 MB, or justify itself
some other way. That is the honest comparison, and it is the reason the later
stages are framed around *latency and size at matched accuracy* rather than
around chasing a leaderboard.

Per class, linear SVM (full per-model reports in `reports/`):

| label | precision | recall | F1 | support |
| --- | ---: | ---: | ---: | ---: |
| `student_loan` | 0.928 | 0.911 | **0.919** | 5,011 |
| `mortgage` | 0.879 | 0.912 | **0.895** | 3,201 |
| `credit_reporting` | 0.816 | 0.849 | 0.832 | 8,924 |
| `vehicle_or_consumer_loan` | 0.801 | 0.823 | 0.812 | 5,127 |
| `card` | 0.785 | 0.768 | 0.776 | 6,485 |
| `bank_account` | 0.773 | 0.776 | 0.775 | 7,797 |
| `money_transfer_or_crypto` | 0.756 | 0.779 | 0.768 | 6,086 |
| `debt_collection` | 0.707 | 0.722 | 0.715 | 4,547 |
| `payday_or_personal_loan` | 0.738 | **0.570** | 0.643 | 3,144 |

### Where it fails, and why that shaped the service

`student_loan` and `mortgage` are easy: the vocabulary is distinctive
("deferment", "escrow", "forbearance"). The failure is
`payday_or_personal_loan`, at 0.570 recall — it is being swallowed by the other
loan categories, which is unsurprising since a title loan, a personal loan and a
vehicle loan share most of their language.

This is why the companion service routes on **margin** and not confidence alone.
A prediction of 0.55 with the runner-up at 0.54 and one of 0.55 with the
runner-up at 0.08 are both "55% confident" and should be treated completely
differently. The first is an adjacent-category confusion and belongs with a
person; sending it to a queue is how a complaint bounces between two desks for a
week.

## Layout

```
src/complaint_triage/     importable package - all logic lives here
  config.py               paths, label taxonomy, thresholds
  data.py                 streaming loader, cleaning, dedup, splits
  features.py             word/char TF-IDF, LSA
  classical.py            SVM, random forest, gradient boosting
  clustering.py           k-means sweep, NMF topics
  neural.py               Keras feed-forward, CNN, BiLSTM, student
  distill.py              offline distillation, cached teacher logits
  quantize.py             TFLite float16/int8, ONNX export
  evaluate.py             shared metrics, results ledger
scripts/                  one runner per stage
tests/                    26 tests
reports/                  results, per-class reports, figures
```

Notebooks narrate and import the package; they do not contain the logic.

## Running it

```bash
uv venv --python 3.12 && uv pip install -e '.[neural,teacher,deploy,analysis]'
python scripts/download_data.py      # ~880 MB of parquet shards
python scripts/prefetch.py           # also caches teacher weights, for offline use
python -m pytest                     # 26 tests
python scripts/run_classical.py      # fits and scores the baselines
```

After `prefetch.py`, everything runs with no network:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## Engineering notes

A few decisions that are not obvious from the code:

**Streaming, not loading.** The corpus is 1.7 million narratives. Cleaning them
all and only then capping per class would hold several gigabytes of Python
strings, so the per-class cap is applied *during* the single pass, using
reservoir sampling. Taking the first N per class would be cheaper and wrong: the
export is roughly chronological, so it would return only the oldest complaints
and leave the temporal split with nothing recent to test on.

**Offline distillation.** The teacher is fine-tuned once and its logits cached to
disk; the student then trains against those cached soft targets. This keeps the
teacher's forward pass off every student epoch, keeps PyTorch and TensorFlow off
a 6 GB GPU simultaneously, and leaves student training as a plain `model.fit`.

**Token indices stay `int32` through quantisation.** The usual full-integer
recipe rescales inputs, which is meaningless for a vocabulary index — index 7 is
not a quantity. Weights and activations quantise; the input dtype does not.

**Tokenisation lives outside the exported model.** Keras `TextVectorization`
relies on string ops that do not survive TFLite conversion cleanly, and the Java
service must tokenise anyway. The contract is integer sequences in, logits out,
with the vocabulary and the tokenisation *rules* exported alongside so the two
implementations cannot silently diverge.

**Two bugs worth recording.** An earlier downloader resumed a partial file with a
range request; the source regenerated the archive between transfers, and the
result had exactly the expected byte count with a spliced, unreadable
compression stream — the size check passed and the data was garbage. Downloads
are now validated by being parsed, not measured. Separately, a random forest with
400 unbounded trees and `n_jobs=-1` on 200,000 rows was killed by the OOM reaper;
tree depth, leaf size and worker count are now bounded, since each worker holds
its own tree-building state.

## Licence

Code: MIT, see [LICENSE](LICENSE).
Data: not redistributed here. The snapshot used is CC0-1.0; the underlying
complaints are published by the CFPB, which states the data is freely available
to use, analyse and build on. This repository is not affiliated with the CFPB.
