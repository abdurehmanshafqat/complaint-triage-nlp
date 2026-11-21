"""Central configuration: paths, label taxonomy, and split settings.

Keeping these in one module means notebooks and scripts cannot drift apart on
what "the dataset" or "the label set" means.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORT_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURE_DIR: Final[Path] = REPORT_DIR / "figures"

# --------------------------------------------------------------------------
# Data source
# --------------------------------------------------------------------------
# A note on provenance, because it is not obvious and it matters.
#
# The CFPB's own bulk export (files.consumerfinance.gov/ccdb/complaints.csv.zip)
# and its search API both return 15 columns and *no* complaint narrative. The
# free-text field that this project classifies is no longer published there.
#
# We therefore train on a CC0-licensed snapshot taken while the narratives were
# still public. Its column names are identical to the CFPB export's, so nothing
# downstream has to know the difference. Being a frozen snapshot is a feature
# here: results stay reproducible instead of shifting under a live feed.
DATASET_REPO: Final[str] = "BEE-spoke-data/consumer-finance-complaints"
DATASET_CONFIG: Final[str] = "has-text"  # pre-filtered to rows that have text
DATASET_LICENSE: Final[str] = "CC0-1.0"
DATASET_ROWS: Final[int] = 1_689_573

_SHARD_BASE: Final[str] = (
    f"https://huggingface.co/api/datasets/{DATASET_REPO}/parquet/{DATASET_CONFIG}/train"
)
N_SHARDS: Final[int] = 3
DATA_SHARD_URLS: Final[tuple[str, ...]] = tuple(
    f"{_SHARD_BASE}/{index}.parquet" for index in range(N_SHARDS)
)

SHARD_DIR: Final[Path] = RAW_DIR / "parquet"

DATA_SOURCE_PAGE: Final[str] = f"https://huggingface.co/datasets/{DATASET_REPO}"
CFPB_SOURCE_PAGE: Final[str] = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/"
)
CFPB_DATA_USE_PAGE: Final[str] = "https://www.consumerfinance.gov/complaint/data-use/"

# --------------------------------------------------------------------------
# Source columns we care about (CFPB bulk CSV header names)
# --------------------------------------------------------------------------
COL_NARRATIVE: Final[str] = "Consumer complaint narrative"
COL_PRODUCT: Final[str] = "Product"
COL_SUBPRODUCT: Final[str] = "Sub-product"
COL_ISSUE: Final[str] = "Issue"
COL_DATE: Final[str] = "Date received"
COL_COMPANY: Final[str] = "Company"
COL_STATE: Final[str] = "State"
COL_ID: Final[str] = "Complaint ID"

USE_COLUMNS: Final[tuple[str, ...]] = (
    COL_ID,
    COL_DATE,
    COL_PRODUCT,
    COL_SUBPRODUCT,
    COL_ISSUE,
    COL_NARRATIVE,
    COL_COMPANY,
    COL_STATE,
)

# --------------------------------------------------------------------------
# Label taxonomy
# --------------------------------------------------------------------------
# The CFPB has renamed and re-grouped its `Product` values several times since
# 2011. Training on the raw strings silently splits a single real-world routing
# destination across two or three labels, which depresses accuracy for reasons
# that have nothing to do with the text. We therefore normalise to a stable
# taxonomy that reflects how a complaint would actually be routed to a team.
#
# Every merge below is deliberate and is documented in the README.
PRODUCT_TO_CANONICAL: Final[dict[str, str]] = {
    # --- credit reporting -------------------------------------------------
    "Credit reporting": "credit_reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_reporting",
    "Credit reporting or other personal consumer reports": "credit_reporting",
    # --- debt collection --------------------------------------------------
    "Debt collection": "debt_collection",
    "Debt or credit management": "debt_collection",
    # --- mortgage ---------------------------------------------------------
    "Mortgage": "mortgage",
    # --- cards ------------------------------------------------------------
    # "Credit card or prepaid card" cannot be split back into its two
    # predecessors, so all card products collapse into one destination.
    "Credit card": "card",
    "Prepaid card": "card",
    "Credit card or prepaid card": "card",
    # --- deposit accounts -------------------------------------------------
    "Bank account or service": "bank_account",
    "Checking or savings account": "bank_account",
    # --- student loans ----------------------------------------------------
    "Student loan": "student_loan",
    # --- vehicle & general consumer loans ---------------------------------
    "Consumer Loan": "vehicle_or_consumer_loan",
    "Vehicle loan or lease": "vehicle_or_consumer_loan",
    # --- short-term / personal loans --------------------------------------
    "Payday loan": "payday_or_personal_loan",
    "Payday loan, title loan, or personal loan": "payday_or_personal_loan",
    "Payday loan, title loan, personal loan, or advance loan": "payday_or_personal_loan",
    # --- money movement ---------------------------------------------------
    "Money transfers": "money_transfer_or_crypto",
    "Money transfer, virtual currency, or money service": "money_transfer_or_crypto",
    "Virtual currency": "money_transfer_or_crypto",
    # --- everything else --------------------------------------------------
    "Other financial service": "other_financial_service",
}

CANONICAL_LABELS: Final[tuple[str, ...]] = (
    "bank_account",
    "card",
    "credit_reporting",
    "debt_collection",
    "money_transfer_or_crypto",
    "mortgage",
    "payday_or_personal_loan",
    "student_loan",
    "vehicle_or_consumer_loan",
    "other_financial_service",
)

# --------------------------------------------------------------------------
# Cleaning / split settings
# --------------------------------------------------------------------------
MIN_NARRATIVE_CHARS: Final[int] = 40
RANDOM_SEED: Final[int] = 42
TEST_FRACTION: Final[float] = 0.15
VAL_FRACTION: Final[float] = 0.15

# Narratives are redacted by the CFPB with runs of 'X'. Long runs carry no
# signal and waste model capacity, so they are collapsed during cleaning.
REDACTION_PATTERN: Final[str] = r"X{2,}"
