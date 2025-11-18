"""
Configuration and hyperparameters for the Ricci curvature early-warning project.
"""

from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

SP500_STOCKS_FILE = DATA_DIR / "sp500_stocks.csv"
SP500_COMPANIES_FILE = DATA_DIR / "sp500_companies.csv"
SP500_INDEX_FILE = DATA_DIR / "sp500_index.csv"

# Date range (can be adjusted to your dataset)
START_DATE = "2014-01-01"
END_DATE = "2024-12-31"

# Rolling window length in trading days
WINDOW_SIZE = 252  # ~1 year

# Validation / evaluation split
VALIDATION_END_DATE = "2018-12-31"  # adjust once you inspect your data

# Universe construction
MAX_MISSING_RATIO = 0.05  # at least 95% availability

# Correlation graph hyperparameters
SHRINKAGE_TARGET = "constant_correlation"  # or "identity"
CORR_BETA = 1
TARGET_AVG_DEGREE = 20
LENGTH_EXPONENT = 0.75
LENGTH_EPS = 1e-4

# Precision graph hyperparameters
GLASSO_LAMBDA = 0.1

# Curvature hyperparameters
ORC_ALPHA = 0.5
RANDOM_WALK_ETA = 0.1

# Label thresholds and horizons
DD_THRESHOLDS = [0.03, 0.05, 0.07]
VOL_QUANTILE = 0.8
HORIZONS = [5, 10, 20]

# Random state for reproducibility
RANDOM_STATE = 42
