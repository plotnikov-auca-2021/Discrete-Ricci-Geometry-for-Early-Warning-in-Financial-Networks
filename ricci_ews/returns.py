"""
Return computation and cleaning.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def compute_log_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns per ticker:
        r_{i,t} = log(P_{i,t}) - log(P_{i,t-1})

    Parameters
    ----------
    price_panel : DataFrame
        index = date, columns = tickers

    Returns
    -------
    returns_panel : DataFrame
        log returns, aligned with price_panel index (first row NaN).
    """
    log_prices = np.log(price_panel)
    rets = log_prices.diff()
    return rets


def align_and_rectangularize(
    returns_panel: pd.DataFrame,
    universe: List[str],
) -> pd.DataFrame:
    """
    Keep only tickers in universe and drop all dates with any missing values.

    This enforces a rectangular panel (no NaNs) for the fixed universe U,
    which is required by the graph construction.

    Parameters
    ----------
    returns_panel : DataFrame
        index = date, columns = tickers
    universe : list[str]
        Fixed asset universe.

    Returns
    -------
    rectangular_returns : DataFrame
        index = date, columns = universe, no missing values.
    """
    sub = returns_panel[universe]
    # First row is typically NaN due to differencing; drop any rows with NaNs
    sub = sub.dropna(how="any")
    return sub


def winsorize_returns(
    returns_panel: pd.DataFrame,
    lower_q: float = 0.001,
    upper_q: float = 0.999,
) -> pd.DataFrame:
    """
    Winsorize returns at given quantiles across all assets.

    Parameters
    ----------
    returns_panel : DataFrame
    lower_q, upper_q : float
        Quantiles between 0 and 1.

    Returns
    -------
    DataFrame of same shape as returns_panel.
    """
    flattened = returns_panel.values.flatten()
    lo = np.nanquantile(flattened, lower_q)
    hi = np.nanquantile(flattened, upper_q)
    clipped = np.clip(returns_panel.values, lo, hi)
    return pd.DataFrame(clipped, index=returns_panel.index, columns=returns_panel.columns)


def standardize_within_window(R_window: np.ndarray) -> np.ndarray:
    """
    Standardize returns within a window: zero mean and unit variance per column.

    Parameters
    ----------
    R_window : np.ndarray of shape (W, N)

    Returns
    -------
    standardized : np.ndarray of shape (W, N)
    """
    mean = np.mean(R_window, axis=0, keepdims=True)
    std = np.std(R_window, axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    return (R_window - mean) / std
