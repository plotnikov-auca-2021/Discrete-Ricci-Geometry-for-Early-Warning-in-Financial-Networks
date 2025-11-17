"""
Universe construction.

Select a fixed universe U of stocks with sufficiently long and liquid history.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from . import config


def compute_availability_mask(price_panel: pd.DataFrame) -> pd.Series:
    """
    Compute per-ticker availability as fraction of non-missing days.

    Parameters
    ----------
    price_panel : DataFrame
        index = date, columns = tickers

    Returns
    -------
    availability : Series
        index = ticker, values = fraction of non-NaN observations.
    """
    availability = price_panel.notna().mean(axis=0)
    return availability


def select_universe(
    price_panel: pd.DataFrame,
    max_missing_ratio: float | None = None
) -> List[str]:
    """
    Select a fixed universe of tickers with at most max_missing_ratio missing data.

    Parameters
    ----------
    price_panel : DataFrame
        index = date, columns = tickers
    max_missing_ratio : float, optional
        e.g. 0.05 means at least 95% availability. If None, falls back to config.MAX_MISSING_RATIO.

    Returns
    -------
    list[str]
        Sorted list of tickers in the universe.
    """
    if max_missing_ratio is None:
        max_missing_ratio = config.MAX_MISSING_RATIO

    availability = compute_availability_mask(price_panel)
    universe = availability[availability >= 1.0 - max_missing_ratio].index.tolist()
    return sorted(universe)
