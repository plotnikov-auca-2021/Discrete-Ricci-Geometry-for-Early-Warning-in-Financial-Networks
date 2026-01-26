# ricci_ews/windows.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingWindow:
    """
    A single rolling window over a returns matrix.

    R_win is a numpy array with shape (W, N) and dates_win is a DatetimeIndex of length W.
    """
    window_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    start_idx: int
    end_idx: int
    dates_win: pd.DatetimeIndex
    R_win: np.ndarray


def build_date_to_pos(dates: pd.DatetimeIndex) -> Dict[pd.Timestamp, int]:
    """
    Map each timestamp in dates -> integer position.
    Use this for O(1) endpoint lookup.
    """
    # Ensure uniqueness & monotonicity are not strictly required, but recommended
    return {pd.Timestamp(d): i for i, d in enumerate(dates)}


def get_window_bounds_for_endpoint(
    end_date: pd.Timestamp,
    date_to_pos: Dict[pd.Timestamp, int],
    window_size: int,
) -> Optional[Tuple[int, int]]:
    """
    For a given end_date, return (start_idx, end_idx) inclusive bounds for a window of length window_size.
    Returns None if:
      - end_date is not in date_to_pos
      - not enough history for a full window
    """
    end_date = pd.Timestamp(end_date)
    if end_date not in date_to_pos:
        return None

    end_idx = date_to_pos[end_date]
    start_idx = end_idx - window_size + 1
    if start_idx < 0:
        return None

    return start_idx, end_idx


def iter_analysis_return_windows(
    rets_rect: pd.DataFrame,
    analysis_dates: pd.DatetimeIndex,
    window_size: int = 252,
    stride: int = 1,
    start_window_id: int = 0,
) -> Iterator[RollingWindow]:
    """
    Iterate rolling windows whose endpoints are ONLY in analysis_dates.

    Notes:
      - rets_rect should include warmup+analysis data (so early analysis endpoints can have full history).
      - analysis_dates should be a subset of rets_rect.index (after warmup).
      - stride is applied over analysis_dates (e.g., stride=5 means every 5th analysis endpoint).

    Yields:
      RollingWindow objects with (W, N) numpy arrays.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if rets_rect.empty:
        return

    dates = rets_rect.index
    date_to_pos = build_date_to_pos(dates)

    # Ensure analysis_dates are sorted and unique-like
    analysis_dates = pd.DatetimeIndex(pd.to_datetime(analysis_dates)).sort_values()

    window_id = start_window_id

    # Iterate only over analysis endpoints, applying stride there
    for end_date in analysis_dates[::stride]:
        bounds = get_window_bounds_for_endpoint(end_date, date_to_pos, window_size)
        if bounds is None:
            continue

        start_idx, end_idx = bounds
        dates_win = dates[start_idx : end_idx + 1]

        # Extract W x N values (as numpy) for speed
        R_win = rets_rect.iloc[start_idx : end_idx + 1].to_numpy(dtype=float, copy=False)

        # Skip if shape mismatch (should not happen, but safe)
        if R_win.shape[0] != window_size:
            continue

        yield RollingWindow(
            window_id=window_id,
            start_date=pd.Timestamp(dates_win[0]),
            end_date=pd.Timestamp(dates_win[-1]),
            start_idx=start_idx,
            end_idx=end_idx,
            dates_win=dates_win,
            R_win=R_win,
        )
        window_id += 1


def iter_analysis_windows_with_node_names(
    rets_rect: pd.DataFrame,
    analysis_dates: pd.DatetimeIndex,
    window_size: int = 252,
    stride: int = 1,
    start_window_id: int = 0,
):
    """
    Convenience iterator: yields (RollingWindow, node_names)
    node_names are the tickers / columns in rets_rect.
    """
    node_names = list(rets_rect.columns)
    for win in iter_analysis_return_windows(
        rets_rect=rets_rect,
        analysis_dates=analysis_dates,
        window_size=window_size,
        stride=stride,
        start_window_id=start_window_id,
    ):
        yield win, node_names
