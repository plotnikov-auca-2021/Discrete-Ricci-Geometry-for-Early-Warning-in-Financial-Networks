"""
Rolling-window generation and time-based splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from . import config


@dataclass
class RollingWindow:
    end_date: pd.Timestamp
    start_date: pd.Timestamp
    data: np.ndarray  # shape (W, N)


class RollingWindowGenerator:
    """
    Helper to iterate over overlapping rolling windows of fixed length W.
    """

    def __init__(self, returns_panel: pd.DataFrame, window_size: int | None = None):
        self.returns_panel = returns_panel.sort_index()
        self.window_size = window_size or config.WINDOW_SIZE

    def iter_windows(self) -> Iterator[RollingWindow]:
        dates = self.returns_panel.index
        W = self.window_size
        for i in range(W - 1, len(dates)):
            end = dates[i]
            start = dates[i - W + 1]
            window_df = self.returns_panel.iloc[i - W + 1 : i + 1]
            yield RollingWindow(
                end_date=end,
                start_date=start,
                data=window_df.to_numpy(copy=True),
            )


def split_validation_evaluation(dates: pd.Index,
                                validation_end: str | None = None) -> Tuple[pd.Index, pd.Index]:
    """
    Split dates into validation and evaluation sets.

    Parameters
    ----------
    dates : DatetimeIndex
    validation_end : str or None
        If None, use config.VALIDATION_END_DATE.

    Returns
    -------
    validation_dates, evaluation_dates : DatetimeIndex
    """
    if validation_end is None:
        validation_end = config.VALIDATION_END_DATE
    cut = pd.to_datetime(validation_end)
    validation = dates[dates <= cut]
    evaluation = dates[dates > cut]
    return validation, evaluation
