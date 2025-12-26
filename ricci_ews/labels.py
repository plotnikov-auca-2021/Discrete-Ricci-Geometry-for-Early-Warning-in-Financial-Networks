"""
labels.py
Notation
--------

Let r_t denote the log-return on day t (with date index d_t), and h > 0
be a forward horizon in days.

1. Forward realized volatility over horizon h:

   RV_{t,h} = sqrt( 252 / h * sum_{k=1}^h r_{t+k}^2 )

   We then define a binary "high volatility" event label by thresholding
   RV_{t,h} at a high quantile over the sample (e.g. 80%):

       y^{Vol}_{t,h} = 1{ RV_{t,h} >= q_{Vol}(h) }

   where q_{Vol}(h) is the VOL_QUANTILE quantile of {RV_{t,h}}_t.


2. Forward maximum drawdown over horizon h:

   Consider the forward path of log prices starting at t:

       S_0 = 0
       S_k = sum_{i=1}^k r_{t+i},   k = 1, ..., h

   The corresponding simple-price path is P_k = exp(S_k), up to a
   multiplicative constant. We define the forward maximum drawdown as

       DD_{t,h} = max_{0 <= k <= h} (1 - P_k / max_{0 <= j <= k} P_j)
                = max_k (1 - exp( S_k - max_{j<=k} S_j ))

   which is the largest peak-to-trough percentage loss over the next
   h days, measured relative to the running maximum.

   We then define binary drawdown-event labels for a set of fixed
   thresholds δ in DD_THRESHOLDS, e.g. {3%, 5%, 7%}:

       y^{DD,δ}_{t,h} = 1{ DD_{t,h} >= δ }

Core building blocks:

    compute_forward_realized_volatility(index_log_returns, horizons=None)
    compute_forward_max_drawdown(index_log_returns, horizons=None)

These return forward RV_{t,h} and DD_{t,h} series for each horizon h.

Binary labels:

    make_volatility_labels(rv_df, vol_quantile=None, prefix="y_vol")
    make_drawdown_labels(dd_df, dd_thresholds=None, prefix="y_dd")

High-level convenience:

    build_all_labels(index_log_returns, horizons=None,
                     dd_thresholds=None, vol_quantile=None)

which returns:

    labels_df, meta

where labels_df contains forward RV/DD and their binary labels, indexed
by date t (the "current" day used for features), and meta contains
threshold information for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config


# ----------------------------------------------------------------------
# Helpers for defaults from config
# ----------------------------------------------------------------------


def _get_default_horizons() -> Tuple[int, ...]:
    """
    Horizon set H used for forward-looking labels.

    Defaults to config.HORIZONS if present, otherwise (5, 10, 20).
    """
    default = (5, 10, 20)
    horizons = getattr(config, "HORIZONS", default)
    if isinstance(horizons, (list, tuple)):
        return tuple(int(h) for h in horizons)
    return default


def _get_default_dd_thresholds() -> Tuple[float, ...]:
    """
    Drawdown thresholds δ used for binary labels.

    Defaults to config.DD_THRESHOLDS if present, otherwise (0.03, 0.05, 0.07).
    """
    default = (0.03, 0.05, 0.07)
    dd = getattr(config, "DD_THRESHOLDS", default)
    if isinstance(dd, (list, tuple)):
        return tuple(float(x) for x in dd)
    return default


def _get_default_vol_quantile() -> float:
    """
    Quantile level q for high-volatility labels.

    Defaults to config.VOL_QUANTILE if present, otherwise 0.8.
    """
    return float(getattr(config, "VOL_QUANTILE", 0.8))


# ----------------------------------------------------------------------
# Forward realized volatility and drawdown
# ----------------------------------------------------------------------


def compute_forward_realized_volatility(
    index_log_returns: pd.Series,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Compute forward realized volatility RV_{t,h} for each horizon h.

    Parameters
    ----------
    index_log_returns : pd.Series
        Log-returns of the index, r_t, indexed by date (e.g. S&P 500).
        These should be aligned with the dates of the stock-return panel
        used for feature windows (i.e. same index as rets_rect).
    horizons : sequence of int, optional
        Forward horizons h (in trading days). If None, uses config.HORIZONS.

    Returns
    -------
    rv_df : pd.DataFrame
        DataFrame indexed by date t, with one column per horizon:

            rv_h5, rv_h10, rv_h20, ...

        At date t, rv_h{h}[t] = RV_{t,h} uses returns r_{t+1}..r_{t+h}.
        The last h rows are NaN because not enough forward data.
    """
    if horizons is None:
        horizons = _get_default_horizons()

    r = index_log_returns.sort_index().astype(float)
    dates = r.index
    arr = r.values
    T = len(arr)

    out: Dict[str, np.ndarray] = {}

    for h in horizons:
        h = int(h)
        if h <= 0:
            continue
        rv = np.full(T, np.nan, dtype=float)
        # For each t, use future returns r_{t+1},...,r_{t+h}
        for t in range(T - h):
            window = arr[t + 1 : t + 1 + h]
            if window.size != h:
                continue
            rv[t] = np.sqrt(252.0 / h * np.sum(window ** 2))
        out[f"rv_h{h}"] = rv

    rv_df = pd.DataFrame(out, index=dates)
    return rv_df


def compute_forward_max_drawdown(
    index_log_returns: pd.Series,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Compute forward maximum drawdown DD_{t,h} for each horizon h.

    We work purely in log-return space, as described in the module
    docstring: starting from S_0 = 0, S_k = sum_{i=1}^k r_{t+i},
    and then converting back to simple drawdowns.

    Parameters
    ----------
    index_log_returns : pd.Series
        Log-returns of the index, r_t, indexed by date.
    horizons : sequence of int, optional
        Forward horizons h (in trading days). If None, uses config.HORIZONS.

    Returns
    -------
    dd_df : pd.DataFrame
        DataFrame indexed by date t, with one column per horizon:

            dd_h5, dd_h10, dd_h20, ...

        At date t, dd_h{h}[t] = DD_{t,h} ∈ [0,1] is the largest
        peak-to-trough percentage loss over the next h days.
        The last h rows are NaN because not enough forward data.
    """
    if horizons is None:
        horizons = _get_default_horizons()

    r = index_log_returns.sort_index().astype(float)
    dates = r.index
    arr = r.values
    T = len(arr)

    out: Dict[str, np.ndarray] = {}

    for h in horizons:
        h = int(h)
        if h <= 0:
            continue
        dd = np.full(T, np.nan, dtype=float)
        # For each t, use future returns r_{t+1},...,r_{t+h}
        for t in range(T - h):
            window = arr[t + 1 : t + 1 + h]
            if window.size == 0:
                continue
            # Log-price path relative to S_0=0, including initial level
            cum = np.concatenate(([0.0], np.cumsum(window)))  # length h+1
            running_max = np.maximum.accumulate(cum)
            dd_log = running_max - cum  # >= 0
            # Convert log drawdowns to simple percentage losses:
            # If Δ = log(P_max / P), then drop = 1 - exp(-Δ).
            dd_simple = 1.0 - np.exp(-dd_log)
            dd[t] = float(dd_simple.max())
        out[f"dd_h{h}"] = dd

    dd_df = pd.DataFrame(out, index=dates)
    return dd_df


# ----------------------------------------------------------------------
# Binary labels: volatility and drawdown
# ----------------------------------------------------------------------


def make_volatility_labels(
    rv_df: pd.DataFrame,
    vol_quantile: float | None = None,
    prefix: str = "y_vol",
) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """
    Construct high-volatility binary labels y^{Vol}_{t,h} from forward
    realized volatility RV_{t,h}.

    Parameters
    ----------
    rv_df : pd.DataFrame
        Output of compute_forward_realized_volatility, with columns
        'rv_h{h}' for each horizon h, indexed by date t.
    vol_quantile : float, optional
        Quantile level q for "high volatility" events. If None, uses
        config.VOL_QUANTILE.
    prefix : str
        Prefix for label column names. Labels for horizon h will be
        named '{prefix}_h{h}', e.g. 'y_vol_h5'.

    Returns
    -------
    labels_df : pd.DataFrame
        DataFrame with one binary column per horizon, indexed by date t.
    thresholds : dict
        Mapping h -> q_{Vol}(h), the numeric threshold used for each
        horizon.
    """
    if vol_quantile is None:
        vol_quantile = _get_default_vol_quantile()

    labels = {}
    thresholds: Dict[int, float] = {}

    for col in rv_df.columns:
        if not col.startswith("rv_h"):
            continue
        # Parse horizon h from column name 'rv_h{h}'
        try:
            h = int(col.split("h", 1)[1])
        except Exception:
            continue

        series = rv_df[col].astype(float)
        # Use only finite values to estimate threshold
        finite_vals = series[np.isfinite(series)]
        if finite_vals.empty:
            thresh = np.nan
            labels_col = pd.Series(np.nan, index=series.index)
        else:
            thresh = float(finite_vals.quantile(vol_quantile))
            labels_col = (series >= thresh).astype(float)
        labels[f"{prefix}_h{h}"] = labels_col
        thresholds[h] = thresh

    labels_df = pd.DataFrame(labels, index=rv_df.index)
    return labels_df, thresholds


def make_drawdown_labels(
    dd_df: pd.DataFrame,
    dd_thresholds: Sequence[float] | None = None,
    prefix: str = "y_dd",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Construct binary drawdown-event labels y^{DD,δ}_{t,h} from forward
    maximum drawdown DD_{t,h}, for a set of fixed thresholds δ.

    Parameters
    ----------
    dd_df : pd.DataFrame
        Output of compute_forward_max_drawdown, with columns 'dd_h{h}'.
    dd_thresholds : sequence of float, optional
        Drawdown thresholds δ ∈ (0,1), e.g. [0.03, 0.05, 0.07].
        If None, uses config.DD_THRESHOLDS.
    prefix : str
        Prefix for label column names. Labels for horizon h and
        threshold δ will be named:

            '{prefix}_h{h}_thr{b}'

        where b = int(100 * δ), e.g. 'y_dd_h20_thr5' for 5% DD over 20 days.

    Returns
    -------
    labels_df : pd.DataFrame
        DataFrame with one binary column per (h, δ) combination.
    thresholds : dict
        Mapping from label column name -> δ (numeric threshold) so the
        exact definition of each label is explicit.
    """
    if dd_thresholds is None:
        dd_thresholds = _get_default_dd_thresholds()

    labels = {}
    thresholds: Dict[str, float] = {}

    for col in dd_df.columns:
        if not col.startswith("dd_h"):
            continue
        try:
            h = int(col.split("h", 1)[1])
        except Exception:
            continue

        series = dd_df[col].astype(float)

        for delta in dd_thresholds:
            delta = float(delta)
            # binary: event if DD_{t,h} >= δ
            lab_col_name = f"{prefix}_h{h}_thr{int(round(delta * 100))}"
            labels[lab_col_name] = (series >= delta).astype(float)
            thresholds[lab_col_name] = delta

    labels_df = pd.DataFrame(labels, index=dd_df.index)
    return labels_df, thresholds


# ----------------------------------------------------------------------
# High-level label construction
# ----------------------------------------------------------------------


@dataclass
class LabelMetadata:
    """
    Metadata describing how labels were constructed.

    Attributes
    ----------
    horizons : tuple of int
        Horizons h used.
    dd_thresholds : tuple of float
        Drawdown thresholds δ used.
    vol_quantile : float
        Volatility quantile q used.
    vol_thresholds : dict
        Mapping h -> q_{Vol}(h) (numeric volatility thresholds).
    dd_label_thresholds : dict
        Mapping label_name -> δ (drawdown thresholds per label column).
    """

    horizons: Tuple[int, ...]
    dd_thresholds: Tuple[float, ...]
    vol_quantile: float
    vol_thresholds: Dict[int, float]
    dd_label_thresholds: Dict[str, float]


def build_all_labels(
    index_log_returns: pd.Series,
    horizons: Sequence[int] | None = None,
    dd_thresholds: Sequence[float] | None = None,
    vol_quantile: float | None = None,
) -> Tuple[pd.DataFrame, LabelMetadata]:
    """
    Convenience function to build forward RV/DD series and all binary
    labels in one pass.

    Parameters
    ----------
    index_log_returns : pd.Series
        Index log-returns r_t, indexed by date.
    horizons : sequence of int, optional
        Forward horizons h (days). If None, uses config.HORIZONS.
    dd_thresholds : sequence of float, optional
        Drawdown thresholds δ (in [0,1]). If None, config.DD_THRESHOLDS.
    vol_quantile : float, optional
        Volatility quantile q ∈ (0,1). If None, config.VOL_QUANTILE.

    Returns
    -------
    labels_df : pd.DataFrame
        DataFrame indexed by date t, containing:
          - forward RV series:  rv_h{h}
          - forward DD series:  dd_h{h}
          - volatility labels:   y_vol_h{h}
          - drawdown labels:     y_dd_h{h}_thr{b}
    meta : LabelMetadata
        Metadata object with horizons, thresholds, and quantile info.

    """
    if horizons is None:
        horizons = _get_default_horizons()
    else:
        horizons = tuple(int(h) for h in horizons)

    if dd_thresholds is None:
        dd_thresholds = _get_default_dd_thresholds()
    else:
        dd_thresholds = tuple(float(x) for x in dd_thresholds)

    if vol_quantile is None:
        vol_quantile = _get_default_vol_quantile()

    # 1) Continuous forward targets
    rv_df = compute_forward_realized_volatility(index_log_returns, horizons=horizons)
    dd_df = compute_forward_max_drawdown(index_log_returns, horizons=horizons)

    # 2) Binary labels
    vol_labels_df, vol_thr = make_volatility_labels(rv_df, vol_quantile=vol_quantile)
    dd_labels_df, dd_thr_map = make_drawdown_labels(dd_df, dd_thresholds=dd_thresholds)

    # 3) Assemble all into a single DataFrame
    labels_df = pd.concat([rv_df, dd_df, vol_labels_df, dd_labels_df], axis=1)

    meta = LabelMetadata(
        horizons=tuple(horizons),
        dd_thresholds=tuple(dd_thresholds),
        vol_quantile=float(vol_quantile),
        vol_thresholds=vol_thr,
        dd_label_thresholds=dd_thr_map,
    )

    return labels_df, meta
