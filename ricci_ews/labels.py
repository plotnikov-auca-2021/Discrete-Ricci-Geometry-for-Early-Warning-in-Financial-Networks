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
# ricci_ews/labels.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def _pct_to_tag(p: float) -> str:
    """
    Convert pct (e.g., 0.03) into a compact tag.
      0.03 -> "3"
      0.025 -> "2p5"
      0.001 -> "0p1"
    Interpreted as *percent*, i.e., multiply by 100.
    """
    x = 100.0 * float(p)
    # close to integer?
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    # close to 1 decimal?
    if abs(x * 10 - round(x * 10)) < 1e-9:
        s = f"{x:.1f}"
        return s.replace(".", "p").rstrip("0").rstrip("p")
    # fallback: 2 decimals
    s = f"{x:.2f}"
    s = s.rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _forward_matrix(rets: pd.Series, h: int) -> pd.DataFrame:
    """
    Build T x h matrix of future returns:
      col k contains r_{t+k+1}
    """
    cols = [rets.shift(-(k + 1)) for k in range(h)]
    mat = pd.concat(cols, axis=1)
    mat.columns = [f"t+{k+1}" for k in range(h)]
    return mat


def _forward_realized_vol(rets: pd.Series, h: int) -> pd.Series:
    """
    Forward realized volatility over next h days:
      vol_t = std( r_{t+1}, ..., r_{t+h} )
    """
    M = _forward_matrix(rets, h)
    return M.std(axis=1, ddof=0)


def _forward_max_drawdown_mag(rets: pd.Series, h: int) -> pd.Series:
    """
    Forward max drawdown magnitude over next h days, computed on the
    forward price path starting at 1.0.

    For each t:
      f = [r_{t+1},...,r_{t+h}]
      P_0 = 1
      P_k = exp(sum_{i=1..k} f_i)
      DD = min_k (P_k / max_{j<=k} P_j - 1)
      magnitude = -DD (>=0)
    """
    r = rets.values.astype(float)
    T = len(r)
    out = np.full(T, np.nan, dtype=float)

    # compute with small loop (h is small: 5/10/15/20 etc)
    for i in range(T - h):
        f = r[i + 1 : i + h + 1]
        if np.any(~np.isfinite(f)):
            continue
        p = np.concatenate([[1.0], np.exp(np.cumsum(f))])
        run_max = np.maximum.accumulate(p)
        dd = np.min(p / run_max - 1.0)
        out[i] = -dd  # magnitude
    return pd.Series(out, index=rets.index, name=f"mdd_h{h}")


def build_all_labels(
    index_log_rets: pd.Series,
    *,
    analysis_dates: Optional[pd.DatetimeIndex] = None,
    vol_top_pct: float = 0.03,
    dd_tail_pcts: Tuple[float, ...] = (0.03, 0.05, 0.07),
    label_horizons: Tuple[int, ...] = (5, 10, 20),
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build ALL binary event labels for the pipeline, parameterized by:
      - vol_top_pct (top X% of forward realized vol within analysis_dates)
      - dd_tail_pcts (top X% worst forward max drawdown magnitudes within analysis_dates)
      - label_horizons (forward horizon days)

    Naming:
      - volatility label for horizon h:
          y_vol_top{TAG}_h{h}
        where TAG is derived from vol_top_pct (e.g., 0.03 -> 3)

      - drawdown label for horizon h, tail pct p:
          y_dd_h{h}_thr{TAG}
        where TAG is derived from p (e.g., 0.05 -> 5)

    Thresholds are computed ONLY on analysis_dates (not warmup).
    Output index is restricted to analysis_dates if provided; otherwise full index.

    Returns:
      labels_df, meta_dict
    """
    if not isinstance(index_log_rets, pd.Series):
        raise TypeError("index_log_rets must be a pd.Series of log returns indexed by date.")
    rets = index_log_rets.sort_index().astype(float)

    # define analysis slice for threshold computation
    if analysis_dates is None:
        analysis_dates = pd.DatetimeIndex(rets.index)
    else:
        analysis_dates = pd.DatetimeIndex(analysis_dates)
        analysis_dates = analysis_dates.intersection(rets.index)

    if len(analysis_dates) < 30:
        raise ValueError(f"analysis_dates too small (n={len(analysis_dates)}). Need >= ~30.")

    vol_tag = _pct_to_tag(vol_top_pct)

    labels = {}
    meta: Dict = {
        "vol_top_pct": float(vol_top_pct),
        "dd_tail_pcts": tuple(float(x) for x in dd_tail_pcts),
        "label_horizons": tuple(int(h) for h in label_horizons),
        "thresholds": {"vol": {}, "dd": {}},
    }

    for h in label_horizons:
        h = int(h)
        if h <= 0:
            continue

        # --- volatility ---
        vol = _forward_realized_vol(rets, h)
        vol_a = vol.loc[analysis_dates].dropna()
        if len(vol_a) > 0:
            thr_vol = float(vol_a.quantile(1.0 - float(vol_top_pct)))
        else:
            thr_vol = np.nan
        meta["thresholds"]["vol"][h] = thr_vol

        vol_col = f"y_vol_top{vol_tag}_h{h}"
        labels[vol_col] = (vol >= thr_vol).astype(float)  # float to allow NaN later
        # restore NaNs where vol itself is NaN
        labels[vol_col] = labels[vol_col].where(np.isfinite(vol), np.nan)

        # --- drawdowns ---
        mdd = _forward_max_drawdown_mag(rets, h)
        mdd_a = mdd.loc[analysis_dates].dropna()

        meta["thresholds"]["dd"][h] = {}
        for p in dd_tail_pcts:
            p = float(p)
            tag = _pct_to_tag(p)
            if len(mdd_a) > 0:
                thr_dd = float(mdd_a.quantile(1.0 - p))
            else:
                thr_dd = np.nan
            meta["thresholds"]["dd"][h][p] = thr_dd

            dd_col = f"y_dd_h{h}_thr{tag}"
            y = (mdd >= thr_dd).astype(float)
            y = y.where(np.isfinite(mdd), np.nan)
            labels[dd_col] = y

    df_labels = pd.DataFrame(labels, index=rets.index).sort_index()

    # Restrict output rows to analysis_dates (recommended for your pipeline)
    df_labels = df_labels.loc[analysis_dates].copy()

    # Cast to Int64 (nullable) for clean CSVs and downstream handling
    for c in df_labels.columns:
        df_labels[c] = df_labels[c].round().astype("Int64")

    return df_labels, meta
