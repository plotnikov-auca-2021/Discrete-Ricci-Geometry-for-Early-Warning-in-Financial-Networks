# pipeline/pipeline.py
"""
Rolling-window pipeline for Discrete Ricci Geometry Early-Warning project.

Steps (CLI subcommands)
-----------------------

1) corr_graphs
   - From raw S&P 500 data up to correlation-based graphs.
   - For each rolling window with endpoint in analysis period:
        * build correlation graph G_corr
        * compute basic graph metrics (nodes, edges, density, etc.)
   - Output: correlation_graphs_over_time.csv

2) prec_graphs
   - Same as above but for precision-based graphs (Graphical Lasso).
   - Output: precision_graphs_over_time.csv

3) curvature
   - For each window endpoint in analysis period:
        * build G_corr and G_prec
        * compute FRC for both graphs
        * optionally compute ORC for both graphs (cfg.compute_orc)
        * aggregate into scalar curvature features (features.compute_curvature_features_for_date)
   - Output:
        * curvature_over_time.csv   (meta + FRC (+ ORC if enabled))
        * curvature_frc_only.csv    (meta + FRC columns)
        * curvature_orc_only.csv    (meta + ORC columns)  [only if enabled and present]

4) features
   - For each window endpoint in analysis period:
        * baseline features (market-only, topology-only, eigenmode) on correlation graph
   - Merge with curvature_over_time.csv
   - Add EWMA transforms (features.add_ewma_features)
   - Add labels from labels.build_all_labels using cfg.vol_top_pct, cfg.dd_tail_pcts, cfg.label_horizons
     Thresholds are computed ONLY over analysis period (no warmup labels)
   - Output: features_and_labels.csv

5) models
   - Load features_and_labels.csv
   - Discover labels matching current cfg (supports both old and new naming)
   - Train default models (models.get_default_model_specs) with chronological train/test split
   - Save:
        * model_predictions.csv (per date, y_true, p_hat, model, label)
        * model_metrics.csv     (per model x label)

6) mech_probes
   - Load model_predictions.csv
   - Compute mechanistic probes for ALL predicted dates:
        * rebuild window
        * compute probes on G_corr and G_prec (diffusion/mixing, MFPT, commute time, etc.)
   - Output: mechanistic_probes.csv

Usage
-----
$env:PIPE_VOL_TOP_PCT="0.05"
$env:PIPE_DD_TAIL_PCTS="0.02,0.05"
$env:PIPE_LABEL_HORIZONS="5,10,15"


python -m pipeline.pipeline corr_graphs --out-dir outputs
python -m pipeline.pipeline prec_graphs --out-dir outputs
python -m pipeline.pipeline curvature   --out-dir outputs
python -m pipeline.pipeline features    --out-dir outputs
python -m pipeline.pipeline models      --out-dir outputs
python -m pipeline.pipeline mech_probes --out-dir outputs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .pipeline_config import get_pipeline_config, PipelineConfig

from ricci_ews import (
    config as core_config,
    data_io,
    universe,
    returns,
    graphs_correlation,
    graphs_precision,
    curvature,
    features,
    labels,
    models,
    probes,
)

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None


TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _pct_to_tag(p: float) -> str:
    """Convert pct (e.g., 0.03) -> tag used in label names (e.g., '3', '2p5')."""
    x = 100.0 * float(p)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    if abs(x * 10 - round(x * 10)) < 1e-9:
        s = f"{x:.1f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _call_first_existing(module, names: Sequence[str], *args, **kwargs):
    """
    Call the first callable attribute in `names` that exists on `module`.
    Avoids hard references that trigger IDE "Cannot find reference" warnings.
    """
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    raise AttributeError(f"{module.__name__}: none of these callables exist: {list(names)}")


def _project_root() -> Path:
    # pipeline/ is at project_root/pipeline/
    return Path(__file__).resolve().parents[1]


def _get_window_size_default() -> int:
    return int(getattr(core_config, "WINDOW_SIZE", 252))


def _safe_to_datetime_index(x) -> pd.DatetimeIndex:
    if isinstance(x, pd.DatetimeIndex):
        return x
    return pd.DatetimeIndex(pd.to_datetime(x))


# ---------------------------------------------------------------------
# Data loading and slicing: warmup + analysis
# ---------------------------------------------------------------------

def _slice_warmup_analysis_by_trading_days(
    dates: pd.DatetimeIndex,
    warmup_years: int,
    analysis_years: int,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Slice from the beginning by trading days:
      warmup_days = warmup_years * 252
      analysis_days = analysis_years * 252
      keep warmup+analysis (for windowing), but "analysis_dates" are endpoints we model/label.

    Returns:
      dates_slice: warmup+analysis dates
      analysis_dates: analysis-only dates (endpoints to process)
    """
    dates = _safe_to_datetime_index(dates)
    warmup_days = int(warmup_years) * TRADING_DAYS_PER_YEAR
    analysis_days = int(analysis_years) * TRADING_DAYS_PER_YEAR

    if warmup_days < 0 or analysis_days <= 0:
        raise ValueError("warmup_years must be >=0 and analysis_years must be >0")

    total_days = warmup_days + analysis_days
    if len(dates) < total_days:
        # allow shorter total; still require enough to have warmup + at least some analysis
        if len(dates) <= warmup_days + 20:
            raise ValueError(
                f"Not enough dates for warmup_years={warmup_years} and analysis_years={analysis_years}. "
                f"Have {len(dates)}."
            )
        dates_slice = dates
    else:
        dates_slice = dates[:total_days]

    if len(dates_slice) <= warmup_days:
        raise ValueError(
            f"After slicing, not enough analysis dates. "
            f"Have {len(dates_slice)} total, warmup_days={warmup_days}."
        )

    analysis_dates = dates_slice[warmup_days:]
    return dates_slice, analysis_dates


def _tidy_to_wide_prices(prices_tidy: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a tidy price table into a wide (date x ticker) price matrix.

    Expected tidy columns include:
      - date (or Date)
      - ticker (or symbol/Symbol)
      - adj_close/close/price-like column
    """
    df = prices_tidy.copy()

    # date
    date_col = None
    for c in ["date", "Date", "DATE"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("Could not find a date column in prices data.")

    # ticker
    ticker_col = None
    for c in ["ticker", "Ticker", "symbol", "Symbol", "SYMBOL"]:
        if c in df.columns:
            ticker_col = c
            break
    if ticker_col is None:
        raise ValueError("Could not find a ticker/symbol column in prices data.")

    # price column
    price_col = None
    for c in ["adj_close", "Adj Close", "adjclose", "close", "Close", "price", "Price"]:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        # fallback to first numeric column not date/ticker
        cand = [c for c in df.columns if c not in [date_col, ticker_col]]
        num = df[cand].select_dtypes(include=[np.number]).columns.tolist()
        if not num:
            raise ValueError("Could not find a numeric price column in prices data.")
        price_col = num[0]

    df[date_col] = pd.to_datetime(df[date_col])
    wide = (
        df.pivot_table(index=date_col, columns=ticker_col, values=price_col, aggfunc="last")
        .sort_index()
    )
    wide.columns = [str(c) for c in wide.columns]
    return wide


def _compute_log_returns_from_prices_wide(prices_wide: pd.DataFrame) -> pd.DataFrame:
    prices_wide = prices_wide.sort_index()
    # avoid log(0)
    px = prices_wide.astype(float).replace(0.0, np.nan)
    rets = np.log(px).diff()
    return rets

def _make_pseudo_index_log_returns(rets_rect: pd.DataFrame) -> pd.Series:
    """
    Fallback when no index price series is available in data_io:
    use equal-weighted 'market' log return = cross-sectional mean of stock log returns.

    This yields a Series aligned to rets_rect.index.
    """
    s = rets_rect.mean(axis=1, skipna=True)
    s.name = "pseudo_index_log_ret"
    return s


def _load_base_data(cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Load prices, compute returns, align dates, slice warmup+analysis.

    IMPORTANT CHANGE:
    - If ricci_ews.data_io does NOT provide an index loader, we create a pseudo-index
      log return series as the equal-weight mean of stock log returns.
    """

    # -------- load raw prices (tidy OR wide depending on your data_io) --------
    prices_tidy = _call_first_existing(
        data_io,
        ["load_prices", "load_stocks", "load_sp500_stocks", "load_stock_prices"],
    )

    # -------- build universe (optional) --------
    df_universe = None
    try:
        df_universe = _call_first_existing(
            universe,
            ["build_universe", "make_universe", "get_universe"],
            prices_tidy,
            cfg=cfg,
        )
    except TypeError:
        # function doesn't accept cfg
        try:
            df_universe = _call_first_existing(
                universe,
                ["build_universe", "make_universe", "get_universe"],
                prices_tidy,
            )
        except Exception:
            df_universe = None
    except Exception:
        df_universe = None

    # -------- compute stock returns (prefer returns module; fallback to manual) --------
    rets_rect: Optional[pd.DataFrame] = None
    fn_rect = getattr(returns, "compute_log_returns_rect", None)

    if callable(fn_rect) and df_universe is not None:
        rets_rect = fn_rect(prices_tidy, df_universe)
    else:
        # fallback: convert tidy->wide if needed, then log-diff
        if isinstance(prices_tidy, pd.DataFrame) and (
            ("ticker" in prices_tidy.columns) or ("Symbol" in prices_tidy.columns) or ("symbol" in prices_tidy.columns)
        ):
            prices_wide = _tidy_to_wide_prices(prices_tidy)
        else:
            prices_wide = prices_tidy.copy()
            if not isinstance(prices_wide.index, pd.DatetimeIndex):
                for c in ["date", "Date", "DATE"]:
                    if c in prices_wide.columns:
                        prices_wide[c] = pd.to_datetime(prices_wide[c])
                        prices_wide = prices_wide.set_index(c)
                        break
                prices_wide.index = pd.to_datetime(prices_wide.index)
            prices_wide = prices_wide.sort_index()

        # if universe exists, filter columns
        if df_universe is not None:
            tcol = None
            for c in ["ticker", "Ticker", "symbol", "Symbol"]:
                if c in df_universe.columns:
                    tcol = c
                    break
            if tcol is not None:
                keep = [str(x) for x in df_universe[tcol].dropna().unique().tolist()]
                keep_set = set(keep)
                common = [c for c in prices_wide.columns if str(c) in keep_set]
                if common:
                    prices_wide = prices_wide[common]

        rets_rect = _compute_log_returns_from_prices_wide(prices_wide)

    if rets_rect is None or rets_rect.empty:
        raise ValueError("Failed to compute stock return matrix rets_rect.")

    rets_rect = rets_rect.sort_index()

    # -------- compute index log returns --------
    # Try to load an index price series if data_io provides it; otherwise build pseudo-index from rets_rect.
    index_df = None
    try:
        index_df = _call_first_existing(
            data_io,
            ["load_index", "load_index_prices", "load_sp500_index", "load_market_index"],
        )
    except AttributeError:
        index_df = None

    index_log_rets: Optional[pd.Series] = None
    fn_idx = getattr(returns, "compute_log_returns", None)

    if index_df is not None and callable(fn_idx):
        idx = fn_idx(index_df)
        if isinstance(idx, pd.DataFrame):
            num_cols = idx.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                raise ValueError("Index returns DataFrame has no numeric columns.")
            index_log_rets = idx[num_cols[0]].copy()
        else:
            index_log_rets = idx.copy()

    elif index_df is not None:
        # manual from index prices
        if isinstance(index_df, pd.DataFrame):
            df = index_df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                for c in ["date", "Date", "DATE"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c])
                        df = df.set_index(c)
                        break
                df.index = pd.to_datetime(df.index)

            if "close" in df.columns:
                s = df["close"]
            else:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not num_cols:
                    raise ValueError("Index dataframe has no numeric columns to compute returns.")
                s = df[num_cols[0]]

            index_log_rets = np.log(s.astype(float).replace(0.0, np.nan)).diff()
            index_log_rets.name = "index_log_ret"

        elif isinstance(index_df, pd.Series):
            s = index_df.copy()
            s.index = pd.to_datetime(s.index)
            index_log_rets = np.log(s.astype(float).replace(0.0, np.nan)).diff()
            index_log_rets.name = "index_log_ret"

        else:
            raise ValueError("index_df must be a DataFrame or Series.")

    else:
        # FINAL fallback: pseudo-index from cross-sectional mean of stock log returns
        index_log_rets = _make_pseudo_index_log_returns(rets_rect)

    index_log_rets = index_log_rets.sort_index()
    if index_log_rets.empty:
        raise ValueError("Index log returns are empty after computation.")

    # -------- align dates --------
    common_dates = rets_rect.index.intersection(index_log_rets.index)
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between stock returns and index returns.")

    rets_rect = rets_rect.loc[common_dates]
    index_log_rets = index_log_rets.loc[common_dates]

    # -------- slice warmup + analysis by trading days --------
    dates_slice, analysis_dates = _slice_warmup_analysis_by_trading_days(
        rets_rect.index, cfg.warmup_years, cfg.analysis_years
    )

    rets_rect = rets_rect.loc[dates_slice]
    index_log_rets = index_log_rets.loc[dates_slice]

    base_info = {
        "dates_slice": dates_slice,
        "analysis_dates": analysis_dates,
        "analysis_start_date": analysis_dates[0],
        "analysis_end_date": analysis_dates[-1],
        "n_dates_slice": int(len(dates_slice)),
        "n_dates_analysis": int(len(analysis_dates)),
        "index_source": "data_io" if index_df is not None else "pseudo_from_stocks",
    }
    return rets_rect, index_log_rets, base_info

# ---------------------------------------------------------------------
# Window generation (endpoints only in analysis period)
# ---------------------------------------------------------------------

def _iter_windows_for_end_dates(
    dates_slice: pd.DatetimeIndex,
    end_dates: pd.DatetimeIndex,
    window_size: int,
) -> List[Dict]:
    """
    Build window descriptors for given end_dates (each must be in dates_slice).
    Returns list of dicts (stable order).
    """
    dates_slice = _safe_to_datetime_index(dates_slice)
    end_dates = _safe_to_datetime_index(end_dates).intersection(dates_slice)

    date_to_idx = {d: i for i, d in enumerate(dates_slice)}
    win_infos: List[Dict] = []
    wid = 0
    for end_date in end_dates:
        end_idx = date_to_idx.get(end_date, None)
        if end_idx is None:
            continue
        start_idx = end_idx - window_size + 1
        if start_idx < 0:
            continue
        win_infos.append(
            {
                "window_id": wid,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_date": dates_slice[start_idx],
                "end_date": dates_slice[end_idx],
            }
        )
        wid += 1
    return win_infos


def _analysis_endpoints(base_info: Dict, stride: int) -> pd.DatetimeIndex:
    """
    Analysis endpoints are analysis_dates subsampled by stride.
    If stride=1 => daily endpoints; stride=5 => ~weekly (51 per year).
    """
    ad = _safe_to_datetime_index(base_info["analysis_dates"])
    stride = int(stride)
    if stride <= 0:
        stride = 1
    return ad[::stride]


# ---------------------------------------------------------------------
# STEP 1: Correlation graphs
# ---------------------------------------------------------------------

def run_step_corr_graphs(out_dir: str, cfg: PipelineConfig) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "correlation_graphs_over_time.csv")

    rets_rect, index_log_rets, base_info = _load_base_data(cfg)
    window_size = _get_window_size_default()
    endpoints = _analysis_endpoints(base_info, cfg.window_stride_days)

    print(f"[corr_graphs] warmup_years={cfg.warmup_years}, analysis_years={cfg.analysis_years}")
    print(f"[corr_graphs] total_days={len(rets_rect)}, assets={rets_rect.shape[1]}")
    print(f"[corr_graphs] endpoints={len(endpoints)} (stride={cfg.window_stride_days}), window_size={window_size}")

    win_infos = _iter_windows_for_end_dates(rets_rect.index, endpoints, window_size)

    rows: List[Dict] = []
    for wi in win_infos:
        R_win = rets_rect.iloc[wi["start_idx"] : wi["end_idx"] + 1, :].dropna(axis=1, how="any")
        node_names = list(R_win.columns)

        G = graphs_correlation.build_correlation_graph(R_win.values, node_names)

        A = np.asarray(G.adjacency).astype(bool)
        W = np.asarray(G.W).astype(float)

        n = int(A.shape[0])
        m = int(A.sum() // 2)
        deg = A.sum(axis=1)
        avg_deg = float(np.mean(deg)) if n > 0 else np.nan
        density = float(m) / (n * (n - 1) / 2.0) if n > 1 else np.nan

        ew = W[A]
        total_weight = float(np.sum(ew)) if ew.size > 0 else 0.0
        avg_weight = float(np.mean(ew)) if ew.size > 0 else np.nan

        rows.append(
            {
                "date": wi["end_date"],
                "window_id": wi["window_id"],
                "window_start": wi["start_date"],
                "window_end": wi["end_date"],
                "n_days": int(R_win.shape[0]),
                "n_assets": int(R_win.shape[1]),
                "corr_num_edges": m,
                "corr_avg_degree": avg_deg,
                "corr_density": density,
                "corr_total_weight": total_weight,
                "corr_avg_weight": avg_weight,
            }
        )

    df_out = pd.DataFrame(rows).sort_values("date")
    df_out.to_csv(out_path, index=False)
    print(f"[corr_graphs] wrote {out_path} rows={len(df_out)}")


# ---------------------------------------------------------------------
# STEP 2: Precision graphs
# ---------------------------------------------------------------------

def run_step_prec_graphs(out_dir: str, cfg: PipelineConfig) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "precision_graphs_over_time.csv")

    rets_rect, index_log_rets, base_info = _load_base_data(cfg)
    window_size = _get_window_size_default()
    endpoints = _analysis_endpoints(base_info, cfg.window_stride_days)

    print(f"[prec_graphs] warmup_years={cfg.warmup_years}, analysis_years={cfg.analysis_years}")
    print(f"[prec_graphs] total_days={len(rets_rect)}, assets={rets_rect.shape[1]}")
    print(f"[prec_graphs] endpoints={len(endpoints)} (stride={cfg.window_stride_days}), window_size={window_size}")

    win_infos = _iter_windows_for_end_dates(rets_rect.index, endpoints, window_size)

    rows: List[Dict] = []
    for wi in win_infos:
        R_win = rets_rect.iloc[wi["start_idx"] : wi["end_idx"] + 1, :].dropna(axis=1, how="any")
        node_names = list(R_win.columns)

        G = graphs_precision.build_precision_graph(R_win.values, node_names)

        A = np.asarray(G.adjacency).astype(bool)
        W = np.asarray(G.W).astype(float)

        n = int(A.shape[0])
        m = int(A.sum() // 2)
        deg = A.sum(axis=1)
        avg_deg = float(np.mean(deg)) if n > 0 else np.nan
        density = float(m) / (n * (n - 1) / 2.0) if n > 1 else np.nan

        ew = W[A]
        total_weight = float(np.sum(ew)) if ew.size > 0 else 0.0
        avg_weight = float(np.mean(ew)) if ew.size > 0 else np.nan

        rows.append(
            {
                "date": wi["end_date"],
                "window_id": wi["window_id"],
                "window_start": wi["start_date"],
                "window_end": wi["end_date"],
                "n_days": int(R_win.shape[0]),
                "n_assets": int(R_win.shape[1]),
                "prec_num_edges": m,
                "prec_avg_degree": avg_deg,
                "prec_density": density,
                "prec_total_weight": total_weight,
                "prec_avg_weight": avg_weight,
            }
        )

    df_out = pd.DataFrame(rows).sort_values("date")
    df_out.to_csv(out_path, index=False)
    print(f"[prec_graphs] wrote {out_path} rows={len(df_out)}")


# ---------------------------------------------------------------------
# STEP 3: Curvature (FRC always, ORC optional)
# ---------------------------------------------------------------------

def _compute_curvature_for_graph(G, compute_orc: bool):
    """
    Tries to compute:
      - FRC always
      - ORC only if compute_orc is True

    We avoid passing flags into curvature.compute_curvature_for_graph because
    earlier you hit: unexpected keyword argument.
    """
    if compute_orc:
        # expect dict-like return OR a structured object used by features.py
        return curvature.compute_curvature_for_graph(G)

    # FRC-only mode: prefer a dedicated FRC function if it exists
    fn_frc = getattr(curvature, "compute_frc_for_graph", None)
    if callable(fn_frc):
        return fn_frc(G)

    # fallback: compute full curvature then ignore ORC downstream
    return curvature.compute_curvature_for_graph(G)


def _drop_orc_keys(d: Dict) -> Dict:
    """Remove ORC-related keys by naming heuristics."""
    out = {}
    for k, v in d.items():
        kl = str(k).lower()
        if ("_k_" in k) or ("orc" in kl) or ("kappa" in kl) or ("ollivier" in kl):
            continue
        out[k] = v
    return out


def _curvature_worker(rets_rect: pd.DataFrame, wi: Dict, cfg: PipelineConfig) -> Dict:
    R_win = rets_rect.iloc[wi["start_idx"] : wi["end_idx"] + 1, :].dropna(axis=1, how="any")
    node_names = list(R_win.columns)

    G_corr = graphs_correlation.build_correlation_graph(R_win.values, node_names)
    G_prec = graphs_precision.build_precision_graph(R_win.values, node_names)

    curv_corr = _compute_curvature_for_graph(G_corr, cfg.compute_orc)
    curv_prec = _compute_curvature_for_graph(G_prec, cfg.compute_orc)

    # Aggregate into scalar features using features.py if available.
    fn = getattr(features, "compute_curvature_features_for_date", None)
    if callable(fn):
        curv_feats = fn(G_corr=G_corr, curv_corr=curv_corr, G_prec=G_prec, curv_prec=curv_prec)
        if (not cfg.compute_orc) and isinstance(curv_feats, dict):
            curv_feats = _drop_orc_keys(curv_feats)
    else:
        # Minimal fallback: if curvature returns dict with summary stats, keep them
        if isinstance(curv_corr, dict) and isinstance(curv_prec, dict):
            curv_feats = {}
            for k, v in _drop_orc_keys(curv_corr).items():
                curv_feats[f"corr_{k}"] = v
            for k, v in _drop_orc_keys(curv_prec).items():
                curv_feats[f"prec_{k}"] = v
        else:
            curv_feats = {}

    row = {
        "date": wi["end_date"],
        "window_id": wi["window_id"],
        "window_start": wi["start_date"],
        "window_end": wi["end_date"],
        "n_days": int(R_win.shape[0]),
        "n_assets": int(R_win.shape[1]),
    }
    if isinstance(curv_feats, dict):
        row.update(curv_feats)
    return row


def run_step_curvature(out_dir: str, cfg: PipelineConfig) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_all = os.path.join(out_dir, "curvature_over_time.csv")
    out_frc = os.path.join(out_dir, "curvature_frc_only.csv")
    out_orc = os.path.join(out_dir, "curvature_orc_only.csv")

    rets_rect, index_log_rets, base_info = _load_base_data(cfg)
    window_size = _get_window_size_default()
    endpoints = _analysis_endpoints(base_info, cfg.window_stride_days)

    print(f"[curvature] compute_orc={cfg.compute_orc}")
    print(f"[curvature] endpoints={len(endpoints)} (stride={cfg.window_stride_days}), window_size={window_size}")
    print(f"[curvature] n_workers={cfg.n_workers}")

    win_infos = _iter_windows_for_end_dates(rets_rect.index, endpoints, window_size)

    if cfg.n_workers != 1 and Parallel is not None:
        rows = Parallel(n_jobs=cfg.n_workers)(
            delayed(_curvature_worker)(rets_rect, wi, cfg) for wi in win_infos
        )
    else:
        rows = [_curvature_worker(rets_rect, wi, cfg) for wi in win_infos]

    df_all = pd.DataFrame(rows).sort_values("date")
    df_all.to_csv(out_all, index=False)
    print(f"[curvature] wrote {out_all} rows={len(df_all)}")

    meta_cols = ["date", "window_id", "window_start", "window_end", "n_days", "n_assets"]
    feature_cols = [c for c in df_all.columns if c not in meta_cols]

    # Use naming heuristics (works with your current conventions)
    frc_cols = [c for c in feature_cols if ("_f_" in c) or ("frc" in c.lower()) or ("forman" in c.lower())]
    orc_cols = [c for c in feature_cols if ("_k_" in c) or ("orc" in c.lower()) or ("kappa" in c.lower())]

    df_frc = df_all[meta_cols + frc_cols] if frc_cols else df_all[meta_cols]
    df_frc.to_csv(out_frc, index=False)
    print(f"[curvature] wrote FRC-only: {out_frc}")

    if cfg.compute_orc and len(orc_cols) > 0:
        df_orc = df_all[meta_cols + orc_cols]
        df_orc.to_csv(out_orc, index=False)
        print(f"[curvature] wrote ORC-only: {out_orc}")
    else:
        print("[curvature] ORC disabled or no ORC columns detected -> skipping curvature_orc_only.csv")


# ---------------------------------------------------------------------
# STEP 4: Features + labels (works with FRC-only or FRC+ORC)
# ---------------------------------------------------------------------

def run_step_features(out_dir: str, cfg: PipelineConfig) -> None:
    os.makedirs(out_dir, exist_ok=True)
    curv_path = os.path.join(out_dir, "curvature_over_time.csv")
    out_path = os.path.join(out_dir, "features_and_labels.csv")

    if not os.path.exists(curv_path):
        raise FileNotFoundError(f"Missing {curv_path}. Run `curvature` step first.")

    df_curv = pd.read_csv(curv_path, parse_dates=["date", "window_start", "window_end"])

    rets_rect, index_log_rets, base_info = _load_base_data(cfg)
    window_size = _get_window_size_default()
    endpoints = _analysis_endpoints(base_info, cfg.window_stride_days)

    print(f"[features] building baselines for windows on analysis_dates={len(endpoints)}")
    win_infos = _iter_windows_for_end_dates(rets_rect.index, endpoints, window_size)

    baseline_rows: List[Dict] = []
    for wi in win_infos:
        R_win = rets_rect.iloc[wi["start_idx"] : wi["end_idx"] + 1, :].dropna(axis=1, how="any")
        node_names = list(R_win.columns)

        # correlation graph for topology features
        G_corr = graphs_correlation.build_correlation_graph(R_win.values, node_names)

        # correlation matrix and index returns window for market baseline
        corr_mat = pd.DataFrame(R_win).corr().to_numpy()
        idx_win = index_log_rets.iloc[wi["start_idx"] : wi["end_idx"] + 1].values

        # Compute baselines (fallbacks if functions are absent)
        market_feats = {}
        topo_feats = {}
        eigen_feats = {}

        fn_mkt = getattr(features, "compute_market_baseline_features", None)
        if callable(fn_mkt):
            market_feats = fn_mkt(corr_matrix=corr_mat, market_returns_window=idx_win)

        fn_top = getattr(features, "compute_topology_baseline_features", None)
        if callable(fn_top):
            topo_feats = fn_top(G_corr)

        fn_eig = getattr(features, "compute_eigenmode_baseline_features", None)
        if callable(fn_eig):
            eigen_feats = fn_eig(corr_mat)

        row = {
            "date": wi["end_date"],
            "window_id": wi["window_id"],
            "window_start": wi["start_date"],
            "window_end": wi["end_date"],
            "n_days": int(R_win.shape[0]),
            "n_assets": int(R_win.shape[1]),
        }
        if isinstance(market_feats, dict):
            row.update(market_feats)
        if isinstance(topo_feats, dict):
            row.update(topo_feats)
        if isinstance(eigen_feats, dict):
            row.update(eigen_feats)

        baseline_rows.append(row)

    df_base = pd.DataFrame(baseline_rows).set_index("date").sort_index()
    df_curv_idx = df_curv.set_index("date").sort_index()

    # Join curvature + baselines on date (curvature may be FRC-only or FRC+ORC)
    df_feats_raw = df_curv_idx.join(df_base, how="inner", rsuffix="_base")

    # Identify numeric feature columns (exclude meta)
    meta_cols = ["window_id", "window_start", "window_end", "n_days", "n_assets"]
    meta_cols = [c for c in meta_cols if c in df_feats_raw.columns]

    numeric_cols = [
        c for c in df_feats_raw.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(df_feats_raw[c])
    ]

    # Add EWMA features
    fn_ewma = getattr(features, "add_ewma_features", None)
    if callable(fn_ewma):
        df_feats = fn_ewma(df_feats_raw, columns=numeric_cols)
    else:
        df_feats = df_feats_raw

    # Build labels ONLY over analysis period and using cfg parameters
    analysis_dates = _safe_to_datetime_index(base_info["analysis_dates"])
    labels_out = labels.build_all_labels(
        index_log_rets,
        analysis_dates=analysis_dates,
        vol_top_pct=cfg.vol_top_pct,
        dd_tail_pcts=cfg.dd_tail_pcts,
        label_horizons=cfg.label_horizons,
    )
    if isinstance(labels_out, tuple) and len(labels_out) >= 1:
        labels_df = labels_out[0]
    else:
        labels_df = labels_out

    # Align features and labels on date
    df_full = df_feats.join(labels_df, how="inner")

    df_full = df_full.reset_index().rename(columns={"index": "date"})
    df_full.to_csv(out_path, index=False)

    print(f"[features] wrote {out_path} shape={df_full.shape}")
    print(f"[features] labels: vol_top_pct={cfg.vol_top_pct}, dd_tail_pcts={cfg.dd_tail_pcts}, horizons={cfg.label_horizons}")


# ---------------------------------------------------------------------
# STEP 5: Models (labels fully parameterized)
# ---------------------------------------------------------------------

def run_step_models(out_dir: str, cfg: "PipelineConfig") -> None:
    """
    Train/evaluate models for all label columns in features_and_labels.csv.

    CHANGE REQUEST (from user):
    ---------------------------
    Instead of splitting chronologically by time, split by *events* so that
    BOTH train and test contain (approximately) the SAME NUMBER OF POSITIVE EVENTS.

    Implementation details:
    - We still keep a time-order within each split (no shuffling of samples),
      but we choose a cutoff date such that cumulative positives up to cutoff
      is ~ half of total positives (for the chosen label).
    - Because each label has different event frequency, the split is computed
      PER LABEL (most robust), producing potentially different split dates
      across labels.
      This avoids impossible constraints when event rates differ.

    Robustness:
    - Drops rows where y is NaN (label horizon tail).
    - Imputes NaN/Inf features with train medians.
    - If y_train becomes single-class (rare with event-balanced split, but can happen
      for extremely sparse labels), falls back to constant predictor.
    - Produces:
        * model_predictions.csv (per date, y_true, p_hat, model, label)
        * model_metrics.csv     (per model x label)

    Notes:
    - For AUC you need both classes in y_true. If test has only one class,
      AUC is left NaN.
    - Some sklearn models expose decision_function; many do not.
      Our evaluation uses predicted probabilities; no decision_function required.
    """
    import os
    import re
    import numpy as np
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)

    feats_path = os.path.join(out_dir, "features_and_labels.csv")
    preds_path = os.path.join(out_dir, "model_predictions.csv")
    metrics_path = os.path.join(out_dir, "model_metrics.csv")

    if not os.path.exists(feats_path):
        raise FileNotFoundError(
            f"Features CSV not found: {feats_path}. Run `features` step first."
        )

    df = pd.read_csv(feats_path, parse_dates=["date"])
    if df.empty:
        raise ValueError("features_and_labels.csv is empty.")
    df = df.sort_values("date").set_index("date")

    # ---------------------------
    # Discover labels based on your naming conventions
    # ---------------------------
    # volatility: y_vol_top{pct*100}_h{h}
    vol_pat = re.compile(r"^y_vol_top(\d+)_h(\d+)$", re.IGNORECASE)
    # drawdown: y_dd_h{h}_thr{thr}
    dd_pat = re.compile(r"^y_dd_h(\d+)_thr(\d+)$", re.IGNORECASE)

    label_info_map = {}
    for c in df.columns:
        m1 = vol_pat.match(c)
        if m1:
            label_info_map[c] = {
                "event_type": "vol",
                "top_pct_int": int(m1.group(1)),
                "horizon_days": int(m1.group(2)),
                "threshold": None,
            }
            continue
        m2 = dd_pat.match(c)
        if m2:
            label_info_map[c] = {
                "event_type": "dd",
                "top_pct_int": None,
                "horizon_days": int(m2.group(1)),
                "threshold": int(m2.group(2)),
            }

    found_labels = list(label_info_map.keys())
    print(f"[models] Discovered labels ({len(found_labels)}): {found_labels}")

    # Use config-driven horizons/thresholds if available
    desired_horizons = set(getattr(cfg, "label_horizons", (5, 10, 20)))
    desired_dd_thresholds = set(getattr(cfg, "dd_tail_pcts", (0.03, 0.05, 0.07)))
    # dd_tail_pcts are floats; your label names use thr3,thr5,thr7 (integers)
    # We'll map 0.03->3, 0.05->5, 0.07->7 if floats are given:
    dd_thr_ints = set()
    for x in desired_dd_thresholds:
        if isinstance(x, float):
            dd_thr_ints.add(int(round(100 * x)))
        else:
            dd_thr_ints.add(int(x))

    # volatility label naming uses top_pct in integer percent (e.g. 3 for top 3%)
    vol_top_pct = getattr(cfg, "vol_top_pct", 0.03)
    vol_top_int = int(round(100 * vol_top_pct))

    label_cols = []
    for c, info in label_info_map.items():
        if info["horizon_days"] not in desired_horizons:
            continue
        if info["event_type"] == "vol":
            if info["top_pct_int"] == vol_top_int:
                label_cols.append(c)
        elif info["event_type"] == "dd":
            if info["threshold"] in dd_thr_ints:
                label_cols.append(c)

    def _sort_key(name: str):
        info = label_info_map[name]
        if info["event_type"] == "vol":
            return (0, info["horizon_days"], -1)
        return (1, info["horizon_days"], info["threshold"])

    label_cols = sorted(label_cols, key=_sort_key)

    if not label_cols:
        raise ValueError(
            "No labels selected. Check that features_and_labels.csv contains "
            "labels matching your configured horizons/thresholds."
        )

    print("[models] Using labels:")
    for c in label_cols:
        info = label_info_map[c]
        print(
            f"  - {c} (event={info['event_type']}, horizon={info['horizon_days']}, thr={info['threshold']}, top={info.get('top_pct_int')})"
        )

    # ---------------------------
    # Feature columns
    # ---------------------------
    meta_cols = ["window_id", "window_start", "window_end", "n_days", "n_assets"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in label_cols]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding meta+label columns.")

    X_all = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    print(f"[models] Total samples: {len(df)}")
    print(f"[models] Features: {len(feature_cols)}")

    # ---------------------------
    # Models
    # ---------------------------
    specs = models.get_default_model_specs()
    if not specs:
        raise ValueError("models.get_default_model_specs() returned empty list.")

    # ---------------------------
    # Utility: event-balanced chronological split
    # ---------------------------
    def _event_balanced_split_dates(y_series: pd.Series) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, dict]:
        """
        Given a label series y indexed by date, choose a cutoff date such that
        train/test have ~equal number of positive events.

        Returns train_dates, test_dates, meta dict.

        Strategy:
        - Drop NaNs in y (horizon tail).
        - Let total_pos = sum(y==1).
        - If total_pos < 2, can't split by events -> fallback to 70/30 time split.
        - Find earliest date where cumulative positives >= ceil(total_pos/2),
          use that as last train date. Everything after is test.
        - Ensure both sets non-empty and contain at least 1 positive if possible.
        """
        y_nonan = y_series.dropna().astype(int)
        dates = y_nonan.index
        n = len(y_nonan)
        if n < 50:
            # too tiny overall: fallback simple
            split_idx = max(int(0.7 * n), 1)
            return dates[:split_idx], dates[split_idx:], {
                "split_mode": "fallback_time_small_n",
                "total_pos": int(y_nonan.sum()),
                "train_pos": int(y_nonan.iloc[:split_idx].sum()),
                "test_pos": int(y_nonan.iloc[split_idx:].sum()),
                "cutoff_date": dates[split_idx - 1] if split_idx > 0 else None,
            }

        total_pos = int(y_nonan.sum())
        if total_pos < 2:
            # Not enough events to split meaningfully; fallback chronological
            split_idx = int(0.7 * n)
            split_idx = min(max(split_idx, 1), n - 1)
            return dates[:split_idx], dates[split_idx:], {
                "split_mode": "fallback_time_few_events",
                "total_pos": total_pos,
                "train_pos": int(y_nonan.iloc[:split_idx].sum()),
                "test_pos": int(y_nonan.iloc[split_idx:].sum()),
                "cutoff_date": dates[split_idx - 1],
            }

        target_train_pos = int(np.ceil(total_pos / 2))
        cpos = y_nonan.cumsum()

        # first index where cum positives >= half
        idx = int(np.searchsorted(cpos.values, target_train_pos, side="left"))
        idx = min(max(idx, 0), n - 2)  # keep at least 1 obs for test
        cutoff_date = dates[idx]

        train_dates = dates[: idx + 1]
        test_dates = dates[idx + 1 :]

        train_pos = int(y_nonan.loc[train_dates].sum())
        test_pos = int(y_nonan.loc[test_dates].sum())

        # If test ended up with 0 positives (possible when events are clustered early),
        # push cutoff earlier until test has at least 1 positive (if possible).
        if test_pos == 0 and total_pos > 0:
            # move cutoff backwards until test has positives or train becomes too small
            j = idx
            while j > 10 and int(y_nonan.iloc[j + 1 :].sum()) == 0:
                j -= 1
            if j != idx:
                cutoff_date = dates[j]
                train_dates = dates[: j + 1]
                test_dates = dates[j + 1 :]
                train_pos = int(y_nonan.loc[train_dates].sum())
                test_pos = int(y_nonan.loc[test_dates].sum())

        return train_dates, test_dates, {
            "split_mode": "event_balanced",
            "total_pos": total_pos,
            "target_train_pos": target_train_pos,
            "train_pos": train_pos,
            "test_pos": test_pos,
            "cutoff_date": cutoff_date,
        }

    # ---------------------------
    # Metrics helper (no decision_function needed)
    # ---------------------------
    def _safe_auc(y_true: np.ndarray, p_hat: np.ndarray) -> float:
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) < 2:
                return np.nan
            return float(roc_auc_score(y_true, p_hat))
        except Exception:
            return np.nan

    def _basic_metrics(y_true: np.ndarray, p_hat: np.ndarray) -> dict:
        eps = 1e-12
        p = np.clip(np.asarray(p_hat, dtype=float), eps, 1 - eps)
        y = np.asarray(y_true, dtype=int)

        y_pred = (p >= 0.5).astype(int)

        acc = float((y_pred == y).mean()) if len(y) else np.nan
        brier = float(np.mean((p - y) ** 2)) if len(y) else np.nan
        logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else np.nan

        tp = int(((y_pred == 1) & (y == 1)).sum())
        fp = int(((y_pred == 1) & (y == 0)).sum())
        fn = int(((y_pred == 0) & (y == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        auc = _safe_auc(y, p)

        return {
            "accuracy": acc,
            "brier": brier,
            "logloss": logloss,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1) if (len(np.unique(y)) >= 2) else np.nan,  # f1 is defined even for one class, but keep consistent
            "auc": auc,
            "error": "",
        }

    def _constant_proba(y_train_l: pd.Series, n_test: int) -> np.ndarray:
        p = float(np.clip(y_train_l.mean(), 0.0, 1.0))
        return np.full(n_test, p, dtype=float)

    pred_frames = []
    metrics_rows = []
    skip_summary = []

    # ---------------------------
    # Train/eval loop (PER LABEL event-balanced split)
    # ---------------------------
    for label_name in label_cols:
        info = label_info_map[label_name]
        y_raw = df[label_name]

        # Determine split dates based on events
        train_dates, test_dates, split_meta = _event_balanced_split_dates(y_raw)

        # Build X/y for those dates (drop NaN labels)
        y_nonan = y_raw.dropna().astype(int)
        train_dates = pd.DatetimeIndex([d for d in train_dates if d in y_nonan.index])
        test_dates = pd.DatetimeIndex([d for d in test_dates if d in y_nonan.index])

        if len(train_dates) == 0 or len(test_dates) == 0:
            skip_summary.append((label_name, "empty_train_or_test_after_dropna"))
            continue

        X_train = X_all.loc[train_dates].copy()
        X_test = X_all.loc[test_dates].copy()
        y_train = y_nonan.loc[train_dates]
        y_test = y_nonan.loc[test_dates]

        # Impute missing features using train medians (per label split)
        med = X_train.median(axis=0, numeric_only=True)
        X_train = X_train.fillna(med)
        X_test = X_test.fillna(med)

        # Sanity: still may contain NaNs if a column is entirely NaN in train
        X_train = X_train.fillna(0.0)
        X_test = X_test.fillna(0.0)

        y_train_unique = sorted(set(y_train.unique().tolist()))
        y_test_unique = sorted(set(y_test.unique().tolist()))

        if len(X_train) < 50 or len(X_test) < 30:
            skip_summary.append((label_name, f"too_few_samples train={len(X_train)} test={len(X_test)}"))
            continue

        train_single_class = (len(y_train_unique) < 2)

        print(
            f"[models] Label={label_name} | split={split_meta.get('split_mode')} "
            f"| total_pos={split_meta.get('total_pos')} "
            f"| train_pos={split_meta.get('train_pos')} test_pos={split_meta.get('test_pos')} "
            f"| train_n={len(X_train)} test_n={len(X_test)} "
            f"| cutoff={split_meta.get('cutoff_date')}"
        )

        for spec in specs:
            fit_status = "ok"
            eval_result = {}
            p_test = None

            if train_single_class:
                # constant fallback
                p_test = _constant_proba(y_train, len(X_test))
                eval_result = _basic_metrics(y_test.values, p_test)
                fit_status = "constant_fallback_single_class_train"
            else:
                try:
                    fitted = models.fit_model(spec, X_train, y_train)
                    p_test = models.predict_proba(fitted, X_test)

                    # Prefer package evaluation if it works; otherwise fallback
                    try:
                        eval_result = models.evaluate_model(fitted, X_test, y_test)
                        if "error" not in eval_result:
                            eval_result["error"] = ""
                        # Ensure auc exists (some evaluate_model versions omit it)
                        if "auc" not in eval_result or eval_result.get("auc") is None:
                            eval_result["auc"] = _safe_auc(y_test.values, np.asarray(p_test))
                    except Exception as e_eval:
                        eval_result = _basic_metrics(y_test.values, p_test)
                        eval_result["error"] = f"evaluate_failed: {str(e_eval)}"

                except Exception as e_fit:
                    p_test = _constant_proba(y_train, len(X_test))
                    eval_result = _basic_metrics(y_test.values, p_test)
                    fit_status = "constant_fallback_fit_failed"
                    eval_result["error"] = f"fit_failed: {str(e_fit)}"

            # Metrics row
            mrow = dict(eval_result)
            mrow.update(
                {
                    "fit_status": fit_status,
                    "model_name": spec.name,
                    "label_name": label_name,
                    "event_type": info["event_type"],
                    "horizon_days": info["horizon_days"],
                    "threshold": info["threshold"],
                    "vol_top_pct_int": info.get("top_pct_int"),
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                    "y_train_unique": ",".join(map(str, y_train_unique)),
                    "y_test_unique": ",".join(map(str, y_test_unique)),
                    "split_mode": split_meta.get("split_mode"),
                    "split_cutoff_date": split_meta.get("cutoff_date"),
                    "total_pos": split_meta.get("total_pos"),
                    "train_pos": split_meta.get("train_pos"),
                    "test_pos": split_meta.get("test_pos"),
                }
            )
            metrics_rows.append(mrow)

            # Predictions rows
            df_pred = pd.DataFrame(
                {
                    "date": X_test.index,
                    "model_name": spec.name,
                    "label_name": label_name,
                    "event_type": info["event_type"],
                    "horizon_days": info["horizon_days"],
                    "threshold": info["threshold"],
                    "vol_top_pct_int": info.get("top_pct_int"),
                    "y_true": y_test.values.astype(int),
                    "p_hat": np.asarray(p_test, dtype=float),
                    "fit_status": fit_status,
                    "split_mode": split_meta.get("split_mode"),
                    "split_cutoff_date": split_meta.get("cutoff_date"),
                }
            )
            pred_frames.append(df_pred)

    if not pred_frames:
        msg = (
            "No predictions were produced.\n"
            f"skip_summary(first 20)={skip_summary[:20]}\n"
            "Try: verify labels are not mostly NaN/constant; ensure enough samples; "
            "or relax the hard minimums (train>=50/test>=30) inside run_step_models.\n"
        )
        raise ValueError(msg)

    preds_all = (
        pd.concat(pred_frames, axis=0)
        .sort_values(["event_type", "horizon_days", "threshold", "label_name", "model_name", "date"])
        .reset_index(drop=True)
    )
    metrics_all = (
        pd.DataFrame(metrics_rows)
        .sort_values(["event_type", "horizon_days", "threshold", "label_name", "model_name"])
        .reset_index(drop=True)
    )

    preds_all.to_csv(preds_path, index=False)
    metrics_all.to_csv(metrics_path, index=False)

    print(f"[models] Wrote predictions to {preds_path} (rows={len(preds_all)})")
    print(f"[models] Wrote metrics to {metrics_path} (rows={len(metrics_all)})")

# ---------------------------------------------------------------------
# STEP 6: Mechanistic probes (ALL predicted dates)
# ---------------------------------------------------------------------

def _probe_worker(rets_rect: pd.DataFrame, date: pd.Timestamp, window_size: int) -> Dict:
    dates_all = _safe_to_datetime_index(rets_rect.index)
    if date not in dates_all:
        return {}

    end_idx = int(dates_all.get_loc(date))
    start_idx = end_idx - window_size + 1
    if start_idx < 0:
        return {}

    R_win = rets_rect.iloc[start_idx : end_idx + 1, :].dropna(axis=1, how="any")
    node_names = list(R_win.columns)

    G_corr = graphs_correlation.build_correlation_graph(R_win.values, node_names)
    G_prec = graphs_precision.build_precision_graph(R_win.values, node_names)

    mech_corr = probes.mechanistic_probes_for_graph(G_corr, prefix="corr_")
    mech_prec = probes.mechanistic_probes_for_graph(G_prec, prefix="prec_")

    row = {
        "date": date,
        "window_start": dates_all[start_idx],
        "window_end": date,
        "n_days": int(R_win.shape[0]),
        "n_assets": int(R_win.shape[1]),
    }
    if isinstance(mech_corr, dict):
        row.update(mech_corr)
    if isinstance(mech_prec, dict):
        row.update(mech_prec)
    return row


def run_step_mech_probes(out_dir: str, cfg: PipelineConfig) -> None:
    os.makedirs(out_dir, exist_ok=True)

    preds_path = os.path.join(out_dir, "model_predictions.csv")
    out_path = os.path.join(out_dir, "mechanistic_probes.csv")

    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Missing {preds_path}. Run `models` step first.")

    preds = pd.read_csv(preds_path, parse_dates=["date"])
    if preds.empty:
        raise ValueError("model_predictions.csv is empty.")

    unique_dates = sorted(pd.to_datetime(preds["date"]).dropna().unique())
    print(f"[mech_probes] computing probes for ALL predicted dates: {len(unique_dates)}")
    print(f"[mech_probes] n_workers={cfg.n_workers}")

    rets_rect, index_log_rets, base_info = _load_base_data(cfg)
    window_size = _get_window_size_default()

    if cfg.n_workers != 1 and Parallel is not None:
        rows = Parallel(n_jobs=cfg.n_workers)(
            delayed(_probe_worker)(rets_rect, pd.Timestamp(d), window_size) for d in unique_dates
        )
    else:
        rows = [_probe_worker(rets_rect, pd.Timestamp(d), window_size) for d in unique_dates]

    rows = [r for r in rows if r]
    df_out = pd.DataFrame(rows).sort_values("date")
    df_out.to_csv(out_path, index=False)
    print(f"[mech_probes] wrote {out_path} rows={len(df_out)}")


# ---------------------------------------------------------------------
# MAIN CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rolling-window pipeline for Ricci EWS project.")
    parser.add_argument(
        "step",
        choices=["corr_graphs", "prec_graphs", "curvature", "features", "models", "mech_probes"],
        help="Which pipeline step to run.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Directory to write CSV outputs (default: %(default)s)",
    )

    args = parser.parse_args()
    cfg = get_pipeline_config()

    if args.step == "corr_graphs":
        run_step_corr_graphs(args.out_dir, cfg)
    elif args.step == "prec_graphs":
        run_step_prec_graphs(args.out_dir, cfg)
    elif args.step == "curvature":
        run_step_curvature(args.out_dir, cfg)
    elif args.step == "features":
        run_step_features(args.out_dir, cfg)
    elif args.step == "models":
        run_step_models(args.out_dir, cfg)
    elif args.step == "mech_probes":
        run_step_mech_probes(args.out_dir, cfg)
    else:
        raise ValueError(f"Unknown step: {args.step}")


if __name__ == "__main__":
    main()