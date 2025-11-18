"""
test_curvature_over_time.py

Compute Ollivier–Ricci (ORC) and Forman–Ricci (FRC) curvature over multiple
rolling windows, for BOTH:
    - correlation-based graphs
    - precision-based (graphical lasso) graphs

Run from project root:

    python -m scripts.test_curvature_over_time

Outputs:
    curvature_metrics_over_time.csv

Each row = one rolling window.
Columns include summary stats for:
    - corr_edge_orc_*   (edge-level ORC on correlation graph)
    - corr_edge_frc_*   (edge-level FRC on correlation graph)
    - corr_node_orc_*   (node-level ORC on correlation graph)
    - corr_node_frc_*   (node-level FRC on correlation graph)
    - prec_edge_orc_*, prec_edge_frc_*, prec_node_orc_*, prec_node_frc_*
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ricci_ews import (
    config,
    data_io,
    universe,
    returns,
    windows,
    graphs_correlation,
    graphs_precision,
    curvature,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _stats_1d(x: np.ndarray) -> dict:
    """Return simple stats (mean, median, q10, q90) for a 1D array."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "q10": np.nan,
            "q90": np.nan,
        }
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q90": float(np.quantile(x, 0.90)),
    }


def summarize_curvature_for_graph(
    G: graphs_correlation.GraphData,
    curv: curvature.CurvatureResult,
    prefix: str,
) -> dict:
    """
    Summarize curvature for one graph (edge + node, ORC + FRC).

    Parameters
    ----------
    G : GraphData
        Graph structure.
    curv : CurvatureResult
        Output of curvature.compute_curvature_for_graph for this graph.
    prefix : str
        Prefix to add to column names (e.g. 'corr_' or 'prec_').

    Returns
    -------
    metrics : dict
        Flat dict of summary statistics with prefixed keys.
    """
    A = G.adjacency.astype(bool)

    edge_orc_vals = curv.edge_orc[A]
    edge_frc_vals = curv.edge_frc[A]
    node_orc_vals = curv.node_orc
    node_frc_vals = curv.node_frc

    eo = _stats_1d(edge_orc_vals)
    ef = _stats_1d(edge_frc_vals)
    no = _stats_1d(node_orc_vals)
    nf = _stats_1d(node_frc_vals)

    out = {}

    # Edge ORC
    out[f"{prefix}edge_orc_mean"] = eo["mean"]
    out[f"{prefix}edge_orc_median"] = eo["median"]
    out[f"{prefix}edge_orc_q10"] = eo["q10"]
    out[f"{prefix}edge_orc_q90"] = eo["q90"]

    # Edge FRC
    out[f"{prefix}edge_frc_mean"] = ef["mean"]
    out[f"{prefix}edge_frc_median"] = ef["median"]
    out[f"{prefix}edge_frc_q10"] = ef["q10"]
    out[f"{prefix}edge_frc_q90"] = ef["q90"]

    # Node ORC
    out[f"{prefix}node_orc_mean"] = no["mean"]
    out[f"{prefix}node_orc_median"] = no["median"]
    out[f"{prefix}node_orc_q10"] = no["q10"]
    out[f"{prefix}node_orc_q90"] = no["q90"]

    # Node FRC
    out[f"{prefix}node_frc_mean"] = nf["mean"]
    out[f"{prefix}node_frc_median"] = nf["median"]
    out[f"{prefix}node_frc_q10"] = nf["q10"]
    out[f"{prefix}node_frc_q90"] = nf["q90"]

    return out


# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------


def main():
    # -----------------------------
    # 0. Config for this test
    # -----------------------------
    MAX_WINDOWS = 5     # max number of windows to process
    WINDOW_STRIDE = 1    # use every k-th window to keep runtime manageable

    # -----------------------------
    # 1. Load data
    # -----------------------------
    print("Loading data...")
    prices_tidy = data_io.load_prices()
    index_series = data_io.load_index_series()
    print(f"Loaded {len(prices_tidy)} rows of stock prices.")
    print(f"Index series length: {len(index_series)} rows.")

    # -----------------------------
    # 2. Price panel & universe
    # -----------------------------
    print("Building price panel...")
    price_panel = data_io.build_price_panel(prices_tidy)
    print(f"Price panel shape: {price_panel.shape} (dates x tickers)")

    print("Selecting universe...")
    U = universe.select_universe(price_panel)
    print(f"Universe size: {len(U)} tickers")

    # -----------------------------
    # 3. Returns & rectangular panel
    # -----------------------------
    print("Computing log returns...")
    rets = returns.compute_log_returns(price_panel)

    print("Winsorizing returns...")
    rets_wins = returns.winsorize_returns(rets)

    print("Rectangularizing returns for universe U...")
    rets_rect = returns.align_and_rectangularize(rets_wins, U)
    print(f"Rectangular return panel shape: {rets_rect.shape}")

    node_names = list(rets_rect.columns)
    n_nodes = len(node_names)
    print(f"Curvature will be computed on graphs with {n_nodes} nodes.")

    # -----------------------------
    # 4. Rolling windows
    # -----------------------------
    print(f"Generating rolling windows with W = {config.WINDOW_SIZE}...")
    rw_gen = windows.RollingWindowGenerator(
        rets_rect,
        window_size=config.WINDOW_SIZE,
    )

    rows = []

    print(
        f"\nIterating over up to {MAX_WINDOWS} windows "
        f"(stride = {WINDOW_STRIDE})..."
    )

    for idx, rw in enumerate(rw_gen.iter_windows()):
        if idx % WINDOW_STRIDE != 0:
            continue
        if len(rows) >= MAX_WINDOWS:
            break

        print(
            f"\nWindow {idx}: {rw.start_date.date()} → {rw.end_date.date()}, "
            f"shape: {rw.data.shape} (W x N)"
        )

        row = {
            "window_index": idx,
            "start_date": rw.start_date,
            "end_date": rw.end_date,
            "num_nodes": n_nodes,
        }

        # --- 4.1 Correlation-based graph + curvature ---
        try:
            G_corr = graphs_correlation.build_correlation_graph(
                R_window=rw.data,
                node_names=node_names,
                beta=config.CORR_BETA,
                target_avg_degree=config.TARGET_AVG_DEGREE,
                gamma=config.LENGTH_EXPONENT,
                eps=config.LENGTH_EPS,
                shrinkage_target=config.SHRINKAGE_TARGET,
            )
            curv_corr = curvature.compute_curvature_for_graph(G_corr)
            row.update(
                summarize_curvature_for_graph(G_corr, curv_corr, prefix="corr_")
            )
            row["corr_success"] = True
        except Exception as e:
            print(f"  [WARN] Failed to compute correlation-graph curvature: {e}")
            row["corr_success"] = False

        # --- 4.2 Precision-based graph + curvature ---
        try:
            G_prec = graphs_precision.build_precision_graph(
                R_window=rw.data,
                node_names=node_names,
                lam=config.GLASSO_LAMBDA,
                beta=config.CORR_BETA,
                target_avg_degree=config.TARGET_AVG_DEGREE,
                gamma=config.LENGTH_EXPONENT,
                eps=config.LENGTH_EPS,
            )
            curv_prec = curvature.compute_curvature_for_graph(G_prec)
            row.update(
                summarize_curvature_for_graph(G_prec, curv_prec, prefix="prec_")
            )
            row["prec_success"] = True
        except Exception as e:
            print(f"  [WARN] Failed to compute precision-graph curvature: {e}")
            row["prec_success"] = False

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No windows were processed. Possibly not enough data to form "
            f"a single window of length {config.WINDOW_SIZE}."
        )

    metrics_df = pd.DataFrame(rows).set_index("end_date").sort_index()

    # -----------------------------
    # 5. Summary + save
    # -----------------------------
    print("\n=== Curvature summary over processed windows ===")
    print(f"Number of windows processed: {len(metrics_df)}")
    print("Columns:", list(metrics_df.columns))

    print("\nHead of curvature metrics:")
    print(metrics_df.head())

    print("\nBasic stats (example: corr_edge_orc_mean, prec_edge_orc_mean):")
    cols_to_show = [
        c for c in metrics_df.columns
        if c in ("corr_edge_orc_mean", "prec_edge_orc_mean")
    ]
    if cols_to_show:
        print(metrics_df[cols_to_show].describe())
    else:
        print("No curvature columns found (all failures?).")

    out_path = "curvature_metrics_over_time.csv"
    metrics_df.to_csv(out_path)
    print(f"\nSaved curvature metrics to {out_path}")


if __name__ == "__main__":
    main()
