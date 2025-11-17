"""
Test precision-based graph over multiple rolling windows.

Run from project root:
    python -m scripts.test_precision_graph_over_time

This script:
- loads S&P 500 data
- builds the universe & rectangular return panel
- iterates over multiple rolling windows
- for each window, constructs the precision-based graph (graphical lasso)
- records degree and weight statistics over time
- saves metrics to 'precision_graph_metrics_over_time.csv'
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
)
from ricci_ews import graphs_precision


def compute_graph_metrics(G: graphs_precision.GraphData) -> dict:
    """
    Compute simple diagnostics for a graph: degrees and weight distribution.
    """
    W = G.W
    A = G.adjacency

    n = W.shape[0]
    num_edges = int(A.sum() // 2)  # undirected
    degrees = A.sum(axis=1)

    avg_degree = float(degrees.mean())
    min_degree = int(degrees.min())
    max_degree = int(degrees.max())

    # Non-zero weights only
    w_vals = W[A]
    if w_vals.size == 0:
        weight_mean = np.nan
        weight_median = np.nan
        w_q10 = np.nan
        w_q90 = np.nan
    else:
        weight_mean = float(w_vals.mean())
        weight_median = float(np.median(w_vals))
        w_q10 = float(np.quantile(w_vals, 0.10))
        w_q90 = float(np.quantile(w_vals, 0.90))

    return {
        "num_nodes": n,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "weight_mean": weight_mean,
        "weight_median": weight_median,
        "weight_q10": w_q10,
        "weight_q90": w_q90,
    }


def main():
    # ------------------------------------------------------------------
    # 0. Config for this test
    # ------------------------------------------------------------------
    # Max number of windows to evaluate (to keep runtime reasonable)
    MAX_WINDOWS = 30
    # Optionally, sample every k-th window instead of every single one:
    WINDOW_STRIDE = 1  # set to 5 or 10 if runtime too long

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    prices_tidy = data_io.load_prices()
    index_series = data_io.load_index_series()
    print(f"Loaded {len(prices_tidy)} rows of stock prices.")
    print(f"Index series length: {len(index_series)} rows.")

    # ------------------------------------------------------------------
    # 2. Price panel & universe
    # ------------------------------------------------------------------
    print("Building price panel...")
    price_panel = data_io.build_price_panel(prices_tidy)
    print(f"Price panel shape: {price_panel.shape} (dates x tickers)")

    print("Selecting universe...")
    U = universe.select_universe(price_panel)
    print(f"Universe size: {len(U)} tickers")

    # ------------------------------------------------------------------
    # 3. Returns & rectangular panel
    # ------------------------------------------------------------------
    print("Computing log returns...")
    rets = returns.compute_log_returns(price_panel)

    print("Winsorizing returns...")
    rets_wins = returns.winsorize_returns(rets)

    print("Rectangularizing returns for universe U...")
    rets_rect = returns.align_and_rectangularize(rets_wins, U)
    print(f"Rectangular return panel shape: {rets_rect.shape}")

    # ------------------------------------------------------------------
    # 4. Rolling windows
    # ------------------------------------------------------------------
    print(f"Generating rolling windows with W = {config.WINDOW_SIZE}...")
    rw_gen = windows.RollingWindowGenerator(
        rets_rect,
        window_size=config.WINDOW_SIZE,
    )

    metrics_list = []
    node_names = list(rets_rect.columns)

    print(f"\nIterating over up to {MAX_WINDOWS} windows (stride = {WINDOW_STRIDE})...")

    for idx, rw in enumerate(rw_gen.iter_windows()):
        # Implement stride (e.g., use every k-th window)
        if idx % WINDOW_STRIDE != 0:
            continue
        if len(metrics_list) >= MAX_WINDOWS:
            break

        print(
            f"Window {idx}: {rw.start_date.date()} → {rw.end_date.date()}, "
            f"shape: {rw.data.shape} (W x N)"
        )

        # Build precision-based graph
        G = graphs_precision.build_precision_graph(
            R_window=rw.data,
            node_names=node_names,
            lam=config.GLASSO_LAMBDA,
            beta=config.CORR_BETA,
            target_avg_degree=config.TARGET_AVG_DEGREE,
            gamma=config.LENGTH_EXPONENT,
            eps=config.LENGTH_EPS,
        )

        # Compute metrics
        m = compute_graph_metrics(G)
        m["window_index"] = idx
        m["start_date"] = rw.start_date
        m["end_date"] = rw.end_date
        m["lambda"] = config.GLASSO_LAMBDA
        metrics_list.append(m)

    if not metrics_list:
        raise RuntimeError(
            "No windows were processed. Possibly not enough data to form "
            f"a single window of length {config.WINDOW_SIZE}."
        )

    metrics_df = pd.DataFrame(metrics_list).set_index("end_date").sort_index()

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print("\n=== Summary over processed windows (precision graph) ===")
    print(f"Number of windows processed: {len(metrics_df)}")
    print("Columns:", list(metrics_df.columns))

    print("\nHead of metrics:")
    print(metrics_df.head())

    print("\nTail of metrics:")
    print(metrics_df.tail())

    print("\nBasic stats for avg_degree and weight_mean:")
    print(metrics_df[["avg_degree", "weight_mean"]].describe())

    # Save to CSV for plotting later
    out_path = "precision_graph_metrics_over_time.csv"
    metrics_df.to_csv(out_path)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
