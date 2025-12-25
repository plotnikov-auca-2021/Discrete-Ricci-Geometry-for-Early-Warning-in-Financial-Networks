"""
build_features_over_time.py

Construct the full feature vector over time,
without labels, as described in Section 3.8 of the proposal.

For each rolling window [t-W+1, t]:

  - Build correlation-based graph (ρ) and precision-based graph (Θ)
  - Compute Ollivier–Ricci (ORC) and Forman–Ricci (FRC) curvature
    for both graph families
  - Build curvature feature vector x_t from edge- and vertex-level
    aggregates for each graph family (ρ, Θ)
  - Build non-curvature baselines z_t:
      * market_abs_corr_mean, market_rv_short, market_ret_recent
      * topo_avg_degree, topo_weighted_clustering, topo_algebraic_conn,
        topo_spectral_radius (on correlation graph)
      * eigen_largest_corr (largest correlation eigenvalue)
  - Assemble φ_t and collect over time

Outputs:

  - features_raw_over_time.csv
      φ_t without any EWMA transforms.

  - features_with_ewma_over_time.csv
      φ_t plus EWMA transforms of curvature features x_t with
      half-lives H ∈ {5, 10, 20}.
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
    features,
)


def main():
    # --------------------------------------------------------------
    # 0. Config for this script
    # --------------------------------------------------------------
    # Limit the number of windows for runtime reasons.
    # Set to None to use all windows (may be slow due to ORC).
    MAX_WINDOWS = 5
    # Use every k-th window (stride > 1 reduces runtime).
    WINDOW_STRIDE = 1

    # Short-horizon (in days) for market RV and recent return baseline
    H_SHORT = 5

    # --------------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------------
    print("Loading stock and index data...")
    prices_tidy = data_io.load_prices()         # tidy stock prices
    index_series = data_io.load_index_series()  # S&P 500 index prices

    print(f"Loaded {len(prices_tidy)} rows of stock prices.")
    print(f"Index series length: {len(index_series)} rows.")

    # Ensure index_series is a Series of prices, indexed by date
    index_prices = getattr(index_series, "squeeze", lambda: index_series)()
    index_prices = index_prices.sort_index()
    # Log returns for the index
    index_log_rets = np.log(index_prices).diff().dropna()

    # --------------------------------------------------------------
    # 2. Build price panel & universe
    # --------------------------------------------------------------
    print("Building price panel...")
    price_panel = data_io.build_price_panel(prices_tidy)
    print(f"Price panel shape: {price_panel.shape} (dates x tickers)")

    print("Selecting universe...")
    U = universe.select_universe(price_panel)
    print(f"Universe size: {len(U)} tickers")

    # --------------------------------------------------------------
    # 3. Returns & rectangular panel
    # --------------------------------------------------------------
    print("Computing log returns for stocks...")
    rets = returns.compute_log_returns(price_panel)

    print("Winsorizing returns...")
    rets_wins = returns.winsorize_returns(rets)

    print("Rectangularizing returns for universe U...")
    rets_rect = returns.align_and_rectangularize(rets_wins, U)
    print(f"Rectangular return panel shape: {rets_rect.shape}")

    node_names = list(rets_rect.columns)
    n_nodes = len(node_names)
    print(f"Graphs will have {n_nodes} nodes.")

    # --------------------------------------------------------------
    # 4. Rolling windows
    # --------------------------------------------------------------
    print(f"Generating rolling windows with W = {config.WINDOW_SIZE}...")
    rw_gen = windows.RollingWindowGenerator(
        rets_rect,
        window_size=config.WINDOW_SIZE,
    )

    rows = []

    print(
        f"\nIterating over rolling windows (stride = {WINDOW_STRIDE}, "
        f"max windows = {MAX_WINDOWS or 'ALL'})..."
    )

    for idx, rw in enumerate(rw_gen.iter_windows()):
        # Apply stride
        if idx % WINDOW_STRIDE != 0:
            continue
        if (MAX_WINDOWS is not None) and (len(rows) >= MAX_WINDOWS):
            break

        print(
            f"\nWindow {idx}: {rw.start_date.date()} → {rw.end_date.date()}, "
            f"shape: {rw.data.shape} (W x N)"
        )

        R_window = np.asarray(rw.data, dtype=float)  # (W, N)

        # Correlation matrix for this window (for baseline features)
        # We use plain sample correlation here; graph construction uses
        # shrinkage internally, which is fine.
        corr_matrix = np.corrcoef(R_window, rowvar=False)

        # Market (index) returns window: align by date range
        mask_idx = (index_log_rets.index >= rw.start_date) & (
            index_log_rets.index <= rw.end_date
        )
        market_returns_window = index_log_rets.loc[mask_idx].values

        # ----------------------------------------------------------
        # 4.1 Build graphs and curvature
        # ----------------------------------------------------------
        try:
            # Correlation-based graph ρ
            G_corr = graphs_correlation.build_correlation_graph(
                R_window=R_window,
                node_names=node_names,
                beta=config.CORR_BETA,
                target_avg_degree=config.TARGET_AVG_DEGREE,
                gamma=config.LENGTH_EXPONENT,
                eps=config.LENGTH_EPS,
                shrinkage_target=config.SHRINKAGE_TARGET,
            )
            curv_corr = curvature.compute_curvature_for_graph(G_corr)

            # Precision-based graph Θ
            G_prec = graphs_precision.build_precision_graph(
                R_window=R_window,
                node_names=node_names,
                lam=config.GLASSO_LAMBDA,
                beta=config.CORR_BETA,
                target_avg_degree=config.TARGET_AVG_DEGREE,
                gamma=config.LENGTH_EXPONENT,
                eps=config.LENGTH_EPS,
            )
            curv_prec = curvature.compute_curvature_for_graph(G_prec)

        except Exception as e:
            print(f"  [WARN] Failed to build graphs/curvature for window {idx}: {e}")
            # Skip this window entirely
            continue

        # ----------------------------------------------------------
        # 4.2 Curvature feature vector x_t (both graph families)
        # ----------------------------------------------------------
        x_t = features.compute_curvature_features_for_date(
            G_corr=G_corr,
            curv_corr=curv_corr,
            G_prec=G_prec,
            curv_prec=curv_prec,
        )

        # ----------------------------------------------------------
        # 4.3 Baseline feature vector z_t
        # ----------------------------------------------------------
        # Market-only baseline
        z_market = features.compute_market_baseline_features(
            corr_matrix=corr_matrix,
            market_returns_window=market_returns_window,
            h_short=H_SHORT,
        )

        # Topology-only baseline (using correlation graph)
        z_topo = features.compute_topology_baseline_features(G_corr)

        # Eigenmode baseline (largest correlation eigenvalue)
        z_eigen = features.compute_eigenmode_baseline_features(corr_matrix)

        # Assemble baselines
        z_t = {}
        z_t.update(z_market)
        z_t.update(z_topo)
        z_t.update(z_eigen)

        # ----------------------------------------------------------
        # 4.4 Full feature vector φ_t = [x_t^⊤, z_t^⊤]^⊤
        # ----------------------------------------------------------
        phi_t = features.assemble_feature_vector(x_t, z_t)
        phi_t["window_index"] = idx
        phi_t["start_date"] = rw.start_date
        phi_t["end_date"] = rw.end_date
        phi_t["num_nodes"] = n_nodes

        rows.append(phi_t)

    if not rows:
        raise RuntimeError(
            "No windows were processed. Possibly not enough data or all windows failed."
        )

    # --------------------------------------------------------------
    # 5. Build DataFrame over time
    # --------------------------------------------------------------
    df_raw = pd.DataFrame(rows)
    # Use end_date as index (t corresponds to window end)
    df_raw = df_raw.set_index("end_date").sort_index()

    print("\n=== Raw feature matrix φ_t (no EWMA) ===")
    print(f"Shape: {df_raw.shape}")
    print("Columns (first 20):", list(df_raw.columns[:20]))

    # Save raw features
    raw_out_path = "features_raw_over_time.csv"
    df_raw.to_csv(raw_out_path)
    print(f"\nSaved raw features (no EWMA) to {raw_out_path}")

    # --------------------------------------------------------------
    # 6. Add EWMA features for curvature components only
    # --------------------------------------------------------------
    # Curvature feature columns are those starting with 'corr_' or 'prec_'
    curvature_cols = [
        c for c in df_raw.columns
        if c.startswith("corr_") or c.startswith("prec_")
    ]

    print(f"\nNumber of curvature feature columns for EWMA: {len(curvature_cols)}")

    df_with_ewma = features.add_ewma_features(
        df_raw,
        columns=curvature_cols,
        half_lives=None,  # use default (5,10,20) from config or features
    )

    print("\n=== Feature matrix with EWMA ===")
    print(f"Shape: {df_with_ewma.shape}")
    print("Columns (first 20):", list(df_with_ewma.columns[:20]))

    ewma_out_path = "features_with_ewma_over_time.csv"
    df_with_ewma.to_csv(ewma_out_path)
    print(f"\nSaved features with EWMA to {ewma_out_path}")


if __name__ == "__main__":
    main()
