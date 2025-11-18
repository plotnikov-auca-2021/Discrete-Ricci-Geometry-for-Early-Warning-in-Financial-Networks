"""
Test of build_correlation_graph on a single rolling window.

Run from project root:
    python -m scripts.test_correlation_graph
"""

from ricci_ews import (
    config,
    data_io,
    universe,
    returns,
    windows,
)
from ricci_ews import graphs_correlation


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    prices_tidy = data_io.load_prices()          # sp500_stocks.csv
    index_series = data_io.load_index_series()   # sp500_index.csv (not used here but good sanity check)

    print(f"Loaded {len(prices_tidy)} rows of stock prices.")
    print(f"Index series length: {len(index_series)} rows.")

    # ------------------------------------------------------------------
    # 2. Build price panel & universe
    # ------------------------------------------------------------------
    print("Building price panel...")
    price_panel = data_io.build_price_panel(prices_tidy)
    print(f"Price panel shape: {price_panel.shape} (dates x tickers)")

    print("Selecting universe...")
    U = universe.select_universe(price_panel)
    print(f"Universe size: {len(U)} tickers")

    # ------------------------------------------------------------------
    # 3. Compute returns & clean
    # ------------------------------------------------------------------
    print("Computing log returns...")
    rets = returns.compute_log_returns(price_panel)

    print("Winsorizing returns...")
    rets_wins = returns.winsorize_returns(rets)

    print("Rectangularizing returns for universe U...")
    rets_rect = returns.align_and_rectangularize(rets_wins, U)
    print(f"Rectangular return panel shape: {rets_rect.shape}")

    # ------------------------------------------------------------------
    # 4. Generate first rolling window
    # ------------------------------------------------------------------
    print(f"Generating rolling windows with W = {config.WINDOW_SIZE}...")
    rw_gen = windows.RollingWindowGenerator(
        rets_rect,
        window_size=config.WINDOW_SIZE,
    )

    try:
        first_window = next(rw_gen.iter_windows())
    except StopIteration:
        raise RuntimeError(
            "Not enough data to form a single rolling window. "
            f"Need at least {config.WINDOW_SIZE} days in the rectangular panel."
        )

    print(
        f"First window from {first_window.start_date.date()} "
        f"to {first_window.end_date.date()}, "
        f"shape: {first_window.data.shape} (W x N)"
    )

    # ------------------------------------------------------------------
    # 5. Build correlation-based graph on this window
    # ------------------------------------------------------------------
    node_names = list(rets_rect.columns)
    print("Building correlation-based graph on first window...")

    G = graphs_correlation.build_correlation_graph(
        R_window=first_window.data,
        node_names=node_names,
        beta=config.CORR_BETA,
        target_avg_degree=config.TARGET_AVG_DEGREE,
        gamma=config.LENGTH_EXPONENT,
        eps=config.LENGTH_EPS,
        shrinkage_target=config.SHRINKAGE_TARGET,
    )

    # ------------------------------------------------------------------
    # 6. Print diagnostics
    # ------------------------------------------------------------------
    W = G.W
    L = G.L
    A = G.adjacency

    n = W.shape[0]
    num_edges = A.sum() // 2  # undirected

    degrees = A.sum(axis=1)
    avg_degree = degrees.mean()
    min_degree = degrees.min()
    max_degree = degrees.max()

    print("\n=== Graph diagnostics ===")
    print(f"Number of nodes:   {n}")
    print(f"Number of edges:   {num_edges}")
    print(f"Average degree:    {avg_degree:.2f}")
    print(f"Min / max degree:  {min_degree} / {max_degree}")

    print("\nSample of edge weights (first 5 non-zero edges):")
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                print(
                    f"  ({G.nodes[i]}, {G.nodes[j]}): "
                    f"W_ij = {W[i, j]:.4f}, L_ij = {L[i, j]:.4f}"
                )
                count += 1
                if count >= 5:
                    break
        if count >= 5:
            break

    print("\nTest complete. If the graph stats look reasonable "
          "(non-zero edges, reasonable degrees), "
          "your correlation graph construction is working.")


if __name__ == "__main__":
    main()
