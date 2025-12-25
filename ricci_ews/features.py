"""
features.py
1. Curvature feature vector x_t
   - Built from edge- and vertex-level Ollivier–Ricci (ORC) and
     Forman–Ricci (FRC) aggregates for BOTH graph families:
       g = ρ    (correlation-based graph)
       g = Θ    (precision-based / graphical-lasso graph)
   - Includes:
       * Edge-level (for each g, separately for κ and F):
           mean, median, q10, q05, LTM_0.10, frac_negative
       * Vertex-level (for each g, separately for κ and F):
           mean, median, q10
       * Plus EWMA transforms of ALL raw scalar curvature features with
         half-lives H ∈ {5, 10, 20}.

2. Non-curvature baselines z_t
   - Market-only: moving absolute correlation, short-horizon realized
     volatility, recent market return.
   - Topology-only: average degree, weighted clustering, algebraic
     connectivity, spectral radius.
   - Eigenmode: largest correlation eigenvalue.

This module does NOT train models; it only produces feature vectors.

Typical usage pattern
---------------------

Given for each date t:

  - G_corr_t: GraphData for correlation graph (ρ)
  - curv_corr_t: CurvatureResult for G_corr_t
  - G_prec_t: GraphData for precision graph (Θ)
  - curv_prec_t: CurvatureResult for G_prec_t


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .graphs_correlation import GraphData
from .curvature import CurvatureResult


# ----------------------------------------------------------------------
# Basic helpers for statistics on 1D arrays
# ----------------------------------------------------------------------


def _to_1d(x: np.ndarray) -> np.ndarray:
    """Flatten and drop NaNs/infs."""
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def _mean(x: np.ndarray) -> float:
    x = _to_1d(x)
    return float(x.mean()) if x.size else np.nan


def _median(x: np.ndarray) -> float:
    x = _to_1d(x)
    return float(np.median(x)) if x.size else np.nan


def _quantile(x: np.ndarray, q: float) -> float:
    x = _to_1d(x)
    return float(np.quantile(x, q)) if x.size else np.nan


def _lower_tail_mean(x: np.ndarray, q: float = 0.10) -> float:
    """
    LTM_p(S) = mean of elements <= q_p(S), as in the proposal.
    """
    x = _to_1d(x)
    if x.size == 0:
        return np.nan
    q_val = np.quantile(x, q)
    tail = x[x <= q_val]
    if tail.size == 0:
        return np.nan
    return float(tail.mean())


def _frac_negative(x: np.ndarray) -> float:
    """
    frac^-(S) = |{x<0}| / |S|, as in the proposal.
    """
    x = _to_1d(x)
    if x.size == 0:
        return np.nan
    return float((x < 0).mean())


# ----------------------------------------------------------------------
# RAW CURVATURE FEATURES FOR A SINGLE GRAPH (ONE FAMILY g)
# ----------------------------------------------------------------------


@dataclass
class RawCurvatureBlock:
    """
    Raw curvature features (no EWMA) for a single graph family g.

    Attributes follow the notation in the proposal:

      - Edge-level κ (ORC):
          k_edge_mean, k_edge_median, k_edge_q10, k_edge_q05,
          k_edge_ltm10, k_edge_frac_neg

      - Edge-level F (Forman):
          f_edge_mean, f_edge_median, f_edge_q10, f_edge_q05,
          f_edge_ltm10, f_edge_frac_neg

      - Vertex-level κ:
          k_vertex_mean, k_vertex_median, k_vertex_q10

      - Vertex-level F:
          f_vertex_mean, f_vertex_median, f_vertex_q10
    """

    # Edge-level ORC
    k_edge_mean: float
    k_edge_median: float
    k_edge_q10: float
    k_edge_q05: float
    k_edge_ltm10: float
    k_edge_frac_neg: float

    # Edge-level FRC
    f_edge_mean: float
    f_edge_median: float
    f_edge_q10: float
    f_edge_q05: float
    f_edge_ltm10: float
    f_edge_frac_neg: float

    # Vertex-level ORC
    k_vertex_mean: float
    k_vertex_median: float
    k_vertex_q10: float

    # Vertex-level FRC
    f_vertex_mean: float
    f_vertex_median: float
    f_vertex_q10: float

    def to_dict(self, prefix: str) -> Dict[str, float]:
        """
        Convert to a flat dict with keys prefixed by prefix.

        Example prefix:
          - 'corr_' for g = ρ (correlation graph)
          - 'prec_' for g = Θ (precision/graphical-lasso graph)

        Output keys (examples with prefix='corr_'):
          - 'corr_k_edge_mean'
          - 'corr_f_edge_q05'
          - 'corr_k_vertex_q10', etc.
        """
        return {
            f"{prefix}k_edge_mean": self.k_edge_mean,
            f"{prefix}k_edge_median": self.k_edge_median,
            f"{prefix}k_edge_q10": self.k_edge_q10,
            f"{prefix}k_edge_q05": self.k_edge_q05,
            f"{prefix}k_edge_ltm10": self.k_edge_ltm10,
            f"{prefix}k_edge_frac_neg": self.k_edge_frac_neg,
            f"{prefix}f_edge_mean": self.f_edge_mean,
            f"{prefix}f_edge_median": self.f_edge_median,
            f"{prefix}f_edge_q10": self.f_edge_q10,
            f"{prefix}f_edge_q05": self.f_edge_q05,
            f"{prefix}f_edge_ltm10": self.f_edge_ltm10,
            f"{prefix}f_edge_frac_neg": self.f_edge_frac_neg,
            f"{prefix}k_vertex_mean": self.k_vertex_mean,
            f"{prefix}k_vertex_median": self.k_vertex_median,
            f"{prefix}k_vertex_q10": self.k_vertex_q10,
            f"{prefix}f_vertex_mean": self.f_vertex_mean,
            f"{prefix}f_vertex_median": self.f_vertex_median,
            f"{prefix}f_vertex_q10": self.f_vertex_q10,
        }


def _raw_curvature_block_for_graph(
    G: GraphData,
    curv: CurvatureResult,
) -> RawCurvatureBlock:
    """
    Compute the raw curvature block for a single graph family g
    given its GraphData and CurvatureResult.

    This implements equations in Section 3.7/3.8:
      - edge-level aggregates over {κ_ij}, {F_ij}
      - vertex-level aggregates over {κ_i}, {F_i}
    """
    A = G.adjacency.astype(bool)

    # Edge values: restrict to edges (i,j) where adjacency>0
    k_edges = curv.edge_orc[A]
    f_edges = curv.edge_frc[A]

    # Vertex values: directly from node-level arrays
    k_vertices = curv.node_orc
    f_vertices = curv.node_frc

    # Edge-level ORC
    k_edge_mean = _mean(k_edges)
    k_edge_median = _median(k_edges)
    k_edge_q10 = _quantile(k_edges, 0.10)
    k_edge_q05 = _quantile(k_edges, 0.05)
    k_edge_ltm10 = _lower_tail_mean(k_edges, 0.10)
    k_edge_frac_neg = _frac_negative(k_edges)

    # Edge-level FRC
    f_edge_mean = _mean(f_edges)
    f_edge_median = _median(f_edges)
    f_edge_q10 = _quantile(f_edges, 0.10)
    f_edge_q05 = _quantile(f_edges, 0.05)
    f_edge_ltm10 = _lower_tail_mean(f_edges, 0.10)
    f_edge_frac_neg = _frac_negative(f_edges)

    # Vertex-level ORC
    k_vertex_mean = _mean(k_vertices)
    k_vertex_median = _median(k_vertices)
    k_vertex_q10 = _quantile(k_vertices, 0.10)

    # Vertex-level FRC
    f_vertex_mean = _mean(f_vertices)
    f_vertex_median = _median(f_vertices)
    f_vertex_q10 = _quantile(f_vertices, 0.10)

    return RawCurvatureBlock(
        k_edge_mean=k_edge_mean,
        k_edge_median=k_edge_median,
        k_edge_q10=k_edge_q10,
        k_edge_q05=k_edge_q05,
        k_edge_ltm10=k_edge_ltm10,
        k_edge_frac_neg=k_edge_frac_neg,
        f_edge_mean=f_edge_mean,
        f_edge_median=f_edge_median,
        f_edge_q10=f_edge_q10,
        f_edge_q05=f_edge_q05,
        f_edge_ltm10=f_edge_ltm10,
        f_edge_frac_neg=f_edge_frac_neg,
        k_vertex_mean=k_vertex_mean,
        k_vertex_median=k_vertex_median,
        k_vertex_q10=k_vertex_q10,
        f_vertex_mean=f_vertex_mean,
        f_vertex_median=f_vertex_median,
        f_vertex_q10=f_vertex_q10,
    )


# ----------------------------------------------------------------------
# FULL CURVATURE FEATURE VECTOR x_t (BOTH GRAPH FAMILIES)
# ----------------------------------------------------------------------


def compute_curvature_features_for_date(
    G_corr: GraphData,
    curv_corr: CurvatureResult,
    G_prec: GraphData,
    curv_prec: CurvatureResult,
) -> Dict[str, float]:
    """
    Compute the full *raw* curvature feature vector x_t for one date t,
    as specified in Section 3.8:

        x_t = [
          edge-level, correlation (ORC, FRC);
          vertex-level, correlation (ORC, FRC);
          edge-level, precision (ORC, FRC);
          vertex-level, precision (ORC, FRC)
        ]

    Output is a flat dict whose keys correspond to scalar components
    of x_t, with the following naming scheme:

      - corr_k_edge_mean        (ρ, edge, ORC)
      - corr_f_edge_mean        (ρ, edge, FRC)
      - corr_k_vertex_q10       (ρ, vertex, ORC)
      - prec_f_edge_ltm10       (Θ, edge, FRC), etc.

    EWMA transforms (with half-lives 5,10,20) are NOT added here;
    use add_ewma_features(...) on the resulting DataFrame over time.
    """
    # Graph family g = ρ (correlation)
    block_corr = _raw_curvature_block_for_graph(G_corr, curv_corr)
    d_corr = block_corr.to_dict(prefix="corr_")

    # Graph family g = Θ (precision)
    block_prec = _raw_curvature_block_for_graph(G_prec, curv_prec)
    d_prec = block_prec.to_dict(prefix="prec_")

    # Combine
    out = {}
    out.update(d_corr)
    out.update(d_prec)
    return out


# ----------------------------------------------------------------------
# EWMA transforms for curvature features
# ----------------------------------------------------------------------


def _get_ewma_half_lives() -> Tuple[int, ...]:
    """
    EWMA half-lives H ∈ {5,10,20} as in the proposal.

    If config.EWMA_HALFLIVES exists, use that; otherwise default to
    (5, 10, 20).
    """
    default = (5, 10, 20)
    hl = getattr(config, "EWMA_HALFLIVES", default)
    if isinstance(hl, (list, tuple)):
        return tuple(int(h) for h in hl)
    return default


def add_ewma_features(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    half_lives: Sequence[int] | None = None,
    suffix_template: str = "_ewma_h{H}",
) -> pd.DataFrame:
    """
    Given a DataFrame of raw scalar features over time, augment it
    with EWMA transforms s_t^{(H)} for H ∈ {5,10,20} (by default).

    This version is optimized to avoid pandas fragmentation by
    building all new columns first and then concatenating once.

    Parameters
    ----------
    df : pd.DataFrame
        Index: dates t (sorted). Columns: raw features (e.g. the outputs
        of compute_curvature_features_for_date assembled over time).
    columns : sequence of str, optional
        Which columns to transform. If None, all numeric columns are used.
    half_lives : sequence of int, optional
        Half-lives H for EWMAs. Default from config.EWMA_HALFLIVES or
        (5,10,20).
    suffix_template : str
        Suffix pattern for new columns; {H} will be replaced with the
        integer half-life. Example: "_ewma_h{H}" -> "corr_k_edge_mean_ewma_h5".

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with additional EWMA columns appended.
    """
    if half_lives is None:
        half_lives = _get_ewma_half_lives()

    if columns is None:
        # take numeric columns only by default
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Collect new EWMA columns here
    new_cols: dict[str, pd.Series] = {}

    for col in columns:
        s = df[col].astype(float)
        for H in half_lives:
            if H <= 0:
                continue
            lam = 1.0 - 2.0 ** (-1.0 / H)
            s_ewma = s.ewm(alpha=lam, adjust=False).mean()
            new_col_name = f"{col}{suffix_template.format(H=H)}"
            new_cols[new_col_name] = s_ewma

    if not new_cols:
        # nothing to add
        return df.copy()

    ewma_df = pd.DataFrame(new_cols, index=df.index)
    # Single concat → no fragmentation warning
    df_out = pd.concat([df, ewma_df], axis=1)

    return df_out


# ----------------------------------------------------------------------
# NON-CURVATURE BASELINE FEATURES z_t (hooks / scaffold)
# ----------------------------------------------------------------------


def compute_market_baseline_features(
    corr_matrix: np.ndarray,
    market_returns_window: np.ndarray,
    h_short: int = 5,
) -> Dict[str, float]:
    """
    Market-only baseline features (z_t subset), as in Section 3.8:

      - moving absolute correlation
      - short-horizon realized volatility
      - recent market return

    Parameters
    ----------
    corr_matrix : np.ndarray
        Empirical correlation matrix for the current window (N x N).
        Typically computed from asset returns over [t-W+1, t].
    market_returns_window : np.ndarray
        1D array of market (index) log-returns over the same window.
        Last element corresponds to date t.
    h_short : int, default 5
        Short horizon for realized volatility and recent return.

    Returns
    -------
    dict with keys:
      - market_abs_corr_mean
      - market_rv_short
      - market_ret_recent
    """
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    N = corr_matrix.shape[0]
    # use upper triangle without diagonal
    iu = np.triu_indices(N, k=1)
    abs_corr_vals = np.abs(corr_matrix[iu])
    market_abs_corr_mean = _mean(abs_corr_vals)

    r = np.asarray(market_returns_window, dtype=float)
    # last h_short returns (backward-looking approximation)
    if r.size >= h_short:
        r_short = r[-h_short:]
    else:
        r_short = r

    if r_short.size == 0:
        market_rv_short = np.nan
        market_ret_recent = np.nan
    else:
        # "short-horizon realized volatility": RMS scaled to annual units,
        # mirroring the forward RV^{(h)} definition but backward-looking.
        market_rv_short = float(
            np.sqrt(252.0 / r_short.size * np.sum(r_short ** 2))
        )
        # simple % price move over the last h_short days
        market_ret_recent = float(np.exp(np.sum(r_short)) - 1.0)

    return {
        "market_abs_corr_mean": market_abs_corr_mean,
        "market_rv_short": market_rv_short,
        "market_ret_recent": market_ret_recent,
    }


def compute_topology_baseline_features(
    G: GraphData,
) -> Dict[str, float]:
    """
    Topology-only baseline features from Section 3.8:

      - average degree
      - weighted clustering (global)
      - algebraic connectivity (2nd-smallest eigenvalue of Laplacian)
      - spectral radius (largest eigenvalue of adjacency / weights)

    Parameters
    ----------
    G : GraphData
        Graph for which to compute these metrics. In the proposal,
        these baselines are applied to the correlation graph, but you
        can apply them to other graphs if desired.

    Returns
    -------
    dict with keys:
      - topo_avg_degree
      - topo_weighted_clustering
      - topo_algebraic_conn
      - topo_spectral_radius
    """
    A = G.adjacency.astype(float)
    W = G.W.astype(float)
    N = A.shape[0]

    # Average (unweighted) degree
    degrees = A.sum(axis=1)
    topo_avg_degree = _mean(degrees)

    # Weighted clustering (simple approximation):
    #   C_i^w = (1 / (deg_i * (deg_i - 1))) * sum_{j,k} (w_ij w_ik w_jk)^{1/3}
    # on undirected graphs. We compute a global average over nodes with deg_i >= 2.
    deg = degrees
    clustering_vals = []
    for i in range(N):
        if deg[i] < 2:
            continue
        neigh = np.where(A[i] > 0)[0]
        if neigh.size < 2:
            continue
        tri_sum = 0.0
        count = 0
        for idx_j in range(len(neigh)):
            j = neigh[idx_j]
            for idx_k in range(idx_j + 1, len(neigh)):
                k = neigh[idx_k]
                if A[j, k] <= 0:
                    continue
                wij = W[i, j]
                wik = W[i, k]
                wjk = W[j, k]
                tri_sum += (wij * wik * wjk) ** (1.0 / 3.0)
                count += 1
        if count > 0:
            clustering_vals.append(tri_sum / count)
    topo_weighted_clustering = (
        _mean(np.array(clustering_vals)) if clustering_vals else np.nan
    )

    # Algebraic connectivity: second-smallest eigenvalue of weighted Laplacian
    w_deg = W.sum(axis=1)
    L = np.diag(w_deg) - W
    L = 0.5 * (L + L.T)
    try:
        evals_L = np.linalg.eigvalsh(L)
        evals_L_sorted = np.sort(evals_L)
        topo_algebraic_conn = (
            float(evals_L_sorted[1]) if evals_L_sorted.size >= 2 else np.nan
        )
    except np.linalg.LinAlgError:
        topo_algebraic_conn = np.nan

    # Spectral radius: largest eigenvalue of W (weights)
    W_sym = 0.5 * (W + W.T)
    try:
        evals_W = np.linalg.eigvalsh(W_sym)
        topo_spectral_radius = float(np.max(evals_W))
    except np.linalg.LinAlgError:
        topo_spectral_radius = np.nan

    return {
        "topo_avg_degree": topo_avg_degree,
        "topo_weighted_clustering": topo_weighted_clustering,
        "topo_algebraic_conn": topo_algebraic_conn,
        "topo_spectral_radius": topo_spectral_radius,
    }


def compute_eigenmode_baseline_features(
    corr_matrix: np.ndarray,
) -> Dict[str, float]:
    """
    Eigenmode baseline feature (Section 3.8):

      - largest correlation eigenvalue.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix (N x N) of asset returns.

    Returns
    -------
    dict with key:
      - eigen_largest_corr
    """
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    try:
        evals = np.linalg.eigvalsh(corr_matrix)
        eigen_largest_corr = float(np.max(evals))
    except np.linalg.LinAlgError:
        eigen_largest_corr = np.nan

    return {"eigen_largest_corr": eigen_largest_corr}


# ----------------------------------------------------------------------
# ASSEMBLY OF FULL FEATURE VECTOR φ_t = [x_t^⊤, z_t^⊤]^⊤
# ----------------------------------------------------------------------


def assemble_feature_vector(
    curvature_features: Mapping[str, float],
    baseline_features: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    """
    Assemble the full feature vector φ_t = [x_t^⊤, z_t^⊤]^⊤
    for one date t, from curvature features (x_t) and optional
    non-curvature baselines (z_t).

    Parameters
    ----------
    curvature_features : mapping
        Output of compute_curvature_features_for_date (or equivalent).
    baseline_features : mapping, optional
        Any combination of market/topology/eigenmode baseline features
        computed above. If None, φ_t consists only of x_t.

    Returns
    -------
    dict
        Flat dict of all features for one date (keys are column names).
    """
    out = dict(curvature_features)
    if baseline_features is not None:
        out.update(baseline_features)
    return out
