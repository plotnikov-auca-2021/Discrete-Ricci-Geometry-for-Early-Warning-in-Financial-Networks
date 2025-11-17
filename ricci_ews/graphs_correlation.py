"""
graphs_correlation.py

Correlation-based graph construction with Ledoit–Wolf shrinkage.

Steps:
1. Estimate shrinkage covariance Σ̂ using Ledoit–Wolf, preferably with
   a constant-correlation target (via PyPortfolioOpt if available).
2. Convert Σ̂ to correlation matrix ρ̂.
3. Define weights w_ij = |ρ̂_ij|^beta (zero on the diagonal).
4. Sparsify to achieve a target average degree by selecting the strongest
   edges globally, then enforce connectivity.
5. Define edge lengths ℓ_ij = 1 / (w_ij^γ) + ε for existing edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config

# Optional: PyPortfolioOpt for constant-correlation shrinkage target
try:
    from pypfopt import risk_models as _risk_models  # type: ignore

    _HAVE_PYPORTFOLIOOPT = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_PYPORTFOLIOOPT = False

# Fallback: sklearn's LedoitWolf (constant-variance target)
try:
    from sklearn.covariance import LedoitWolf as _LedoitWolf  # type: ignore

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


@dataclass
class GraphData:
    """
    Container for correlation-based graph.

    Attributes
    ----------
    W : np.ndarray
        Weight matrix (N x N), symmetric, zeros on diagonal.
    L : np.ndarray
        Length matrix (N x N), symmetric, positive for edges, np.inf for no edge.
    adjacency : np.ndarray
        Boolean adjacency matrix (N x N), symmetric, False on diagonal.
    nodes : list[str]
        Node labels in the same order as matrix indices.
    """

    W: np.ndarray
    L: np.ndarray
    adjacency: np.ndarray
    nodes: List[str]


# ----------------------------------------------------------------------
# Covariance and correlation
# ----------------------------------------------------------------------


def ledoit_wolf_cov(
    R_window: np.ndarray,
    shrinkage_target: str = "constant_correlation",
) -> np.ndarray:
    """
    Estimate covariance matrix using Ledoit–Wolf shrinkage.

    Parameters
    ----------
    R_window : np.ndarray
        2D array of shape (W, N) with returns for W days and N assets.
    shrinkage_target : {'constant_correlation', 'constant_variance'}
        Desired target. 'constant_correlation' uses PyPortfolioOpt (if available);
        fallback is 'constant_variance' (sklearn LedoitWolf).

    Returns
    -------
    Sigma_hat : np.ndarray
        Estimated covariance matrix (N x N).
    """
    n_samples, n_assets = R_window.shape

    if shrinkage_target == "constant_correlation" and _HAVE_PYPORTFOLIOOPT:
        # Use PyPortfolioOpt's CovarianceShrinkage implementation
        df = pd.DataFrame(R_window)
        cs = _risk_models.CovarianceShrinkage(df)
        Sigma_hat = cs.ledoit_wolf(shrinkage_target="constant_correlation")
        Sigma_hat = np.asarray(Sigma_hat, dtype=float)
        return Sigma_hat

    if not _HAVE_SKLEARN:
        raise ImportError(
            "scikit-learn is required for LedoitWolf fallback. "
            "Install with: pip install scikit-learn"
        )

    # Fallback: standard LedoitWolf (constant-variance / identity target)
    lw = _LedoitWolf()
    lw.fit(R_window)
    Sigma_hat = lw.covariance_
    return np.asarray(Sigma_hat, dtype=float)


def covariance_to_correlation(Sigma_hat: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix Σ̂ to correlation matrix ρ̂.

    Parameters
    ----------
    Sigma_hat : np.ndarray
        Covariance matrix (N x N).

    Returns
    -------
    rho_hat : np.ndarray
        Correlation matrix (N x N), with ones on the diagonal.
    """
    diag = np.diag(Sigma_hat)
    std = np.sqrt(diag)
    std[std == 0] = 1.0
    outer_std = np.outer(std, std)
    rho_hat = Sigma_hat / outer_std
    np.fill_diagonal(rho_hat, 1.0)
    # Numerical safety
    rho_hat = np.clip(rho_hat, -1.0, 1.0)
    return rho_hat


def correlation_weights(rho_hat: np.ndarray, beta: int = 1) -> np.ndarray:
    """
    Compute correlation-based weights:

        w_ij = |ρ̂_ij|^beta, with zeros on the diagonal.

    Parameters
    ----------
    rho_hat : np.ndarray
        Correlation matrix (N x N).
    beta : int
        Exponent to accentuate strong correlations (typically 1 or 2).

    Returns
    -------
    W : np.ndarray
        Weight matrix (N x N), symmetric, zeros on diagonal.
    """
    W = np.abs(rho_hat) ** beta
    np.fill_diagonal(W, 0.0)
    return W


# ----------------------------------------------------------------------
# Sparsification and connectivity
# ----------------------------------------------------------------------


def _init_union_find(n: int) -> Tuple[np.ndarray, np.ndarray]:
    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    return parent, rank


def _find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union(parent: np.ndarray, rank: np.ndarray, i: int, j: int) -> None:
    ri, rj = _find(parent, i), _find(parent, j)
    if ri == rj:
        return
    if rank[ri] < rank[rj]:
        parent[ri] = rj
    elif rank[ri] > rank[rj]:
        parent[rj] = ri
    else:
        parent[rj] = ri
        rank[ri] += 1


def build_sparse_graph_from_weights(
    W_full: np.ndarray,
    target_avg_degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sparsify full weight matrix to achieve a target average degree.

    Strategy:
    - Collect all off-diagonal edges with W_ij > 0 for i < j.
    - Sort them by weight (descending).
    - Keep the top M edges, where M ~ target_avg_degree * N / 2.
    - Then enforce connectivity by adding the strongest remaining edges
      that connect different components.

    Parameters
    ----------
    W_full : np.ndarray
        Full weight matrix (N x N), symmetric, zeros on diagonal.
    target_avg_degree : int
        Desired average degree per node.

    Returns
    -------
    W_sparse : np.ndarray
        Sparse weight matrix (N x N), symmetric, zeros on diagonal for missing edges.
    adjacency : np.ndarray
        Boolean adjacency matrix (N x N).
    """
    n = W_full.shape[0]
    assert W_full.shape == (n, n)

    # Collect all potential edges (i < j) with positive weight
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = W_full[i, j]
            if w > 0:
                edges.append((w, i, j))

    if not edges:
        # No positive weights; return empty graph
        W_sparse = np.zeros_like(W_full)
        adjacency = np.zeros_like(W_full, dtype=bool)
        return W_sparse, adjacency

    # Sort edges by descending weight
    edges.sort(key=lambda x: x[0], reverse=True)

    max_possible_edges = len(edges)
    target_edges = int(round(target_avg_degree * n / 2))
    target_edges = max(1, min(target_edges, max_possible_edges))

    # Initialize union-find for connectivity
    parent, rank = _init_union_find(n)

    # Step 1: Pick the strongest edges up to target_edges
    chosen = [False] * max_possible_edges
    W_sparse = np.zeros_like(W_full)
    adjacency = np.zeros_like(W_full, dtype=bool)

    for idx in range(target_edges):
        w, i, j = edges[idx]
        chosen[idx] = True
        W_sparse[i, j] = W_sparse[j, i] = w
        adjacency[i, j] = adjacency[j, i] = True
        _union(parent, rank, i, j)

    # Step 2: Ensure connectivity by adding bridging edges if needed
    # If there are multiple components, keep adding the strongest
    # not-yet-chosen edges that link different components.
    for idx in range(target_edges, max_possible_edges):
        # Check if graph already connected
        roots = {_find(parent, i) for i in range(n)}
        if len(roots) == 1:
            break
        w, i, j = edges[idx]
        ri, rj = _find(parent, i), _find(parent, j)
        if ri != rj:
            chosen[idx] = True
            W_sparse[i, j] = W_sparse[j, i] = w
            adjacency[i, j] = adjacency[j, i] = True
            _union(parent, rank, i, j)

    return W_sparse, adjacency


def compute_lengths_from_weights(
    W: np.ndarray,
    gamma: float,
    eps: float,
) -> np.ndarray:
    """
    Compute edge lengths from weights:

        ℓ_ij = 1 / (W_ij^γ) + eps,  for W_ij > 0
        ℓ_ij = inf,                 otherwise

    Parameters
    ----------
    W : np.ndarray
        Weight matrix (N x N).
    gamma : float
        Exponent controlling nonlinearity (e.g. 0.75).
    eps : float
        Small positive constant added for numerical stability.

    Returns
    -------
    L : np.ndarray
        Length matrix (N x N).
    """
    L = np.full_like(W, np.inf, dtype=float)
    mask = W > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        L[mask] = 1.0 / (np.power(W[mask], gamma)) + eps
    np.fill_diagonal(L, 0.0)
    return L


# ----------------------------------------------------------------------
# Public API: build full correlation-based graph
# ----------------------------------------------------------------------


def build_correlation_graph(
    R_window: np.ndarray,
    node_names: Sequence[str],
    beta: int | None = None,
    target_avg_degree: int | None = None,
    gamma: float | None = None,
    eps: float | None = None,
    shrinkage_target: str | None = None,
) -> GraphData:
    """
    Build correlation-based graph from a window of returns.

    Parameters
    ----------
    R_window : np.ndarray
        Returns in the current window, shape (W, N).
    node_names : Sequence[str]
        List of node labels corresponding to columns (assets) in R_window.
    beta : int, optional
        Exponent for correlation weights |ρ|^beta (default: config.CORR_BETA).
    target_avg_degree : int, optional
        Desired average degree (default: config.TARGET_AVG_DEGREE).
    gamma : float, optional
        Exponent in length transform (default: config.LENGTH_EXPONENT).
    eps : float, optional
        Small offset in length transform (default: config.LENGTH_EPS).
    shrinkage_target : str, optional
        Shrinkage target for covariance ('constant_correlation' or 'constant_variance').
        Default: config.SHRINKAGE_TARGET.

    Returns
    -------
    GraphData
        Contains W (weights), L (lengths), adjacency, and node labels.
    """
    if beta is None:
        beta = config.CORR_BETA
    if target_avg_degree is None:
        target_avg_degree = config.TARGET_AVG_DEGREE
    if gamma is None:
        gamma = config.LENGTH_EXPONENT
    if eps is None:
        eps = config.LENGTH_EPS
    if shrinkage_target is None:
        shrinkage_target = config.SHRINKAGE_TARGET

    R_window = np.asarray(R_window, dtype=float)
    n_samples, n_assets = R_window.shape

    if len(node_names) != n_assets:
        raise ValueError(
            f"node_names length ({len(node_names)}) does not match "
            f"number of assets ({n_assets})."
        )

    # 1) Shrinkage covariance
    Sigma_hat = ledoit_wolf_cov(R_window, shrinkage_target=shrinkage_target)

    # 2) Correlation matrix
    rho_hat = covariance_to_correlation(Sigma_hat)

    # 3) Weight matrix
    W_full = correlation_weights(rho_hat, beta=beta)

    # 4) Sparsify and ensure connectivity
    W_sparse, adjacency = build_sparse_graph_from_weights(
        W_full, target_avg_degree=target_avg_degree
    )

    # 5) Length matrix
    L = compute_lengths_from_weights(W_sparse, gamma=gamma, eps=eps)

    return GraphData(
        W=W_sparse,
        L=L,
        adjacency=adjacency,
        nodes=list(node_names),
    )
