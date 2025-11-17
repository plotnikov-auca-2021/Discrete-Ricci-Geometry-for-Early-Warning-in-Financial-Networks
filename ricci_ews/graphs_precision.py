"""
graphs_precision.py

Sparse precision-based graph construction using graphical lasso.

Pipeline for a given return window R_window (W x N):
1. Standardize returns within the window (zero mean, unit variance per asset).
2. Fit graphical lasso on standardized returns to estimate precision matrix Θ̂.
3. Convert Θ̂ to partial correlations ρ̃_ij.
4. Define weights w_ij = |ρ̃_ij|^beta.
5. Sparsify to achieve a target average degree (reusing correlation-graph sparsifier).
6. Define edge lengths ℓ_ij = 1 / (w_ij^γ) + ε for existing edges.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import config
from .returns import standardize_within_window
from .graphs_correlation import (
    GraphData,
    build_sparse_graph_from_weights,
    compute_lengths_from_weights,
)

# Graphical lasso from scikit-learn
try:
    from sklearn.covariance import GraphicalLasso  # type: ignore

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_SKLEARN = False


# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------


def graphical_lasso_precision(
    Z_window: np.ndarray,
    lam: float,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Fit graphical lasso on standardized returns and return precision matrix Θ̂.

    - First, tries GraphicalLasso with lam, lam*2, lam*5.
    - If all attempts fail (e.g. Non-SPD errors), falls back to
      a ridge-regularized covariance inversion.

    Parameters
    ----------
    Z_window : np.ndarray
        Standardized returns of shape (W, N) (zero mean, unit variance per column).
    lam : float
        Baseline regularization parameter (alpha in GraphicalLasso).
    max_iter : int
        Maximum number of iterations for the solver.
    tol : float
        Convergence tolerance for the optimization.

    Returns
    -------
    theta_hat : np.ndarray
        Estimated precision matrix (N x N), symmetric.
    """
    if not _HAVE_SKLEARN:
        raise ImportError(
            "scikit-learn is required for graphical lasso precision graphs. "
            "Install with: pip install scikit-learn"
        )

    Z_window = np.asarray(Z_window, dtype=float)
    _, n_features = Z_window.shape

    last_exception: Exception | None = None

    # --- 1) Try GraphicalLasso with increasing alpha ---
    alphas_to_try = [lam, lam * 2.0, lam * 5.0]

    for alpha in alphas_to_try:
        try:
            model = GraphicalLasso(alpha=alpha, max_iter=max_iter, tol=tol)
            model.fit(Z_window)
            theta_hat = np.asarray(model.precision_, dtype=float)
            # Symmetrize for safety
            theta_hat = 0.5 * (theta_hat + theta_hat.T)
            return theta_hat
        except FloatingPointError as e:
            # Non-SPD or ill-conditioned system
            last_exception = e
            continue
        except Exception as e:
            # Any other numerical issue
            last_exception = e
            continue

    # --- 2) Fallback: ridge-regularized covariance inversion ---
    # Empirical covariance
    S = np.cov(Z_window, rowvar=False)

    # We will iteratively add jitter to the diagonal until inversion works
    jitter = 1e-3
    for _ in range(6):
        try:
            S_reg = S + jitter * np.eye(n_features)
            theta_hat = np.linalg.inv(S_reg)
            theta_hat = 0.5 * (theta_hat + theta_hat.T)
            return theta_hat
        except np.linalg.LinAlgError:
            jitter *= 10.0  # increase regularization and retry

    # If even the fallback fails, raise a clear error
    raise RuntimeError(
        "graphical_lasso_precision failed even with fallback. "
        f"Last exception: {repr(last_exception)}"
    )


def partial_correlations(theta_hat: np.ndarray) -> np.ndarray:
    """
    Convert precision matrix Θ̂ to partial correlation matrix ρ̃.

    Formula (for i != j):
        ρ̃_ij = - Θ̂_ij / sqrt(Θ̂_ii * Θ̂_jj)

    Diagonal entries are set to 1.0.

    Parameters
    ----------
    theta_hat : np.ndarray
        Precision matrix (N x N).

    Returns
    -------
    pcorr : np.ndarray
        Partial correlation matrix (N x N), with ones on the diagonal,
        values clipped to [-1, 1].
    """
    diag = np.diag(theta_hat)
    # Guard against non-positive diagonals
    diag = np.where(diag <= 0, np.nan, diag)
    d = np.sqrt(diag)
    # Replace any NaNs or zeros with a small positive number to avoid division by zero
    d = np.where(np.isnan(d) | (d == 0), 1e-8, d)

    outer_d = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        pcorr = -theta_hat / outer_d

    np.fill_diagonal(pcorr, 1.0)
    pcorr = np.clip(pcorr, -1.0, 1.0)
    return pcorr


def precision_weights(pcorr: np.ndarray, beta: int = 1) -> np.ndarray:
    """
    Compute weights from partial correlations:

        w_ij = |ρ̃_ij|^beta, zeros on diagonal.

    Parameters
    ----------
    pcorr : np.ndarray
        Partial correlation matrix (N x N).
    beta : int
        Exponent for emphasis on strong connections (typically 1 or 2).

    Returns
    -------
    W : np.ndarray
        Weight matrix (N x N), symmetric, zeros on diagonal.
    """
    W = np.abs(pcorr) ** beta
    np.fill_diagonal(W, 0.0)
    return W


# ----------------------------------------------------------------------
# Public API: build precision-based graph
# ----------------------------------------------------------------------


def build_precision_graph(
    R_window: np.ndarray,
    node_names: Sequence[str],
    lam: float | None = None,
    beta: int | None = None,
    target_avg_degree: int | None = None,
    gamma: float | None = None,
    eps: float | None = None,
) -> GraphData:
    """
    Build a precision-based graph from a window of returns.

    Parameters
    ----------
    R_window : np.ndarray
        Returns in the current window, shape (W, N).
    node_names : Sequence[str]
        List of node labels corresponding to columns (assets) in R_window.
    lam : float, optional
        Graphical lasso regularization parameter (alpha). Default: config.GLASSO_LAMBDA.
    beta : int, optional
        Exponent for weights |ρ̃_ij|^beta (default: config.CORR_BETA).
    target_avg_degree : int, optional
        Desired average degree per node (default: config.TARGET_AVG_DEGREE).
    gamma : float, optional
        Exponent in length transform (default: config.LENGTH_EXPONENT).
    eps : float, optional
        Small offset in length transform (default: config.LENGTH_EPS).

    Returns
    -------
    GraphData
        Contains W (weights), L (lengths), adjacency, and node labels.
    """
    if lam is None:
        lam = config.GLASSO_LAMBDA
    if beta is None:
        # Reuse CORR_BETA unless you want a separate parameter in config
        beta = config.CORR_BETA
    if target_avg_degree is None:
        target_avg_degree = config.TARGET_AVG_DEGREE
    if gamma is None:
        gamma = config.LENGTH_EXPONENT
    if eps is None:
        eps = config.LENGTH_EPS

    R_window = np.asarray(R_window, dtype=float)
    n_samples, n_assets = R_window.shape

    if len(node_names) != n_assets:
        raise ValueError(
            f"node_names length ({len(node_names)}) does not match "
            f"number of assets ({n_assets})."
        )

    # 1) Standardize within window (zero mean, unit variance per column)
    Z = standardize_within_window(R_window)

    # 2) Graphical lasso precision matrix (with robust fallback)
    theta_hat = graphical_lasso_precision(Z, lam=lam)

    # 3) Partial correlations
    pcorr = partial_correlations(theta_hat)

    # 4) Weights from partial correlations
    W_full = precision_weights(pcorr, beta=beta)

    # 5) Sparsify and ensure connectivity (reuse correlation-graph logic)
    W_sparse, adjacency = build_sparse_graph_from_weights(
        W_full, target_avg_degree=target_avg_degree
    )

    # 6) Length matrix
    L = compute_lengths_from_weights(W_sparse, gamma=gamma, eps=eps)

    return GraphData(
        W=W_sparse,
        L=L,
        adjacency=adjacency,
        nodes=list(node_names),
    )
