"""
Mechanistic probes for interpreting financial networks, plus simple
univariate logistic probes.
--------------------------------------

Work on a weighted, undirected graph with weight matrix W:

(A) Diffusion & mixing-time probes
    - Build a lazy random walk with transition matrix P:
        P_ij = eta                       if j == i
             = (1 - eta) * w_ij / d_i    if j != i and i~j
             = 0                         otherwise
      where d_i = sum_j w_ij.

    - Stationary distribution (for undirected W):
        pi_i = d_i / vol(G),  vol(G) = sum_i d_i.

    - Spectral gap:
        gamma = 1 - lambda_2,
      where lambda_2 is the second-largest eigenvalue of P.

      We compute eigenvalues via the symmetric similar matrix
        S = eta I + (1 - eta) D^{-1/2} W D^{-1/2},
      which has the same spectrum as P.

    - Proxy mixing time (for epsilon = 0.01 as in the proposal):
        t_mix_hat = (1 / gamma) * log(1 / (eps * pi_min)),
      where pi_min = min_i pi_i (over nodes with positive degree).

    Probes (per graph):
        - diff_gamma
        - diff_tmix_proxy
        - diff_pi_min
        - diff_lambda2   (for debugging / interpretation)

(B) Mean first-passage time (MFPT) and commute time (CT)
    - Weighted Laplacian:
        L = D - W, where D = diag(d_i), d_i = sum_j w_ij.

    - Pseudoinverse L^+ via eigen-decomposition:
        L = U diag(lambda) U^T,   L^+ = U diag(1/lambda_i for lambda_i>0) U^T.

    - Commute time:
        CT(i, j) = vol(G) * <e_i - e_j, L^+(e_i - e_j)>
                  = vol(G) * (L^+_ii + L^+_jj - 2 L^+_ij).

    - Mean first-passage time:
        MFPT(i -> j) = vol(G) * (L^+_jj - L^+_ij).

    In practice, we cannot store/use all N^2 entries in the ML part
    for every window, so we provide:
        - full CT and MFPT matrices (for selected windows / inspection)
        - scalar summaries over all i != j:
              ct_mean,  ct_median,  ct_q10,  ct_q90
              mfpt_mean, mfpt_median, mfpt_q10, mfpt_q90

    These can be used as mechanistic time-series probes and also fed
    into logistic probes or the ML pipeline if desired.


Univariate logistic probes
--------------------------

We also provide simple 1D logistic probes:

    P(y=1 | f) = sigma(alpha + beta * f_std),

for a single feature column f_t and binary label y_t.  These are useful
to get mechanistic, low-capacity relationships between a feature and a
label:

    - ProbeResult with coef, intercept, metrics, calibration table.
    - Batch mode over many features.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

from .graphs_correlation import GraphData


# ======================================================================
# Helpers for metrics & calibration (used by logistic probes)
# ======================================================================


def _safe_metric(fn, y_true, y_score, default=np.nan) -> float:
    """Safely compute a metric; on failure return default."""
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float(default)


def _compute_metrics(
    y_true: np.ndarray,
    p: np.ndarray,
) -> Dict[str, float]:
    """
    Basic binary-probability metrics for a probe:
      - roc_auc
      - pr_auc
      - brier
      - log_loss
      - pos_rate
      - n_samples
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(p)
    if not mask.any():
        return {
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "pos_rate": np.nan,
            "n_samples": 0,
        }

    y = y_true[mask]
    p = np.clip(p[mask], 1e-6, 1.0 - 1e-6)

    pos_rate = float(y.mean())
    roc_auc = _safe_metric(roc_auc_score, y, p)
    pr_auc = _safe_metric(average_precision_score, y, p)
    brier = _safe_metric(brier_score_loss, y, p)
    ll = _safe_metric(log_loss, y, p)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "log_loss": ll,
        "pos_rate": pos_rate,
        "n_samples": int(len(y)),
    }


def _calibration_table(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Calibration table for a 1D probability model:

      For each probability bin, report:
        - bin_lower, bin_upper, bin_center
        - n_samples
        - mean_pred (mean predicted probability)
        - emp_rate (empirical event rate)

    Bins are quantile-based in predicted probability to get roughly
    balanced bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(p)
    if not mask.any():
        return pd.DataFrame(
            columns=[
                "bin_lower",
                "bin_upper",
                "bin_center",
                "n_samples",
                "mean_pred",
                "emp_rate",
            ]
        )

    y = y_true[mask]
    p = np.clip(p[mask], 1e-6, 1.0 - 1e-6)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(p, quantiles))

    # If too few unique edges, fall back to fixed bins
    if bin_edges.size <= 2:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_idx = np.digitize(p, bin_edges, right=True)

    rows = []
    for b in range(1, len(bin_edges)):
        mask_b = bin_idx == b
        if not mask_b.any():
            continue
        p_b = p[mask_b]
        y_b = y[mask_b]

        lower = bin_edges[b - 1]
        upper = bin_edges[b]
        center = 0.5 * (lower + upper)

        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "bin_center": float(center),
                "n_samples": int(mask_b.sum()),
                "mean_pred": float(p_b.mean()),
                "emp_rate": float(y_b.mean()),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================
# MECHANISTIC PROBES
# ======================================================================

# ---------------- Diffusion & mixing-time probes ----------------------


def diffusion_mixing_probes_for_graph(
    G: GraphData,
    eta: float = 0.5,
    eps: float = 0.01,
) -> Dict[str, float]:
    """
    Compute diffusion & mixing-time probes for a single weighted graph.

    Parameters
    ----------
    G : GraphData
        Graph with adjacency and W (weights). Assumed undirected and
        (approximately) connected.
    eta : float, default 0.5
        Self-loop probability for the lazy random walk.
    eps : float, default 0.01
        Total-variation epsilon used in the mixing-time bound:
            t_mix <= (1/gamma) * log(1 / (eps * pi_min))

    Returns
    -------
    dict with keys:
        - diff_gamma        : spectral gap gamma_t
        - diff_tmix_proxy   : proxy mixing time t_mix_hat_t
        - diff_pi_min       : min_i pi_i, pi_i ∝ d_i
        - diff_lambda2      : second-largest eigenvalue of P
    """
    W = np.asarray(G.W, dtype=float)
    W = 0.5 * (W + W.T)  # symmetrize for safety

    d = W.sum(axis=1)
    vol = float(d.sum())
    if vol <= 0:
        return {
            "diff_gamma": np.nan,
            "diff_tmix_proxy": np.nan,
            "diff_pi_min": np.nan,
            "diff_lambda2": np.nan,
        }

    # Stationary distribution pi_i ∝ d_i (for undirected weighted graphs)
    pi = d / vol
    # Avoid zero-degree nodes in pi_min; use positive entries
    positive_pi = pi[pi > 0]
    if positive_pi.size == 0:
        pi_min = np.nan
    else:
        pi_min = float(positive_pi.min())

    N = W.shape[0]
    I = np.eye(N)

    # Symmetric similar matrix for P:
    #   S = eta I + (1 - eta) D^{-1/2} W D^{-1/2}
    # P and S share the same eigenvalues, but S is symmetric.
    D_sqrt_inv = np.zeros_like(d)
    mask = d > 0
    D_sqrt_inv[mask] = 1.0 / np.sqrt(d[mask])

    S = eta * I + (1.0 - eta) * (D_sqrt_inv[:, None] * W * D_sqrt_inv[None, :])
    S = 0.5 * (S + S.T)

    try:
        evals = np.linalg.eigvalsh(S)
    except np.linalg.LinAlgError:
        return {
            "diff_gamma": np.nan,
            "diff_tmix_proxy": np.nan,
            "diff_pi_min": pi_min,
            "diff_lambda2": np.nan,
        }

    # Sort descending: lambda_1 >= lambda_2 >= ...
    evals_sorted = np.sort(evals)[::-1]
    if evals_sorted.size < 2:
        return {
            "diff_gamma": np.nan,
            "diff_tmix_proxy": np.nan,
            "diff_pi_min": pi_min,
            "diff_lambda2": np.nan,
        }

    lambda1 = float(evals_sorted[0])
    lambda2 = float(evals_sorted[1])

    # Spectral gap gamma = 1 - lambda_2.
    # For a lazy chain, lambda_1 ≈ 1, lambda_2 <= 1.
    gamma = 1.0 - lambda2
    # Numerical safety: avoid division by zero
    if gamma <= 1e-12:
        tmix = np.nan
    elif not np.isfinite(pi_min) or pi_min <= 0:
        tmix = np.nan
    else:
        tmix = float((1.0 / gamma) * np.log(1.0 / (eps * pi_min)))

    return {
        "diff_gamma": float(gamma),
        "diff_tmix_proxy": tmix,
        "diff_pi_min": float(pi_min),
        "diff_lambda2": lambda2,
    }


# ---------------- Laplacian, MFPT & commute time ----------------------


def _laplacian_and_pseudoinverse(
    W: np.ndarray,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute Laplacian L, its pseudoinverse L^+, node strengths d, and
    graph volume vol(G) for a weighted undirected graph.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix (symmetric, nonnegative).
    tol : float
        Tolerance below which eigenvalues are treated as zero.

    Returns
    -------
    L : np.ndarray
        Laplacian D - W.
    L_plus : np.ndarray
        Moore–Penrose pseudoinverse of L.
    d : np.ndarray
        Strengths d_i = sum_j w_ij.
    vol : float
        Graph volume = sum_i d_i.
    """
    W = np.asarray(W, dtype=float)
    W = 0.5 * (W + W.T)

    d = W.sum(axis=1)
    L = np.diag(d) - W
    L = 0.5 * (L + L.T)

    vol = float(d.sum())

    try:
        evals, U = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback: no pseudoinverse
        return L, np.full_like(L, np.nan), d, vol

    inv_evals = np.zeros_like(evals)
    mask = evals > tol
    inv_evals[mask] = 1.0 / evals[mask]
    # Eigenvalues close to zero (e.g. connected components) keep 0 in L^+
    # (standard definition of Moore–Penrose pseudoinverse).
    L_plus = (U * inv_evals) @ U.T
    L_plus = 0.5 * (L_plus + L_plus.T)

    return L, L_plus, d, vol


def commute_mfpt_matrices_for_graph(
    G: GraphData,
    tol: float = 1e-10,
) -> Dict[str, np.ndarray]:
    """
    Compute full commute-time and MFPT matrices for a graph.

    Parameters
    ----------
    G : GraphData
        Weighted graph (assumed undirected).
    tol : float
        Eigenvalue tolerance for L^+.

    Returns
    -------
    dict with keys:
        - CT    : ndarray, CT[i,j] = commute time between i and j
        - MFPT  : ndarray, MFPT[i,j] = mean first-passage time i -> j
        - L     : Laplacian matrix (for inspection, same shape)
        - L_plus: Laplacian pseudoinverse
        - d     : strengths (degrees) vector
        - vol   : graph volume (scalar)

    Notes
    -----
    Complexity is O(N^3) for the eigen-decomposition, plus O(N^2) for
    forming the CT and MFPT matrices. This is fine for moderate N
    (e.g. ~100–200), but you probably do NOT want to run this on every
    daily window in a long sample. Instead, call this on a subset of
    windows for mechanistic analysis.
    """
    W = np.asarray(G.W, dtype=float)
    W = 0.5 * (W + W.T)

    L, L_plus, d, vol = _laplacian_and_pseudoinverse(W, tol=tol)

    N = W.shape[0]
    if not np.isfinite(L_plus).all() or vol <= 0:
        CT = np.full((N, N), np.nan, dtype=float)
        MFPT = np.full((N, N), np.nan, dtype=float)
        return {"CT": CT, "MFPT": MFPT, "L": L, "L_plus": L_plus, "d": d, "vol": vol}

    diag_Lp = np.diag(L_plus)

    # Commute time:
    #   CT(i,j) = vol * (L^+_ii + L^+_jj - 2 L^+_ij).
    CT = vol * (diag_Lp[:, None] + diag_Lp[None, :] - 2.0 * L_plus)
    np.fill_diagonal(CT, 0.0)

    # Mean first-passage time:
    #   MFPT(i->j) = vol * (L^+_jj - L^+_ij).
    MFPT = vol * (diag_Lp[None, :] - L_plus)
    np.fill_diagonal(MFPT, 0.0)

    return {"CT": CT, "MFPT": MFPT, "L": L, "L_plus": L_plus, "d": d, "vol": vol}


def summarize_commute_mfpt(
    CT: np.ndarray,
    MFPT: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Summarize commute-time and MFPT matrices into scalar probes:

      For all i != j, compute:
        - ct_mean, ct_median, ct_q10, ct_q90
        - mfpt_mean, mfpt_median, mfpt_q10, mfpt_q90

    Parameters
    ----------
    CT : ndarray
        Commute-time matrix (N x N).
    MFPT : ndarray
        Mean first-passage matrix (N x N).
    prefix : str
        Optional string to prefix all keys with (e.g. 'corr_' or 'prec_').

    Returns
    -------
    dict of scalar probes.
    """
    CT = np.asarray(CT, dtype=float)
    MFPT = np.asarray(MFPT, dtype=float)

    N = CT.shape[0]
    off_diag = ~np.eye(N, dtype=bool)

    ct_vals = CT[off_diag]
    mfpt_vals = MFPT[off_diag]

    def _stats(vals: np.ndarray, base: str) -> Dict[str, float]:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {
                f"{prefix}{base}_mean": np.nan,
                f"{prefix}{base}_median": np.nan,
                f"{prefix}{base}_q10": np.nan,
                f"{prefix}{base}_q90": np.nan,
            }
        return {
            f"{prefix}{base}_mean": float(vals.mean()),
            f"{prefix}{base}_median": float(np.median(vals)),
            f"{prefix}{base}_q10": float(np.quantile(vals, 0.10)),
            f"{prefix}{base}_q90": float(np.quantile(vals, 0.90)),
        }

    out = {}
    out.update(_stats(ct_vals, "ct"))
    out.update(_stats(mfpt_vals, "mfpt"))
    return out


def mechanistic_probes_for_graph(
    G: GraphData,
    eta: float = 0.5,
    eps: float = 0.01,
    tol: float = 1e-10,
    prefix: str = "",
    include_matrices: bool = False,
) -> Dict[str, object]:
    """
    Convenience wrapper: compute *all* mechanistic probes for a graph.

    Parameters
    ----------
    G : GraphData
        Graph to probe.
    eta : float
        Lazy random-walk self-loop probability.
    eps : float
        Epsilon for mixing-time bound.
    tol : float
        Tolerance for Laplacian pseudoinverse.
    prefix : str
        Optional prefix for scalar probe keys (e.g. 'corr_' or 'prec_').
    include_matrices : bool
        If True, include CT and MFPT matrices and Laplacian objects in
        the output dict (under keys f"{prefix}CT", etc.). Use this only
        for a small number of windows (heavy).

    Returns
    -------
    out : dict
        Scalar probes (diffusion + MFPT/CT summaries), and optionally
        full matrices if include_matrices=True.
    """
    # Diffusion and mixing-time probes
    diff = diffusion_mixing_probes_for_graph(G, eta=eta, eps=eps)
    diff_prefixed = {f"{prefix}{k}": v for k, v in diff.items()}

    # Commute/ MFPT matrices and summary
    lap = commute_mfpt_matrices_for_graph(G, tol=tol)
    CT = lap["CT"]
    MFPT = lap["MFPT"]

    summary = summarize_commute_mfpt(CT, MFPT, prefix=prefix)

    out: Dict[str, object] = {}
    out.update(diff_prefixed)
    out.update(summary)

    if include_matrices:
        out[f"{prefix}CT"] = CT
        out[f"{prefix}MFPT"] = MFPT
        out[f"{prefix}L"] = lap["L"]
        out[f"{prefix}L_plus"] = lap["L_plus"]
        out[f"{prefix}d"] = lap["d"]
        out[f"{prefix}vol"] = lap["vol"]

    return out


# ======================================================================
# UNIVARIATE LOGISTIC PROBES (OPTIONAL BUT USEFUL)
# ======================================================================


@dataclass
class ProbeResult:
    """
    Result of fitting a mechanistic *logistic* probe on a single feature.

    Attributes
    ----------
    feature : str
        Feature name used for the probe (column in X).
    coef : float
        Logistic regression coefficient on standardized feature.
    intercept : float
        Logistic regression intercept.
    metrics : dict
        Basic performance metrics (roc_auc, pr_auc, brier, log_loss, etc.)
    calibration : pd.DataFrame
        Calibration table with columns:
          [bin_lower, bin_upper, bin_center, n_samples, mean_pred, emp_rate]
    n_train : int
        Number of samples used to fit the probe.
    """

    feature: str
    coef: float
    intercept: float
    metrics: Dict[str, float]
    calibration: pd.DataFrame
    n_train: int


def fit_univariate_logit_probe(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    feature_name: str,
    C: float = 1.0,
    max_iter: int = 1000,
    n_bins_calibration: int = 10,
) -> ProbeResult:
    """
    Fit a univariate logistic-probe:

        P(y=1 | f) = sigma(alpha + beta f_std),

    where f_std is the standardized feature (zero mean, unit variance).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with feature_name in columns.
    y : array-like
        Binary labels (0/1), aligned with X.
    feature_name : str
        Name of the feature column to probe.
    C : float
        Inverse regularization strength for logistic regression.
    max_iter : int
        Max iterations for the logistic solver.
    n_bins_calibration : int
        Number of bins for the calibration table.

    Returns
    -------
    ProbeResult
    """
    if feature_name not in X.columns:
        raise ValueError(f"Feature '{feature_name}' not found in X columns.")

    f = X[feature_name].astype(float)
    y_arr = np.asarray(y, dtype=float)

    mask = np.isfinite(f.values) & np.isfinite(y_arr)
    f = f[mask]
    y_arr = y_arr[mask]

    if len(f) == 0:
        empty_cal = _calibration_table(np.array([]), np.array([]), n_bins=10)
        return ProbeResult(
            feature=feature_name,
            coef=np.nan,
            intercept=np.nan,
            metrics=_compute_metrics(np.array([]), np.array([])),
            calibration=empty_cal,
            n_train=0,
        )

    scaler = StandardScaler()
    f_std = scaler.fit_transform(f.values.reshape(-1, 1)).ravel()

    clf = LogisticRegression(
        penalty="l2",
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(f_std.reshape(-1, 1), y_arr)

    p = clf.predict_proba(f_std.reshape(-1, 1))[:, 1]
    metrics = _compute_metrics(y_arr, p)
    calib = _calibration_table(y_arr, p, n_bins=n_bins_calibration)

    coef = float(clf.coef_.ravel()[0])
    intercept = float(clf.intercept_.ravel()[0])

    return ProbeResult(
        feature=feature_name,
        coef=coef,
        intercept=intercept,
        metrics=metrics,
        calibration=calib,
        n_train=len(f),
    )


def fit_univariate_logit_probes(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    features: Sequence[str],
    C: float = 1.0,
    max_iter: int = 1000,
    n_bins_calibration: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, ProbeResult]]:
    """
    Fit a univariate logistic probe for each feature in `features`.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Binary labels aligned with X index.
    features : sequence of str
        List of feature names to probe.
    C : float
        Logistic regression C parameter (inverse regularization).
    max_iter : int
        Logistic max_iter.
    n_bins_calibration : int
        Number of bins for calibration curves.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per feature with:
            ['coef', 'intercept',
             'roc_auc', 'pr_auc', 'brier', 'log_loss',
             'pos_rate', 'n_samples', 'n_train'],
        indexed by feature.
    probes : dict
        Mapping feature_name -> ProbeResult (incl. calibration).
    """
    results: Dict[str, ProbeResult] = {}

    rows = []
    for feat in features:
        pr = fit_univariate_logit_probe(
            X=X,
            y=y,
            feature_name=feat,
            C=C,
            max_iter=max_iter,
            n_bins_calibration=n_bins_calibration,
        )
        results[feat] = pr

        row = {
            "feature": pr.feature,
            "coef": pr.coef,
            "intercept": pr.intercept,
            "n_train": pr.n_train,
        }
        row.update(pr.metrics)
        rows.append(row)

    if not rows:
        return pd.DataFrame(), results

    summary_df = (
        pd.DataFrame(rows)
        .set_index("feature")
        .sort_values(by="roc_auc", ascending=False)
    )
    return summary_df, results
