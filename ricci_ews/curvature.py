"""
curvature.py

Ollivier–Ricci (ORC) and Forman–Ricci (FRC) curvature on weighted graphs.

This module operates on GraphData objects

    GraphData(
        W: np.ndarray  # edge weights w_ij >= 0
        L: np.ndarray  # edge lengths ℓ_ij > 0 on edges, np.inf / 0 on non-edges
        adjacency: np.ndarray  # 0/1 adjacency
        nodes: List[str]       # node labels
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import linprog

from . import config
from .graphs_correlation import GraphData


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def _get_default_orc_alpha() -> float:
    """
    Default lazy-walk parameter α for ORC.

    Uses config.ORC_ALPHA if present, otherwise falls back to 0.5.
    """
    return float(getattr(config, "ORC_ALPHA", 0.5))


def _all_pairs_shortest_paths(L: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Compute all-pairs shortest-path distances for a weighted graph.

    Parameters
    ----------
    L : np.ndarray
        Edge length matrix (N x N). Only entries where A[i,j] == 1 are used
        as edge lengths; others are ignored.
    A : np.ndarray
        Adjacency (N x N), 0/1.

    Returns
    -------
    dist : np.ndarray
        Matrix of shortest-path distances (N x N), with dist[i,i] = 0 and
        np.inf for unreachable nodes (should not happen if graph is connected).
    """
    N = L.shape[0]
    dist = np.full((N, N), np.inf, dtype=float)

    for s in range(N):
        # Dijkstra with O(N^2) complexity (fine for N <= few hundred)
        visited = np.zeros(N, dtype=bool)
        dist_s = np.full(N, np.inf, dtype=float)
        dist_s[s] = 0.0

        for _ in range(N):
            # Pick unvisited node with smallest distance
            u = -1
            best = np.inf
            for v in range(N):
                if not visited[v] and dist_s[v] < best:
                    best = dist_s[v]
                    u = v
            if u == -1 or best == np.inf:
                break
            visited[u] = True

            # Relax neighbors
            neighbors = np.where(A[u] > 0)[0]
            for v in neighbors:
                alt = dist_s[u] + L[u, v]
                if alt < dist_s[v]:
                    dist_s[v] = alt

        dist[s] = dist_s

    return dist


def _wasserstein_1(mu_i: np.ndarray, mu_j: np.ndarray, C: np.ndarray) -> float:
    """
    Compute the 1-Wasserstein distance W1(mu_i, mu_j) using linear programming.

    Parameters
    ----------
    mu_i : np.ndarray
        Probability vector on a finite support (length M).
    mu_j : np.ndarray
        Probability vector on the same support (length M).
    C : np.ndarray
        Cost matrix (M x M), C[a,b] = d(x_a, y_b).

    Returns
    -------
    W1 : float
        Earth-mover distance between mu_i and mu_j.
    """
    mu_i = np.asarray(mu_i, dtype=float)
    mu_j = np.asarray(mu_j, dtype=float)
    C = np.asarray(C, dtype=float)

    M = mu_i.shape[0]
    assert mu_j.shape[0] == M
    assert C.shape == (M, M)

    # Flatten cost
    c = C.reshape(-1)

    # Equality constraints: row sums = mu_i, column sums = mu_j
    # Variables: π[a,b] with a,b in {0,...,M-1}, flattened row-major.

    num_vars = M * M
    A_eq = np.zeros((2 * M, num_vars), dtype=float)
    b_eq = np.concatenate([mu_i, mu_j])

    # Row constraints: for each a, sum_b π[a,b] = mu_i[a]
    for a in range(M):
        A_eq[a, a * M:(a + 1) * M] = 1.0

    # Column constraints: for each b, sum_a π[a,b] = mu_j[b]
    for b in range(M):
        A_eq[M + b, b::M] = 1.0

    bounds = [(0.0, None)] * num_vars

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        # Fall back: in the worst case, treat W1 as distance between means.
        # This is conservative (tends to push κ_ij towards 0).
        return float((mu_i @ np.arange(M) - mu_j @ np.arange(M)))

    return float(res.fun)


# ----------------------------------------------------------------------
# Forman–Ricci curvature
# ----------------------------------------------------------------------


def forman_ricci_edge_curvature(G: GraphData) -> np.ndarray:
    """
    Compute Forman–Ricci curvature on edges of a weighted graph.

    Parameters
    ----------
    G : GraphData
        Graph structure with weights W (N x N) and adjacency (N x N).

    Returns
    -------
    F : np.ndarray
        Edge-level FRC matrix (N x N), symmetric, with F[i,j] = 0 where
        there is no edge.
    """
    W = np.asarray(G.W, dtype=float)
    A = np.asarray(G.adjacency, dtype=int)
    N = W.shape[0]

    # Node strengths: w_i = Σ_{k~i} w_ik
    strengths = (W * A).sum(axis=1)

    F = np.zeros_like(W, dtype=float)

    for i in range(N):
        neigh_i = np.where(A[i] > 0)[0]
        w_i = strengths[i]
        for j in neigh_i:
            if j <= i:
                # We will fill each undirected edge once (i < j).
                continue
            w_ij = W[i, j]
            if w_ij <= 0:
                continue

            neigh_j = np.where(A[j] > 0)[0]
            w_j = strengths[j]

            # First two terms: w_i/w_ij + w_j/w_ij
            term_main = w_i / w_ij + w_j / w_ij

            # Sum over k ~ i, k != j
            sum_i = 0.0
            for k in neigh_i:
                if k == j:
                    continue
                w_ik = W[i, k]
                if w_ik <= 0:
                    continue
                sum_i += w_i / np.sqrt(w_ij * w_ik)

            # Sum over ℓ ~ j, ℓ != i
            sum_j = 0.0
            for ell in neigh_j:
                if ell == i:
                    continue
                w_jell = W[j, ell]
                if w_jell <= 0:
                    continue
                sum_j += w_j / np.sqrt(w_ij * w_jell)

            F_ij = w_ij * (term_main - sum_i - sum_j)
            F[i, j] = F_ij
            F[j, i] = F_ij

    return F


# ----------------------------------------------------------------------
# Ollivier–Ricci curvature
# ----------------------------------------------------------------------


def ollivier_ricci_edge_curvature(
    G: GraphData,
    alpha: float | None = None,
    dist_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute Ollivier–Ricci curvature κ_ij on edges of a weighted graph.

    Parameters
    ----------
    G : GraphData
        Graph structure with weights W, lengths L, adjacency A.
    alpha : float, optional
        Lazy-walk parameter α in [0.5, 0.9]; default from config.ORC_ALPHA
        or 0.5 if not set.
    dist_matrix : np.ndarray, optional
        Pre-computed all-pairs shortest-path distances (N x N).
        If None, it will be computed from G.L and G.adjacency.

    Returns
    -------
    kappa : np.ndarray
        Edge-level ORC matrix (N x N), symmetric, with kappa[i,j] = 0 where
        there is no edge.
    """
    W = np.asarray(G.W, dtype=float)
    L = np.asarray(G.L, dtype=float)
    A = np.asarray(G.adjacency, dtype=int)
    N = W.shape[0]

    if alpha is None:
        alpha = _get_default_orc_alpha()

    # Pre-compute all-pairs shortest paths if not provided
    if dist_matrix is None:
        dist = _all_pairs_shortest_paths(L, A)
    else:
        dist = np.asarray(dist_matrix, dtype=float)

    # Precompute node strengths and transition probabilities
    strengths = (W * A).sum(axis=1)

    kappa = np.zeros_like(W, dtype=float)

    for i in range(N):
        neigh_i = np.where(A[i] > 0)[0]
        w_i = strengths[i]
        # Transition probabilities from i
        p_i = np.zeros(N, dtype=float)
        if w_i > 0:
            p_i[neigh_i] = W[i, neigh_i] / w_i

        for j in neigh_i:
            if j <= i:
                continue

            # Edge length d(i,j) is just the shortest path from i to j
            d_ij = dist[i, j]
            if not np.isfinite(d_ij) or d_ij <= 0:
                # Should not happen if the graph is connected and lengths > 0
                continue

            neigh_j = np.where(A[j] > 0)[0]
            w_j = strengths[j]
            p_j = np.zeros(N, dtype=float)
            if w_j > 0:
                p_j[neigh_j] = W[j, neigh_j] / w_j

            # Support S = {i} ∪ N(i) ∪ {j} ∪ N(j)
            support_nodes = np.unique(
                np.concatenate([[i, j], neigh_i, neigh_j])
            )
            M = support_nodes.shape[0]

            # μ_i and μ_j restricted to S
            mu_i = np.zeros(M, dtype=float)
            mu_j = np.zeros(M, dtype=float)
            for idx, node in enumerate(support_nodes):
                if node == i:
                    mu_i[idx] += 1.0 - alpha
                if node == j:
                    mu_j[idx] += 1.0 - alpha
                # neighbor mass
                mu_i[idx] += alpha * p_i[node]
                mu_j[idx] += alpha * p_j[node]

            # Normalize numerically (avoid small drift)
            if mu_i.sum() > 0:
                mu_i /= mu_i.sum()
            if mu_j.sum() > 0:
                mu_j /= mu_j.sum()

            # Cost matrix on S using all-pairs distances
            C = dist[np.ix_(support_nodes, support_nodes)]

            # Compute W1(μ_i, μ_j)
            W1_ij = _wasserstein_1(mu_i, mu_j, C)

            # Edge curvature
            k_ij = 1.0 - (W1_ij / d_ij)
            kappa[i, j] = k_ij
            kappa[j, i] = k_ij

    return kappa


# ----------------------------------------------------------------------
# Node-level curvature
# ----------------------------------------------------------------------


@dataclass
class CurvatureResult:
    """
    Container for curvature on a single graph.

    Attributes
    ----------
    edge_orc : np.ndarray
        Edge-level Ollivier–Ricci curvature (N x N).
    edge_frc : np.ndarray
        Edge-level Forman–Ricci curvature (N x N).
    node_orc : np.ndarray
        Node-level ORC (length N), computed as incident-edge averages.
    node_frc : np.ndarray
        Node-level FRC (length N), computed as incident-edge averages.
    """
    edge_orc: np.ndarray
    edge_frc: np.ndarray
    node_orc: np.ndarray
    node_frc: np.ndarray


def _vertex_curvature_from_edges(edge_curv: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Compute node-level curvature as incident-edge averages.

    Parameters
    ----------
    edge_curv : np.ndarray
        Edge-level curvature (N x N), symmetric.
    A : np.ndarray
        Adjacency (N x N), 0/1.

    Returns
    -------
    node_curv : np.ndarray
        Node-level curvature (length N).
    """
    N = edge_curv.shape[0]
    node_curv = np.zeros(N, dtype=float)

    for i in range(N):
        neigh = np.where(A[i] > 0)[0]
        deg_i = len(neigh)
        if deg_i == 0:
            node_curv[i] = 0.0
        else:
            node_curv[i] = float(edge_curv[i, neigh].mean())

    return node_curv


def compute_curvature_for_graph(
    G: GraphData,
    alpha: float | None = None,
    dist_matrix: np.ndarray | None = None,
) -> CurvatureResult:
    """
    Convenience wrapper to compute both ORC and FRC for a graph,
    including node-level curvature.

    Parameters
    ----------
    G : GraphData
        Graph structure.
    alpha : float, optional
        ORC lazy-walk parameter α; default from config or 0.5.
    dist_matrix : np.ndarray, optional
        Pre-computed all-pairs distance matrix (N x N); if None, computed
        on the fly from G.L and G.adjacency.

    Returns
    -------
    CurvatureResult
        Edge and node curvature (ORC and FRC).
    """
    A = np.asarray(G.adjacency, dtype=int)

    # Edge-level curvature
    edge_frc = forman_ricci_edge_curvature(G)
    edge_orc = ollivier_ricci_edge_curvature(G, alpha=alpha, dist_matrix=dist_matrix)

    # Node-level curvature
    node_frc = _vertex_curvature_from_edges(edge_frc, A)
    node_orc = _vertex_curvature_from_edges(edge_orc, A)

    return CurvatureResult(
        edge_orc=edge_orc,
        edge_frc=edge_frc,
        node_orc=node_orc,
        node_frc=node_frc,
    )
