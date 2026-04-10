"""
Robustness Module — Byzantine-Tolerant Aggregation & Anomaly Detection
========================================================================

Provides three defences that compose with AWFedAvg:

1. **Coordinate-wise Trimmed Mean** (Yin et al., ICML 2018)
   Removes the top/bottom β fraction of each parameter coordinate before
   averaging.  Tolerates up to β < 0.5 fraction of Byzantine clients.

2. **Cosine-Similarity Filtering** (pre-aggregation)
   Computes pairwise cosine similarity of flattened updates; flags any
   client whose median similarity to others falls below a z-score
   threshold.  Flagged clients are down-weighted (not hard-dropped, to
   avoid colluding minorities silencing honest outliers).

3. **Norm-Bounding** (Sun et al., 2019)
   Clips each client's update to a maximum L2 norm before aggregation.
   Prevents a single malicious client from dominating the aggregate.

4. **KRUM / Multi-Krum** (Blanchard et al., NeurIPS 2017)
   Selects the update(s) closest to the majority, discarding outliers.

Integration
-----------
These are *composable filters* applied **before** AWFedAvg's adaptive
weighted averaging:

    results  →  norm_bound  →  cosine_filter  →  trimmed_mean / krum
               (per-client)   (flag outliers)     (robust aggregation)

The ``RobustAggregator`` class wraps all of these and exposes a single
``robust_aggregate()`` method.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

import numpy as np
try:
    import torch
except ImportError:
    import torch_shim as torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: flatten / unflatten parameter arrays
# ─────────────────────────────────────────────────────────────────────────────

def _flatten(param_list: List[np.ndarray]) -> np.ndarray:
    """Flatten a list of ndarrays into a single 1-D vector."""
    return np.concatenate([p.ravel() for p in param_list])


def _unflatten(flat: np.ndarray, shapes: List[tuple]) -> List[np.ndarray]:
    """Reverse of _flatten, restoring original shapes."""
    result, idx = [], 0
    for s in shapes:
        size = int(np.prod(s))
        result.append(flat[idx:idx + size].reshape(s))
        idx += size
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. Norm-Bounding
# ─────────────────────────────────────────────────────────────────────────────

def norm_bound(param_list: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
    """
    Clip the L2 norm of a flattened parameter vector to *max_norm*.

    If ||params||₂ ≤ max_norm the parameters are returned unchanged.
    Otherwise they are scaled down:  params ← params × (max_norm / ||params||₂)
    """
    flat = _flatten(param_list)
    l2 = np.linalg.norm(flat)
    if l2 > max_norm:
        flat = flat * (max_norm / l2)
    shapes = [p.shape for p in param_list]
    return _unflatten(flat, shapes)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cosine-Similarity Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity_matrix(updates: List[np.ndarray]) -> np.ndarray:
    """
    Compute the K×K pairwise cosine similarity matrix for K flattened updates.
    """
    K = len(updates)
    mat = np.zeros((K, K))
    norms = [np.linalg.norm(u) for u in updates]
    for i in range(K):
        for j in range(i, K):
            if norms[i] < 1e-12 or norms[j] < 1e-12:
                sim = 0.0
            else:
                sim = float(np.dot(updates[i], updates[j]) / (norms[i] * norms[j]))
            mat[i, j] = sim
            mat[j, i] = sim
    return mat


def detect_anomalies(
    client_params: List[List[np.ndarray]],
    z_threshold: float = 2.0,
) -> Tuple[List[bool], np.ndarray]:
    """
    Flag clients whose median cosine similarity to others is anomalously low.

    Parameters
    ----------
    client_params : list of parameter-lists (one per client)
    z_threshold   : number of standard deviations below mean to trigger flag

    Returns
    -------
    is_anomalous : list of bool (True = flagged)
    median_sims  : per-client median similarity score
    """
    flat_updates = [_flatten(p) for p in client_params]
    sim_mat = cosine_similarity_matrix(flat_updates)

    K = len(flat_updates)
    median_sims = np.zeros(K)
    for i in range(K):
        others = [sim_mat[i, j] for j in range(K) if j != i]
        median_sims[i] = np.median(others) if others else 1.0

    mu = np.mean(median_sims)
    sigma = np.std(median_sims) + 1e-12
    z_scores = (median_sims - mu) / sigma

    is_anomalous = [z < -z_threshold for z in z_scores]
    if any(is_anomalous):
        flagged = [i for i, a in enumerate(is_anomalous) if a]
        logger.warning(
            "Anomaly detection flagged clients %s  "
            "(median_sim z-scores: %s)",
            flagged,
            [f"{z_scores[i]:+.2f}" for i in flagged],
        )
    return is_anomalous, median_sims


# ─────────────────────────────────────────────────────────────────────────────
# 3. Coordinate-wise Trimmed Mean
# ─────────────────────────────────────────────────────────────────────────────

def trimmed_mean(
    param_lists: List[List[np.ndarray]],
    weights: np.ndarray,
    beta: float = 0.1,
) -> List[np.ndarray]:
    """
    Coordinate-wise β-trimmed weighted mean.

    For each coordinate, sort the K client values, remove the bottom β·K
    and top β·K entries, then compute the weighted mean of the remaining.

    Parameters
    ----------
    param_lists : K lists of parameter arrays
    weights     : shape (K,) — AWFedAvg adaptive weights (will be re-normalised
                  over the kept subset for each coordinate)
    beta        : trim fraction per tail  (0.1 = remove 10% top + 10% bottom)
    """
    K = len(param_lists)
    if K < 3:
        # Not enough clients to trim — fall back to weighted mean
        result = [np.zeros_like(p) for p in param_lists[0]]
        for params, w in zip(param_lists, weights):
            for i, p in enumerate(params):
                result[i] += w * p
        return result

    trim_count = max(1, int(np.floor(beta * K)))
    # Cannot trim more than K-1 from each side
    trim_count = min(trim_count, (K - 1) // 2)

    n_params = len(param_lists[0])
    result = []
    for layer_idx in range(n_params):
        shape = param_lists[0][layer_idx].shape
        stacked = np.stack([param_lists[k][layer_idx].ravel()
                            for k in range(K)], axis=0)  # (K, D)
        sorted_indices = np.argsort(stacked, axis=0)       # (K, D)

        # Build per-coordinate mask: 1 if kept, 0 if trimmed
        mask = np.ones_like(stacked, dtype=float)
        for d in range(stacked.shape[1]):
            for t in range(trim_count):
                mask[sorted_indices[t, d], d] = 0.0           # bottom trim
                mask[sorted_indices[-(t + 1), d], d] = 0.0    # top trim

        # Weighted mean over kept entries
        w_arr = np.array(weights).reshape(K, 1)
        masked_weights = mask * w_arr                          # (K, D)
        denom = np.sum(masked_weights, axis=0, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        normed_weights = masked_weights / denom
        agg = np.sum(normed_weights * stacked, axis=0).reshape(shape)
        result.append(agg)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-Krum Selection
# ─────────────────────────────────────────────────────────────────────────────

def multi_krum(
    param_lists: List[List[np.ndarray]],
    num_byzantine: int = 1,
    num_select: int = None,
) -> List[int]:
    """
    Multi-Krum: select the *num_select* clients whose updates are closest
    to the majority (measured by sum of distances to nearest neighbours).

    Parameters
    ----------
    param_lists   : K lists of parameter arrays
    num_byzantine : assumed max number of Byzantine clients (f)
    num_select    : how many clients to keep (default K - f)

    Returns
    -------
    selected_indices : indices of selected (trustworthy) clients
    """
    K = len(param_lists)
    f = num_byzantine
    m = num_select if num_select else max(1, K - f)
    m = min(m, K)

    flat = [_flatten(p) for p in param_lists]
    # Pairwise squared L2 distances
    dists = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            d = np.linalg.norm(flat[i] - flat[j]) ** 2
            dists[i, j] = d
            dists[j, i] = d

    # For each client, sum distances to its K-f-1 nearest neighbours
    n_neighbours = K - f - 1
    if n_neighbours < 1:
        n_neighbours = 1
    scores = np.zeros(K)
    for i in range(K):
        sorted_d = np.sort(dists[i])  # includes dist to self (0)
        scores[i] = np.sum(sorted_d[1:1 + n_neighbours])

    selected = np.argsort(scores)[:m].tolist()
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 5. RobustAggregator — composable wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RobustAggregator:
    """
    Orchestrates the full robust aggregation pipeline.

    Pipeline order:
        1. Norm-bound each client update
        2. Cosine-similarity anomaly detection → soft down-weight flagged clients
        3. Robust aggregation (trimmed_mean *or* krum + weighted average)

    Parameters
    ----------
    method           : 'trimmed_mean' | 'krum' | 'weighted_mean' (pass-through)
    max_norm         : L2 norm bound per client (None = disabled)
    anomaly_z        : z-score threshold for cosine anomaly detection (None = disabled)
    anomaly_penalty  : multiplicative weight penalty for flagged clients (e.g. 0.1)
    trim_beta        : trim fraction for trimmed_mean
    krum_byzantine   : assumed max Byzantine count for multi-krum
    """

    def __init__(
        self,
        method: str = "trimmed_mean",
        max_norm: Optional[float] = 10.0,
        anomaly_z: Optional[float] = 2.0,
        anomaly_penalty: float = 0.1,
        trim_beta: float = 0.1,
        krum_byzantine: int = 1,
    ):
        assert method in ("trimmed_mean", "krum", "weighted_mean")
        self.method = method
        self.max_norm = max_norm
        self.anomaly_z = anomaly_z
        self.anomaly_penalty = anomaly_penalty
        self.trim_beta = trim_beta
        self.krum_byzantine = krum_byzantine

        # Diagnostics
        self.last_anomalies: List[bool] = []
        self.last_median_sims: np.ndarray = np.array([])
        self.last_krum_selected: List[int] = []

    def robust_aggregate(
        self,
        client_params: List[List[np.ndarray]],
        weights: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Run the full robust pipeline and return (aggregated_params, adjusted_weights).

        Parameters
        ----------
        client_params : list of K parameter-lists
        weights       : shape (K,) adaptive weights from AWFedAvg

        Returns
        -------
        aggregated : list of ndarrays (aggregated model parameters)
        adj_weights: shape (K,) — possibly adjusted weights for logging
        """
        K = len(client_params)
        adj_weights = weights.copy()

        # ── Step 1: Norm bounding ─────────────────────────────────────────
        if self.max_norm is not None:
            client_params = [norm_bound(p, self.max_norm) for p in client_params]

        # ── Step 2: Cosine anomaly detection ──────────────────────────────
        if self.anomaly_z is not None and K >= 3:
            is_anom, med_sims = detect_anomalies(client_params, self.anomaly_z)
            self.last_anomalies = is_anom
            self.last_median_sims = med_sims
            for i, flagged in enumerate(is_anom):
                if flagged:
                    adj_weights[i] *= self.anomaly_penalty
            # Re-normalise
            w_sum = np.sum(adj_weights)
            if w_sum > 1e-12:
                adj_weights = adj_weights / w_sum
            else:
                adj_weights = np.ones(K) / K

        # ── Step 3: Robust aggregation ────────────────────────────────────
        if self.method == "trimmed_mean":
            aggregated = trimmed_mean(client_params, adj_weights, self.trim_beta)

        elif self.method == "krum":
            selected = multi_krum(client_params, self.krum_byzantine)
            self.last_krum_selected = selected
            # Weighted mean over selected subset
            sel_params = [client_params[i] for i in selected]
            sel_w = np.array([adj_weights[i] for i in selected])
            sel_w = sel_w / (np.sum(sel_w) + 1e-12)
            aggregated = [np.zeros_like(p) for p in sel_params[0]]
            for params, w in zip(sel_params, sel_w):
                for idx, p in enumerate(params):
                    aggregated[idx] += w * p

        else:  # weighted_mean — vanilla pass-through
            aggregated = [np.zeros_like(p) for p in client_params[0]]
            for params, w in zip(client_params, adj_weights):
                for idx, p in enumerate(params):
                    aggregated[idx] += w * p

        return aggregated, adj_weights

    def get_diagnostics(self) -> Dict:
        """Return diagnostics from the last aggregation round."""
        return {
            "anomalies_flagged": sum(self.last_anomalies) if self.last_anomalies else 0,
            "anomaly_mask": self.last_anomalies,
            "median_cosine_sims": self.last_median_sims.tolist()
                if self.last_median_sims.size else [],
            "krum_selected": self.last_krum_selected,
            "method": self.method,
        }
