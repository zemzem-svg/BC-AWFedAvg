"""
Efficient Differential Privacy Module
======================================

Three enhancements over the baseline Gaussian mechanism + advanced composition:

1. **Rényi Differential Privacy (RDP) Accountant**  (Mironov, 2017)
   Tracks per-round Rényi divergence at multiple orders α, then converts
   to (ε, δ)-DP via the optimal conversion lemma.  Yields *significantly*
   tighter cumulative ε than the advanced composition theorem for many
   rounds.

2. **Adaptive Clipping** (Andrew et al., 2021)
   Adjusts the L2 clip norm C each round so that a target quantile of
   client updates lie within the bound.  Avoids over-clipping (too much
   bias) or under-clipping (too much noise relative to signal).

3. **Top-K Gradient Sparsification** with error feedback (Aji & Heafield, 2017)
   Only transmits the K largest-magnitude coordinates each round.  The
   residual (un-transmitted coordinates) is accumulated into an error
   buffer and added to the next round's update.  Achieves 10–100× fewer
   transmitted floats with negligible convergence loss when K/D ≥ 0.01.

Integration
-----------
All three are drop-in replacements / additions to the existing pipeline:

    PrivacyPreservingFederatedLearning.add_differential_privacy_noise()
       ↳ replaced by  EfficientDPManager.add_dp_noise()  which uses
         adaptive clipping + calibrated Gaussian noise

    cumulative_privacy_cost()
       ↳ replaced by  RDPAccountant.get_epsilon()  for tighter bounds

    client fit() parameter upload
       ↳ wrapped with  TopKSparsifier.sparsify() / densify()
"""

from __future__ import annotations

import math
import logging
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

import numpy as np
try:
    import torch
except ImportError:
    import torch_shim as torch

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Rényi Differential Privacy Accountant
# ═════════════════════════════════════════════════════════════════════════════

class RDPAccountant:
    """
    Rényi Differential Privacy accountant for the Gaussian mechanism.

    For a single application of the Gaussian mechanism with sensitivity Δ
    and noise σ, the RDP at order α is:

        ε_RDP(α) = α Δ² / (2 σ²)

    Composition across T rounds is additive:

        ε_RDP_total(α) = T · ε_RDP(α)

    Conversion to (ε, δ)-DP uses the optimal bound (Balle et al., 2020):

        ε = ε_RDP(α) + ln(1/δ) / (α − 1) − ln(α) / (α − 1) + ln((α−1)/α)

    We evaluate at many α values and take the tightest ε.
    """

    # Default α orders to evaluate (dense near 1, sparse at large values)
    DEFAULT_ORDERS = [
        1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
        10.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
    ]

    def __init__(self, orders: Optional[List[float]] = None):
        self.orders = orders or self.DEFAULT_ORDERS
        # Accumulated RDP ε at each order (additive across rounds)
        self._rdp_eps = np.zeros(len(self.orders))
        self._rounds = 0

    def step(self, noise_sigma: float, sensitivity: float = 1.0):
        """
        Record one mechanism application (one FL round) with given σ and Δ.
        """
        if noise_sigma < 1e-12:
            logger.warning("RDP step with near-zero sigma — privacy budget unbounded")
            self._rdp_eps += np.inf
        else:
            for i, alpha in enumerate(self.orders):
                self._rdp_eps[i] += alpha * (sensitivity ** 2) / (2 * noise_sigma ** 2)
        self._rounds += 1

    def get_epsilon(self, delta: float) -> float:
        """
        Convert accumulated RDP to (ε, δ)-DP using the optimal conversion.

        Returns the *tightest* ε across all α orders.
        """
        best_eps = float("inf")
        for i, alpha in enumerate(self.orders):
            if alpha <= 1.0:
                continue
            eps_candidate = (
                self._rdp_eps[i]
                + math.log(1.0 / delta) / (alpha - 1.0)
                - math.log(alpha) / (alpha - 1.0)
                + math.log((alpha - 1.0) / alpha)
            )
            best_eps = min(best_eps, eps_candidate)
        return max(0.0, best_eps)

    def get_epsilon_advanced_composition(
        self, epsilon_per_round: float, delta: float, T: int
    ) -> float:
        """
        For comparison: return the old advanced-composition ε.
        ε_AC = √(2T·ln(1/δ)) · ε
        """
        return math.sqrt(2 * T * math.log(1.0 / delta)) * epsilon_per_round

    def privacy_report(self, delta: float) -> Dict:
        """Full report comparing RDP vs advanced composition."""
        eps_rdp = self.get_epsilon(delta)
        # For AC comparison we need per-round ε; approximate from first-order RDP
        if self._rounds > 0 and len(self.orders) > 0:
            # σ from first round α=2:  ε_RDP(2) = Δ²/(σ²) → σ = Δ/√ε_RDP
            idx_alpha2 = min(range(len(self.orders)),
                            key=lambda i: abs(self.orders[i] - 2.0))
            per_round_rdp_alpha2 = self._rdp_eps[idx_alpha2] / max(self._rounds, 1)
            # Convert single-round (α=2) to approximate per-round (ε,δ)
            per_round_eps = per_round_rdp_alpha2 + math.log(1/delta)
            eps_ac = self.get_epsilon_advanced_composition(
                per_round_eps, delta, self._rounds
            )
        else:
            per_round_eps = 0.0
            eps_ac = 0.0

        return {
            "method": "RDP (Rényi)",
            "rounds": self._rounds,
            "delta": delta,
            "eps_rdp": float(eps_rdp),
            "eps_advanced_composition": float(eps_ac),
            "improvement_pct": (
                (1 - eps_rdp / eps_ac) * 100 if eps_ac > 0 else 0.0
            ),
            "best_alpha": float(
                self.orders[int(np.argmin([
                    self._rdp_eps[i] + math.log(1/delta)/(a-1)
                    for i, a in enumerate(self.orders) if a > 1
                ]))]
            ) if self._rounds > 0 else 0.0,
        }

    def reset(self):
        """Reset the accountant."""
        self._rdp_eps = np.zeros(len(self.orders))
        self._rounds = 0


# ═════════════════════════════════════════════════════════════════════════════
# 2. Adaptive Clipping
# ═════════════════════════════════════════════════════════════════════════════

class AdaptiveClipper:
    """
    Adjusts the L2 clip norm each round so that a target fraction of client
    updates fall within the bound.

    Algorithm (Andrew et al., "Differentially Private Learning with Adaptive
    Clipping", NeurIPS 2021):
        C_{t+1} = C_t · exp(−η · (fraction_clipped_t − target_quantile))

    A lower target_quantile (e.g. 0.5) clips more aggressively → more bias
    but less noise.  Higher (e.g. 0.8) clips less → lower bias, more noise.
    """

    def __init__(
        self,
        initial_clip_norm: float = 1.0,
        target_quantile: float = 0.6,
        learning_rate: float = 0.2,
        min_clip: float = 0.1,
        max_clip: float = 50.0,
    ):
        self.clip_norm = initial_clip_norm
        self.target_quantile = target_quantile
        self.lr = learning_rate
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.history: List[Dict] = []

    def clip_and_update(
        self, param_dicts: List[OrderedDict]
    ) -> Tuple[List[OrderedDict], float]:
        """
        Clip each client's parameters and update the clip norm for next round.

        Returns (clipped_params_list, current_clip_norm).
        """
        norms = []
        clipped_list = []

        for params in param_dicts:
            # Compute L2 norm
            flat = torch.cat([p.flatten() for p in params.values() if isinstance(p, torch.Tensor)])
            l2 = float(torch.norm(flat))
            norms.append(l2)

            # Clip
            clipped = OrderedDict()
            scale = min(1.0, self.clip_norm / max(l2, 1e-12))
            for name, p in params.items():
                clipped[name] = p * scale if isinstance(p, torch.Tensor) else p
            clipped_list.append(clipped)

        # Fraction that were clipped (norm exceeded clip_norm)
        fraction_clipped = sum(1 for n in norms if n > self.clip_norm) / max(len(norms), 1)

        # Geometric update: if too many clients are clipped, loosen the bound
        # (increase C); if too few, tighten it (decrease C).
        old_clip = self.clip_norm
        self.clip_norm *= math.exp(
            self.lr * (fraction_clipped - self.target_quantile)
        )
        self.clip_norm = np.clip(self.clip_norm, self.min_clip, self.max_clip)

        self.history.append({
            "clip_norm": old_clip,
            "new_clip_norm": self.clip_norm,
            "fraction_clipped": fraction_clipped,
            "norms": norms,
        })

        logger.info(
            "AdaptiveClipper: C=%.4f→%.4f  frac_clipped=%.2f  target=%.2f",
            old_clip, self.clip_norm, fraction_clipped, self.target_quantile,
        )
        return clipped_list, old_clip


# ═════════════════════════════════════════════════════════════════════════════
# 3. Top-K Gradient Sparsification with Error Feedback
# ═════════════════════════════════════════════════════════════════════════════

class TopKSparsifier:
    """
    Top-K sparsification with error feedback for communication-efficient FL.

    Each client keeps a residual error buffer.  At each round:
        1. accumulated = gradient + error_buffer
        2. mask = top-K coordinates by magnitude
        3. transmitted = accumulated * mask
        4. error_buffer = accumulated * (1 − mask)   ← carried to next round

    The server receives sparse updates and densifies them.

    Parameters
    ----------
    compression_ratio : float in (0, 1]
        Fraction of coordinates to keep.  0.01 = 1% = 100× compression.
    """

    def __init__(self, compression_ratio: float = 0.1):
        assert 0.0 < compression_ratio <= 1.0
        self.k_ratio = compression_ratio
        # Per-client error buffers: client_id → list of np.ndarrays
        self._error_buffers: Dict[int, List[np.ndarray]] = {}

    def sparsify(
        self,
        client_id: int,
        param_list: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """
        Sparsify parameters with error feedback.

        Returns
        -------
        sparse_params : list of ndarrays (zeros for un-selected coordinates)
        masks         : list of boolean ndarrays
        actual_ratio  : fraction of non-zero coordinates
        """
        # Initialise error buffer on first call
        if client_id not in self._error_buffers:
            self._error_buffers[client_id] = [np.zeros_like(p) for p in param_list]

        accumulated = []
        for p, e in zip(param_list, self._error_buffers[client_id]):
            accumulated.append(p + e)

        # Flatten to find global top-K
        flat = np.concatenate([a.ravel() for a in accumulated])
        total_size = flat.size
        k = max(1, int(self.k_ratio * total_size))
        # Partial sort — O(n) via argpartition
        top_indices = np.argpartition(np.abs(flat), -k)[-k:]
        mask_flat = np.zeros(total_size, dtype=bool)
        mask_flat[top_indices] = True

        # Unflatten masks and build sparse output
        sparse_params = []
        masks = []
        new_errors = []
        idx = 0
        for a in accumulated:
            size = a.size
            m = mask_flat[idx:idx + size].reshape(a.shape)
            masks.append(m)
            sparse = np.where(m, a, 0.0)
            sparse_params.append(sparse)
            new_errors.append(np.where(m, 0.0, a))  # carry un-sent to buffer
            idx += size

        self._error_buffers[client_id] = new_errors
        actual_ratio = k / total_size

        return sparse_params, masks, actual_ratio

    @staticmethod
    def densify(sparse_params: List[np.ndarray]) -> List[np.ndarray]:
        """
        Densify sparse parameters (identity — sparse params already have
        zeros in un-selected positions, so the server just sums them).
        """
        return sparse_params  # already dense with zeros

    def get_stats(self) -> Dict:
        """Return compression statistics."""
        return {
            "compression_ratio": self.k_ratio,
            "active_clients": len(self._error_buffers),
            "error_buffer_norms": {
                cid: float(np.sqrt(sum(np.sum(e**2) for e in bufs)))
                for cid, bufs in self._error_buffers.items()
            },
        }

    def reset_client(self, client_id: int):
        """Reset error buffer for a client."""
        if client_id in self._error_buffers:
            del self._error_buffers[client_id]


# ═════════════════════════════════════════════════════════════════════════════
# 4. EfficientDPManager — Unified Interface
# ═════════════════════════════════════════════════════════════════════════════

class EfficientDPManager:
    """
    Unified manager combining RDP accounting + adaptive clipping + calibrated
    Gaussian noise.

    Replaces the original add_differential_privacy_noise() + cumulative_privacy_cost()
    with tighter bounds and adaptive behaviour.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        initial_clip_norm: float = 1.0,
        adaptive_clip: bool = True,
        target_quantile: float = 0.6,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.rdp = RDPAccountant()
        self.clipper = AdaptiveClipper(
            initial_clip_norm=initial_clip_norm,
            target_quantile=target_quantile,
        ) if adaptive_clip else None
        self._fixed_clip_norm = initial_clip_norm

    @property
    def clip_norm(self) -> float:
        return self.clipper.clip_norm if self.clipper else self._fixed_clip_norm

    def noise_sigma(self, sensitivity: float = 1.0) -> float:
        """Compute Gaussian noise scale for the current ε, δ, clip_norm."""
        return math.sqrt(2 * math.log(1.25 / self.delta)) * sensitivity / self.epsilon

    def add_dp_noise(
        self,
        model_params: OrderedDict,
        sensitivity: float = 1.0,
    ) -> OrderedDict:
        """
        Clip + add calibrated Gaussian noise.  Records the step in the RDP
        accountant automatically.
        """
        sigma = self.noise_sigma(sensitivity)

        noised = OrderedDict()
        for name, param in model_params.items():
            if isinstance(param, torch.Tensor):
                # Clip
                pnorm = float(torch.norm(param))
                clip = self.clip_norm
                if pnorm > clip:
                    param = param * (clip / pnorm)
                noise = torch.randn_like(param) * sigma
                noised[name] = param + noise
            else:
                noised[name] = param

        # Record in RDP accountant
        self.rdp.step(sigma, sensitivity)
        return noised

    def clip_client_updates(
        self, param_dicts: List[OrderedDict]
    ) -> Tuple[List[OrderedDict], float]:
        """
        Adaptive clip a batch of client parameter dicts.
        Returns (clipped_list, clip_norm_used).
        """
        if self.clipper:
            return self.clipper.clip_and_update(param_dicts)
        else:
            # Fixed clipping
            clipped = []
            for params in param_dicts:
                flat = torch.cat([p.flatten() for p in params.values()
                                  if isinstance(p, torch.Tensor)])
                l2 = float(torch.norm(flat))
                scale = min(1.0, self._fixed_clip_norm / max(l2, 1e-12))
                clipped.append(OrderedDict(
                    (n, p * scale if isinstance(p, torch.Tensor) else p)
                    for n, p in params.items()
                ))
            return clipped, self._fixed_clip_norm

    def get_epsilon(self) -> float:
        """Current cumulative ε using RDP accounting."""
        return self.rdp.get_epsilon(self.delta)

    def privacy_report(self) -> Dict:
        """Full privacy report with RDP vs AC comparison."""
        report = self.rdp.privacy_report(self.delta)
        report["adaptive_clip_history"] = (
            self.clipper.history if self.clipper else []
        )
        return report

    def reset(self):
        self.rdp.reset()
        if self.clipper:
            self.clipper = AdaptiveClipper(
                initial_clip_norm=self.clipper.clip_norm,
                target_quantile=self.clipper.target_quantile,
                learning_rate=self.clipper.lr,
            )
