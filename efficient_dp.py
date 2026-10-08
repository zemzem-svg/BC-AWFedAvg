"""
Privacy utilities for BC-AWFedAvg
================================

Corrected implementation of:
    1. Gaussian DP with RDP accounting
    2. Adaptive clipping utilities
    3. Top-K update sparsification with error feedback

Important protocol distinction
------------------------------
BC-AWFedAvg clips and protects a CLIENT UPDATE, not the complete model:

    delta_k = theta_k - theta_global
    delta_hat_k = Clip(delta_k, C) + N(0, sigma^2 I)

The global model itself must not be clipped to C. Clipping the complete
parameter vector would generally destroy the learned model scale.

For the two-phase weighted secure-aggregation protocol, the recommended
order is:

    local update
      -> client-update clipping
      -> Gaussian DP noise
      -> aggregation weight w_k
      -> pairwise masking
      -> server sum

Adaptive clipping note
----------------------
A global target-quantile update of C requires cross-client update-norm
information. That information cannot simply be collected in plaintext in a
privacy-preserving protocol. Therefore the BC-AWFedAvg integration keeps the
thesis' fixed C=1 configuration by default. AdaptiveClipper is provided as a
separate utility for experiments where a protected norm-estimation mechanism
is available.

Top-K note
----------
Top-K is applied to updates/contributions, not absolute model parameters.
Error feedback is persistent per client. The sparse tensor remains dense in
memory with zeros on non-selected coordinates so it can still be summed after
secure masking.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

TensorDict = OrderedDict[str, torch.Tensor]


# ============================================================================
# 1. Rényi Differential Privacy Accountant
# ============================================================================


class RDPAccountant:
    """RDP accountant for Gaussian mechanisms.

    For Gaussian noise with standard deviation ``sigma`` and L2 sensitivity
    ``Delta``, the Gaussian mechanism has RDP at order alpha > 1:

        eps_RDP(alpha) = alpha * Delta^2 / (2 * sigma^2)

    RDP composes additively across rounds. The standard conversion used here
    is the valid bound

        eps_(epsilon,delta) <= eps_RDP(alpha)
                                + log(1/delta)/(alpha - 1)

    evaluated over the configured orders.
    """

    DEFAULT_ORDERS = [
        1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
        10.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
    ]

    def __init__(self, orders: Optional[Sequence[float]] = None):
        raw_orders = list(orders) if orders is not None else list(self.DEFAULT_ORDERS)
        self.orders = [float(a) for a in raw_orders if float(a) > 1.0]
        if not self.orders:
            raise ValueError("At least one RDP order alpha > 1 is required.")
        self._rdp_eps = np.zeros(len(self.orders), dtype=np.float64)
        self._rounds = 0
        self._round_records: List[Dict[str, float]] = []

    @property
    def rounds(self) -> int:
        return self._rounds

    def step(self, noise_sigma: float, sensitivity: float = 1.0) -> None:
        """Record one Gaussian mechanism application."""
        sigma = float(noise_sigma)
        delta_sens = float(sensitivity)
        if sigma <= 0.0:
            logger.warning("RDP step with non-positive sigma: privacy is unbounded")
            self._rdp_eps[:] = np.inf
        else:
            increment = np.array(
                [alpha * (delta_sens ** 2) / (2.0 * sigma ** 2)
                 for alpha in self.orders],
                dtype=np.float64,
            )
            self._rdp_eps += increment
        self._rounds += 1
        self._round_records.append({
            "sigma": sigma,
            "sensitivity": delta_sens,
        })

    def epsilon_by_order(self, delta: float) -> Dict[float, float]:
        """Return converted epsilon for every configured RDP order."""
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must satisfy 0 < delta < 1.")
        return {
            alpha: float(
                self._rdp_eps[i] + math.log(1.0 / delta) / (alpha - 1.0)
            )
            for i, alpha in enumerate(self.orders)
        }

    def get_epsilon(self, delta: float) -> float:
        """Return the tightest converted cumulative epsilon."""
        values = self.epsilon_by_order(delta)
        if not values:
            return 0.0
        best = min(values.values())
        return max(0.0, float(best))

    def best_alpha(self, delta: float) -> float:
        """Return the RDP order producing the tightest conversion."""
        values = self.epsilon_by_order(delta)
        return float(min(values, key=values.get))

    @staticmethod
    def advanced_composition_epsilon(
        epsilon_per_round: float,
        delta_prime: float,
        rounds: int,
    ) -> float:
        """Standard advanced-composition upper bound for comparison.

        This is a comparison utility only. It assumes every round is
        (epsilon_per_round, delta_i)-DP and uses an additional delta_prime.
        The per-round deltas are not included in the returned epsilon.
        """
        eps = float(epsilon_per_round)
        dp = float(delta_prime)
        t = int(rounds)
        if eps < 0.0 or dp <= 0.0 or dp >= 1.0 or t < 0:
            raise ValueError("Invalid advanced-composition arguments.")
        if t == 0 or eps == 0.0:
            return 0.0
        return float(
            math.sqrt(2.0 * t * math.log(1.0 / dp)) * eps
            + t * eps * (math.exp(eps) - 1.0)
        )

    def privacy_report(
        self,
        delta: float,
        nominal_epsilon_per_round: Optional[float] = None,
        advanced_comp_delta_prime: Optional[float] = None,
    ) -> Dict:
        """Return an auditable RDP report.

        ``nominal_epsilon_per_round`` is optional. If supplied, a standard
        advanced-composition comparison is included. No per-round epsilon is
        inferred from an RDP order, avoiding the misleading reverse-calibration
        used by the old implementation.
        """
        eps_rdp = self.get_epsilon(delta)
        best_alpha = self.best_alpha(delta) if self._rounds else 0.0

        report = {
            "method": "RDP (Renyi)",
            "rounds": int(self._rounds),
            "delta": float(delta),
            "eps_rdp": float(eps_rdp),
            "best_alpha": float(best_alpha),
            "rdp_orders": list(self.orders),
            "per_order_epsilon": self.epsilon_by_order(delta) if self._rounds else {},
        }

        if nominal_epsilon_per_round is not None and self._rounds:
            dp_prime = (
                float(advanced_comp_delta_prime)
                if advanced_comp_delta_prime is not None
                else float(delta)
            )
            eps_ac = self.advanced_composition_epsilon(
                epsilon_per_round=float(nominal_epsilon_per_round),
                delta_prime=dp_prime,
                rounds=self._rounds,
            )
            report["eps_advanced_composition"] = float(eps_ac)
            report["advanced_composition_delta_prime"] = dp_prime
            report["improvement_pct"] = (
                (1.0 - eps_rdp / eps_ac) * 100.0 if eps_ac > 0.0 else 0.0
            )
        else:
            report["eps_advanced_composition"] = None
            report["advanced_composition_delta_prime"] = None
            report["improvement_pct"] = None

        return report

    def reset(self) -> None:
        self._rdp_eps.fill(0.0)
        self._rounds = 0
        self._round_records.clear()


# ============================================================================
# 2. Tensor/update helpers
# ============================================================================


def _tensor_items(params: Mapping[str, torch.Tensor]):
    for name, value in params.items():
        if torch.is_tensor(value):
            yield name, value


def global_l2_norm(params: Mapping[str, torch.Tensor]) -> float:
    """Compute the L2 norm of all tensor coordinates in a parameter mapping."""
    sq = 0.0
    found = False
    for _, tensor in _tensor_items(params):
        found = True
        x = tensor.detach().float().cpu().numpy().astype(np.float64, copy=False)
        sq += float(np.sum(x * x))
    return math.sqrt(max(sq, 0.0)) if found else 0.0


def subtract_state(
    local_params: Mapping[str, torch.Tensor],
    global_params: Mapping[str, torch.Tensor],
) -> TensorDict:
    """Return the client update ``local - global``."""
    if list(local_params.keys()) != list(global_params.keys()):
        raise ValueError("local_params and global_params must have identical keys/order.")

    update = OrderedDict()
    for name in local_params:
        local = local_params[name]
        base = global_params[name]
        if torch.is_tensor(local) and torch.is_tensor(base):
            update[name] = local.detach().float() - base.detach().float()
        else:
            update[name] = local
    return update


def clip_update(
    update: Mapping[str, torch.Tensor],
    clip_norm: float,
) -> Tuple[TensorDict, float, float]:
    """Clip a complete client update vector to L2 norm ``clip_norm``.

    Returns ``(clipped_update, original_norm, scale)``.
    """
    C = float(clip_norm)
    if C <= 0.0:
        raise ValueError("clip_norm must be positive.")

    norm = global_l2_norm(update)
    scale = min(1.0, C / max(norm, 1e-12))
    clipped = OrderedDict()
    for name, value in update.items():
        if torch.is_tensor(value):
            clipped[name] = value.detach().float() * scale
        else:
            clipped[name] = value
    return clipped, norm, scale


# ============================================================================
# 3. Adaptive clipping utility
# ============================================================================


class AdaptiveClipper:
    """Adaptive clip-norm controller.

    ``updates`` must already be CLIENT UPDATES (local minus global), not full
    model states.

    The controller updates C from the fraction of client updates whose norm is
    above C. In a private federated deployment, the norm statistics should be
    collected through a protected mechanism before this controller is used.
    """

    def __init__(
        self,
        initial_clip_norm: float = 1.0,
        target_quantile: float = 0.6,
        learning_rate: float = 0.2,
        min_clip: float = 0.1,
        max_clip: float = 50.0,
    ):
        if initial_clip_norm <= 0:
            raise ValueError("initial_clip_norm must be positive.")
        if not (0.0 < target_quantile < 1.0):
            raise ValueError("target_quantile must be in (0,1).")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if min_clip <= 0 or max_clip < min_clip:
            raise ValueError("Invalid clip range.")

        self.clip_norm = float(initial_clip_norm)
        self.target_quantile = float(target_quantile)
        self.lr = float(learning_rate)
        self.min_clip = float(min_clip)
        self.max_clip = float(max_clip)
        self.history: List[Dict] = []

    def clip_and_update(
        self,
        updates: Sequence[Mapping[str, torch.Tensor]],
    ) -> Tuple[List[TensorDict], float]:
        """Clip a batch of client updates and update C for the next round."""
        norms = [global_l2_norm(u) for u in updates]
        clipped_list: List[TensorDict] = []

        old_clip = float(self.clip_norm)
        for update in updates:
            clipped, _, _ = clip_update(update, old_clip)
            clipped_list.append(clipped)

        fraction_clipped = (
            sum(n > old_clip for n in norms) / max(len(norms), 1)
        )

        # Increase C when too many updates are clipped; decrease C when too
        # few are clipped. This controls the clip norm used NEXT round.
        new_clip = old_clip * math.exp(
            self.lr * (fraction_clipped - self.target_quantile)
        )
        self.clip_norm = float(np.clip(new_clip, self.min_clip, self.max_clip))

        self.history.append({
            "clip_norm_used": old_clip,
            "clip_norm_next": self.clip_norm,
            "fraction_clipped": float(fraction_clipped),
            "target_quantile": self.target_quantile,
            "norms": [float(x) for x in norms],
        })

        return clipped_list, old_clip


# ============================================================================
# 4. Top-K sparsification with error feedback
# ============================================================================


class TopKSparsifier:
    """Top-K sparsification of CLIENT UPDATES with persistent error feedback."""

    def __init__(self, compression_ratio: float = 0.1):
        ratio = float(compression_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError("compression_ratio must be in (0, 1].")
        self.k_ratio = ratio
        self._error_buffers: Dict[int, List[np.ndarray]] = {}
        self._shapes: Dict[int, List[Tuple[int, ...]]] = {}

    def sparsify(
        self,
        client_id: int,
        update_list: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """Sparsify an update vector with error feedback.

        All arrays are copied to float32 for deterministic residual handling.
        The residual is maintained per client and therefore requires persistent
        client objects across rounds.
        """
        cid = int(client_id)
        arrays = [np.asarray(x, dtype=np.float32) for x in update_list]
        if not arrays:
            raise ValueError("update_list must not be empty.")
        if any(a.size == 0 for a in arrays):
            raise ValueError("Empty tensors are not supported.")

        if cid not in self._error_buffers:
            self._error_buffers[cid] = [np.zeros_like(a) for a in arrays]
            self._shapes[cid] = [a.shape for a in arrays]
        else:
            if self._shapes[cid] != [a.shape for a in arrays]:
                raise ValueError(f"Client {cid} parameter shapes changed.")
            if len(self._error_buffers[cid]) != len(arrays):
                raise ValueError(f"Client {cid} parameter count changed.")

        accumulated = [a + e for a, e in zip(arrays, self._error_buffers[cid])]
        flat = np.concatenate([a.reshape(-1) for a in accumulated])
        total_size = int(flat.size)
        k = min(total_size, max(1, int(math.ceil(self.k_ratio * total_size))))

        top_indices = np.argpartition(np.abs(flat), -k)[-k:]
        mask_flat = np.zeros(total_size, dtype=bool)
        mask_flat[top_indices] = True

        sparse_params: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        new_errors: List[np.ndarray] = []
        offset = 0

        for a in accumulated:
            size = int(a.size)
            m = mask_flat[offset:offset + size].reshape(a.shape)
            sparse = np.where(m, a, np.float32(0.0)).astype(np.float32, copy=False)
            residual = np.where(m, np.float32(0.0), a).astype(np.float32, copy=False)
            sparse_params.append(sparse)
            masks.append(m.copy())
            new_errors.append(residual)
            offset += size

        self._error_buffers[cid] = new_errors
        actual_ratio = float(k / total_size)
        return sparse_params, masks, actual_ratio

    @staticmethod
    def densify(sparse_params: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Convert sparse arrays to float32 dense arrays for server summation."""
        return [np.asarray(x, dtype=np.float32).copy() for x in sparse_params]

    def get_stats(self) -> Dict:
        return {
            "compression_ratio": self.k_ratio,
            "active_clients": len(self._error_buffers),
            "error_buffer_norms": {
                cid: float(
                    math.sqrt(sum(float(np.sum(e.astype(np.float64) ** 2))
                                  for e in bufs))
                )
                for cid, bufs in self._error_buffers.items()
            },
        }

    def reset_client(self, client_id: int) -> None:
        self._error_buffers.pop(int(client_id), None)
        self._shapes.pop(int(client_id), None)

    def reset(self) -> None:
        self._error_buffers.clear()
        self._shapes.clear()


# ============================================================================
# 5. Unified Efficient DP manager
# ============================================================================


class EfficientDPManager:
    """Unified RDP + Gaussian DP manager.

    For BC-AWFedAvg the recommended configuration is:
        epsilon = 1.0
        delta = 1e-5
        initial_clip_norm = 1.0
        adaptive_clip = False

    ``add_dp_noise`` expects a CLIENT UPDATE, not a complete model state.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        initial_clip_norm: float = 1.0,
        adaptive_clip: bool = False,
        target_quantile: float = 0.6,
        rdp_orders: Optional[Sequence[float]] = None,
    ):
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must satisfy 0 < delta < 1.")
        if initial_clip_norm <= 0.0:
            raise ValueError("initial_clip_norm must be positive.")

        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.rdp = RDPAccountant(orders=rdp_orders)
        self.clipper = (
            AdaptiveClipper(
                initial_clip_norm=initial_clip_norm,
                target_quantile=target_quantile,
            )
            if adaptive_clip else None
        )
        self._fixed_clip_norm = float(initial_clip_norm)
        self.nominal_epsilon_per_round = float(epsilon)

    @property
    def clip_norm(self) -> float:
        return float(self.clipper.clip_norm if self.clipper else self._fixed_clip_norm)

    def noise_sigma(
        self,
        sensitivity: Optional[float] = None,
    ) -> float:
        """Return Gaussian standard deviation.

        Uses the same calibration convention as the thesis configuration:

            sigma = C * sqrt(2 ln(1.25/delta)) / epsilon

        where ``C`` is the client-update clipping bound.
        """
        C = self.clip_norm if sensitivity is None else float(sensitivity)
        if C <= 0.0:
            raise ValueError("sensitivity/clip norm must be positive.")
        return float(
            C * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon
        )

    def add_dp_noise(
        self,
        client_update: Mapping[str, torch.Tensor],
        sensitivity: Optional[float] = None,
        clip_norm: Optional[float] = None,
    ) -> Tuple[TensorDict, float]:
        """Clip and noise a client update.

        Parameters
        ----------
        client_update:
            Local model minus current global model.
        sensitivity:
            Optional sensitivity C. Defaults to the current clip norm.
        clip_norm:
            Optional explicit clip norm. Defaults to the current clip norm.

        Returns
        -------
        protected_update, sigma
        """
        C = self.clip_norm if clip_norm is None else float(clip_norm)
        if C <= 0.0:
            raise ValueError("clip_norm must be positive.")

        clipped, _, _ = clip_update(client_update, C)
        sens = C if sensitivity is None else float(sensitivity)
        sigma = self.noise_sigma(sensitivity=sens)

        protected = OrderedDict()
        for name, value in clipped.items():
            if torch.is_tensor(value):
                protected[name] = value.detach().float() + torch.randn_like(value.float()) * sigma
            else:
                protected[name] = value

        self.rdp.step(noise_sigma=sigma, sensitivity=sens)
        return protected, sigma

    def clip_client_updates(
        self,
        client_updates: Sequence[Mapping[str, torch.Tensor]],
    ) -> Tuple[List[TensorDict], float]:
        """Clip a batch of CLIENT UPDATES.

        With adaptive clipping enabled, the resulting C is used for the next
        call, exactly as recorded in ``clipper.history``.
        """
        if self.clipper is not None:
            return self.clipper.clip_and_update(client_updates)

        clipped: List[TensorDict] = []
        for update in client_updates:
            c, _, _ = clip_update(update, self._fixed_clip_norm)
            clipped.append(c)
        return clipped, self._fixed_clip_norm

    def get_epsilon(self) -> float:
        return self.rdp.get_epsilon(self.delta)

    def privacy_report(self) -> Dict:
        report = self.rdp.privacy_report(
            delta=self.delta,
            nominal_epsilon_per_round=self.nominal_epsilon_per_round,
        )
        report["epsilon_per_round_nominal"] = self.nominal_epsilon_per_round
        report["clip_norm_current"] = self.clip_norm
        report["adaptive_clipping"] = self.clipper is not None
        report["adaptive_clip_history"] = (
            list(self.clipper.history) if self.clipper else []
        )
        return report

    def reset(self) -> None:
        self.rdp.reset()
        if self.clipper is not None:
            current = self.clipper.clip_norm
            self.clipper = AdaptiveClipper(
                initial_clip_norm=current,
                target_quantile=self.clipper.target_quantile,
                learning_rate=self.clipper.lr,
                min_clip=self.clipper.min_clip,
                max_clip=self.clipper.max_clip,
            )


# ============================================================================
# 6. Backward-compatible helper
# ============================================================================


def add_differential_privacy_noise(
    client_update: Mapping[str, torch.Tensor],
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
) -> Tuple[TensorDict, float]:
    """Stateless compatibility wrapper.

    The input is explicitly a CLIENT UPDATE. It returns ``(protected, sigma)``.
    """
    manager = EfficientDPManager(
        epsilon=epsilon,
        delta=delta,
        initial_clip_norm=clip_norm,
        adaptive_clip=False,
    )
    return manager.add_dp_noise(client_update)


__all__ = [
    "RDPAccountant",
    "AdaptiveClipper",
    "TopKSparsifier",
    "EfficientDPManager",
    "global_l2_norm",
    "subtract_state",
    "clip_update",
    "add_differential_privacy_noise",
]
