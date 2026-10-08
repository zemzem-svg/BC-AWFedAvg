"""
BC-AWFedAvg — corrected two-phase weighted secure-aggregation integration.

This file keeps the existing AWFedAvg PPO client/server implementation but
changes the secure-aggregation flow to match the Chapter 4 protocol:

Phase 1:
    local PPO training -> QoS/learning metrics only -> server computes 5-criterion weights

Phase 2:
    client DP -> apply final weight -> pairwise masking -> server sums masked contributions

The standard one-shot Flower `Client.fit -> Strategy.aggregate_fit` path cannot
implement this protocol correctly because the final adaptive weights are unknown
until all client metrics have been collected. Therefore the provided sequential
simulator explicitly orchestrates the two phases.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

import flwr as fl
import numpy as np
import torch
from flwr.common import (
    Context,
    EvaluateIns,
    GetParametersIns,
    Parameters,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from stable_baselines3 import PPO

from adaptive_weighted_fedavg import (
    AdaptiveWeightedFedAvg,
    AdaptiveWeightCalculator,
    EnhancedFlowerClient,
    ResourceMonitor,
    PPONetwork,
    CLIENT_CONFIGS,
    NUM_CLIENTS,
    CLIENTS_PER_ROUND,
    TOTAL_ROUNDS,
    LOCAL_EPOCHS,
    EVALUATION_EPISODES,
    DEVICE,
    BASE_MODEL_PATH,
    create_phy_env,
    evaluate_model_simple,
    sb3_ppo_to_pytorch,
    pytorch_to_sb3_ppo,
    save_awfedavg_results_with_resources,
    print_resource_summary,
    get_client_hyperparams,
)

from privacy_blockchain_fl import PrivacyPreservingFederatedLearning

from efficient_dp import EfficientDPManager, subtract_state, TopKSparsifier

from secure_aggregation import (
    add_pairwise_masks,
    aggregate_masked_parameters,
    generate_pairwise_secrets,
)


PPONETWORK_LAYER_KEYS = [
    "policy_net.0.weight", "policy_net.0.bias",
    "policy_net.2.weight", "policy_net.2.bias",
    "value_net.0.weight", "value_net.0.bias",
    "value_net.2.weight", "value_net.2.bias",
    "action_net.weight", "action_net.bias",
    "value_head.weight", "value_head.bias",
]


# ============================================================================
# CONVERSION HELPERS
# ============================================================================


def ndarrays_to_ordered_dict(
    arrays: List[np.ndarray],
    keys: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> OrderedDict:
    """Convert NumPy arrays to an OrderedDict of float32 tensors."""
    dev = device or DEVICE
    if keys is None:
        keys = [f"param_{i}" for i in range(len(arrays))]
    if len(keys) != len(arrays):
        raise ValueError("Number of keys must equal number of arrays.")
    return OrderedDict(
        (key, torch.as_tensor(arr, dtype=torch.float32, device=dev).clone())
        for key, arr in zip(keys, arrays)
    )


def ordered_dict_to_ndarrays(params: Mapping[str, torch.Tensor]) -> List[np.ndarray]:
    return [tensor.detach().cpu().numpy() for tensor in params.values()]


# ============================================================================
# CLIENT-UPDATE DP
# ============================================================================

# DP is implemented by EfficientDPManager in efficient_dp.py.
# This module intentionally keeps no second DP implementation, preventing
# divergence between the protocol and the privacy accounting code.


# ============================================================================
# REPUTATION-AWARE WEIGHT CALCULATOR
# ============================================================================


class BlockchainAdaptiveWeightCalculator(AdaptiveWeightCalculator):
    """AWFedAvg four criteria plus persistent blockchain reputation."""

    def __init__(
        self,
        alpha_embb: float,
        alpha_urllc: float,
        alpha_activation: float,
        alpha_stability: float,
        alpha_reputation: float,
        smoothing_factor: float = 0.7,
        reputation_beta: float = 0.85,
        reputation_scale: float = 1000.0,
    ):
        total = (
            alpha_embb
            + alpha_urllc
            + alpha_activation
            + alpha_stability
            + alpha_reputation
        )
        if total <= 0:
            raise ValueError("At least one aggregation coefficient must be positive.")

        self.alpha_embb = alpha_embb / total
        self.alpha_urllc = alpha_urllc / total
        self.alpha_activation = alpha_activation / total
        self.alpha_stability = alpha_stability / total
        self.alpha_reputation = alpha_reputation / total

        self.performance_history: Dict[int, List[float]] = {}
        self.weight_history: List[np.ndarray] = []
        self.reputation_history: List[Dict[int, float]] = []
        self.reputations: Dict[int, float] = {}
        self.smoothing_factor = float(smoothing_factor)
        self.reputation_beta = float(reputation_beta)
        self.reputation_scale = float(reputation_scale)
        if not (0.0 <= self.reputation_beta <= 1.0):
            raise ValueError("reputation_beta must be in [0, 1].")
        if self.reputation_scale <= 0:
            raise ValueError("reputation_scale must be positive.")

    def update_reputation_from_chain(self, reputations: Mapping[int, int | float]) -> None:
        self.reputations = {
            int(cid): float(np.clip(float(rep), 0.0, self.reputation_scale))
            for cid, rep in reputations.items()
        }
        self.reputation_history.append(dict(self.reputations))

    @staticmethod
    def _inverse_normalized(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return values
        if np.std(values) <= 1e-12:
            return np.ones(len(values), dtype=float) / len(values)
        inv = 1.0 / (np.maximum(values, 0.0) + 1e-8)
        total = float(np.sum(inv))
        return inv / total if total > 0 else np.ones(len(values)) / len(values)

    def _component_scores(
        self,
        client_metrics: Mapping[int, Mapping[str, float]],
        client_configs,
    ) -> tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        client_ids = [int(cid) for cid in client_metrics.keys()]
        n = len(client_ids)
        if n == 0:
            empty = np.asarray([], dtype=float)
            return client_ids, empty, empty, empty, empty

        embb = np.array([
            float(client_metrics[cid].get("avg_embb_outage_counter", 0.0))
            for cid in client_ids
        ], dtype=float)
        urllc = np.array([
            float(client_metrics[cid].get("avg_residual_urllc_pkt", 0.0))
            for cid in client_ids
        ], dtype=float)

        activations = np.array([
            float(client_configs[cid % len(client_configs)]["activation"])
            for cid in client_ids
        ], dtype=float)

        rewards = np.array([
            float(client_metrics[cid].get("average_reward", 0.0))
            for cid in client_ids
        ], dtype=float)

        # 1. eMBB outage, lower is better.
        e_score = self._inverse_normalized(embb)

        # 2. URLLC residual, lower is better.
        u_score = self._inverse_normalized(urllc)

        # 3. Activation diversity, inherited from AWFedAvg.
        diversity = np.abs(activations - np.mean(activations))
        diversity_score = (1.0 + diversity)
        a_total = float(np.sum(diversity_score))
        a_score = (
            diversity_score / a_total
            if a_total > 0
            else np.ones(n, dtype=float) / n
        )

        # 4. Stability. Prefer an explicit update-variance metric when provided;
        # otherwise retain the AWFedAvg reward-history variance used by the code.
        s_values = np.zeros(n, dtype=float)
        for i, cid in enumerate(client_ids):
            explicit_var = client_metrics[cid].get("update_variance")
            if explicit_var is not None:
                variance = max(float(explicit_var), 0.0)
            else:
                history = self.performance_history.setdefault(cid, [])
                history.append(float(rewards[i]))
                recent = history[-5:]
                variance = float(np.var(recent)) if len(recent) > 1 else 0.0
            s_values[i] = 1.0 / (1.0 + variance)
        s_total = float(np.sum(s_values))
        s_score = s_values / s_total if s_total > 0 else np.ones(n) / n

        return client_ids, e_score, u_score, a_score, s_score

    def calculate_adaptive_weights(
        self,
        client_metrics: Mapping[int, Mapping[str, float]],
        client_configs,
        round_num: int,
    ) -> List[float]:
        client_ids, e_score, u_score, a_score, s_score = self._component_scores(
            client_metrics, client_configs
        )
        n = len(client_ids)
        if n == 0:
            return []

        rho = np.array([
            float(self.reputations.get(cid, self.reputation_scale / n))
            for cid in client_ids
        ], dtype=float)
        rho = np.clip(rho, 0.0, self.reputation_scale)
        rho_sum = float(np.sum(rho))
        r_score = rho / rho_sum if rho_sum > 0 else np.ones(n) / n

        preliminary = (
            self.alpha_embb * e_score
            + self.alpha_urllc * u_score
            + self.alpha_activation * a_score
            + self.alpha_stability * s_score
            + self.alpha_reputation * r_score
        )

        weights = np.maximum(preliminary, 1e-12)
        weights /= np.sum(weights)

        if self.weight_history:
            prev = np.asarray(self.weight_history[-1], dtype=float)
            if len(prev) == len(weights):
                eta = self.smoothing_factor
                weights = eta * weights + (1.0 - eta) * prev
                weights /= np.sum(weights)

        self.weight_history.append(weights.copy())
        return weights.tolist()

    def calculate_reputation_targets(
        self,
        client_metrics: Mapping[int, Mapping[str, float]],
        client_configs,
    ) -> Dict[int, float]:
        """Compute the bounded g(E,R,S) target used by Eq. (4.4.4).

        The thesis defines g(.) as a bounded function rewarding low outage, low
        uRLLC residuals and low update variance, but it does not provide a unique
        closed-form expression. For reproducibility, the canonical implementation
        uses the arithmetic mean of the three normalized E/R/S scores and maps it
        to the contract's 0..1000 representation.
        """
        client_ids, e_score, u_score, _a_score, s_score = self._component_scores(
            client_metrics, client_configs
        )
        if not client_ids:
            return {}

        g = np.clip((e_score + u_score + s_score) / 3.0, 0.0, 1.0)
        targets = self.reputation_scale * g
        return {cid: float(target) for cid, target in zip(client_ids, targets)}


# ============================================================================
# BLOCKCHAIN STRATEGY
# ============================================================================


class BlockchainAdaptiveWeightedFedAvg(AdaptiveWeightedFedAvg):
    """BC-AWFedAvg with reputation-aware weighting and two-phase SecAgg."""

    def __init__(
        self,
        ppfl: PrivacyPreservingFederatedLearning,
        ppfl_config: Optional[Dict] = None,
        apply_coordinator_dp: bool = True,
        compression: bool = True,
        blockchain_enabled: bool = True,
        alpha_reputation: float = 0.05,
        reputation_beta: float = 0.85,
        reputation_scale: float = 1000.0,
        isolation_threshold_factor: float = 0.5,
        strict_blockchain: bool = True,
        **awfedavg_kwargs,
    ):
        # Remove BC-only arguments before entering the original four-criterion
        # AWFedAvg constructor.
        alpha_embb = awfedavg_kwargs.pop("alpha_embb")
        alpha_urllc = awfedavg_kwargs.pop("alpha_urllc")
        alpha_activation = awfedavg_kwargs.pop("alpha_activation")
        alpha_stability = awfedavg_kwargs.pop("alpha_stability")

        super().__init__(
            alpha_embb=alpha_embb,
            alpha_urllc=alpha_urllc,
            alpha_activation=alpha_activation,
            alpha_stability=alpha_stability,
            **awfedavg_kwargs,
        )

        self.weight_calculator = BlockchainAdaptiveWeightCalculator(
            alpha_embb=alpha_embb,
            alpha_urllc=alpha_urllc,
            alpha_activation=alpha_activation,
            alpha_stability=alpha_stability,
            alpha_reputation=alpha_reputation,
            smoothing_factor=0.7,
            reputation_beta=reputation_beta,
            reputation_scale=reputation_scale,
        )

        self.ppfl = ppfl
        self.ppfl_config = ppfl_config or {}
        self.apply_coordinator_dp = bool(apply_coordinator_dp)
        self.compression = bool(compression)
        self.blockchain_enabled = bool(blockchain_enabled)
        self.strict_blockchain = bool(strict_blockchain)

        self.alpha_reputation = float(alpha_reputation)
        self.reputation_beta = float(reputation_beta)
        self.reputation_scale = float(reputation_scale)
        self.isolation_threshold_factor = float(isolation_threshold_factor)
        self.last_round_contribution_hashes: Dict[int, str] = {}
        self.last_global_ipfs_hash = ""
        self.blockchain_round_records: List[Dict[str, Any]] = []
        self._client_addrs_map: Dict[int, str] = {}
        self.current_round_weights: Dict[int, float] = {}
        self.current_round_reputations: Dict[int, float] = {}
        self.current_round_reputation_targets: Dict[int, float] = {}
        self.reputation_history_by_round: Dict[int, Dict[int, float]] = {}
        self.first_isolation_round: Dict[int, int] = {}
        self.isolation_history: List[Dict[str, Any]] = []
        self.current_round_isolation: Dict[int, Dict[str, Any]] = {}
        self.current_round_tx = ""

    def _fetch_reputations(self, client_ids: List[int]) -> Dict[int, float]:
        reputations = {}
        contract = getattr(self.ppfl, "contract", None)
        n = max(len(client_ids), 1)
        default_rep = self.reputation_scale / n

        for cid in client_ids:
            rep = default_rep
            address = self._client_addrs_map.get(cid)
            if self.blockchain_enabled and contract is not None and address:
                try:
                    _, chain_rep, _, _ = contract.functions.getClientInfo(address).call()
                    rep = float(chain_rep)
                except Exception as exc:
                    warnings.warn(
                        f"[Blockchain] reputation lookup failed for client {cid}: {exc}; "
                        f"using uniform default {default_rep:.3f}."
                    )
            reputations[cid] = float(np.clip(rep, 0.0, self.reputation_scale))

        self.current_round_reputations = reputations
        self.weight_calculator.update_reputation_from_chain(reputations)
        return reputations

    def begin_round(self, server_round: int) -> str:
        """Open the blockchain round; no model parameters are handled here."""
        if not self.blockchain_enabled:
            self.current_round_tx = ""
            return ""

        try:
            tx = self.ppfl.start_round_on_chain(
                previous_model_ipfs_hash=self.last_global_ipfs_hash
            )
            self.current_round_tx = tx or ""
            print(f"\n🔗 Round {server_round} opened on blockchain  tx={tx}")
            return self.current_round_tx
        except Exception as exc:
            self.current_round_tx = ""
            if self.strict_blockchain:
                raise RuntimeError(
                    f"Blockchain startRound failed for round {server_round}: {exc}"
                ) from exc
            warnings.warn(
                f"[Blockchain] start_round_on_chain failed (round {server_round}): {exc}"
            )
            return ""

    def get_eligible_client_ids(self, candidate_ids: List[int]) -> List[int]:
        """Return active clients only; isolation is influence control, not exclusion.

        The thesis describes reputation-driven isolation as a gradual reduction of
        aggregation influence. A weight crossing theta_iso therefore does not
        deactivate the client or block its contribution in the next round.
        """
        if not self.blockchain_enabled or self.ppfl.contract is None:
            return [int(cid) for cid in candidate_ids]

        eligible = []
        for cid in candidate_ids:
            address = self._client_addrs_map.get(int(cid))
            if not address:
                continue
            try:
                info = self.ppfl.get_client_info(address)
                if info["is_active"]:
                    eligible.append(int(cid))
            except Exception as exc:
                if self.strict_blockchain:
                    raise RuntimeError(
                        f"Could not determine blockchain activity for client {cid}: {exc}"
                    ) from exc
                warnings.warn(f"[Blockchain] activity lookup failed for client {cid}: {exc}")
        return eligible

    def record_round_contributions(
        self,
        server_round: int,
        masked_results: List[tuple],
    ) -> str:
        """Anchor contribution hashes after Phase 2 and before global publication."""
        if not self.blockchain_enabled:
            self.last_round_contribution_hashes = {}
            return ""
        addresses = []
        hashes = []
        sizes = []
        for cid, param_list, _num_examples, _metrics in masked_results:
            address = self._client_addrs_map.get(int(cid))
            if not address:
                raise RuntimeError(f"Missing blockchain address for client {cid}.")
            digest = hashlib.sha256()
            total_bytes = 0
            for arr in param_list:
                raw = np.asarray(arr)
                digest.update(str(raw.dtype).encode("utf-8"))
                digest.update(str(tuple(raw.shape)).encode("utf-8"))
                digest.update(raw.tobytes(order="C"))
                total_bytes += raw.nbytes
            digest.update(f"round={server_round}|client={int(cid)}".encode("utf-8"))
            addresses.append(address)
            hashes.append(digest.digest())
            sizes.append(int(total_bytes))

        tx = self.ppfl.record_contributions_batch_on_chain(
            client_addresses=addresses,
            round_number=server_round,
            update_hashes=hashes,
            data_sizes=sizes,
        )
        self.last_round_contribution_hashes = {
            int(cid): h.hex() for cid, h in zip(
                [x[0] for x in masked_results], hashes
            )
        }
        return tx

    def compute_reputation_targets_from_metrics(
        self,
        client_metrics: Mapping[int, Mapping[str, float]],
    ) -> Dict[int, float]:
        """Compute bounded g(E,R,S) targets for the on-chain leaky integrator."""
        targets = self.weight_calculator.calculate_reputation_targets(
            client_metrics=client_metrics,
            client_configs=CLIENT_CONFIGS,
        )
        self.current_round_reputation_targets = dict(targets)
        return targets

    def update_round_reputations(
        self,
        server_round: int,
        participant_ids: List[int],
        reputation_targets: Optional[Mapping[int, float]] = None,
    ) -> str:
        """Apply Eq. (4.4.4) on-chain using bounded E/R/S target signals."""
        targets = {int(cid): float(v) for cid, v in (reputation_targets or {}).items()}
        if self.blockchain_enabled:
            missing = [cid for cid in participant_ids if cid not in targets]
            if missing:
                raise ValueError(f"Missing reputation targets for clients: {missing}")

            addresses = []
            target_values = []
            for cid in participant_ids:
                address = self._client_addrs_map.get(int(cid))
                if not address:
                    raise RuntimeError(f"Missing blockchain address for client {cid}.")
                addresses.append(address)
                target_values.append(
                    int(round(np.clip(targets[int(cid)], 0.0, self.reputation_scale)))
                )

            tx = self.ppfl.update_reputations_batch_on_chain(
                client_addresses=addresses,
                reputation_signals=target_values,
            )
        else:
            tx = ""

        self._fetch_reputations([int(cid) for cid in participant_ids])
        self.reputation_history_by_round[int(server_round)] = dict(
            self.current_round_reputations
        )
        return tx

    def update_isolation_tracking(
        self,
        server_round: int,
        weights: Mapping[int, float],
        participant_ids: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """Record first threshold crossing without hard-excluding the client."""
        k = max(len(participant_ids), 1)
        theta_iso = self.isolation_threshold_factor / k
        current: Dict[int, Dict[str, Any]] = {}
        for cid in participant_ids:
            weight = float(weights[int(cid)])
            if int(cid) not in self.first_isolation_round and weight < theta_iso:
                self.first_isolation_round[int(cid)] = int(server_round)
            first = self.first_isolation_round.get(int(cid))
            current[int(cid)] = {
                "weight": weight,
                "theta_iso": float(theta_iso),
                "isolated": first is not None and int(first) <= int(server_round),
                "tti_round": first,
            }
        self.current_round_isolation = current
        self.isolation_history.append({
            "round": int(server_round),
            "theta_iso": float(theta_iso),
            "clients": current,
        })
        return current

    def compute_adaptive_weights_from_metrics(
        self,
        server_round: int,
        client_metrics: Mapping[int, Mapping[str, float]],
    ) -> Dict[int, float]:
        """Phase 1 server operation: metrics -> reputation -> final weights."""
        ids = [int(cid) for cid in client_metrics.keys()]
        self._fetch_reputations(ids)

        weights = self.weight_calculator.calculate_adaptive_weights(
            client_metrics=dict(client_metrics),
            client_configs=CLIENT_CONFIGS,
            round_num=server_round,
        )
        if len(weights) != len(ids):
            raise RuntimeError("Weight calculator returned an unexpected number of client weights.")
        self.current_round_weights = dict(zip(ids, weights))
        if not math.isclose(sum(self.current_round_weights.values()), 1.0, abs_tol=1e-9):
            raise RuntimeError("Adaptive aggregation weights must sum to 1.")

        print(
            "  Adaptive weights: "
            + ", ".join(f"client {cid}={w:.4f}" for cid, w in self.current_round_weights.items())
        )
        print(
            "  Reputation: "
            + ", ".join(f"client {cid}={self.current_round_reputations[cid]:.0f}" for cid in ids)
        )
        return self.current_round_weights.copy()

    def aggregate_masked_fit(
        self,
        server_round: int,
        masked_results: List[tuple],
        client_metrics: Mapping[int, Mapping[str, float]],
        base_parameters: Parameters,
        failures: Optional[List[Any]] = None,
    ):
        """Phase 2: sum masked weighted UPDATES and add them to the global model."""
        if not masked_results:
            return None, {}

        expected_ids = set(self.current_round_weights)
        received_ids = {int(x[0]) for x in masked_results}
        if expected_ids != received_ids:
            raise RuntimeError(
                f"Secure aggregation participant mismatch: expected {sorted(expected_ids)}, "
                f"received {sorted(received_ids)}"
            )

        masked_dicts = []
        client_ids = []
        communication_sizes = []
        training_times = []
        rewards = []

        for cid, param_list, num_examples, metrics in masked_results:
            params = ndarrays_to_ordered_dict(
                param_list,
                keys=PPONETWORK_LAYER_KEYS[:len(param_list)],
                device=DEVICE,
            )
            masked_dicts.append(params)
            client_ids.append(int(cid))
            communication_sizes.append(
                sum(np.asarray(p).nbytes for p in param_list)
            )
            training_times.append(float(metrics.get("training_time", 0.0)))
            rewards.append(float(metrics.get("average_reward", 0.0)))

        # The masked payloads are weighted CLIENT UPDATES, not complete model
        # states. Pairwise masks cancel under summation, leaving
        #     sum_k w_k * DP(update_k).
        aggregate_update = aggregate_masked_parameters(
            masked_dicts,
            output_dtype=torch.float32,
        )

        base_arrays = parameters_to_ndarrays(base_parameters)
        base_dict = ndarrays_to_ordered_dict(
            base_arrays,
            keys=PPONETWORK_LAYER_KEYS[:len(base_arrays)],
            device=DEVICE,
        )
        global_dict = OrderedDict()
        if list(base_dict.keys()) != list(aggregate_update.keys()):
            raise ValueError(
                "Base model and aggregated update have different parameter keys."
            )
        for name in base_dict:
            global_dict[name] = base_dict[name].detach().float() + aggregate_update[name].detach().float()

        aggregated_parameters = ndarrays_to_parameters(
            ordered_dict_to_ndarrays(global_dict)
        )

        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        avg_train = float(np.mean(training_times)) if training_times else 0.0
        max_train = float(np.max(training_times)) if training_times else 0.0
        comm_mb = float(sum(communication_sizes) / (1024 * 1024))

        round_metrics = {
            "average_reward": avg_reward,
            "adaptive_weights": [self.current_round_weights[cid] for cid in client_ids],
            "client_ids": client_ids,
            "reputations": [self.current_round_reputations.get(cid, 500.0) for cid in client_ids],
            "round_training_time": max_train,
            "average_client_training_time": avg_train,
            "communication_mb": comm_mb,
            "secure_aggregation": True,
            "weighted_before_masking": True,
        }

        # Populate the resource/reporting structures expected by existing AWFedAvg
        # result-saving code.
        round_resource_data = {
            "round": server_round,
            "timing": {
                "total_round_time": 0.0,
                "collection_time": 0.0,
                "weight_calculation_time": 0.0,
                "aggregation_time": 0.0,
                "metrics_calculation_time": 0.0,
                "client_training_times": {
                    "average": avg_train,
                    "maximum": max_train,
                    "individual": {
                        int(cid): float(client_metrics[cid].get("training_time", 0.0))
                        for cid in client_ids
                    },
                },
            },
            "communication": {
                "total_parameters_size_mb": comm_mb,
                "average_params_size_mb": comm_mb / max(1, len(client_ids)),
                "total_communication_overhead": 0.0,
                "estimated_network_time": 0.0,
            },
            "resources": {},
            "client_metrics": dict(client_metrics),
            "adaptive_weights": [self.current_round_weights[cid] for cid in client_ids],
            "reputations": dict(self.current_round_reputations),
            "global_performance": avg_reward,
        }

        self.round_resources.append(round_resource_data)
        self.performance_history.append(avg_reward)
        self.final_parameters = aggregated_parameters
        self.round_metrics.append({
            "round": server_round,
            "client_metrics": dict(client_metrics),
            "aggregated_metrics": round_metrics,
            "adaptive_weights": [self.current_round_weights[cid] for cid in client_ids],
            "reputations": dict(self.current_round_reputations),
            "resource_data": round_resource_data,
        })

        return aggregated_parameters, round_metrics

    def publish_global_model(
        self,
        server_round: int,
        aggregated_parameters: Parameters,
    ) -> Dict[str, Any]:
        """Apply server-side DP for archival, encrypt, upload and anchor CID."""
        ipfs_hash = ""
        agg_tx = ""
        dp_applied = False
        upload_size_kb = 0.0
        storage_reduction_pct = 0.0
        t0 = time.time()

        if self.blockchain_enabled:
            try:
                arrays = parameters_to_ndarrays(aggregated_parameters)
                params_dict = ndarrays_to_ordered_dict(
                    arrays,
                    keys=PPONETWORK_LAYER_KEYS[:len(arrays)],
                    device=DEVICE,
                )
                original_size = float(sum(p.numel() * 4 for p in params_dict.values()))

                if self.apply_coordinator_dp:
                    params_dict = self.ppfl.add_differential_privacy_noise(
                        params_dict,
                        sensitivity=1.0,
                    )
                    dp_applied = True

                encrypted_data, _sym_key = self.ppfl.encrypt_model(
                    params_dict,
                    compression=self.compression,
                )
                upload_size_kb = len(encrypted_data) / 1024.0
                if original_size > 0:
                    storage_reduction_pct = (
                        1.0 - len(encrypted_data) / original_size
                    ) * 100.0

                ipfs_hash = self.ppfl.upload_to_ipfs(encrypted_data, pin=True)
                model_hash = hashlib.sha256(encrypted_data).digest()
                agg_tx = self.ppfl.submit_aggregated_model_on_chain(
                    round_number=server_round,
                    ipfs_hash=ipfs_hash,
                    model_hash=model_hash,
                )
                self.last_global_ipfs_hash = ipfs_hash

                print(
                    f"  📦 Global model → IPFS {ipfs_hash} "
                    f"({upload_size_kb:.1f} KB; storage reduction={storage_reduction_pct:+.1f}%)"
                )
                print(f"  🔗 Model metadata anchored on-chain  tx={agg_tx}")

            except Exception as exc:
                warnings.warn(
                    f"[Blockchain] publication failed (round {server_round}): {exc}. "
                    "The live aggregated model is preserved."
                )

        return {
            "ipfs_hash": ipfs_hash,
            "blockchain_open_tx": self.current_round_tx,
            "blockchain_agg_tx": agg_tx,
            "dp_applied": dp_applied,
            "compression_ratio_pct": storage_reduction_pct,
            "ipfs_upload_size_kb": upload_size_kb,
            "blockchain_overhead_s": time.time() - t0,
            "epsilon": float(getattr(self.ppfl, "epsilon", 0.0)),
            "delta": float(getattr(self.ppfl, "delta", 0.0)),
        }

    def aggregate_fit(self, server_round, results, failures):
        """Disable the unsafe one-phase Flower aggregation path."""
        raise RuntimeError(
            "BC-AWFedAvg secure aggregation requires the two-phase protocol. "
            "Use _run_sequential_simulation()/compute_adaptive_weights_from_metrics() "
            "and aggregate_masked_fit()."
        )

    def get_isolation_summary(self) -> Dict[str, Any]:
        """Return gradual first-crossing TTI records without excluding clients."""
        if not self.first_isolation_round:
            return {
                "isolated_clients": 0,
                "first_isolation_round": {},
                "theta_iso": None,
            }
        latest_k = 0
        if self.isolation_history:
            latest_k = len(self.isolation_history[-1].get("clients", {}))
        theta = self.isolation_threshold_factor / latest_k if latest_k else None
        return {
            "isolated_clients": len(self.first_isolation_round),
            "first_isolation_round": dict(self.first_isolation_round),
            "theta_iso": theta,
        }

    def get_blockchain_summary(self) -> Dict[str, Any]:
        """Summarize IPFS/model-publication transactions recorded by this runner."""
        if not self.blockchain_round_records:
            return {}
        overheads = [r["blockchain_overhead_s"] for r in self.blockchain_round_records]
        sizes = [r["ipfs_upload_size_kb"] for r in self.blockchain_round_records]
        return {
            "rounds_with_blockchain": len(self.blockchain_round_records),
            "total_blockchain_overhead_s": float(sum(overheads)),
            "avg_blockchain_overhead_s": float(np.mean(overheads)),
            "total_ipfs_upload_kb": float(sum(sizes)),
            "avg_ipfs_upload_kb": float(np.mean(sizes)),
            "ipfs_hashes": [r["ipfs_hash"] for r in self.blockchain_round_records],
        }

    def print_blockchain_performance_summary(self):
        resource_summary = self.get_comprehensive_resource_summary()
        if resource_summary:
            print_resource_summary(resource_summary)

        iso = self.get_isolation_summary()
        if iso.get("isolated_clients", 0) > 0:
            print("\n" + "=" * 70)
            print("GRADUAL ISOLATION / TTI SUMMARY")
            print("=" * 70)
            print(f"  Isolation threshold : {iso['theta_iso']:.4f}")
            print(f"  Isolated clients    : {iso['isolated_clients']}")
            print(f"  First TTI rounds    : {iso['first_isolation_round']}")
            print("  Mode                : influence suppression, not hard exclusion")

        bc = self.get_blockchain_summary()
        if bc:
            print("\n" + "=" * 70)
            print("BLOCKCHAIN / IPFS OVERHEAD SUMMARY")
            print("=" * 70)
            print(f"  Rounds with blockchain : {bc['rounds_with_blockchain']}")
            print(f"  Total BC overhead      : {bc['total_blockchain_overhead_s']:.2f}s")
            print(f"  Avg BC overhead/round  : {bc['avg_blockchain_overhead_s']:.2f}s")
            print(f"  Total IPFS uploads     : {bc['total_ipfs_upload_kb']:.1f} KB")
            print(f"  Avg IPFS upload/round  : {bc['avg_ipfs_upload_kb']:.1f} KB")

        try:
            priv = self.ppfl.privacy_report()
            print("\n" + "=" * 70)
            print("PRIVACY ACCOUNTING")
            print("=" * 70)
            print(f"  ε per round            : {priv.get('eps_per_round', 0.0):.4f}")
            print(f"  δ                      : {priv.get('delta', 0.0):.1e}")
            print(f"  Rounds elapsed         : {priv.get('rounds_elapsed', 0)}")
            print(f"  ε_total / RDP value    : {priv.get('eps_total_so_far', 0.0):.4f}")
        except Exception as exc:
            print(f"  Privacy report unavailable: {exc}")

        try:
            self.ppfl.print_performance_summary()
        except Exception:
            pass


# ============================================================================
# FLOWER CLIENT
# ============================================================================


class BlockchainEnhancedFlowerClient(EnhancedFlowerClient):
    """Persistent client supporting the two-phase BC-AWFedAvg protocol."""

    def __init__(
        self,
        client_id: int,
        client_address: str,
        client_private_key: str,
        ppfl_config: Dict,
        blockchain_enabled: bool = True,
        stake_amount: float = 0.01,
        num_clients: int = NUM_CLIENTS,
        secure_aggregation: bool = True,
        pairwise_secrets: Optional[Mapping[int, bytes]] = None,
        attack_type: str = "none",
        attack_fraction: float = 0.0,
        attack_strength: float = 1.0,
    ):
        super().__init__(client_id)
        self.client_address = client_address
        self.client_private_key = client_private_key
        self._ppfl_config = dict(ppfl_config)
        self._ppfl = None
        self.blockchain_enabled = bool(blockchain_enabled)
        self.stake_amount = float(stake_amount)
        self._num_clients = int(num_clients)
        self._secure_aggregation = bool(secure_aggregation)
        self._pairwise_secrets = dict(pairwise_secrets or {})
        self._attack_type = attack_type
        self._attack_fraction = float(attack_fraction)
        self._attack_strength = float(attack_strength)
        self._param_history: List[List[np.ndarray]] = []
        self._pending_params: Optional[List[np.ndarray]] = None
        self._pending_global_params: Optional[List[np.ndarray]] = None
        self._pending_num_examples: int = 0
        self._pending_metrics: Dict[str, Any] = {}
        self._dp_manager = EfficientDPManager(
            epsilon=float(self._ppfl_config.get("epsilon", 1.0)),
            delta=float(self._ppfl_config.get("delta", 1e-5)),
            initial_clip_norm=float(self._ppfl_config.get("clip_norm", 1.0)),
            adaptive_clip=bool(self._ppfl_config.get("adaptive_clip", False)),
            target_quantile=float(self._ppfl_config.get("target_quantile", 0.6)),
        )
        topk_ratio = self._ppfl_config.get("topk_ratio", None)
        self._topk = (
            TopKSparsifier(float(topk_ratio))
            if topk_ratio is not None else None
        )

    @property
    def ppfl(self) -> PrivacyPreservingFederatedLearning:
        if self._ppfl is None:
            cfg = self._ppfl_config
            self._ppfl = PrivacyPreservingFederatedLearning(
                blockchain_provider=cfg.get("blockchain_provider", "http://127.0.0.1:8545"),
                contract_address=cfg.get("contract_address"),
                contract_abi_path=cfg.get("contract_abi_path"),
                ipfs_addr=cfg.get("ipfs_addr", "/ip4/127.0.0.1/tcp/5001"),
                epsilon=cfg.get("epsilon", 1.0),
                delta=cfg.get("delta", 1e-5),
                clip_norm=cfg.get("clip_norm", 1.0),
                coordinator_private_key=cfg.get("coordinator_private_key"),
                require_connection=cfg.get("blockchain_enabled", True),
            )
        return self._ppfl

    def prepare_fit(self, parameters, config: Mapping[str, Any]) -> Dict[str, Any]:
        """Phase 1: local training and metric submission only."""
        # Keep the received global model so Phase 2 can protect the CLIENT
        # UPDATE (local_model - global_model), rather than the full model.
        self._pending_global_params = [
            np.array(p, copy=True) for p in parameters_to_ndarrays(parameters)
        ]
        param_list, num_examples, metrics = super().fit(parameters, dict(config))

        if self._attack_type != "none":
            try:
                from experiments import apply_attack_to_params
                server_round = int(config.get("server_round", 0))
                param_list = apply_attack_to_params(
                    param_list,
                    self.client_id,
                    {
                        "attack_type": self._attack_type,
                        "attack_fraction": self._attack_fraction,
                        "attack_strength": self._attack_strength,
                        "num_clients": self._num_clients,
                        "server_round": server_round,
                    },
                    self._param_history,
                )
            except Exception as exc:
                warnings.warn(f"[Attack] Client {self.client_id}: {exc}")

        self._param_history.append([np.array(p, copy=True) for p in param_list])
        if len(self._param_history) > 5:
            self._param_history.pop(0)

        self._pending_params = [np.array(p, copy=True) for p in param_list]
        self._pending_num_examples = int(num_examples)
        self._pending_metrics = dict(metrics)
        return {
            "client_id": int(self.client_id),
            "num_examples": int(num_examples),
            "metrics": dict(metrics),
        }

    def finalize_fit(
        self,
        aggregation_weight: float,
        server_round: int,
        participant_ids: Optional[List[int]] = None,
    ) -> tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Phase 2 client operation: client DP -> weight -> pairwise masks."""
        if self._pending_params is None:
            raise RuntimeError(
                f"Client {self.client_id} has no pending local update for round {server_round}."
            )
        if aggregation_weight < 0:
            raise ValueError("Aggregation weight must be non-negative.")

        device = DEVICE
        raw_dict = ndarrays_to_ordered_dict(
            self._pending_params,
            keys=PPONETWORK_LAYER_KEYS[:len(self._pending_params)],
            device=device,
        )

        if self._pending_global_params is None:
            raise RuntimeError(
                f"Client {self.client_id}: global model is missing for update computation."
            )
        global_dict = ndarrays_to_ordered_dict(
            self._pending_global_params,
            keys=PPONETWORK_LAYER_KEYS[:len(self._pending_global_params)],
            device=device,
        )
        client_update = subtract_state(raw_dict, global_dict)

        protected_update, sigma = self._dp_manager.add_dp_noise(client_update)

        weighted = OrderedDict(
            (name, tensor * float(aggregation_weight))
            for name, tensor in protected_update.items()
        )

        sparse_ratio = 1.0
        if self._topk is not None:
            sparse_list, _, sparse_ratio = self._topk.sparsify(
                self.client_id, ordered_dict_to_ndarrays(weighted)
            )
            weighted = ndarrays_to_ordered_dict(
                sparse_list,
                keys=list(weighted.keys()),
                device=device,
            )

        if self._secure_aggregation:
            if not self._pairwise_secrets:
                raise RuntimeError(
                    f"Client {self.client_id}: pairwise secrets are not configured."
                )
            participants = list(participant_ids) if participant_ids is not None else list(range(self._num_clients))
            masked = add_pairwise_masks(
                params=weighted,
                client_id=self.client_id,
                all_client_ids=participants,
                round_num=server_round,
                pairwise_secrets=self._pairwise_secrets,
                mask_scale=float(self._ppfl_config.get("secagg_mask_scale", 1.0)),
            )
        else:
            masked = weighted

        metrics = dict(self._pending_metrics)
        metrics.update({
            "aggregation_weight": float(aggregation_weight),
            "secure_aggregation": bool(self._secure_aggregation),
            "weighted_before_masking": bool(self._secure_aggregation),
            "client_dp_applied": True,
            "client_dp_sigma": float(sigma),
            "client_dp_accountant_epsilon": float(self._dp_manager.get_epsilon()),
            "protected_object": "client_update",
            "topk_enabled": bool(self._topk is not None),
            "topk_actual_ratio": float(sparse_ratio),
        })

        arrays = ordered_dict_to_ndarrays(masked)
        return arrays, self._pending_num_examples, metrics

    def fit(self, parameters, config):
        """Prevent accidental use of the unsafe one-phase Flower protocol."""
        raise RuntimeError(
            "BlockchainEnhancedFlowerClient requires the two-phase BC-AWFedAvg "
            "runner. Call prepare_fit() followed by finalize_fit(weight, round)."
        )


# ============================================================================
# FACTORY
# ============================================================================


_DEFAULT_CLIENT_ADDRESSES = []


def create_blockchain_awfedavg_strategy(
    blockchain_provider: str = "http://localhost:8545",
    ipfs_addr: str = "/ip4/127.0.0.1/tcp/5001",
    contract_address: Optional[str] = None,
    contract_abi_path: Optional[str] = None,
    coordinator_private_key: Optional[str] = None,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    apply_coordinator_dp: bool = True,
    compression: bool = True,
    blockchain_enabled: bool = True,
    alpha_embb: float = 0.22,
    alpha_urllc: float = 0.38,
    alpha_activation: float = 0.20,
    alpha_stability: float = 0.15,
    alpha_reputation: float = 0.05,
    reputation_beta: float = 0.85,
    reputation_scale: float = 1000.0,
    isolation_threshold_factor: float = 0.5,
    min_fit_clients: int = CLIENTS_PER_ROUND,
    min_evaluate_clients: int = CLIENTS_PER_ROUND,
    min_available_clients: int = NUM_CLIENTS,
) -> BlockchainAdaptiveWeightedFedAvg:
    """Create BC-AWFedAvg with exactly normalized five-criterion coefficients."""
    coeffs = [
        alpha_embb,
        alpha_urllc,
        alpha_activation,
        alpha_stability,
        alpha_reputation,
    ]
    if blockchain_enabled and min_fit_clients < 2:
        raise ValueError("BC-AWFedAvg secure aggregation requires at least two clients per round.")
    if any(c < 0 for c in coeffs):
        raise ValueError("All aggregation coefficients must be non-negative.")
    if not math.isclose(sum(coeffs), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "BC-AWFedAvg coefficients must sum to 1. "
            f"Received {sum(coeffs):.6f}."
        )

    ppfl_config = {
        "blockchain_provider": str(blockchain_provider),
        "contract_address": str(contract_address) if contract_address else None,
        "contract_abi_path": str(contract_abi_path) if contract_abi_path else None,
        "ipfs_addr": str(ipfs_addr),
        "epsilon": float(epsilon),
        "delta": float(delta),
        "clip_norm": float(clip_norm),
        "coordinator_private_key": str(coordinator_private_key) if coordinator_private_key else None,
        "blockchain_enabled": bool(blockchain_enabled),
        "adaptive_clip": False,
        "target_quantile": 0.6,
        "reputation_beta": float(reputation_beta),
        "reputation_scale": float(reputation_scale),
        "isolation_threshold_factor": float(isolation_threshold_factor),
    }

    ppfl = PrivacyPreservingFederatedLearning(
        blockchain_provider=blockchain_provider,
        contract_address=contract_address,
        contract_abi_path=contract_abi_path,
        ipfs_addr=ipfs_addr,
        epsilon=epsilon,
        delta=delta,
        clip_norm=clip_norm,
        coordinator_private_key=coordinator_private_key,
        require_connection=blockchain_enabled,
    )

    return BlockchainAdaptiveWeightedFedAvg(
        ppfl=ppfl,
        ppfl_config=ppfl_config,
        apply_coordinator_dp=apply_coordinator_dp,
        compression=compression,
        blockchain_enabled=blockchain_enabled,
        alpha_reputation=alpha_reputation,
        reputation_beta=reputation_beta,
        reputation_scale=reputation_scale,
        isolation_threshold_factor=isolation_threshold_factor,
        alpha_embb=alpha_embb,
        alpha_urllc=alpha_urllc,
        alpha_activation=alpha_activation,
        alpha_stability=alpha_stability,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )


def make_blockchain_client_fn(
    ppfl_config: Dict,
    client_addresses: Optional[List[Dict]] = None,
    blockchain_enabled: bool = True,
    num_clients: int = NUM_CLIENTS,
    stake_amount: float = 0.01,
    secure_aggregation: bool = True,
    attack_type: str = "none",
    attack_fraction: float = 0.0,
    attack_strength: float = 1.0,
):
    """Create a persistent client factory for the sequential two-phase runner."""
    if blockchain_enabled and not client_addresses:
        raise ValueError(
            "Explicit client_addresses with real blockchain identities are required "
            "when blockchain_enabled=True. Do not use fabricated addresses or keys."
        )

    addresses = client_addresses or _DEFAULT_CLIENT_ADDRESSES
    addrs = [
        {
            "address": str(a["address"]),
            "private_key": str(a.get("private_key") or ""),
        }
        for a in addresses
    ]

    cfg = {
        "blockchain_provider": str(ppfl_config.get("blockchain_provider", "http://127.0.0.1:8545")),
        "contract_address": ppfl_config.get("contract_address"),
        "contract_abi_path": ppfl_config.get("contract_abi_path"),
        "ipfs_addr": str(ppfl_config.get("ipfs_addr", "/ip4/127.0.0.1/tcp/5001")),
        "epsilon": float(ppfl_config.get("epsilon", 1.0)),
        "delta": float(ppfl_config.get("delta", 1e-5)),
        "clip_norm": float(ppfl_config.get("clip_norm", 1.0)),
        "coordinator_private_key": ppfl_config.get("coordinator_private_key"),
        "blockchain_enabled": bool(ppfl_config.get("blockchain_enabled", blockchain_enabled)),
        "secagg_mask_scale": float(ppfl_config.get("secagg_mask_scale", 1.0)),
        "adaptive_clip": bool(ppfl_config.get("adaptive_clip", False)),
        "target_quantile": float(ppfl_config.get("target_quantile", 0.6)),
        "topk_ratio": ppfl_config.get("topk_ratio", None),
    }

    pairwise_store = (
        generate_pairwise_secrets(range(num_clients))
        if secure_aggregation
        else {cid: {} for cid in range(num_clients)}
    )

    raw_clients: Dict[int, BlockchainEnhancedFlowerClient] = {}
    wrappers: Dict[int, fl.client.Client] = {}

    def get_raw_client(cid: int) -> BlockchainEnhancedFlowerClient:
        cid = int(cid) % num_clients
        if cid not in raw_clients:
            info = addrs[cid % len(addrs)]
            raw_clients[cid] = BlockchainEnhancedFlowerClient(
                client_id=cid,
                client_address=info["address"],
                client_private_key=info["private_key"],
                ppfl_config=cfg,
                blockchain_enabled=blockchain_enabled,
                num_clients=num_clients,
                stake_amount=stake_amount,
                secure_aggregation=secure_aggregation,
                pairwise_secrets=pairwise_store[cid],
                attack_type=attack_type,
                attack_fraction=attack_fraction,
                attack_strength=attack_strength,
            )
        return raw_clients[cid]

    def client_fn(context: Context) -> fl.client.Client:
        cid = int(context.node_id) % num_clients
        if cid not in wrappers:
            wrappers[cid] = get_raw_client(cid).to_client()
        return wrappers[cid]

    client_fn.get_raw_client = get_raw_client
    client_fn.get_all_raw_clients = lambda: dict(raw_clients)
    client_fn.pairwise_store = pairwise_store
    return client_fn


# ============================================================================
# SEQUENTIAL TWO-PHASE FLOWER SIMULATOR
# ============================================================================


class _Ctx:
    def __init__(self, node_id):
        self.node_id = str(node_id)
        self.node_config = {}
        self.run_config = {}
        self.state = fl.common.RecordSet()


class _History:
    def __init__(self):
        self.losses_distributed = []
        self.losses_centralized = []
        self.metrics_distributed = {"fit": [], "evaluate": []}
        self.metrics_centralized = {}


class _DummyClientManager:
    def __init__(self, client_ids):
        self._ids = list(client_ids)

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        class _Proxy:
            def __init__(self, cid):
                self.cid = str(cid)
        return [_Proxy(i) for i in self._ids[:num_clients]]

    def num_available(self):
        return len(self._ids)


def _run_sequential_simulation(
    client_fn,
    strategy: BlockchainAdaptiveWeightedFedAvg,
    num_clients: int,
    num_rounds: int,
    reputation_targets_by_round: Optional[Mapping[int, Mapping[int, float]]] = None,
):
    """Run BC-AWFedAvg with explicit two-phase learning and on-chain governance."""
    hist = _History()
    raw_clients = {
        cid: client_fn.get_raw_client(cid)
        for cid in range(num_clients)
    }

    init_client = raw_clients[0]
    current_parameters = ndarrays_to_parameters(init_client.get_parameters({}))

    print(
        "\n[Sim] Two-phase BC-AWFedAvg protocol enabled "
        "(metrics → weights → client DP → weight → mask → sum → governance)"
    )

    candidate_ids = list(range(num_clients))

    for rnd in range(1, num_rounds + 1):
        print("\n" + "=" * 70)
        print(f"[Sim] Round {rnd}/{num_rounds}")
        print("=" * 70)

        # Determine the eligible federation before opening the on-chain round so
        # a round is never started when the contract cannot be satisfied.
        participant_ids = strategy.get_eligible_client_ids(candidate_ids)

        min_required = int(getattr(strategy, "min_fit_clients", 1))
        if len(participant_ids) < min_required:
            message = (
                f"Round {rnd}: only {len(participant_ids)} eligible clients remain; "
                f"minimum required is {min_required}."
            )
            if strategy.blockchain_enabled and strategy.ppfl.contract is not None:
                raise RuntimeError(message)
            print("⚠️ " + message)
            break

        strategy.begin_round(rnd)

        # ---------------- Phase 1 ----------------
        client_metrics: Dict[int, Dict[str, float]] = {}
        phase1_start = time.time()
        for cid in participant_ids:
            client = raw_clients[cid]
            info = client.prepare_fit(
                current_parameters,
                {"server_round": rnd, "local_epochs": LOCAL_EPOCHS},
            )
            client_metrics[cid] = dict(info["metrics"])
            print(
                f"  Phase 1 — client {cid}: "
                f"reward={client_metrics[cid].get('average_reward', 0.0):.4f}, "
                f"eMBB={client_metrics[cid].get('avg_embb_outage_counter', 0.0):.4f}, "
                f"uRLLC={client_metrics[cid].get('avg_residual_urllc_pkt', 0.0):.4f}"
            )
        phase1_time = time.time() - phase1_start

        # Server: reputation-aware weights.
        weight_start = time.time()
        weights = strategy.compute_adaptive_weights_from_metrics(rnd, client_metrics)
        weight_time = time.time() - weight_start
        reputation_targets = strategy.compute_reputation_targets_from_metrics(client_metrics)
        if reputation_targets_by_round and rnd in reputation_targets_by_round:
            reputation_targets.update({
                int(cid): float(value)
                for cid, value in reputation_targets_by_round[rnd].items()
                if int(cid) in reputation_targets
            })
        isolation_state = strategy.update_isolation_tracking(
            server_round=rnd,
            weights=weights,
            participant_ids=participant_ids,
        )

        # ---------------- Phase 2 ----------------
        masked_results = []
        phase2_start = time.time()
        for cid in participant_ids:
            client = raw_clients[cid]
            masked_params, num_examples, metrics = client.finalize_fit(
                aggregation_weight=weights[cid],
                server_round=rnd,
                participant_ids=participant_ids,
            )
            masked_results.append((cid, masked_params, num_examples, metrics))
        phase2_time = time.time() - phase2_start

        # Record metadata BEFORE aggregation finalization. No model parameters are sent on-chain.
        contribution_tx = strategy.record_round_contributions(rnd, masked_results)

        aggregated_parameters, agg_metrics = strategy.aggregate_masked_fit(
            server_round=rnd,
            masked_results=masked_results,
            client_metrics=client_metrics,
            base_parameters=current_parameters,
            failures=[],
        )
        if aggregated_parameters is None:
            print(f"[Sim] Round {rnd}: aggregation failed; stopping.")
            break

        # Apply the gradual on-chain reputation recurrence so the updated
        # reputation affects the NEXT round.
        reputation_tx = strategy.update_round_reputations(
            server_round=rnd,
            participant_ids=participant_ids,
            reputation_targets=reputation_targets,
        )

        # Publication / governance layer.
        publish_metrics = strategy.publish_global_model(
            server_round=rnd,
            aggregated_parameters=aggregated_parameters,
        )
        strategy.blockchain_round_records.append({
            "round": rnd,
            **publish_metrics,
            "weights": dict(weights),
            "reputations_before_update": dict(strategy.reputation_history_by_round.get(rnd - 1, strategy.current_round_reputations)),
            "reputation_targets": dict(reputation_targets),
            "reputations_after_update": dict(strategy.current_round_reputations),
            "reputation_decay_beta": strategy.reputation_beta,
            "isolation_threshold": (strategy.isolation_threshold_factor / max(len(participant_ids), 1)),
            "isolation_state": isolation_state,
            "contribution_tx": contribution_tx,
            "reputation_tx": reputation_tx,
            "contribution_hashes": dict(strategy.last_round_contribution_hashes),
            "participant_ids": list(participant_ids),
            "phase1_time_s": phase1_time,
            "weight_calculation_time_s": weight_time,
            "phase2_time_s": phase2_time,
        })

        agg_metrics.update(publish_metrics)
        agg_metrics.update({
            "participant_count": len(participant_ids),
            "contribution_tx": contribution_tx,
            "reputation_tx": reputation_tx,
            "reputation_targets": dict(reputation_targets),
        })
        hist.metrics_distributed["fit"].append((rnd, agg_metrics))
        current_parameters = aggregated_parameters

        print(
            f"  Phase 2 complete: secure sum of {len(masked_results)} masked contributions"
        )
        print(f"  Weights sum: {sum(weights.values()):.6f}")
        print(f"  Round protocol time: {phase1_time + weight_time + phase2_time:.2f}s")
        if contribution_tx:
            print(f"  🔗 Contributions recorded: {contribution_tx}")
        if reputation_tx:
            print(f"  🔗 Reputation updated: {reputation_tx}")

        # ---------------- Evaluation ----------------
        eval_results = []
        for cid in candidate_ids:
            try:
                loss, n_eval, eval_metrics = raw_clients[cid].evaluate(
                    current_parameters,
                    {"server_round": rnd},
                )
                eval_results.append((cid, float(loss), int(n_eval), eval_metrics))
            except Exception as exc:
                warnings.warn(f"[Eval] client {cid} failed: {exc}")

        if eval_results:
            total_n = sum(x[2] for x in eval_results)
            weighted_loss = sum(x[1] * x[2] for x in eval_results) / max(total_n, 1)
            eval_metrics = {
                "average_reward": float(
                    np.mean([x[3].get("average_reward", 0.0) for x in eval_results])
                )
            }
            hist.losses_distributed.append((rnd, weighted_loss))
            hist.metrics_distributed["evaluate"].append((rnd, eval_metrics))

    return hist


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================


def run_blockchain_awfedavg_experiment(
    strategy: Optional[BlockchainAdaptiveWeightedFedAvg] = None,
    client_fn=None,
    num_rounds: int = TOTAL_ROUNDS,
    num_clients: int = NUM_CLIENTS,
    **strategy_kwargs,
):
    """Run the corrected two-phase BC-AWFedAvg experiment."""
    print("\n" + "=" * 80)
    print("BC-AWFedAvg — TWO-PHASE WEIGHTED SECURE AGGREGATION")
    print("=" * 80)

    local_results = {}

    client_addresses = strategy_kwargs.pop("client_addresses", None)
    secure_agg = strategy_kwargs.pop("secure_aggregation", True)
    attack_type = strategy_kwargs.pop("attack_type", "none")
    attack_fraction = strategy_kwargs.pop("attack_fraction", 0.0)
    attack_strength = strategy_kwargs.pop("attack_strength", 1.0)
    reputation_targets_by_round = strategy_kwargs.pop("reputation_targets_by_round", None)
    legacy_validity_by_round = strategy_kwargs.pop("validity_by_round", None)
    if legacy_validity_by_round is not None:
        warnings.warn(
            "validity_by_round is deprecated in BC-AWFedAvg. The canonical protocol "
            "uses gradual reputation targets derived from E/R/S metrics."
        )

    if strategy is None:
        strategy = create_blockchain_awfedavg_strategy(**strategy_kwargs)

    if client_fn is None:
        client_fn = make_blockchain_client_fn(
            ppfl_config=strategy.ppfl_config,
            client_addresses=client_addresses,
            blockchain_enabled=strategy.blockchain_enabled,
            num_clients=num_clients,
            secure_aggregation=secure_agg,
            attack_type=attack_type,
            attack_fraction=attack_fraction,
            attack_strength=attack_strength,
        )

    print("\n⚙️ Configuration")
    wc = strategy.weight_calculator
    print(f"  eMBB       : {wc.alpha_embb:.2%}")
    print(f"  URLLC      : {wc.alpha_urllc:.2%}")
    print(f"  Activation : {wc.alpha_activation:.2%}")
    print(f"  Stability  : {wc.alpha_stability:.2%}")
    print(f"  Reputation : {wc.alpha_reputation:.2%}")
    print(f"  SecAgg     : {'enabled' if secure_agg else 'disabled'}")
    print(f"  Client DP  : enabled")
    print(f"  Server DP  : {'enabled' if strategy.apply_coordinator_dp else 'disabled'}")
    print(f"  Blockchain : {'enabled' if strategy.blockchain_enabled else 'disabled'}")
    print(f"  Reputation β: {strategy.reputation_beta:.2f}")
    print(f"  Isolation τ  : {strategy.isolation_threshold_factor / max(num_clients, 1):.4f}")
    print("  Isolation    : gradual influence control; no hard participation gate")

    if strategy.blockchain_enabled and client_addresses:
        strategy._client_addrs_map = {
            i: client_addresses[i % len(client_addresses)]["address"]
            for i in range(num_clients)
        }
        print("\n📝 Ensuring MVNOs are registered on blockchain…")
        for i, info in enumerate(client_addresses[:num_clients]):
            try:
                address = info["address"]
                if strategy.ppfl.is_client_registered(address):
                    print(f"  ✓ MVNO {i} already registered")
                    continue
                public_key = strategy.ppfl.generate_client_keypair(str(i))[1]
                tx = strategy.ppfl.register_client_on_chain(
                    client_address=address,
                    client_private_key=info.get("private_key"),
                    public_key_pem=public_key,
                    stake_amount=0.01,
                )
                print(f"  ✓ MVNO {i} registered: {tx}")
            except Exception as exc:
                if strategy.strict_blockchain:
                    raise RuntimeError(f"[Blockchain] MVNO {i} registration failed: {exc}") from exc
                warnings.warn(f"[Blockchain] MVNO {i} registration failed: {exc}")

    monitor = ResourceMonitor()
    monitor.start_monitoring(interval=1.0)
    fed_start = time.time()

    try:
        hist = _run_sequential_simulation(
            client_fn=client_fn,
            strategy=strategy,
            num_clients=num_clients,
            num_rounds=num_rounds,
            reputation_targets_by_round=reputation_targets_by_round,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        monitor.stop_monitoring()
        print(f"❌ Federated training failed: {exc}")
        return None, local_results, None, None
    finally:
        system_resource_summary = monitor.stop_monitoring()

    print(f"\n✅ Federated training complete in {time.time() - fed_start:.1f}s")

    # Final global-model evaluation.
    federated_results = None
    final_parameters = strategy.get_final_parameters()
    if final_parameters is not None:
        try:
            env = create_phy_env(0)
            input_dim = env.observation_space.shape[0]
            output_dim = env.action_space.n
            pyt_model = PPONetwork(input_dim, output_dim).to(DEVICE)

            arrays = parameters_to_ndarrays(final_parameters)
            model_keys = list(pyt_model.state_dict().keys())
            if len(arrays) == len(model_keys):
                state_dict = OrderedDict(
                    (k, torch.as_tensor(v, device=DEVICE))
                    for k, v in zip(model_keys, arrays)
                )
                pyt_model.load_state_dict(state_dict, strict=True)
                fed_model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                )
                fed_model = pytorch_to_sb3_ppo(pyt_model, fed_model)
                federated_results = evaluate_model_simple(
                    fed_model,
                    env,
                    num_episodes=EVALUATION_EPISODES,
                )
                fed_model.save(
                    f"{BASE_MODEL_PATH}/blockchain_awfedavg_global_model.zip"
                )
                print(f"  Final reward   : {federated_results['average_reward']:.4f}")
                print(f"  Final stability: {federated_results['stability_score']:.4f}")
            env.close()
        except Exception as exc:
            print(f"❌ Final evaluation failed: {exc}")

    try:
        save_awfedavg_results_with_resources(
            strategy,
            local_results,
            federated_results,
        )
    except Exception as exc:
        warnings.warn(f"Results save failed: {exc}")

    bc_records_path = (
        f"{BASE_MODEL_PATH}/blockchain_records_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(bc_records_path, "w", encoding="utf-8") as fh:
        json.dump(strategy.blockchain_round_records, fh, indent=2, default=str)

    print(f"\n📂 Blockchain records saved: {bc_records_path}")
    strategy.print_blockchain_performance_summary()

    return hist, local_results, federated_results, system_resource_summary


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    print("🚀 BC-AWFedAvg corrected two-phase integration")
    print("  ✅ Reputation-aware five-criterion weighting")
    print("  ✅ Persistent clients in sequential simulation")
    print("  ✅ Client DP before weighting")
    print("  ✅ Weight applied before pairwise masking")
    print("  ✅ Server performs sum-only secure aggregation")
    print("  ✅ Persistent leaky-integrator reputation (β=0.85)")
    print("  ✅ Gradual isolation tracking at θ_iso=1/(2K), without hard exclusion")
    print("  ✅ Blockchain/IPFS publication kept separate from live aggregation")

    config_path = os.path.join(os.path.dirname(__file__), "contract_info.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        blockchain_enabled = True
        contract_address = cfg["contract_address"]
        contract_abi_path = config_path
        coordinator_private_key = cfg["coordinator_private_key"]
        client_addresses = cfg["clients"]
    else:
        blockchain_enabled = False
        contract_address = None
        contract_abi_path = None
        coordinator_private_key = None
        client_addresses = None
        print("⚠️ contract_info.json not found — blockchain disabled")

    run_blockchain_awfedavg_experiment(
        blockchain_enabled=blockchain_enabled,
        blockchain_provider="http://127.0.0.1:8545",
        contract_address=contract_address,
        contract_abi_path=contract_abi_path,
        coordinator_private_key=coordinator_private_key,
        ipfs_addr="/ip4/127.0.0.1/tcp/5001",
        epsilon=1.0,
        delta=1e-5,
        clip_norm=1.0,
        alpha_embb=0.22,
        alpha_urllc=0.38,
        alpha_activation=0.20,
        alpha_stability=0.15,
        alpha_reputation=0.05,
        reputation_beta=0.85,
        reputation_scale=1000.0,
        isolation_threshold_factor=0.5,
        secure_aggregation=True,
        num_rounds=TOTAL_ROUNDS,
        num_clients=NUM_CLIENTS,
        client_addresses=client_addresses,
    )
