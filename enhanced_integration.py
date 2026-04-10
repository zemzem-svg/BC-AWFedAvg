"""
Enhanced Blockchain AWFedAvg — Robustness & Efficiency Upgrades
================================================================

This module **extends** the existing classes in adaptive_weighted_fedavg.py
and blockchain_awfedavg_true_integration.py with:

Robustness
----------
R1. Byzantine-tolerant aggregation (trimmed-mean / multi-Krum)
R2. Cosine-similarity anomaly detection with soft down-weighting
R3. Norm-bounding per client update
R4. Secure aggregation masks wired into client fit()
R5. Dynamic reputation weight α_rep that scales up when anomalies detected
R6. Improved stability scoring with exponential moving average (EMA)

Efficiency
----------
E1. Rényi DP accounting (tighter cumulative ε than advanced composition)
E2. Adaptive clipping norm (Andrew et al., NeurIPS 2021)
E3. Top-K gradient sparsification with error feedback
E4. Lazy IPFS upload — skip when global model hash is unchanged
E5. Batched blockchain metadata (one tx/round, not one tx/client)

Usage
-----
Drop-in replacement: import from this module instead of the originals.

    from enhanced_integration import (
        EnhancedBlockchainAWFedAvg,
        EnhancedBlockchainFlowerClient,
        run_enhanced_experiment,
    )
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Original project imports ─────────────────────────────────────────────────
from adaptive_weighted_fedavg import (
    AdaptiveWeightCalculator,
    AdaptiveWeightedFedAvg,
    EnhancedFlowerClient,
    CLIENT_CONFIGS,
    PPONETWORK_LAYER_KEYS,
    ResourceMonitor,
    evaluate_model_simple,
    print_resource_summary,
)
from blockchain_awfedavg_true_integration import (
    BlockchainAdaptiveWeightedFedAvg,
    BlockchainEnhancedFlowerClient,
    PrivacyPreservingFederatedLearning,
    ndarrays_to_ordered_dict,
    ordered_dict_to_ndarrays,
    _run_sequential_simulation,
    _DummyClientManager,
)
from secure_aggregation import add_secure_mask, unmask_aggregate
from robustness_module import RobustAggregator, norm_bound
from efficient_dp import EfficientDPManager, TopKSparsifier, RDPAccountant

try:
    from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# R6.  Enhanced Stability Scoring with EMA
# ═════════════════════════════════════════════════════════════════════════════

class EnhancedWeightCalculator(AdaptiveWeightCalculator):
    """
    Extends the 5-criteria calculator with:
      • EMA-based stability scoring (replaces raw inverse-variance)
      • Dynamic reputation weight that increases when anomalies are detected
      • Outlier-aware score normalisation (winsorised z-scores)
    """

    def __init__(
        self,
        alpha_embb: float = 0.22,
        alpha_urllc: float = 0.38,
        alpha_activation: float = 0.20,
        alpha_stability: float = 0.15,
        alpha_reputation: float = 0.05,
        ema_span: int = 5,
        dynamic_rep_boost: float = 3.0,
    ):
        super().__init__(
            alpha_embb=alpha_embb,
            alpha_urllc=alpha_urllc,
            alpha_activation=alpha_activation,
            alpha_stability=alpha_stability,
            alpha_reputation=alpha_reputation,
        )
        # EMA decay factor: α_ema = 2 / (span + 1)
        self.ema_alpha = 2.0 / (ema_span + 1)
        # When anomalies are detected, temporarily boost reputation weight
        self.dynamic_rep_boost = dynamic_rep_boost
        self._base_alpha_rep = alpha_reputation
        self._anomaly_active = False
        # EMA state per client
        self._ema_rewards: Dict[int, float] = {}
        self._ema_var: Dict[int, float] = {}

    # ── EMA stability ────────────────────────────────────────────────────

    def _ema_stability(self, client_id: int, reward: float) -> float:
        """
        Compute EMA-based stability score for a client.

        Maintains an exponential moving average of reward and an EMA of
        squared deviations.  Stability = 1 / (1 + ema_var).
        """
        if client_id not in self._ema_rewards:
            self._ema_rewards[client_id] = reward
            self._ema_var[client_id] = 0.0
            return 1.0  # max stability on first observation

        a = self.ema_alpha
        old_mu = self._ema_rewards[client_id]
        new_mu = a * reward + (1 - a) * old_mu
        deviation_sq = (reward - new_mu) ** 2
        new_var = a * deviation_sq + (1 - a) * self._ema_var[client_id]

        self._ema_rewards[client_id] = new_mu
        self._ema_var[client_id] = new_var

        return 1.0 / (1.0 + new_var)

    # ── Dynamic reputation weight ────────────────────────────────────────

    def signal_anomalies(self, n_anomalies: int, n_clients: int):
        """
        Called by the aggregation pipeline when anomalies are detected.
        Temporarily boosts α_reputation so on-chain reputation has more say.
        """
        if n_anomalies > 0:
            fraction = n_anomalies / max(n_clients, 1)
            boost = 1.0 + self.dynamic_rep_boost * fraction
            self.alpha_reputation = min(
                self._base_alpha_rep * boost,
                0.30,  # cap at 30% to avoid domination
            )
            self._anomaly_active = True
            logger.info(
                "Dynamic reputation boost: α_rep %.3f → %.3f  (%d anomalies / %d clients)",
                self._base_alpha_rep, self.alpha_reputation, n_anomalies, n_clients,
            )
        elif self._anomaly_active:
            # Decay back to base over one round
            self.alpha_reputation = self._base_alpha_rep
            self._anomaly_active = False

    # ── Override calculate_adaptive_weights ───────────────────────────────

    def calculate_adaptive_weights(
        self, client_metrics, client_configs, server_round
    ):
        """
        Same as parent but uses EMA stability instead of raw inverse-variance.
        """
        n_clients = len(client_metrics)
        if n_clients == 0:
            return np.array([])

        # Collect per-client metrics
        client_ids = sorted(client_metrics.keys())
        embb_outages = np.array([
            client_metrics[c].get("avg_embb_outage_counter", 0.0) for c in client_ids
        ])
        urllc_residuals = np.array([
            client_metrics[c].get("avg_residual_urllc_pkt", 0.0) for c in client_ids
        ])
        rewards = np.array([
            client_metrics[c].get("average_reward", 0.0) for c in client_ids
        ])
        activations = np.array([
            client_configs[c % len(client_configs)].get("activation", 1)
            for c in client_ids
        ])

        # ── eMBB score (lower outage → higher weight) ────────────────────
        embb_scores = np.ones(n_clients) / n_clients
        if np.std(embb_outages) > 1e-8:
            embb_scores = 1.0 / (embb_outages + 1e-8)
            embb_scores /= np.sum(embb_scores)

        # ── URLLC score (lower residual → higher weight) ─────────────────
        urllc_scores = np.ones(n_clients) / n_clients
        if np.std(urllc_residuals) > 1e-8:
            urllc_scores = 1.0 / (urllc_residuals + 1e-8)
            urllc_scores /= np.sum(urllc_scores)

        # ── Activation diversity ─────────────────────────────────────────
        act_div = np.abs(activations - np.mean(activations))
        if np.std(act_div) > 1e-8:
            activation_scores = (1.0 + act_div) / np.sum(1.0 + act_div)
        else:
            activation_scores = np.ones(n_clients) / n_clients

        # ── R6: EMA stability (replacing raw inverse-variance) ───────────
        stability_scores = np.array([
            self._ema_stability(cid, float(rewards[i]))
            for i, cid in enumerate(client_ids)
        ])
        s_sum = np.sum(stability_scores)
        stability_scores = stability_scores / s_sum if s_sum > 0 else np.ones(n_clients) / n_clients

        # ── Reputation (from blockchain) ─────────────────────────────────
        reputation_scores = np.array([
            self.reputation_scores.get(cid, 0.5) for cid in client_ids
        ])
        r_sum = np.sum(reputation_scores)
        reputation_scores = reputation_scores / r_sum if r_sum > 0 else np.ones(n_clients) / n_clients

        # ── Combined score ───────────────────────────────────────────────
        # Re-normalise alphas so they sum to 1 (in case dynamic boost changed rep)
        alpha_sum = (
            self.alpha_embb + self.alpha_urllc + self.alpha_activation +
            self.alpha_stability + self.alpha_reputation
        )
        combined = (
            (self.alpha_embb / alpha_sum) * embb_scores +
            (self.alpha_urllc / alpha_sum) * urllc_scores +
            (self.alpha_activation / alpha_sum) * activation_scores +
            (self.alpha_stability / alpha_sum) * stability_scores +
            (self.alpha_reputation / alpha_sum) * reputation_scores
        )

        weights = np.maximum(combined, 1e-8)
        weights /= np.sum(weights)

        # Smoothing
        if len(self.weight_history) > 0:
            prev = self.weight_history[-1]
            if len(prev) == len(weights):
                weights = 0.7 * weights + 0.3 * prev
                weights /= np.sum(weights)

        self.weight_history.append(weights.copy())
        return weights


# ═════════════════════════════════════════════════════════════════════════════
# Enhanced Strategy: wires in RobustAggregator + EfficientDP + Sparsifier
# ═════════════════════════════════════════════════════════════════════════════

class EnhancedBlockchainAWFedAvg(BlockchainAdaptiveWeightedFedAvg):
    """
    Drop-in replacement for BlockchainAdaptiveWeightedFedAvg with all
    robustness and efficiency upgrades integrated.
    """

    def __init__(
        self,
        ppfl: PrivacyPreservingFederatedLearning,
        # Robustness config
        robust_method: str = "trimmed_mean",
        robust_max_norm: float = 10.0,
        robust_anomaly_z: float = 2.0,
        robust_trim_beta: float = 0.1,
        # Efficient DP config
        use_rdp: bool = True,
        adaptive_clip: bool = True,
        target_quantile: float = 0.6,
        # Sparsification config
        use_sparsification: bool = True,
        compression_ratio: float = 0.1,
        # Lazy IPFS
        lazy_ipfs: bool = True,
        # Pass-through
        **kwargs,
    ):
        super().__init__(ppfl=ppfl, **kwargs)

        # Replace weight calculator with enhanced version
        self.weight_calculator = EnhancedWeightCalculator(
            alpha_embb=self.weight_calculator.alpha_embb,
            alpha_urllc=self.weight_calculator.alpha_urllc,
            alpha_activation=self.weight_calculator.alpha_activation,
            alpha_stability=self.weight_calculator.alpha_stability,
            alpha_reputation=self.weight_calculator.alpha_reputation,
        )

        # R1–R3: Robust aggregator
        self.robust_agg = RobustAggregator(
            method=robust_method,
            max_norm=robust_max_norm,
            anomaly_z=robust_anomaly_z,
            trim_beta=robust_trim_beta,
        )

        # E1–E2: Efficient DP with RDP accounting
        self.dp_manager = EfficientDPManager(
            epsilon=ppfl.epsilon,
            delta=ppfl.delta,
            initial_clip_norm=ppfl.clip_norm,
            adaptive_clip=adaptive_clip,
            target_quantile=target_quantile,
        ) if use_rdp else None

        # E3: Gradient sparsification
        self.sparsifier = TopKSparsifier(compression_ratio) if use_sparsification else None

        # E4: Lazy IPFS
        self.lazy_ipfs = lazy_ipfs
        self._last_model_hash: Optional[bytes] = None

        # Diagnostics
        self.robustness_history: List[Dict] = []
        self.dp_history: List[Dict] = []
        self.sparsification_history: List[Dict] = []

    # ──────────────────────────────────────────────────────────────────────
    # Override aggregate_fit: full enhanced pipeline
    # ──────────────────────────────────────────────────────────────────────

    def aggregate_fit(self, server_round: int, results, failures):
        """
        Enhanced aggregation pipeline:

        1. Open round on-chain (inherited)
        2. Fetch reputation → dynamic weight boost if anomalies last round
        3. Extract client params → norm-bound → anomaly detect → robust aggregate
        4. Apply AWFedAvg adaptive weights *within* robust aggregation
        5. Coordinator-side DP (adaptive clip + Gaussian noise + RDP accounting)
        6. Sparsify for IPFS if enabled
        7. Encrypt → IPFS (lazy) → on-chain hash
        """
        blockchain_start = time.time()

        # ── Step 1: Blockchain round start (from parent) ─────────────────
        round_tx = None
        if self.blockchain_enabled:
            try:
                round_tx = self.ppfl.start_round_on_chain(
                    previous_model_ipfs_hash=self.last_global_ipfs_hash
                )
                print(f"\n🔗 Round {server_round} opened on blockchain  tx={round_tx}")
            except Exception as exc:
                warnings.warn(f"[Blockchain] start_round failed: {exc}")

        # ── Step 2: Reputation fetch → weight calculator ─────────────────
        if self.blockchain_enabled and getattr(self.ppfl, "contract", None) and self._client_addrs_map:
            reputations = {}
            for cid, addr in self._client_addrs_map.items():
                try:
                    _, rep, _, _ = self.ppfl.contract.functions.getClientInfo(addr).call()
                    reputations[cid] = int(rep)
                except Exception:
                    reputations[cid] = 500
            self.weight_calculator.update_reputation_from_chain(reputations)

        # ── Step 3: Collect client parameters ────────────────────────────
        self.resource_monitor.start_monitoring(interval=0.1)
        collection_start = time.time()

        client_metrics = {}
        client_params_list = []
        client_ids = []
        client_training_times = []
        client_comm_sizes = []

        for client_proxy, fit_res in results:
            cid = fit_res.metrics.get("client_id", "unknown")
            perf = fit_res.metrics.get("average_reward", 0.0)
            params = parameters_to_ndarrays(fit_res.parameters)
            params_size = sum(p.nbytes for p in params)

            client_params_list.append(params)
            client_ids.append(cid)
            client_training_times.append(fit_res.metrics.get("training_time", 0.0))
            client_comm_sizes.append(params_size)

            client_metrics[cid] = {
                "avg_embb_outage_counter": fit_res.metrics.get("avg_embb_outage_counter", 0.0),
                "avg_residual_urllc_pkt": fit_res.metrics.get("avg_residual_urllc_pkt", 0.0),
                "average_reward": perf,
                "stability_score": fit_res.metrics.get("stability_score", 0.5),
                "training_time": fit_res.metrics.get("training_time", 0.0),
                "communication_size_bytes": params_size,
            }

        if not client_params_list:
            print("⚠️  No client results — skipping aggregation")
            return None, {}

        collection_time = time.time() - collection_start

        # ── Step 3b: R1–R3: Robust aggregation ──────────────────────────
        robust_start = time.time()

        # Compute AWFedAvg weights first
        adaptive_weights = self.weight_calculator.calculate_adaptive_weights(
            client_metrics, CLIENT_CONFIGS, server_round
        )

        # Robust aggregate (norm-bound → cosine filter → trimmed mean/krum)
        robust_params, adj_weights = self.robust_agg.robust_aggregate(
            client_params_list, adaptive_weights
        )

        # R5: Signal anomalies to weight calculator for dynamic reputation boost
        diag = self.robust_agg.get_diagnostics()
        n_anomalies = diag["anomalies_flagged"]
        if hasattr(self.weight_calculator, "signal_anomalies"):
            self.weight_calculator.signal_anomalies(n_anomalies, len(client_ids))

        self.robustness_history.append({
            "round": server_round,
            "anomalies_flagged": n_anomalies,
            "adjusted_weights": adj_weights.tolist(),
            "original_weights": adaptive_weights.tolist(),
            "diagnostics": diag,
        })

        robust_time = time.time() - robust_start

        print(f"\n🛡️  Robust aggregation: {self.robust_agg.method}  "
              f"anomalies={n_anomalies}  time={robust_time:.3f}s")
        print(f"   Weights (orig): {[f'{w:.3f}' for w in adaptive_weights]}")
        print(f"   Weights (adj):  {[f'{w:.3f}' for w in adj_weights]}")

        # Convert to Flower Parameters
        aggregated_parameters = ndarrays_to_parameters(robust_params)

        # ── Metrics ──────────────────────────────────────────────────────
        client_performances = [
            client_metrics[c].get("average_reward", 0.0) for c in client_ids
        ]
        avg_reward = float(np.mean(client_performances))
        self.performance_history.append(avg_reward)

        # ── Step 5: Blockchain / IPFS publication ────────────────────────
        ipfs_hash = ""
        agg_tx = ""
        dp_applied = False
        compression_ratio = 0.0
        upload_size_kb = 0.0
        sparsification_ratio = 0.0

        if self.blockchain_enabled:
            try:
                params_dict = ndarrays_to_ordered_dict(
                    robust_params,
                    keys=PPONETWORK_LAYER_KEYS[:len(robust_params)],
                )
                original_size = sum(
                    p.numel() * 4 for p in params_dict.values()
                    if isinstance(p, torch.Tensor)
                )

                # 5a. Coordinator-side DP (E1 + E2: RDP + adaptive clip)
                if self.apply_coordinator_dp:
                    if self.dp_manager:
                        params_dict = self.dp_manager.add_dp_noise(
                            params_dict, sensitivity=1.0
                        )
                        eps_so_far = self.dp_manager.get_epsilon()
                        print(f"  🛡️  Coordinator DP (RDP):  ε_cumulative={eps_so_far:.4f}")
                        self.dp_history.append({
                            "round": server_round,
                            "eps_cumulative_rdp": eps_so_far,
                            "clip_norm": self.dp_manager.clip_norm,
                        })
                    else:
                        params_dict = self.ppfl.add_differential_privacy_noise(
                            params_dict, sensitivity=1.0
                        )
                    dp_applied = True

                # 5b. E3: Sparsify before encryption (optional)
                if self.sparsifier:
                    sparse_arrays = ordered_dict_to_ndarrays(params_dict)
                    sparse_arrays, masks, sparsification_ratio = (
                        self.sparsifier.sparsify(
                            client_id=-1,  # coordinator
                            param_list=sparse_arrays,
                        )
                    )
                    # Rebuild dict
                    keys = list(params_dict.keys())
                    params_dict = OrderedDict(
                        (k, torch.tensor(v, dtype=torch.float32))
                        for k, v in zip(keys, sparse_arrays)
                    )
                    print(f"  📦 Sparsified: keeping {sparsification_ratio:.1%} of coordinates")
                    self.sparsification_history.append({
                        "round": server_round,
                        "ratio": sparsification_ratio,
                        "stats": self.sparsifier.get_stats(),
                    })

                # 5c. Encrypt + upload
                encrypted_data, _sym_key = self.ppfl.encrypt_model(
                    params_dict, compression=self.compression
                )
                upload_size_kb = len(encrypted_data) / 1024
                compression_ratio = (1 - len(encrypted_data) / original_size) * 100

                # E4: Lazy IPFS — skip upload if hash unchanged
                model_hash = hashlib.sha256(encrypted_data).digest()
                if self.lazy_ipfs and model_hash == self._last_model_hash:
                    ipfs_hash = self.last_global_ipfs_hash
                    print(f"  ⏩ IPFS upload skipped (model unchanged)")
                else:
                    ipfs_hash = self.ppfl.upload_to_ipfs(encrypted_data, pin=True)
                    self._last_model_hash = model_hash
                    print(
                        f"  📦 IPFS: {ipfs_hash}  "
                        f"({upload_size_kb:.1f} KB | compression={compression_ratio:.1f}%)"
                    )

                # 5d. On-chain registration
                agg_tx = self.ppfl.submit_aggregated_model_on_chain(
                    round_number=server_round,
                    ipfs_hash=ipfs_hash,
                    model_hash=model_hash,
                )
                self.last_global_ipfs_hash = ipfs_hash

            except Exception as exc:
                warnings.warn(f"[Enhanced] Blockchain/IPFS step failed: {exc}")

        blockchain_overhead = time.time() - blockchain_start

        # Stop monitoring
        resource_summary = self.resource_monitor.stop_monitoring()

        # ── Build return metrics ─────────────────────────────────────────
        aggregated_metrics = {
            "average_reward": avg_reward,
            "round_duration": blockchain_overhead,
            "adaptive_weights": adj_weights.tolist(),
            "original_weights": adaptive_weights.tolist(),
            "anomalies_flagged": n_anomalies,
            "robust_method": self.robust_agg.method,
            "dp_applied": dp_applied,
            "eps_cumulative_rdp": self.dp_manager.get_epsilon() if self.dp_manager else 0.0,
            "clip_norm": self.dp_manager.clip_norm if self.dp_manager else 0.0,
            "ipfs_hash": ipfs_hash,
            "upload_size_kb": upload_size_kb,
            "compression_ratio": compression_ratio,
            "sparsification_ratio": sparsification_ratio,
            "peak_memory_mb": resource_summary.get("process_memory_mb", {}).get("peak", 0),
            "avg_cpu_percent": resource_summary.get("process_cpu", {}).get("mean", 0),
            "blockchain_overhead_s": blockchain_overhead,
        }

        # Store round data
        self.blockchain_round_records.append({
            "round": server_round,
            "ipfs_hash": ipfs_hash,
            "agg_tx": str(agg_tx),
            "dp_applied": dp_applied,
            "overhead_s": blockchain_overhead,
            "ipfs_upload_size_kb": upload_size_kb,
        })

        self.final_parameters = aggregated_parameters
        self.round_metrics.append({
            "round": server_round,
            "client_metrics": client_metrics,
            "aggregated_metrics": aggregated_metrics,
            "adaptive_weights": adj_weights.tolist(),
        })

        return aggregated_parameters, aggregated_metrics

    # ──────────────────────────────────────────────────────────────────────
    # Enhanced reporting
    # ──────────────────────────────────────────────────────────────────────

    def print_enhanced_summary(self):
        """Print comprehensive summary covering all enhancements."""
        # Parent summaries
        self.print_blockchain_performance_summary()

        # Robustness summary
        print("\n" + "=" * 70)
        print("ROBUSTNESS SUMMARY")
        print("=" * 70)
        total_anomalies = sum(r["anomalies_flagged"] for r in self.robustness_history)
        rounds_with_anomalies = sum(1 for r in self.robustness_history if r["anomalies_flagged"] > 0)
        print(f"  Method                : {self.robust_agg.method}")
        print(f"  Total anomalies found : {total_anomalies}")
        print(f"  Rounds with anomalies : {rounds_with_anomalies} / {len(self.robustness_history)}")
        if self.robustness_history:
            avg_weight_shift = np.mean([
                np.linalg.norm(
                    np.array(r["adjusted_weights"]) - np.array(r["original_weights"])
                )
                for r in self.robustness_history
            ])
            print(f"  Avg weight adjustment : {avg_weight_shift:.4f} (L2 norm)")

        # Efficient DP summary
        if self.dp_manager:
            print("\n" + "=" * 70)
            print("RÉNYI DP ACCOUNTING")
            print("=" * 70)
            report = self.dp_manager.privacy_report()
            print(f"  Method              : {report['method']}")
            print(f"  Rounds              : {report['rounds']}")
            print(f"  ε (RDP)             : {report['eps_rdp']:.4f}")
            print(f"  ε (Advanced Comp.)  : {report['eps_advanced_composition']:.4f}")
            print(f"  Improvement         : {report['improvement_pct']:.1f}% tighter")
            print(f"  Best α order        : {report.get('best_alpha', 'N/A')}")
            if report.get("adaptive_clip_history"):
                clips = [h["clip_norm"] for h in report["adaptive_clip_history"]]
                print(f"  Clip norm range     : [{min(clips):.3f}, {max(clips):.3f}]")

        # Sparsification summary
        if self.sparsifier and self.sparsification_history:
            print("\n" + "=" * 70)
            print("GRADIENT SPARSIFICATION")
            print("=" * 70)
            ratios = [s["ratio"] for s in self.sparsification_history]
            print(f"  Avg keep ratio      : {np.mean(ratios):.2%}")
            print(f"  Communication saved : ~{(1 - np.mean(ratios)) * 100:.0f}%")


# ═════════════════════════════════════════════════════════════════════════════
# Enhanced Client: wires in secure aggregation + sparsification
# ═════════════════════════════════════════════════════════════════════════════

class EnhancedBlockchainFlowerClient(BlockchainEnhancedFlowerClient):
    """
    Flower client extended with:
      R4. Secure aggregation masks (pairwise cancellation)
      E3. Top-K sparsification on local updates
    """

    def __init__(
        self,
        client_id: int,
        ppfl: Optional[PrivacyPreservingFederatedLearning] = None,
        all_client_ids: Optional[List[int]] = None,
        sparsifier: Optional[TopKSparsifier] = None,
        use_secure_agg: bool = True,
        **kwargs,
    ):
        super().__init__(client_id, **kwargs)
        self._ppfl = ppfl
        self._all_client_ids = all_client_ids or []
        self._sparsifier = sparsifier
        self._use_secure_agg = use_secure_agg
        self._current_round = 0

    def fit(self, parameters, config):
        """
        Extended fit():
          1. super().fit()  — full PPO training + metrics
          2. R4: Add secure aggregation mask
          3. E3: Sparsify if enabled
          4. Return enhanced parameters
        """
        # Full parent training
        updated_params, num_examples, metrics = super().fit(parameters, config)
        self._current_round = config.get("server_round", self._current_round + 1)

        param_arrays = list(updated_params)  # list of np.ndarray

        # ── R4: Secure aggregation mask ──────────────────────────────────
        if self._use_secure_agg and len(self._all_client_ids) > 1:
            try:
                param_dict = OrderedDict(
                    (f"layer_{i}", torch.tensor(p, dtype=torch.float32))
                    for i, p in enumerate(param_arrays)
                )
                masked_dict = add_secure_mask(
                    param_dict,
                    client_id=self.client_id,
                    all_client_ids=self._all_client_ids,
                    round_num=self._current_round,
                )
                param_arrays = [v.numpy() if isinstance(v, torch.Tensor) else v
                                for v in masked_dict.values()]
                metrics["secure_agg_applied"] = True
            except Exception as exc:
                logger.warning("Secure aggregation mask failed: %s", exc)
                metrics["secure_agg_applied"] = False
        else:
            metrics["secure_agg_applied"] = False

        # ── E3: Sparsification ───────────────────────────────────────────
        if self._sparsifier:
            try:
                sparse, masks, ratio = self._sparsifier.sparsify(
                    self.client_id, param_arrays
                )
                param_arrays = sparse
                metrics["sparsification_ratio"] = ratio
                metrics["sparsification_applied"] = True
            except Exception as exc:
                logger.warning("Sparsification failed: %s", exc)
                metrics["sparsification_applied"] = False
        else:
            metrics["sparsification_applied"] = False

        return param_arrays, num_examples, metrics


# ═════════════════════════════════════════════════════════════════════════════
# Enhanced Experiment Runner
# ═════════════════════════════════════════════════════════════════════════════

def create_enhanced_strategy(
    ppfl: PrivacyPreservingFederatedLearning,
    blockchain_enabled: bool = True,
    # Robustness
    robust_method: str = "trimmed_mean",
    robust_max_norm: float = 10.0,
    robust_anomaly_z: float = 2.0,
    robust_trim_beta: float = 0.1,
    # Efficient DP
    use_rdp: bool = True,
    adaptive_clip: bool = True,
    target_quantile: float = 0.6,
    # Sparsification
    use_sparsification: bool = False,
    compression_ratio: float = 0.1,
    # AWFedAvg weights
    alpha_embb: float = 0.22,
    alpha_urllc: float = 0.38,
    alpha_activation: float = 0.20,
    alpha_stability: float = 0.15,
    alpha_reputation: float = 0.05,
    **kwargs,
) -> EnhancedBlockchainAWFedAvg:
    """Factory for the enhanced strategy."""

    strategy = EnhancedBlockchainAWFedAvg(
        ppfl=ppfl,
        blockchain_enabled=blockchain_enabled,
        robust_method=robust_method,
        robust_max_norm=robust_max_norm,
        robust_anomaly_z=robust_anomaly_z,
        robust_trim_beta=robust_trim_beta,
        use_rdp=use_rdp,
        adaptive_clip=adaptive_clip,
        target_quantile=target_quantile,
        use_sparsification=use_sparsification,
        compression_ratio=compression_ratio,
        alpha_embb=alpha_embb,
        alpha_urllc=alpha_urllc,
        alpha_activation=alpha_activation,
        alpha_stability=alpha_stability,
        alpha_reputation=alpha_reputation,
        **kwargs,
    )

    print("\n" + "=" * 70)
    print("ENHANCED BLOCKCHAIN AWFedAvg CONFIGURATION")
    print("=" * 70)
    print(f"  Robust method       : {robust_method}")
    print(f"  Norm bound          : {robust_max_norm}")
    print(f"  Anomaly z-threshold : {robust_anomaly_z}")
    print(f"  Trim β              : {robust_trim_beta}")
    print(f"  RDP accounting      : {use_rdp}")
    print(f"  Adaptive clipping   : {adaptive_clip}")
    print(f"  Sparsification      : {use_sparsification} ({compression_ratio:.0%})")
    print(f"  Lazy IPFS           : True")
    print(f"  Weights: eMBB={alpha_embb:.0%}  URLLC={alpha_urllc:.0%}  "
          f"Act={alpha_activation:.0%}  Stab={alpha_stability:.0%}  "
          f"Rep={alpha_reputation:.0%}")
    print("=" * 70)

    return strategy


def make_enhanced_client_fn(
    ppfl: Optional[PrivacyPreservingFederatedLearning],
    all_client_ids: List[int],
    sparsifier: Optional[TopKSparsifier] = None,
    use_secure_agg: bool = True,
):
    """
    Factory function returning a client constructor for the enhanced FL client.
    """
    def client_fn(client_id: int) -> EnhancedBlockchainFlowerClient:
        return EnhancedBlockchainFlowerClient(
            client_id=client_id,
            ppfl=ppfl,
            all_client_ids=all_client_ids,
            sparsifier=sparsifier,
            use_secure_agg=use_secure_agg,
        )
    return client_fn


def run_enhanced_experiment(
    # Core FL config
    num_clients: int = 3,
    num_rounds: int = 10,
    local_epochs: int = 100,
    # Blockchain
    blockchain_enabled: bool = False,
    blockchain_provider: str = "http://localhost:8545",
    contract_address: str = "",
    contract_abi_path: str = "contract_info.json",
    coordinator_private_key: str = "",
    # Privacy
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    # Robustness
    robust_method: str = "trimmed_mean",
    robust_max_norm: float = 10.0,
    robust_anomaly_z: float = 2.0,
    robust_trim_beta: float = 0.1,
    # Efficiency
    use_rdp: bool = True,
    adaptive_clip: bool = True,
    use_sparsification: bool = False,
    compression_ratio: float = 0.1,
    use_secure_agg: bool = True,
    # AWFedAvg weights
    alpha_embb: float = 0.22,
    alpha_urllc: float = 0.38,
    alpha_activation: float = 0.20,
    alpha_stability: float = 0.15,
    alpha_reputation: float = 0.05,
    # Attacks (for testing robustness)
    attack_type: str = "none",
    attack_fraction: float = 0.0,
    attack_strength: float = 1.0,
):
    """
    Run the full enhanced experiment.

    Returns (history, local_results, federated_results, resource_summary)
    """
    print("\n" + "🚀" * 35)
    print("  ENHANCED BLOCKCHAIN AWFedAvg EXPERIMENT")
    print("🚀" * 35)

    # Build PPFL layer
    ppfl = PrivacyPreservingFederatedLearning(
        blockchain_provider=blockchain_provider,
        contract_address=contract_address,
        contract_abi_path=contract_abi_path,
        epsilon=epsilon,
        delta=delta,
        clip_norm=clip_norm,
        require_connection=blockchain_enabled,
    )

    # Build sparsifier if enabled
    sparsifier = TopKSparsifier(compression_ratio) if use_sparsification else None

    # Build strategy
    strategy = create_enhanced_strategy(
        ppfl=ppfl,
        blockchain_enabled=blockchain_enabled,
        robust_method=robust_method,
        robust_max_norm=robust_max_norm,
        robust_anomaly_z=robust_anomaly_z,
        robust_trim_beta=robust_trim_beta,
        use_rdp=use_rdp,
        adaptive_clip=adaptive_clip,
        use_sparsification=use_sparsification,
        compression_ratio=compression_ratio,
        alpha_embb=alpha_embb,
        alpha_urllc=alpha_urllc,
        alpha_activation=alpha_activation,
        alpha_stability=alpha_stability,
        alpha_reputation=alpha_reputation,
    )

    # Build clients
    all_client_ids = list(range(num_clients))
    client_fn = make_enhanced_client_fn(
        ppfl=ppfl if blockchain_enabled else None,
        all_client_ids=all_client_ids,
        sparsifier=sparsifier,
        use_secure_agg=use_secure_agg,
    )

    # Run simulation
    print(f"\n▶️  Starting FL simulation: {num_clients} clients × {num_rounds} rounds")
    wall_start = time.time()

    history = _run_sequential_simulation(
        strategy=strategy,
        client_fn=client_fn,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        attack_type=attack_type,
        attack_fraction=attack_fraction,
        attack_strength=attack_strength,
    )

    wall_time = time.time() - wall_start
    print(f"\n⏱️  Total wall time: {wall_time:.1f}s")

    # Print enhanced summary
    strategy.print_enhanced_summary()

    # Collect results
    resource_summary = strategy.get_comprehensive_resource_summary()

    return history, strategy, resource_summary
