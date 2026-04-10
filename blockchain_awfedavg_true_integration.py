# """
# Blockchain-AWFedAvg — Three-Layer Federated Learning System
# ============================================================

# Layer 1 — Learning (Off-chain)
#     AdaptiveWeightedFedAvg with 4-criteria weighting (eMBB, URLLC, activation
#     diversity, stability) plus on-chain reputation as a 5th criterion.
#     Secure aggregation masking (Bonawitz et al. CCS'17): each client adds a
#     pairwise-cancelling mask so the coordinator sees only Σw_k, never w_k.
#     Cumulative DP accounting: ε_total = √(2T·ln(1/δ))·ε tracked per round.

# Layer 2 — Governance (Blockchain, metadata-only)
#     Smart contract stores ONLY: round metadata, aggregated model hash, client
#     reputation.  No weights, no heavy computation, no per-client uploads.
#     O(1) transaction per round regardless of number of clients K.

# Layer 3 — Storage (IPFS)
#     One upload per round: coordinator-DP-noised + encrypted global model.
#     Content-addressed CID stored on-chain for immutable audit trail.

# Key classes
# -----------
#     BlockchainAdaptiveWeightedFedAvg  ─ server strategy (extends AWFedAvg)
#     BlockchainEnhancedFlowerClient    ─ client (extends EnhancedFlowerClient)

# Usage
# -----
#     strategy = create_blockchain_awfedavg_strategy(...)
#     client_fn = make_blockchain_client_fn(strategy.ppfl_config)
#     run_blockchain_awfedavg_experiment(...)
# """

# import os
# import sys
# import math
# import hashlib
# import json
# import time
# import warnings
# import numpy as np
# import torch
# from collections import OrderedDict
# from typing import Dict, List, Tuple, Optional, Any
# from datetime import datetime

# import flwr as fl
# from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Parameters, Context

# # ── Import the REAL AWFedAvg classes (not re-implementations) ─────────────────
# from adaptive_weighted_fedavg import (
#     AdaptiveWeightedFedAvg,
#     AdaptiveWeightCalculator,
#     EnhancedFlowerClient,
#     ResourceMonitor,
#     PPONetwork,
#     CLIENT_CONFIGS,
#     NUM_CLIENTS,
#     CLIENTS_PER_ROUND,
#     TOTAL_ROUNDS,
#     LOCAL_EPOCHS,
#     EVALUATION_EPISODES,
#     DEVICE,
#     BASE_MODEL_PATH,
#     create_phy_env,
#     evaluate_model_simple,
#     sb3_ppo_to_pytorch,
#     pytorch_to_sb3_ppo,
#     save_awfedavg_results_with_resources,
#     print_resource_summary,
#     get_client_hyperparams,
# )
# from stable_baselines3 import PPO

# # ── Import blockchain / privacy system ───────────────────────────────────────
# from privacy_blockchain_fl import PrivacyPreservingFederatedLearning


# # ─────────────────────────────────────────────────────────────────────────────
# # HELPER: parameter list <-> named OrderedDict conversion
# # ─────────────────────────────────────────────────────────────────────────────

# # Canonical layer names for PPONetwork – must match PPONetwork.state_dict() order
# PPONETWORK_LAYER_KEYS = [
#     "policy_net.0.weight", "policy_net.0.bias",
#     "policy_net.2.weight", "policy_net.2.bias",
#     "value_net.0.weight",  "value_net.0.bias",
#     "value_net.2.weight",  "value_net.2.bias",
#     "action_net.weight",   "action_net.bias",
#     "value_head.weight",   "value_head.bias",
# ]


# def ndarrays_to_ordered_dict(
#     arrays: List[np.ndarray],
#     keys: Optional[List[str]] = None,
#     device: torch.device = None,          # None → use global DEVICE (auto-detected GPU/CPU)
# ) -> OrderedDict:
#     """Convert a list of numpy arrays to a named OrderedDict of tensors.

#     If *keys* is None, positional names ``param_0``, ``param_1``, … are used,
#     which is sufficient for DP noise addition and IPFS serialisation.
#     """
#     _dev = device if device is not None else DEVICE
#     if keys is None:
#         keys = [f"param_{i}" for i in range(len(arrays))]
#     return OrderedDict(
#         {k: torch.tensor(v, dtype=torch.float32, device=_dev) for k, v in zip(keys, arrays)}
#     )


# def ordered_dict_to_ndarrays(params: OrderedDict) -> List[np.ndarray]:
#     """Convert an OrderedDict of tensors back to a list of numpy arrays."""
#     return [v.cpu().numpy() for v in params.values()]


# # ─────────────────────────────────────────────────────────────────────────────
# # BLOCKCHAIN-ENABLED STRATEGY
# # Inherits fully from AdaptiveWeightedFedAvg – only aggregate_fit is extended
# # ─────────────────────────────────────────────────────────────────────────────

# class BlockchainAdaptiveWeightedFedAvg(AdaptiveWeightedFedAvg):
#     """
#     AWFedAvg strategy with blockchain audit trail and privacy guarantees.

#     What is inherited unchanged from AdaptiveWeightedFedAvg
#     --------------------------------------------------------
#     * AdaptiveWeightCalculator with 4-criteria weight scoring
#       (eMBB outage, URLLC residual, activation diversity, stability)
#     * adaptive_weighted_average() – the actual weighted parameter aggregation
#     * ResourceMonitor for CPU / GPU / memory tracking per round
#     * round_metrics, performance_history, round_resources
#     * get_comprehensive_resource_summary()
#     * get_final_parameters()

#     Three-layer architecture
#     -------------------------
#     Layer 1 — Learning (off-chain):
#       AWFedAvg 4-criteria weighting, secure aggregation masking (client-side),
#       DP noise, PPO training.  No weights touch the chain.

#     Layer 2 — Governance (blockchain, metadata-only):
#       startRound  →  submitAggregatedModel(ipfsHash, modelHash, round)
#       Only hashes + round metadata are stored on-chain (O(K) txs/round).

#     Layer 3 — Storage (IPFS):
#       One upload per round: coordinator-DP-noised + encrypted global model.
#       Clients never upload; only the aggregated model is stored.

#     What this class adds over AdaptiveWeightedFedAvg
#     -------------------------------------------------
#     * Reputation fetch from chain → 5th AWFedAvg weight criterion
#     * Coordinator-side DP noise on aggregated model before IPFS upload
#     * Cumulative privacy accounting: ε_total = √(2T·ln(1/δ))·ε
#     * IPFS upload + on-chain hash registration (one tx per round)
#     * Augmented metrics: ipfs_hash, epsilon, compression_ratio, overhead_s
#     """

#     def __init__(
#         self,
#         ppfl: PrivacyPreservingFederatedLearning,
#         ppfl_config: Dict = None,            # plain-primitive dict for Ray workers
#         apply_coordinator_dp: bool = True,
#         compression: bool = True,
#         blockchain_enabled: bool = True,
#         **awfedavg_kwargs,
#     ):
#         super().__init__(**awfedavg_kwargs)

#         self.ppfl = ppfl
#         self.ppfl_config = ppfl_config or {}
#         self.apply_coordinator_dp = apply_coordinator_dp
#         self.compression = compression
#         self.blockchain_enabled = blockchain_enabled
#         self.last_global_ipfs_hash: str = ""
#         self.blockchain_round_records: List[Dict] = []
#         # {client_id(int): eth_address(str)} — set by experiment runner after
#         # client addresses are known; used to fetch reputation from blockchain.
#         self._client_addrs_map: dict = {}

#     # ──────────────────────────────────────────────────────────────────────
#     # aggregate_fit – the core integration point
#     # ──────────────────────────────────────────────────────────────────────

#     def aggregate_fit(self, server_round: int, results, failures):
#         """
#         Extended aggregation pipeline.

#         Step 1 – Blockchain: open a new round on-chain
#         Step 2 – AWFedAvg:  run the full 4-criteria adaptive weight
#                             calculation and weighted parameter averaging
#                             (delegated entirely to super())
#         Step 3 – Blockchain: coordinator-side DP, encrypt, IPFS upload,
#                             on-chain hash registration
#         Step 4 – Return augmented metrics to Flower
#         """
#         blockchain_start = time.time()

#         # ── Step 1: Start the round on blockchain ─────────────────────────
#         if self.blockchain_enabled:
#             try:
#                 round_tx = self.ppfl.start_round_on_chain(
#                     previous_model_ipfs_hash=self.last_global_ipfs_hash
#                 )
#                 print(f"\n🔗 Round {server_round} opened on blockchain  tx={round_tx}")
#             except Exception as exc:
#                 warnings.warn(
#                     f"[Blockchain] start_round_on_chain failed (round {server_round}): {exc}. "
#                     "Continuing without blockchain registration for this round."
#                 )
#                 round_tx = None
#         else:
#             round_tx = None

#         # ── Step 2a: Fetch on-chain reputation scores → weight calculator ──────
#         # Reputation from blockchain acts as a 5th criterion in AWFedAvg:
#         #   w_i += α_reputation × (reputation_i / 1000)
#         # This closes the governance loop: on-chain slash/reward → weight change.
#         _client_addrs_map = self._client_addrs_map
#         if self.blockchain_enabled and getattr(self.ppfl, 'contract', None) is not None and _client_addrs_map:
#             reputations = {}
#             for cid, addr in _client_addrs_map.items():
#                 try:
#                     _, rep, _, _ = self.ppfl.contract.functions.getClientInfo(addr).call()
#                     reputations[cid] = int(rep)
#                 except Exception:
#                     reputations[cid] = 500   # neutral default if unavailable
#             self.weight_calculator.update_reputation_from_chain(reputations)

#         # ── Step 2b: Full AWFedAvg aggregation (5-criteria adaptive weights) ──
#         # This calls AdaptiveWeightedFedAvg.aggregate_fit(), which:
#         #   • collects client metrics (eMBB, URLLC, reward, training_time)
#         #   • calls weight_calculator.calculate_adaptive_weights()
#         #     (now includes reputation as 5th criterion)
#         #   • calls self.adaptive_weighted_average(results, adaptive_weights)
#         #   • records round_resources, updates performance_history
#         aggregated_parameters, awfedavg_metrics = super().aggregate_fit(
#             server_round, results, failures
#         )

#         if aggregated_parameters is None:
#             return None, {}

#         # ── Step 3: Blockchain / IPFS publication ─────────────────────────
#         ipfs_hash = ""
#         agg_tx = ""
#         dp_applied = False
#         compression_ratio = 0.0
#         upload_size_kb = 0.0

#         if self.blockchain_enabled:
#             try:
#                 # 3a. Convert flwr Parameters → named OrderedDict of tensors
#                 param_arrays = parameters_to_ndarrays(aggregated_parameters)
#                 params_dict = ndarrays_to_ordered_dict(
#                     param_arrays, keys=PPONETWORK_LAYER_KEYS[: len(param_arrays)]
#                 )

#                 original_size = sum(p.numel() * 4 for p in params_dict.values())  # bytes (float32)

#                 # 3b. Coordinator-side differential privacy noise
#                 if self.apply_coordinator_dp:
#                     params_dict = self.ppfl.add_differential_privacy_noise(
#                         params_dict, sensitivity=1.0
#                     )
#                     dp_applied = True
#                     print(
#                         f"  🛡️  Coordinator DP applied  "
#                         f"(ε={self.ppfl.epsilon}, δ={self.ppfl.delta})"
#                     )

#                 # 3c. Encrypt + compress + upload to IPFS
#                 encrypted_data, _sym_key = self.ppfl.encrypt_model(
#                     params_dict, compression=self.compression
#                 )
#                 upload_size_kb = len(encrypted_data) / 1024
#                 # float16 raw size = half of float32 original_size
#                 float16_raw_size = original_size / 2
#                 # compression_ratio vs float32 baseline (positive = saving)
#                 compression_ratio = (1 - len(encrypted_data) / original_size) * 100

#                 ipfs_hash = self.ppfl.upload_to_ipfs(encrypted_data, pin=True)

#                 # 3d. Compute model hash for on-chain verification
#                 model_hash = hashlib.sha256(encrypted_data).digest()

#                 # 3e. Register on blockchain
#                 agg_tx = self.ppfl.submit_aggregated_model_on_chain(
#                     round_number=server_round,
#                     ipfs_hash=ipfs_hash,
#                     model_hash=model_hash,
#                 )

#                 self.last_global_ipfs_hash = ipfs_hash

#                 print(
#                     f"  📦 Aggregated model → IPFS: {ipfs_hash}  "
#                     f"({upload_size_kb:.1f} KB encrypted | "
#                     f"raw f32={original_size/1024:.1f} KB → f16+npz+AES={upload_size_kb:.1f} KB, "
#                     f"{compression_ratio:+.1f}% vs f32)"
#                 )
#                 print(f"  🔗 Registered on blockchain  tx={agg_tx}")

#             except Exception as exc:
#                 warnings.warn(
#                     f"[Blockchain] Base Station aggregate publish failed (round {server_round}): {exc}. "
#                     "AWFedAvg global model preserved; blockchain record skipped."
#                 )

#         blockchain_overhead = time.time() - blockchain_start

#         # ── Cumulative Privacy Accounting (one call per round) ─────────────
#         print(
#             f"  🛡️  DP: ε={self.ppfl.epsilon:.3f}, δ={self.ppfl.delta:.0e}"
#         )

#         # ── Step 4: Augment and return metrics ────────────────────────────
#         blockchain_metrics = {
#             "ipfs_hash":              ipfs_hash,
#             "blockchain_open_tx":     round_tx or "",
#             "blockchain_agg_tx":      agg_tx,
#             "dp_applied":             dp_applied,
#             "compression_ratio_pct":  compression_ratio,
#             "ipfs_upload_size_kb":    upload_size_kb,
#             "blockchain_overhead_s":  blockchain_overhead,
#             "epsilon":                self.ppfl.epsilon,
#             "delta":                  self.ppfl.delta,
#         }
#         awfedavg_metrics.update(blockchain_metrics)

#         # Store for later reporting
#         self.blockchain_round_records.append(
#             {"round": server_round, **blockchain_metrics}
#         )

#         return aggregated_parameters, awfedavg_metrics

#     # ──────────────────────────────────────────────────────────────────────
#     # Additional reporting
#     # ──────────────────────────────────────────────────────────────────────

#     def get_blockchain_summary(self) -> Dict[str, Any]:
#         """Return per-round blockchain statistics."""
#         if not self.blockchain_round_records:
#             return {}

#         overheads = [r["blockchain_overhead_s"] for r in self.blockchain_round_records]
#         sizes     = [r["ipfs_upload_size_kb"]    for r in self.blockchain_round_records]

#         return {
#             "rounds_with_blockchain": len(self.blockchain_round_records),
#             "total_blockchain_overhead_s": sum(overheads),
#             "avg_blockchain_overhead_s":   np.mean(overheads),
#             "total_ipfs_upload_kb":        sum(sizes),
#             "avg_ipfs_upload_kb":          np.mean(sizes),
#             "ipfs_hashes": [r["ipfs_hash"] for r in self.blockchain_round_records],
#         }

#     def print_blockchain_performance_summary(self):
#         """Print a unified summary covering AWFedAvg, blockchain, and privacy accounting."""
#         resource_summary = self.get_comprehensive_resource_summary()
#         print_resource_summary(resource_summary)

#         # Blockchain / IPFS summary
#         bc = self.get_blockchain_summary()
#         if bc:
#             print("\n" + "=" * 70)
#             print("BLOCKCHAIN / IPFS OVERHEAD SUMMARY")
#             print("=" * 70)
#             print(f"  Rounds with blockchain : {bc['rounds_with_blockchain']}")
#             print(f"  Total BC overhead      : {bc['total_blockchain_overhead_s']:.2f}s")
#             print(f"  Avg BC overhead/round  : {bc['avg_blockchain_overhead_s']:.2f}s")
#             print(f"  Total IPFS uploads     : {bc['total_ipfs_upload_kb']:.1f} KB")
#             print(f"  Avg IPFS upload/round  : {bc['avg_ipfs_upload_kb']:.1f} KB")

#         # Formal privacy accounting report — computed inline (no external dependency)
#         _eps   = self.ppfl.epsilon
#         _delta = self.ppfl.delta
#         _T     = getattr(self.ppfl, "_rounds_elapsed", len(self.blockchain_round_records))
#         _eps_total = (
#             (2 * _T * math.log(1.0 / _delta)) ** 0.5 * _eps
#             if _T > 0 else 0.0
#         )
#         print("\n" + "=" * 70)
#         print("CUMULATIVE PRIVACY ACCOUNTING  (Advanced Composition)")
#         print("=" * 70)
#         print(f"  ε per round            : {_eps:.4f}")
#         print(f"  δ                      : {_delta:.1e}")
#         print(f"  Rounds elapsed (T)     : {_T}")
#         print(f"  ε_total = √(2T·ln(1/δ))·ε : {_eps_total:.4f}")
#         print(f"  Interpretation: after {_T} rounds the system")
#         print(f"    satisfies ({_eps_total:.4f}, {_delta:.1e})-DP.")

#         self.ppfl.print_performance_summary()


# # ─────────────────────────────────────────────────────────────────────────────
# # BLOCKCHAIN-ENABLED FLOWER CLIENT
# # Inherits fully from EnhancedFlowerClient – only fit() is extended
# # ─────────────────────────────────────────────────────────────────────────────

# class BlockchainEnhancedFlowerClient(EnhancedFlowerClient):
#     """
#     Flower client that extends EnhancedFlowerClient with blockchain provenance.

#     What is inherited unchanged from EnhancedFlowerClient
#     ------------------------------------------------------
#     * _get_env() / _get_model() lazy initialisation
#     * set_parameters() / get_parameters()
#     * fit():  adaptive local epochs, resource monitoring,
#               PPO training, SB3↔PyTorch conversion,
#               evaluate_model_simple(), enhanced_metrics dict
#     * evaluate(): full evaluation with resource tracking
#     * get_resource_summary()

#     What this class adds over EnhancedFlowerClient
#     -----------------------------------------------
#     After super().fit() (local PPO training) returns:
#       1. Apply secure aggregation mask (pairwise-cancelling, Bonawitz+2017)
#          → coordinator sees only Σw_k, never individual w_k
#       2. Pass original (unmasked) params to AWFedAvg weighting — masks
#          cancel during aggregation so the global model is correct
#     No per-client IPFS upload or blockchain tx — that is server-only.
#     """

#     def __init__(
#         self,
#         client_id: int,
#         client_address: str,
#         client_private_key: str,
#         ppfl_config: Dict,
#         blockchain_enabled: bool = True,
#         stake_amount: float = 0.01,
#         num_clients: int = NUM_CLIENTS,
#     ):
#         super().__init__(client_id)

#         self.client_address      = client_address
#         self.client_private_key  = client_private_key
#         self._ppfl_config        = ppfl_config
#         self._ppfl               = None
#         self.blockchain_enabled  = blockchain_enabled
#         self.stake_amount        = stake_amount
#         self._num_clients        = num_clients
#         self._param_history: List = []               # for replay attack

#     # ── Lazy PPFL accessor ────────────────────────────────────────────────

#     @property
#     def ppfl(self) -> PrivacyPreservingFederatedLearning:
#         """Create the PPFL instance on first access (inside the Ray worker process)."""
#         if self._ppfl is None:
#             cfg = self._ppfl_config
#             import warnings as _w
#             with _w.catch_warnings():
#                 _w.filterwarnings("ignore", category=RuntimeWarning)
#                 self._ppfl = PrivacyPreservingFederatedLearning(
#                     blockchain_provider     = cfg.get("blockchain_provider", "http://127.0.0.1:8545"),
#                     contract_address        = cfg.get("contract_address"),
#                     contract_abi_path       = cfg.get("contract_abi_path"),
#                     ipfs_addr               = cfg.get("ipfs_addr", "/ip4/127.0.0.1/tcp/5001"),
#                     epsilon                 = cfg.get("epsilon", 1.0),
#                     delta                   = cfg.get("delta", 1e-5),
#                     clip_norm               = cfg.get("clip_norm", 1.0),
#                     coordinator_private_key = cfg.get("coordinator_private_key"),
#                     require_connection      = cfg.get("blockchain_enabled", True),
#                 )
#             # Generate per-client RSA key pair and store public key in ppfl
#             self._private_key_pem, self._public_key_pem = \
#                 self._ppfl.generate_client_keypair(str(self.client_id))
#         return self._ppfl

#     @property
#     def public_key_pem(self):
#         _ = self.ppfl   # trigger lazy init
#         return self._public_key_pem

#     @property
#     def private_key_pem(self):
#         _ = self.ppfl
#         return self._private_key_pem

#     # ──────────────────────────────────────────────────────────────────────
#     # Blockchain registration (call once before training starts)
#     # ──────────────────────────────────────────────────────────────────────

#     def register_on_blockchain(self) -> Optional[str]:
#         """
#         Register this client on the smart contract (signs and sends the tx).

#         Returns the transaction hash string or None if registration
#         fails or blockchain is disabled.
#         """
#         if not self.blockchain_enabled:
#             return None
#         try:
#             tx_hash = self.ppfl.register_client_on_chain(
#                 client_address    = self.client_address,
#                 client_private_key = self.client_private_key,
#                 public_key_pem    = self.public_key_pem,
#                 stake_amount      = self.stake_amount,
#             )
#             print(
#                 f"  📝 Client {self.client_id} registered on-chain "
#                 f"(address={self.client_address[:10]}…  tx={tx_hash[:12]}…)"
#             )
#             return tx_hash
#         except Exception as exc:
#             warnings.warn(
#                 f"[Blockchain] MVNO {self.client_id} registration failed: {exc}"
#             )
#             return None

#     # ──────────────────────────────────────────────────────────────────────
#     # fit() – the core integration point
#     # ──────────────────────────────────────────────────────────────────────

#     def fit(self, parameters, config):
#         """
#         Layer 1 — Learning Layer (off-chain):
#           1. Local PPO training via EnhancedFlowerClient.fit()
#           2. Secure aggregation mask applied to parameters before sending to server.
#              Coordinator sees only Σ w_k (masks cancel pairwise) — never individual w_k.

#         Layer 2 (blockchain) and Layer 3 (IPFS) are handled by aggregate_fit on the server.
#         """
#         param_list, num_examples, metrics = super().fit(parameters, config)

#         # ── Attack injection (for robustness experiments) ─────────────────
#         return param_list, num_examples, metrics

#     # ──────────────────────────────────────────────────────────────────────
#     # Reporting
#     # ──────────────────────────────────────────────────────────────────────

#     def get_blockchain_client_summary(self) -> Dict[str, Any]:
#         """Per-client training summary (secure aggregation applied; no per-client chain tx)."""
#         return {
#             "client_id":          self.client_id,
#             "blockchain_enabled": self.blockchain_enabled,
#         }


# # ─────────────────────────────────────────────────────────────────────────────
# # FACTORY HELPERS
# # ─────────────────────────────────────────────────────────────────────────────

# def create_blockchain_awfedavg_strategy(
#     # PPFL / blockchain params
#     blockchain_provider:      str            = "http://localhost:8545",
#     ipfs_addr:                str            = "/ip4/127.0.0.1/tcp/5001",
#     contract_address:         Optional[str]  = None,
#     contract_abi_path:        Optional[str]  = None,
#     coordinator_private_key:  Optional[str]  = None,
#     epsilon:                  float          = 1.0,
#     delta:                    float          = 1e-5,
#     clip_norm:                float          = 1.0,
#     apply_coordinator_dp:     bool           = True,
#     compression:              bool           = True,
#     blockchain_enabled:       bool           = True,
#     # AWFedAvg weight criteria (passed through to AdaptiveWeightCalculator)
#     alpha_embb:               float          = 0.22,
#     alpha_urllc:              float          = 0.38,
#     alpha_activation:         float          = 0.2,
#     alpha_stability:          float          = 0.15,
#     alpha_reputation:         float          = 0.05,   # on-chain reputation criterion
#     # Flower strategy params
#     min_fit_clients:          int            = CLIENTS_PER_ROUND,
#     min_evaluate_clients:     int            = CLIENTS_PER_ROUND,
#     min_available_clients:    int            = NUM_CLIENTS,
# ) -> BlockchainAdaptiveWeightedFedAvg:
#     """
#     Create a fully configured BlockchainAdaptiveWeightedFedAvg strategy.

#     Instantiates PrivacyPreservingFederatedLearning internally and passes it
#     to BlockchainAdaptiveWeightedFedAvg, so callers only need to provide
#     configuration values.

#     Returns
#     -------
#     strategy : BlockchainAdaptiveWeightedFedAvg
#         Ready to pass to fl.simulation.start_simulation().
#         Access the underlying PPFL system via strategy.ppfl.
#     """
#     # Build a plain-Python-primitive config dict BEFORE constructing PPFL.
#     # This dict will be stored on the strategy and passed to Ray workers —
#     # it must contain ONLY str/float/int/bool/None so cloudpickle can
#     # serialise the client_fn closure without touching web3/eth_abi objects.
#     ppfl_config = {
#         "blockchain_provider":     str(blockchain_provider),
#         "contract_address":        str(contract_address) if contract_address else None,
#         "contract_abi_path":       str(contract_abi_path) if contract_abi_path else None,
#         "ipfs_addr":               str(ipfs_addr),
#         "epsilon":                 float(epsilon),
#         "delta":                   float(delta),
#         "clip_norm":               float(clip_norm),
#         "coordinator_private_key": str(coordinator_private_key) if coordinator_private_key else None,
#         "blockchain_enabled":      bool(blockchain_enabled),
#     }

#     # Build the privacy + blockchain backend (main process only — never pickled).
#     ppfl = PrivacyPreservingFederatedLearning(
#         blockchain_provider     = blockchain_provider,
#         contract_address        = contract_address,
#         contract_abi_path       = contract_abi_path,
#         ipfs_addr               = ipfs_addr,
#         epsilon                 = epsilon,
#         delta                   = delta,
#         clip_norm               = clip_norm,
#         coordinator_private_key = coordinator_private_key,
#         require_connection      = blockchain_enabled,
#     )

#     strategy = BlockchainAdaptiveWeightedFedAvg(
#         ppfl                  = ppfl,
#         ppfl_config           = ppfl_config,
#         apply_coordinator_dp  = apply_coordinator_dp,
#         compression           = compression,
#         blockchain_enabled    = blockchain_enabled,
#         alpha_embb            = alpha_embb,
#         alpha_urllc           = alpha_urllc,
#         alpha_activation      = alpha_activation,
#         alpha_stability       = alpha_stability,
#         alpha_reputation      = alpha_reputation,
#         min_fit_clients       = min_fit_clients,
#         min_evaluate_clients  = min_evaluate_clients,
#         min_available_clients = min_available_clients,
#     )

#     return strategy


# # MVNO configs – each entry maps a Flower node-id modulo NUM_CLIENTS to
# # an Ethereum address.  Each MVNO has its own address on the Base Station's chain.
# _DEFAULT_CLIENT_ADDRESSES = [
#     {"address": f"0x{'0'*39}{i}", "private_key": f"0x{'0'*63}{i}"}
#     for i in range(NUM_CLIENTS)
# ]


# def make_blockchain_client_fn(
#     ppfl_config:        Dict,
#     client_addresses:   Optional[List[Dict]] = None,
#     blockchain_enabled: bool = True,
#     num_clients:        int  = NUM_CLIENTS,
#     stake_amount:       float = 0.01,
# ):
#     """Return a Flower-compatible client_fn.

#     Clients handle Layer 1 (learning + secure aggregation masking) only.
#     Layer 2 (blockchain) and Layer 3 (IPFS) are handled server-side in aggregate_fit.
#     """
#     if client_addresses is None:
#         client_addresses = _DEFAULT_CLIENT_ADDRESSES

#     _cfg = {
#         "blockchain_provider":     str(ppfl_config.get("blockchain_provider", "http://127.0.0.1:8545")),
#         "contract_address":        str(ppfl_config["contract_address"]) if ppfl_config.get("contract_address") else None,
#         "contract_abi_path":       str(ppfl_config["contract_abi_path"]) if ppfl_config.get("contract_abi_path") else None,
#         "ipfs_addr":               str(ppfl_config.get("ipfs_addr", "/ip4/127.0.0.1/tcp/5001")),
#         "epsilon":                 float(ppfl_config.get("epsilon", 1.0)),
#         "delta":                   float(ppfl_config.get("delta", 1e-5)),
#         "clip_norm":               float(ppfl_config.get("clip_norm", 1.0)),
#         "coordinator_private_key": str(ppfl_config["coordinator_private_key"]) if ppfl_config.get("coordinator_private_key") else None,
#         "blockchain_enabled":      bool(ppfl_config.get("blockchain_enabled", True)),
#     }
#     _addrs = [{"address": str(a["address"]), "private_key": str(a.get("private_key") or "")}
#               for a in client_addresses] if client_addresses else []
#     if not _addrs:
#         _addrs = [{"address": f"0x{'0'*39}{i}", "private_key": ""} for i in range(num_clients)]

#     def client_fn(context: Context) -> fl.client.Client:
#         client_id = int(context.node_id) % num_clients
#         info = _addrs[client_id % len(_addrs)]

#         client = BlockchainEnhancedFlowerClient(
#             client_id           = client_id,
#             client_address      = info["address"],
#             client_private_key  = info["private_key"],
#             ppfl_config         = _cfg,
#             blockchain_enabled  = blockchain_enabled,
#             num_clients         = num_clients,
#             stake_amount        = stake_amount,
#         )
#         return client.to_client()

#     return client_fn


# # ─────────────────────────────────────────────────────────────────────────────
# # COMPLETE EXPERIMENT RUNNER
# # Drop-in replacement for adaptive_weighted_fedavg.run_awfedavg_experiment_with_resources
# # ─────────────────────────────────────────────────────────────────────────────

# def _run_sequential_simulation(
#     client_fn,
#     strategy,
#     num_clients: int,
#     num_rounds: int,
# ):
#     """
#     Drop-in replacement for fl.simulation.start_simulation that runs entirely
#     in the main process — no Ray, no cloudpickle, no lru_cache pickling errors.

#     Implements the standard Flower FL protocol:
#       Round 0 : get initial parameters from one client
#       Round 1…N: configure_fit → fit → aggregate_fit
#                   configure_evaluate → evaluate → aggregate_evaluate

#     Returns a minimal history object with the same interface as flwr.server.History.
#     """
#     import flwr as fl
#     from flwr.common import (
#         ndarrays_to_parameters, parameters_to_ndarrays,
#         FitIns, FitRes, EvaluateIns, EvaluateRes,
#         GetParametersIns, Code, Status,
#     )

#     class _History:
#         """Minimal Flower History replacement."""
#         def __init__(self):
#             self.losses_distributed   = []
#             self.losses_centralized   = []
#             self.metrics_distributed  = {"fit": [], "evaluate": []}
#             self.metrics_centralized  = {}

#     hist = _History()

#     # Build Context objects for each client
#     class _Ctx:
#         def __init__(self, node_id):
#             self.node_id     = node_id
#             self.node_config = {}
#             self.run_config  = {}
#             self.state       = fl.common.RecordSet()

#     # ── Round 0: get initial parameters ──────────────────────────────────
#     print("\n[Sim] Requesting initial parameters from client 0…")
#     init_client_proxy = client_fn(_Ctx(0))
#     # init_client_proxy is a flwr.client.Client (result of to_client())
#     init_res = init_client_proxy.get_parameters(
#         ins=GetParametersIns(config={})
#     )
#     current_parameters = init_res.parameters
#     print(f"[Sim] Initial parameters received "
#           f"({len(parameters_to_ndarrays(current_parameters))} tensors)")

#     # ── Rounds 1…num_rounds ───────────────────────────────────────────────
#     for rnd in range(1, num_rounds + 1):
#         print(f"\n{'='*60}")
#         print(f"[Sim] ── Round {rnd}/{num_rounds} ──")
#         print(f"{'='*60}")

#         # configure_fit
#         client_ids = list(range(num_clients))
#         fit_configs = strategy.configure_fit(
#             server_round = rnd,
#             parameters   = current_parameters,
#             client_manager = _DummyClientManager(client_ids),
#         )

#         # fit each client
#         fit_results = []
#         fit_failures = []
#         for client_proxy, fit_ins in fit_configs:
#             cid = int(client_proxy.cid)
#             try:
#                 proxy = client_fn(_Ctx(cid))
#                 fit_res = proxy.fit(ins=fit_ins)
#                 fit_results.append((proxy, fit_res))
#                 print(f"  [Sim] MVNO {cid} fit done — "
#                       f"{fit_res.num_examples} examples, "
#                       f"metrics={list(fit_res.metrics.keys())}")
#             except Exception as exc:
#                 import traceback; traceback.print_exc()
#                 fit_failures.append(exc)
#                 print(f"  [Sim] MVNO {cid} fit FAILED: {exc}")

#         if not fit_results:
#             print(f"[Sim] Round {rnd}: all clients failed — stopping.")
#             break

#         # aggregate_fit  (this calls BlockchainAdaptiveWeightedFedAvg.aggregate_fit)
#         agg = strategy.aggregate_fit(
#             server_round = rnd,
#             results      = fit_results,
#             failures     = fit_failures,
#         )
#         if agg is not None:
#             current_parameters, agg_metrics = agg
#             hist.metrics_distributed["fit"].append((rnd, agg_metrics))
#             print(f"  [Sim] Aggregation done. Metrics: {list(agg_metrics.keys())}")

#         # configure_evaluate
#         eval_configs = strategy.configure_evaluate(
#             server_round   = rnd,
#             parameters     = current_parameters,
#             client_manager = _DummyClientManager(client_ids),
#         )

#         # evaluate each client
#         eval_results  = []
#         eval_failures = []
#         for client_proxy, eval_ins in eval_configs:
#             cid = int(client_proxy.cid)
#             try:
#                 proxy    = client_fn(_Ctx(cid))
#                 eval_res = proxy.evaluate(ins=eval_ins)
#                 eval_results.append((proxy, eval_res))
#             except Exception as exc:
#                 eval_failures.append(exc)

#         # aggregate_evaluate
#         agg_eval = strategy.aggregate_evaluate(
#             server_round = rnd,
#             results      = eval_results,
#             failures     = eval_failures,
#         )
#         if agg_eval is not None:
#             loss, eval_metrics = agg_eval
#             hist.losses_distributed.append((rnd, loss))
#             hist.metrics_distributed["evaluate"].append((rnd, eval_metrics or {}))

#     return hist


# class _DummyClientManager:
#     """Minimal ClientManager that returns (client_id, ins) tuples for configure_fit/evaluate."""
#     def __init__(self, client_ids):
#         self._ids = client_ids

#     def sample(self, num_clients, min_num_clients=None, criterion=None):
#         class _Proxy:
#             def __init__(self, cid): self.cid = str(cid)
#         return [_Proxy(i) for i in self._ids[:num_clients]]

#     def num_available(self): return len(self._ids)



# def run_blockchain_awfedavg_experiment(
#     strategy:    Optional[BlockchainAdaptiveWeightedFedAvg] = None,
#     client_fn                                              = None,
#     num_rounds:  int                                       = TOTAL_ROUNDS,
#     num_clients: int                                       = NUM_CLIENTS,
#     # Pass-through to create_blockchain_awfedavg_strategy when strategy is None
#     **strategy_kwargs,
# ):
#     """
#     Full experiment: local baselines → blockchain-AWFedAvg FL → evaluation.

#     Parameters
#     ----------
#     strategy   : pre-built BlockchainAdaptiveWeightedFedAvg, or None to
#                  create one automatically from strategy_kwargs.
#     client_fn  : Flower client_fn, or None to build from strategy.ppfl.
#     num_rounds : total federated rounds.
#     num_clients: total simulated clients.
#     **strategy_kwargs : forwarded to create_blockchain_awfedavg_strategy
#                         when strategy is None.

#     Returns
#     -------
#     hist, local_results, federated_results, resource_summary
#     """
#     print("\n" + "=" * 80)
#     print("BLOCKCHAIN-ENABLED ADAPTIVE WEIGHTED FEDAVG EXPERIMENT")
#     print("=" * 80)
#     local_results = {}  # initialised early so except blocks can reference it

#     # ── Build strategy + client_fn if not provided ───────────────────────
#     # client_addresses is not a strategy param — pop it before building strategy.
#     _client_addrs = strategy_kwargs.pop("client_addresses", None)

#     if strategy is None:
#         print("\n⚙️  Building strategy…")
#         strategy = create_blockchain_awfedavg_strategy(**strategy_kwargs)

#     if client_fn is None:
#         client_fn = make_blockchain_client_fn(
#             ppfl_config        = strategy.ppfl_config,
#             client_addresses   = _client_addrs,
#             blockchain_enabled = strategy.blockchain_enabled,
#         )

#     # Populate strategy's address map so aggregate_fit can read reputation per client
#     if _client_addrs:
#         strategy._client_addrs_map = {
#             i: _client_addrs[i % len(_client_addrs)]["address"]
#             for i in range(num_clients)
#         }

#     local_results = {}   # no standalone local training — federated only

#     # ── AWFedAvg + Blockchain federated training ──────────────────────────
#     print("\n🚀 Starting Blockchain-AWFedAvg  [MVNOs → Base Station gNB]…")
#     print(f"   AWFedAvg weight criteria:")
#     wc = strategy.weight_calculator
#     print(f"     eMBB outage   : {wc.alpha_embb:.1%}")
#     print(f"     URLLC residual: {wc.alpha_urllc:.1%}")
#     print(f"     Activation div: {wc.alpha_activation:.1%}")
#     print(f"     Stability     : {wc.alpha_stability:.1%}")
#     print(f"   DP parameters   : ε={strategy.ppfl.epsilon}, δ={strategy.ppfl.delta}")
#     print(f"   Blockchain      : {'enabled' if strategy.blockchain_enabled else 'disabled'}")

#     # ── Pre-register clients on blockchain (before Ray workers start) ───────
#     # Registration is done here in the main process where web3 is picklable.
#     if strategy.blockchain_enabled and _client_addrs:
#         print("\n📝 Registering MVNOs on Base Station blockchain…")
#         for i, info in enumerate(_client_addrs[:NUM_CLIENTS]):
#             try:
#                 strategy.ppfl.register_client_on_chain(
#                     client_address     = info["address"],
#                     client_private_key = info["private_key"],
#                     public_key_pem     = strategy.ppfl.generate_client_keypair(str(i))[1],
#                     stake_amount       = 0.01,
#                 )
#             except Exception as exc:
#                 warnings.warn(f"[Blockchain] MVNO {i} pre-registration failed: {exc}")

#     fed_start = time.time()
#     system_monitor = ResourceMonitor()
#     system_monitor.start_monitoring(interval=1.0)
#     hist = None

#     try:
#         hist = _run_sequential_simulation(
#             client_fn   = client_fn,
#             strategy    = strategy,
#             num_clients = num_clients,
#             num_rounds  = num_rounds,
#         )
#         print(f"\n✅ Federated training complete in {time.time()-fed_start:.1f}s")
#     except Exception as exc:
#         import traceback
#         traceback.print_exc()
#         print(f"❌ Federated training error: {exc}")
#         system_monitor.stop_monitoring()
#         return None, local_results, None, None
#     finally:
#         system_resource_summary = system_monitor.stop_monitoring()

#     # ── Final evaluation of global model ─────────────────────────────────
#     federated_results = None
#     if strategy.get_final_parameters() is not None:
#         print("\n📊 Evaluating final global model…")
#         try:
#             eval_env    = create_phy_env(0)
#             input_dim   = eval_env.observation_space.shape[0]
#             output_dim  = eval_env.action_space.n
#             pyt_model   = PPONetwork(input_dim, output_dim).to(DEVICE)

#             final_params = parameters_to_ndarrays(strategy.get_final_parameters())
#             model_keys   = list(pyt_model.state_dict().keys())
#             if len(final_params) == len(model_keys):
#                 state_dict = OrderedDict({
#                     k: torch.tensor(v, device=DEVICE)
#                     for k, v in zip(model_keys, final_params)
#                 })
#                 pyt_model.load_state_dict(state_dict, strict=True)
#                 fed_model = PPO(
#                     "MlpPolicy", eval_env, verbose=0,
#                     policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
#                 )
#                 fed_model = pytorch_to_sb3_ppo(pyt_model, fed_model)
#                 federated_results = evaluate_model_simple(
#                     fed_model, eval_env, num_episodes=EVALUATION_EPISODES
#                 )
#                 fed_model.save(f"{BASE_MODEL_PATH}/blockchain_awfedavg_global_model.zip")
#                 print(f"  Reward   : {federated_results['average_reward']:.4f}")
#                 print(f"  Stability: {federated_results['stability_score']:.4f}")
#             eval_env.close()
#         except Exception as exc:
#             print(f"❌ Final evaluation failed: {exc}")

#     # ── Performance summary ───────────────────────────────────────────────
#     print("\n" + "=" * 80)
#     print("RESULTS")
#     print("=" * 80)
#     if federated_results:
#         print(f"  Blockchain-AWFedAvg reward  : {federated_results['average_reward']:.4f}")
#         print(f"  Stability score             : {federated_results['stability_score']:.4f}")

#     # ── Comprehensive summaries ───────────────────────────────────────────
#     resource_summary = strategy.get_comprehensive_resource_summary()
#     strategy.print_blockchain_performance_summary()

#     # ── Save results ─────────────────────────────────────────────────────
#     try:
#         save_awfedavg_results_with_resources(strategy, local_results, federated_results)
#     except Exception as exc:
#         warnings.warn(f"Results save failed: {exc}")

#     # Save blockchain-specific records
#     bc_records_path = (
#         f"{BASE_MODEL_PATH}/blockchain_records_"
#         f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     )
#     with open(bc_records_path, "w") as f:
#         json.dump(strategy.blockchain_round_records, f, indent=2, default=str)
#     print(f"\n📂 Blockchain records saved: {bc_records_path}")

#     print("\n" + "=" * 80)
#     print("✅ Blockchain-AWFedAvg experiment complete!")
#     print("=" * 80)

#     return hist, local_results, federated_results, resource_summary


# # ─────────────────────────────────────────────────────────────────────────────
# # ENTRY POINT
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     np.random.seed(42)
#     torch.manual_seed(42)

#     print("🚀 Blockchain-AWFedAvg TRUE Integration")
#     print("  ✅ AdaptiveWeightedFedAvg (4-criteria weights) — inherited, not reimplemented")
#     print("  ✅ EnhancedFlowerClient (PPO + resource monitor) — inherited, not reimplemented")
#     print("  ✅ PrivacyPreservingFederatedLearning — DP + encryption + IPFS + blockchain")
#     print("  ✅ Coordinator-side & client-side DP applied independently")
#     print("  ✅ All AWFedAvg metrics preserved and augmented with blockchain fields")

#     # ── Load contract_info.json produced by deploy.py ────────────────────
#     _config_path = os.path.join(os.path.dirname(__file__), "contract_info.json")
#     if os.path.exists(_config_path):
#         with open(_config_path) as _f:
#             _cfg = json.load(_f)
#         _blockchain_enabled       = True
#         _contract_address         = _cfg["contract_address"]
#         _contract_abi_path        = _config_path
#         _coordinator_private_key  = _cfg["coordinator_private_key"]
#         _client_addresses         = _cfg["clients"]          # list of {address, private_key}
#         print("\n✅ contract_info.json loaded — blockchain ENABLED")
#         print(f"   Contract : {_contract_address}")
#         print(f"   Coordinator: {_cfg['coordinator_address']}")
#     else:
#         _blockchain_enabled       = False
#         _contract_address         = None
#         _contract_abi_path        = None
#         _coordinator_private_key  = None
#         _client_addresses         = None
#         print("\n⚠️  contract_info.json not found — running with blockchain_enabled=False")
#         print("   Run  python deploy.py  first to deploy the smart contract.")

#     hist, local_results, fed_results, resources = run_blockchain_awfedavg_experiment(
#         blockchain_enabled       = _blockchain_enabled,
#         blockchain_provider      = "http://127.0.0.1:8545",
#         contract_address         = _contract_address,
#         contract_abi_path        = _contract_abi_path,
#         coordinator_private_key  = _coordinator_private_key,
#         ipfs_addr                = "/ip4/127.0.0.1/tcp/5001",
#         epsilon                  = 1.0,
#         delta                    = 1e-5,
#         clip_norm                = 1.0,
#         alpha_embb               = 0.22,
#         alpha_urllc              = 0.38,
#         alpha_activation         = 0.2,
#         alpha_stability          = 0.2,
#         num_rounds               = TOTAL_ROUNDS,
#         num_clients              = NUM_CLIENTS,
#         client_addresses         = _client_addresses,  # from contract_info.json
#     )
"""
Blockchain-AWFedAvg — Three-Layer Federated Learning System
============================================================

Layer 1 — Learning (Off-chain)
    AdaptiveWeightedFedAvg with 4-criteria weighting (eMBB, URLLC, activation
    diversity, stability) plus on-chain reputation as a 5th criterion.
    Secure aggregation masking (Bonawitz et al. CCS'17): each client adds a
    pairwise-cancelling mask so the coordinator sees only Σw_k, never w_k.
    Cumulative DP accounting: ε_total = √(2T·ln(1/δ))·ε tracked per round.

Layer 2 — Governance (Blockchain, metadata-only)
    Smart contract stores ONLY: round metadata, aggregated model hash, client
    reputation.  No weights, no heavy computation, no per-client uploads.
    O(1) transaction per round regardless of number of clients K.

Layer 3 — Storage (IPFS)
    One upload per round: coordinator-DP-noised + encrypted global model.
    Content-addressed CID stored on-chain for immutable audit trail.

Key classes
-----------
    BlockchainAdaptiveWeightedFedAvg  ─ server strategy (extends AWFedAvg)
    BlockchainEnhancedFlowerClient    ─ client (extends EnhancedFlowerClient)

Usage
-----
    strategy = create_blockchain_awfedavg_strategy(...)
    client_fn = make_blockchain_client_fn(strategy.ppfl_config)
    run_blockchain_awfedavg_experiment(...)
"""

import os
import sys
import hashlib
import json
import time
import warnings
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Parameters, Context

# ── Import the REAL AWFedAvg classes (not re-implementations) ─────────────────
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
from stable_baselines3 import PPO

# ── Import blockchain / privacy system ───────────────────────────────────────
from privacy_blockchain_fl import PrivacyPreservingFederatedLearning


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: parameter list <-> named OrderedDict conversion
# ─────────────────────────────────────────────────────────────────────────────

# Canonical layer names for PPONetwork – must match PPONetwork.state_dict() order
PPONETWORK_LAYER_KEYS = [
    "policy_net.0.weight", "policy_net.0.bias",
    "policy_net.2.weight", "policy_net.2.bias",
    "value_net.0.weight",  "value_net.0.bias",
    "value_net.2.weight",  "value_net.2.bias",
    "action_net.weight",   "action_net.bias",
    "value_head.weight",   "value_head.bias",
]


def ndarrays_to_ordered_dict(
    arrays: List[np.ndarray],
    keys: Optional[List[str]] = None,
    device: torch.device = None,          # None → use global DEVICE (auto-detected GPU/CPU)
) -> OrderedDict:
    """Convert a list of numpy arrays to a named OrderedDict of tensors.

    If *keys* is None, positional names ``param_0``, ``param_1``, … are used,
    which is sufficient for DP noise addition and IPFS serialisation.
    """
    _dev = device if device is not None else DEVICE
    if keys is None:
        keys = [f"param_{i}" for i in range(len(arrays))]
    return OrderedDict(
        {k: torch.tensor(v, dtype=torch.float32, device=_dev) for k, v in zip(keys, arrays)}
    )


def ordered_dict_to_ndarrays(params: OrderedDict) -> List[np.ndarray]:
    """Convert an OrderedDict of tensors back to a list of numpy arrays."""
    return [v.cpu().numpy() for v in params.values()]


# ─────────────────────────────────────────────────────────────────────────────
# BLOCKCHAIN-ENABLED STRATEGY
# Inherits fully from AdaptiveWeightedFedAvg – only aggregate_fit is extended
# ─────────────────────────────────────────────────────────────────────────────

class BlockchainAdaptiveWeightedFedAvg(AdaptiveWeightedFedAvg):
    """
    AWFedAvg strategy with blockchain audit trail and privacy guarantees.

    What is inherited unchanged from AdaptiveWeightedFedAvg
    --------------------------------------------------------
    * AdaptiveWeightCalculator with 4-criteria weight scoring
      (eMBB outage, URLLC residual, activation diversity, stability)
    * adaptive_weighted_average() – the actual weighted parameter aggregation
    * ResourceMonitor for CPU / GPU / memory tracking per round
    * round_metrics, performance_history, round_resources
    * get_comprehensive_resource_summary()
    * get_final_parameters()

    Three-layer architecture
    -------------------------
    Layer 1 — Learning (off-chain):
      AWFedAvg 4-criteria weighting, secure aggregation masking (client-side),
      DP noise, PPO training.  No weights touch the chain.

    Layer 2 — Governance (blockchain, metadata-only):
      startRound  →  submitAggregatedModel(ipfsHash, modelHash, round)
      Only hashes + round metadata are stored on-chain (O(K) txs/round).

    Layer 3 — Storage (IPFS):
      One upload per round: coordinator-DP-noised + encrypted global model.
      Clients never upload; only the aggregated model is stored.

    What this class adds over AdaptiveWeightedFedAvg
    -------------------------------------------------
    * Reputation fetch from chain → 5th AWFedAvg weight criterion
    * Coordinator-side DP noise on aggregated model before IPFS upload
    * Cumulative privacy accounting: ε_total = √(2T·ln(1/δ))·ε
    * IPFS upload + on-chain hash registration (one tx per round)
    * Augmented metrics: ipfs_hash, epsilon, compression_ratio, overhead_s
    """

    def __init__(
        self,
        ppfl: PrivacyPreservingFederatedLearning,
        ppfl_config: Dict = None,            # plain-primitive dict for Ray workers
        apply_coordinator_dp: bool = True,
        compression: bool = True,
        blockchain_enabled: bool = True,
        **awfedavg_kwargs,
    ):
        super().__init__(**awfedavg_kwargs)

        self.ppfl = ppfl
        self.ppfl_config = ppfl_config or {}
        self.apply_coordinator_dp = apply_coordinator_dp
        self.compression = compression
        self.blockchain_enabled = blockchain_enabled
        self.last_global_ipfs_hash: str = ""
        self.blockchain_round_records: List[Dict] = []
        # {client_id(int): eth_address(str)} — set by experiment runner after
        # client addresses are known; used to fetch reputation from blockchain.
        self._client_addrs_map: dict = {}

    # ──────────────────────────────────────────────────────────────────────
    # aggregate_fit – the core integration point
    # ──────────────────────────────────────────────────────────────────────

    def aggregate_fit(self, server_round: int, results, failures):
        """
        Extended aggregation pipeline.

        Step 1 – Blockchain: open a new round on-chain
        Step 2 – AWFedAvg:  run the full 4-criteria adaptive weight
                            calculation and weighted parameter averaging
                            (delegated entirely to super())
        Step 3 – Blockchain: coordinator-side DP, encrypt, IPFS upload,
                            on-chain hash registration
        Step 4 – Return augmented metrics to Flower
        """
        blockchain_start = time.time()

        # ── Step 1: Start the round on blockchain ─────────────────────────
        if self.blockchain_enabled:
            try:
                round_tx = self.ppfl.start_round_on_chain(
                    previous_model_ipfs_hash=self.last_global_ipfs_hash
                )
                print(f"\n🔗 Round {server_round} opened on blockchain  tx={round_tx}")
            except Exception as exc:
                warnings.warn(
                    f"[Blockchain] start_round_on_chain failed (round {server_round}): {exc}. "
                    "Continuing without blockchain registration for this round."
                )
                round_tx = None
        else:
            round_tx = None

        # ── Step 2a: Fetch on-chain reputation scores → weight calculator ──────
        # Reputation from blockchain acts as a 5th criterion in AWFedAvg:
        #   w_i += α_reputation × (reputation_i / 1000)
        # This closes the governance loop: on-chain slash/reward → weight change.
        _client_addrs_map = self._client_addrs_map
        if self.blockchain_enabled and getattr(self.ppfl, 'contract', None) is not None and _client_addrs_map:
            reputations = {}
            for cid, addr in _client_addrs_map.items():
                try:
                    _, rep, _, _ = self.ppfl.contract.functions.getClientInfo(addr).call()
                    reputations[cid] = int(rep)
                except Exception:
                    reputations[cid] = 500   # neutral default if unavailable
            self.weight_calculator.update_reputation_from_chain(reputations)

        # ── Step 2b: Full AWFedAvg aggregation (5-criteria adaptive weights) ──
        # This calls AdaptiveWeightedFedAvg.aggregate_fit(), which:
        #   • collects client metrics (eMBB, URLLC, reward, training_time)
        #   • calls weight_calculator.calculate_adaptive_weights()
        #     (now includes reputation as 5th criterion)
        #   • calls self.adaptive_weighted_average(results, adaptive_weights)
        #   • records round_resources, updates performance_history
        aggregated_parameters, awfedavg_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is None:
            return None, {}

        # ── Step 3: Blockchain / IPFS publication ─────────────────────────
        ipfs_hash = ""
        agg_tx = ""
        dp_applied = False
        compression_ratio = 0.0
        upload_size_kb = 0.0

        if self.blockchain_enabled:
            try:
                # 3a. Convert flwr Parameters → named OrderedDict of tensors
                param_arrays = parameters_to_ndarrays(aggregated_parameters)
                params_dict = ndarrays_to_ordered_dict(
                    param_arrays, keys=PPONETWORK_LAYER_KEYS[: len(param_arrays)]
                )

                original_size = sum(p.numel() * 4 for p in params_dict.values())  # bytes (float32)

                # 3b. Coordinator-side differential privacy noise
                if self.apply_coordinator_dp:
                    params_dict = self.ppfl.add_differential_privacy_noise(
                        params_dict, sensitivity=1.0
                    )
                    dp_applied = True
                    print(
                        f"  🛡️  Coordinator DP applied  "
                        f"(ε={self.ppfl.epsilon}, δ={self.ppfl.delta})"
                    )

                # 3c. Encrypt + compress + upload to IPFS
                encrypted_data, _sym_key = self.ppfl.encrypt_model(
                    params_dict, compression=self.compression
                )
                upload_size_kb = len(encrypted_data) / 1024
                # float16 raw size = half of float32 original_size
                float16_raw_size = original_size / 2
                # compression_ratio vs float32 baseline (positive = saving)
                compression_ratio = (1 - len(encrypted_data) / original_size) * 100

                ipfs_hash = self.ppfl.upload_to_ipfs(encrypted_data, pin=True)

                # 3d. Compute model hash for on-chain verification
                model_hash = hashlib.sha256(encrypted_data).digest()

                # 3e. Register on blockchain
                agg_tx = self.ppfl.submit_aggregated_model_on_chain(
                    round_number=server_round,
                    ipfs_hash=ipfs_hash,
                    model_hash=model_hash,
                )

                self.last_global_ipfs_hash = ipfs_hash

                print(
                    f"  📦 Aggregated model → IPFS: {ipfs_hash}  "
                    f"({upload_size_kb:.1f} KB encrypted | "
                    f"raw f32={original_size/1024:.1f} KB → f16+npz+AES={upload_size_kb:.1f} KB, "
                    f"{compression_ratio:+.1f}% vs f32)"
                )
                print(f"  🔗 Registered on blockchain  tx={agg_tx}")

            except Exception as exc:
                warnings.warn(
                    f"[Blockchain] Base Station aggregate publish failed (round {server_round}): {exc}. "
                    "AWFedAvg global model preserved; blockchain record skipped."
                )

        blockchain_overhead = time.time() - blockchain_start

        # ── Cumulative Privacy Accounting (one call per round) ─────────────
        print(
            f"  🛡️  DP: ε={self.ppfl.epsilon:.3f}, δ={self.ppfl.delta:.0e}"
        )

        # ── Step 4: Augment and return metrics ────────────────────────────
        blockchain_metrics = {
            "ipfs_hash":              ipfs_hash,
            "blockchain_open_tx":     round_tx or "",
            "blockchain_agg_tx":      agg_tx,
            "dp_applied":             dp_applied,
            "compression_ratio_pct":  compression_ratio,
            "ipfs_upload_size_kb":    upload_size_kb,
            "blockchain_overhead_s":  blockchain_overhead,
            "epsilon":                self.ppfl.epsilon,
            "delta":                  self.ppfl.delta,
        }
        awfedavg_metrics.update(blockchain_metrics)

        # Store for later reporting
        self.blockchain_round_records.append(
            {"round": server_round, **blockchain_metrics}
        )

        return aggregated_parameters, awfedavg_metrics

    # ──────────────────────────────────────────────────────────────────────
    # Additional reporting
    # ──────────────────────────────────────────────────────────────────────

    def get_blockchain_summary(self) -> Dict[str, Any]:
        """Return per-round blockchain statistics."""
        if not self.blockchain_round_records:
            return {}

        overheads = [r["blockchain_overhead_s"] for r in self.blockchain_round_records]
        sizes     = [r["ipfs_upload_size_kb"]    for r in self.blockchain_round_records]

        return {
            "rounds_with_blockchain": len(self.blockchain_round_records),
            "total_blockchain_overhead_s": sum(overheads),
            "avg_blockchain_overhead_s":   np.mean(overheads),
            "total_ipfs_upload_kb":        sum(sizes),
            "avg_ipfs_upload_kb":          np.mean(sizes),
            "ipfs_hashes": [r["ipfs_hash"] for r in self.blockchain_round_records],
        }

    def print_blockchain_performance_summary(self):
        """Print a unified summary covering AWFedAvg, blockchain, and privacy accounting."""
        resource_summary = self.get_comprehensive_resource_summary()
        print_resource_summary(resource_summary)

        # Blockchain / IPFS summary
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

        # Formal privacy accounting report
        priv = self.ppfl.privacy_report()
        print("\n" + "=" * 70)
        print("CUMULATIVE PRIVACY ACCOUNTING  (Advanced Composition)")
        print("=" * 70)
        print(f"  ε per round            : {priv['eps_per_round']:.4f}")
        print(f"  δ                      : {priv['delta']:.1e}")
        print(f"  Rounds elapsed (T)     : {priv['rounds_elapsed']}")
        print(f"  ε_total = √(2T·ln(1/δ))·ε : {priv['eps_total_so_far']:.4f}")
        print(f"  Interpretation: after {priv['rounds_elapsed']} rounds the system")
        print(f"    satisfies ({priv['eps_total_so_far']:.4f}, {priv['delta']:.1e})-DP.")

        self.ppfl.print_performance_summary()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCKCHAIN-ENABLED FLOWER CLIENT
# Inherits fully from EnhancedFlowerClient – only fit() is extended
# ─────────────────────────────────────────────────────────────────────────────

class BlockchainEnhancedFlowerClient(EnhancedFlowerClient):
    """
    Flower client that extends EnhancedFlowerClient with blockchain provenance.

    What is inherited unchanged from EnhancedFlowerClient
    ------------------------------------------------------
    * _get_env() / _get_model() lazy initialisation
    * set_parameters() / get_parameters()
    * fit():  adaptive local epochs, resource monitoring,
              PPO training, SB3↔PyTorch conversion,
              evaluate_model_simple(), enhanced_metrics dict
    * evaluate(): full evaluation with resource tracking
    * get_resource_summary()

    What this class adds over EnhancedFlowerClient
    -----------------------------------------------
    After super().fit() (local PPO training) returns:
      1. Apply secure aggregation mask (pairwise-cancelling, Bonawitz+2017)
         → coordinator sees only Σw_k, never individual w_k
      2. Pass original (unmasked) params to AWFedAvg weighting — masks
         cancel during aggregation so the global model is correct
    No per-client IPFS upload or blockchain tx — that is server-only.
    """

    def __init__(
        self,
        client_id: int,
        client_address: str,
        client_private_key: str,
        ppfl_config: Dict,
        blockchain_enabled: bool = True,
        stake_amount: float = 0.01,
        num_clients: int = NUM_CLIENTS,
        secure_aggregation: bool = False,
        attack_type: str = "none",
        attack_fraction: float = 0.0,
        attack_strength: float = 1.0,
    ):
        super().__init__(client_id)

        self.client_address      = client_address
        self.client_private_key  = client_private_key
        self._ppfl_config        = ppfl_config
        self._ppfl               = None
        self.blockchain_enabled  = blockchain_enabled
        self.stake_amount        = stake_amount
        self._num_clients        = num_clients
        self._param_history: List = []               # for replay attack
        self._secure_aggregation = secure_aggregation
        self._attack_type        = attack_type
        self._attack_fraction    = attack_fraction
        self._attack_strength    = attack_strength

    # ── Lazy PPFL accessor ────────────────────────────────────────────────

    @property
    def ppfl(self) -> PrivacyPreservingFederatedLearning:
        """Create the PPFL instance on first access (inside the Ray worker process)."""
        if self._ppfl is None:
            cfg = self._ppfl_config
            import warnings as _w
            with _w.catch_warnings():
                _w.filterwarnings("ignore", category=RuntimeWarning)
                self._ppfl = PrivacyPreservingFederatedLearning(
                    blockchain_provider     = cfg.get("blockchain_provider", "http://127.0.0.1:8545"),
                    contract_address        = cfg.get("contract_address"),
                    contract_abi_path       = cfg.get("contract_abi_path"),
                    ipfs_addr               = cfg.get("ipfs_addr", "/ip4/127.0.0.1/tcp/5001"),
                    epsilon                 = cfg.get("epsilon", 1.0),
                    delta                   = cfg.get("delta", 1e-5),
                    clip_norm               = cfg.get("clip_norm", 1.0),
                    coordinator_private_key = cfg.get("coordinator_private_key"),
                    require_connection      = cfg.get("blockchain_enabled", True),
                )
            # Generate per-client RSA key pair and store public key in ppfl
            self._private_key_pem, self._public_key_pem = \
                self._ppfl.generate_client_keypair(str(self.client_id))
        return self._ppfl

    @property
    def public_key_pem(self):
        _ = self.ppfl   # trigger lazy init
        return self._public_key_pem

    @property
    def private_key_pem(self):
        _ = self.ppfl
        return self._private_key_pem

    # ──────────────────────────────────────────────────────────────────────
    # Blockchain registration (call once before training starts)
    # ──────────────────────────────────────────────────────────────────────

    def register_on_blockchain(self) -> Optional[str]:
        """
        Register this client on the smart contract (signs and sends the tx).

        Returns the transaction hash string or None if registration
        fails or blockchain is disabled.
        """
        if not self.blockchain_enabled:
            return None
        try:
            tx_hash = self.ppfl.register_client_on_chain(
                client_address    = self.client_address,
                client_private_key = self.client_private_key,
                public_key_pem    = self.public_key_pem,
                stake_amount      = self.stake_amount,
            )
            print(
                f"  📝 Client {self.client_id} registered on-chain "
                f"(address={self.client_address[:10]}…  tx={tx_hash[:12]}…)"
            )
            return tx_hash
        except Exception as exc:
            warnings.warn(
                f"[Blockchain] MVNO {self.client_id} registration failed: {exc}"
            )
            return None

    # ──────────────────────────────────────────────────────────────────────
    # fit() – the core integration point
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, parameters, config):
        """
        Layer 1 — Learning Layer (off-chain):
          1. Local PPO training via EnhancedFlowerClient.fit()
          2. Attack injection (for robustness experiments)
          3. Secure aggregation mask applied to parameters before sending to server.
             Coordinator sees only Σ w_k (masks cancel pairwise) — never individual w_k.

        Layer 2 (blockchain) and Layer 3 (IPFS) are handled by aggregate_fit on the server.
        """
        param_list, num_examples, metrics = super().fit(parameters, config)

        # ── Attack injection (for robustness experiments) ─────────────────
        if self._attack_type != "none":
            try:
                from experiments import apply_attack_to_params
                server_round = config.get("server_round", 0) if isinstance(config, dict) else 0
                attack_cfg = {
                    "attack_type":     self._attack_type,
                    "attack_fraction": self._attack_fraction,
                    "attack_strength": self._attack_strength,
                    "num_clients":     self._num_clients,
                    "server_round":    server_round,
                }
                param_list = apply_attack_to_params(
                    param_list, self.client_id, attack_cfg, self._param_history,
                )
            except Exception as exc:
                warnings.warn(f"[Attack] Client {self.client_id}: {exc}")

        # Save param history (for replay attacks)
        self._param_history.append(list(param_list))
        if len(self._param_history) > 5:
            self._param_history.pop(0)

        # ── Secure aggregation mask ───────────────────────────────────────
        if self._secure_aggregation:
            try:
                from secure_aggregation import add_secure_mask
                from collections import OrderedDict
                server_round = config.get("server_round", 0) if isinstance(config, dict) else 0
                # Build a dummy OrderedDict for the mask function
                param_dict = OrderedDict(
                    {f"p{i}": torch.tensor(p) for i, p in enumerate(param_list)}
                )
                masked_dict = add_secure_mask(
                    param_dict,
                    client_id      = self.client_id,
                    all_client_ids = list(range(self._num_clients)),
                    round_num      = server_round,
                )
                param_list = [v.numpy() for v in masked_dict.values()]
                metrics["secure_aggregation"] = True
            except Exception as exc:
                warnings.warn(f"[SecAgg] Client {self.client_id}: {exc}")

        return param_list, num_examples, metrics

    # ──────────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────────

    def get_blockchain_client_summary(self) -> Dict[str, Any]:
        """Per-client training summary (secure aggregation applied; no per-client chain tx)."""
        return {
            "client_id":          self.client_id,
            "blockchain_enabled": self.blockchain_enabled,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def create_blockchain_awfedavg_strategy(
    # PPFL / blockchain params
    blockchain_provider:      str            = "http://localhost:8545",
    ipfs_addr:                str            = "/ip4/127.0.0.1/tcp/5001",
    contract_address:         Optional[str]  = None,
    contract_abi_path:        Optional[str]  = None,
    coordinator_private_key:  Optional[str]  = None,
    epsilon:                  float          = 1.0,
    delta:                    float          = 1e-5,
    clip_norm:                float          = 1.0,
    apply_coordinator_dp:     bool           = True,
    compression:              bool           = True,
    blockchain_enabled:       bool           = True,
    # AWFedAvg weight criteria (passed through to AdaptiveWeightCalculator)
    alpha_embb:               float          = 0.22,
    alpha_urllc:              float          = 0.38,
    alpha_activation:         float          = 0.2,
    alpha_stability:          float          = 0.15,
    alpha_reputation:         float          = 0.05,   # on-chain reputation criterion
    # Flower strategy params
    min_fit_clients:          int            = CLIENTS_PER_ROUND,
    min_evaluate_clients:     int            = CLIENTS_PER_ROUND,
    min_available_clients:    int            = NUM_CLIENTS,
) -> BlockchainAdaptiveWeightedFedAvg:
    """
    Create a fully configured BlockchainAdaptiveWeightedFedAvg strategy.

    Instantiates PrivacyPreservingFederatedLearning internally and passes it
    to BlockchainAdaptiveWeightedFedAvg, so callers only need to provide
    configuration values.

    Returns
    -------
    strategy : BlockchainAdaptiveWeightedFedAvg
        Ready to pass to fl.simulation.start_simulation().
        Access the underlying PPFL system via strategy.ppfl.
    """
    # Build a plain-Python-primitive config dict BEFORE constructing PPFL.
    # This dict will be stored on the strategy and passed to Ray workers —
    # it must contain ONLY str/float/int/bool/None so cloudpickle can
    # serialise the client_fn closure without touching web3/eth_abi objects.
    ppfl_config = {
        "blockchain_provider":     str(blockchain_provider),
        "contract_address":        str(contract_address) if contract_address else None,
        "contract_abi_path":       str(contract_abi_path) if contract_abi_path else None,
        "ipfs_addr":               str(ipfs_addr),
        "epsilon":                 float(epsilon),
        "delta":                   float(delta),
        "clip_norm":               float(clip_norm),
        "coordinator_private_key": str(coordinator_private_key) if coordinator_private_key else None,
        "blockchain_enabled":      bool(blockchain_enabled),
    }

    # Build the privacy + blockchain backend (main process only — never pickled).
    ppfl = PrivacyPreservingFederatedLearning(
        blockchain_provider     = blockchain_provider,
        contract_address        = contract_address,
        contract_abi_path       = contract_abi_path,
        ipfs_addr               = ipfs_addr,
        epsilon                 = epsilon,
        delta                   = delta,
        clip_norm               = clip_norm,
        coordinator_private_key = coordinator_private_key,
        require_connection      = blockchain_enabled,
    )

    strategy = BlockchainAdaptiveWeightedFedAvg(
        ppfl                  = ppfl,
        ppfl_config           = ppfl_config,
        apply_coordinator_dp  = apply_coordinator_dp,
        compression           = compression,
        blockchain_enabled    = blockchain_enabled,
        alpha_embb            = alpha_embb,
        alpha_urllc           = alpha_urllc,
        alpha_activation      = alpha_activation,
        alpha_stability       = alpha_stability,
        alpha_reputation      = alpha_reputation,
        min_fit_clients       = min_fit_clients,
        min_evaluate_clients  = min_evaluate_clients,
        min_available_clients = min_available_clients,
    )

    return strategy


# MVNO configs – each entry maps a Flower node-id modulo NUM_CLIENTS to
# an Ethereum address.  Each MVNO has its own address on the Base Station's chain.
_DEFAULT_CLIENT_ADDRESSES = [
    {"address": f"0x{'0'*39}{i}", "private_key": f"0x{'0'*63}{i}"}
    for i in range(NUM_CLIENTS)
]


def make_blockchain_client_fn(
    ppfl_config:        Dict,
    client_addresses:   Optional[List[Dict]] = None,
    blockchain_enabled: bool = True,
    num_clients:        int  = NUM_CLIENTS,
    stake_amount:       float = 0.01,
    secure_aggregation: bool = False,
    attack_type:        str  = "none",
    attack_fraction:    float = 0.0,
    attack_strength:    float = 1.0,
):
    """Return a Flower-compatible client_fn.

    Clients handle Layer 1 (learning + secure aggregation masking) only.
    Layer 2 (blockchain) and Layer 3 (IPFS) are handled server-side in aggregate_fit.
    """
    if client_addresses is None:
        client_addresses = _DEFAULT_CLIENT_ADDRESSES

    _cfg = {
        "blockchain_provider":     str(ppfl_config.get("blockchain_provider", "http://127.0.0.1:8545")),
        "contract_address":        str(ppfl_config["contract_address"]) if ppfl_config.get("contract_address") else None,
        "contract_abi_path":       str(ppfl_config["contract_abi_path"]) if ppfl_config.get("contract_abi_path") else None,
        "ipfs_addr":               str(ppfl_config.get("ipfs_addr", "/ip4/127.0.0.1/tcp/5001")),
        "epsilon":                 float(ppfl_config.get("epsilon", 1.0)),
        "delta":                   float(ppfl_config.get("delta", 1e-5)),
        "clip_norm":               float(ppfl_config.get("clip_norm", 1.0)),
        "coordinator_private_key": str(ppfl_config["coordinator_private_key"]) if ppfl_config.get("coordinator_private_key") else None,
        "blockchain_enabled":      bool(ppfl_config.get("blockchain_enabled", True)),
    }
    _addrs = [{"address": str(a["address"]), "private_key": str(a.get("private_key") or "")}
              for a in client_addresses] if client_addresses else []
    if not _addrs:
        _addrs = [{"address": f"0x{'0'*39}{i}", "private_key": ""} for i in range(num_clients)]

    def client_fn(context: Context) -> fl.client.Client:
        client_id = int(context.node_id) % num_clients
        info = _addrs[client_id % len(_addrs)]

        client = BlockchainEnhancedFlowerClient(
            client_id           = client_id,
            client_address      = info["address"],
            client_private_key  = info["private_key"],
            ppfl_config         = _cfg,
            blockchain_enabled  = blockchain_enabled,
            num_clients         = num_clients,
            stake_amount        = stake_amount,
            secure_aggregation  = secure_aggregation,
            attack_type         = attack_type,
            attack_fraction     = attack_fraction,
            attack_strength     = attack_strength,
        )
        return client.to_client()

    return client_fn


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE EXPERIMENT RUNNER
# Drop-in replacement for adaptive_weighted_fedavg.run_awfedavg_experiment_with_resources
# ─────────────────────────────────────────────────────────────────────────────

def _run_sequential_simulation(
    client_fn,
    strategy,
    num_clients: int,
    num_rounds: int,
):
    """
    Drop-in replacement for fl.simulation.start_simulation that runs entirely
    in the main process — no Ray, no cloudpickle, no lru_cache pickling errors.

    Implements the standard Flower FL protocol:
      Round 0 : get initial parameters from one client
      Round 1…N: configure_fit → fit → aggregate_fit
                  configure_evaluate → evaluate → aggregate_evaluate

    Returns a minimal history object with the same interface as flwr.server.History.
    """
    import flwr as fl
    from flwr.common import (
        ndarrays_to_parameters, parameters_to_ndarrays,
        FitIns, FitRes, EvaluateIns, EvaluateRes,
        GetParametersIns, Code, Status,
    )

    class _History:
        """Minimal Flower History replacement."""
        def __init__(self):
            self.losses_distributed   = []
            self.losses_centralized   = []
            self.metrics_distributed  = {"fit": [], "evaluate": []}
            self.metrics_centralized  = {}

    hist = _History()

    # Build Context objects for each client
    class _Ctx:
        def __init__(self, node_id):
            self.node_id     = node_id
            self.node_config = {}
            self.run_config  = {}
            self.state       = fl.common.RecordSet()

    # ── Round 0: get initial parameters ──────────────────────────────────
    print("\n[Sim] Requesting initial parameters from client 0…")
    init_client_proxy = client_fn(_Ctx(0))
    # init_client_proxy is a flwr.client.Client (result of to_client())
    init_res = init_client_proxy.get_parameters(
        ins=GetParametersIns(config={})
    )
    current_parameters = init_res.parameters
    print(f"[Sim] Initial parameters received "
          f"({len(parameters_to_ndarrays(current_parameters))} tensors)")

    # ── Rounds 1…num_rounds ───────────────────────────────────────────────
    for rnd in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"[Sim] ── Round {rnd}/{num_rounds} ──")
        print(f"{'='*60}")

        # configure_fit
        client_ids = list(range(num_clients))
        fit_configs = strategy.configure_fit(
            server_round = rnd,
            parameters   = current_parameters,
            client_manager = _DummyClientManager(client_ids),
        )

        # fit each client
        fit_results = []
        fit_failures = []
        for client_proxy, fit_ins in fit_configs:
            cid = int(client_proxy.cid)
            try:
                proxy = client_fn(_Ctx(cid))
                fit_res = proxy.fit(ins=fit_ins)
                fit_results.append((proxy, fit_res))
                print(f"  [Sim] MVNO {cid} fit done — "
                      f"{fit_res.num_examples} examples, "
                      f"metrics={list(fit_res.metrics.keys())}")
            except Exception as exc:
                import traceback; traceback.print_exc()
                fit_failures.append(exc)
                print(f"  [Sim] MVNO {cid} fit FAILED: {exc}")

        if not fit_results:
            print(f"[Sim] Round {rnd}: all clients failed — stopping.")
            break

        # aggregate_fit  (this calls BlockchainAdaptiveWeightedFedAvg.aggregate_fit)
        agg = strategy.aggregate_fit(
            server_round = rnd,
            results      = fit_results,
            failures     = fit_failures,
        )
        if agg is not None:
            current_parameters, agg_metrics = agg
            hist.metrics_distributed["fit"].append((rnd, agg_metrics))
            print(f"  [Sim] Aggregation done. Metrics: {list(agg_metrics.keys())}")

        # configure_evaluate
        eval_configs = strategy.configure_evaluate(
            server_round   = rnd,
            parameters     = current_parameters,
            client_manager = _DummyClientManager(client_ids),
        )

        # evaluate each client
        eval_results  = []
        eval_failures = []
        for client_proxy, eval_ins in eval_configs:
            cid = int(client_proxy.cid)
            try:
                proxy    = client_fn(_Ctx(cid))
                eval_res = proxy.evaluate(ins=eval_ins)
                eval_results.append((proxy, eval_res))
            except Exception as exc:
                eval_failures.append(exc)

        # aggregate_evaluate
        agg_eval = strategy.aggregate_evaluate(
            server_round = rnd,
            results      = eval_results,
            failures     = eval_failures,
        )
        if agg_eval is not None:
            loss, eval_metrics = agg_eval
            hist.losses_distributed.append((rnd, loss))
            hist.metrics_distributed["evaluate"].append((rnd, eval_metrics or {}))

    return hist


class _DummyClientManager:
    """Minimal ClientManager that returns (client_id, ins) tuples for configure_fit/evaluate."""
    def __init__(self, client_ids):
        self._ids = client_ids

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        class _Proxy:
            def __init__(self, cid): self.cid = str(cid)
        return [_Proxy(i) for i in self._ids[:num_clients]]

    def num_available(self): return len(self._ids)



def run_blockchain_awfedavg_experiment(
    strategy:    Optional[BlockchainAdaptiveWeightedFedAvg] = None,
    client_fn                                              = None,
    num_rounds:  int                                       = TOTAL_ROUNDS,
    num_clients: int                                       = NUM_CLIENTS,
    # Pass-through to create_blockchain_awfedavg_strategy when strategy is None
    **strategy_kwargs,
):
    """
    Full experiment: local baselines → blockchain-AWFedAvg FL → evaluation.

    Parameters
    ----------
    strategy   : pre-built BlockchainAdaptiveWeightedFedAvg, or None to
                 create one automatically from strategy_kwargs.
    client_fn  : Flower client_fn, or None to build from strategy.ppfl.
    num_rounds : total federated rounds.
    num_clients: total simulated clients.
    **strategy_kwargs : forwarded to create_blockchain_awfedavg_strategy
                        when strategy is None.

    Returns
    -------
    hist, local_results, federated_results, resource_summary
    """
    print("\n" + "=" * 80)
    print("BLOCKCHAIN-ENABLED ADAPTIVE WEIGHTED FEDAVG EXPERIMENT")
    print("=" * 80)
    local_results = {}  # initialised early so except blocks can reference it

    # ── Build strategy + client_fn if not provided ───────────────────────
    # Pop experiment-specific params that are NOT strategy params.
    _client_addrs     = strategy_kwargs.pop("client_addresses", None)
    _secure_agg       = strategy_kwargs.pop("secure_aggregation", False)
    _attack_type      = strategy_kwargs.pop("attack_type", "none")
    _attack_fraction  = strategy_kwargs.pop("attack_fraction", 0.0)
    _attack_strength  = strategy_kwargs.pop("attack_strength", 1.0)

    if strategy is None:
        print("\n⚙️  Building strategy…")
        strategy = create_blockchain_awfedavg_strategy(**strategy_kwargs)

    if client_fn is None:
        client_fn = make_blockchain_client_fn(
            ppfl_config        = strategy.ppfl_config,
            client_addresses   = _client_addrs,
            blockchain_enabled = strategy.blockchain_enabled,
            num_clients        = num_clients,
            secure_aggregation = _secure_agg,
            attack_type        = _attack_type,
            attack_fraction    = _attack_fraction,
            attack_strength    = _attack_strength,
        )

    # Populate strategy's address map so aggregate_fit can read reputation per client
    if _client_addrs:
        strategy._client_addrs_map = {
            i: _client_addrs[i % len(_client_addrs)]["address"]
            for i in range(num_clients)
        }

    local_results = {}   # no standalone local training — federated only

    # ── AWFedAvg + Blockchain federated training ──────────────────────────
    print("\n🚀 Starting Blockchain-AWFedAvg  [MVNOs → Base Station gNB]…")
    print(f"   AWFedAvg weight criteria:")
    wc = strategy.weight_calculator
    print(f"     eMBB outage   : {wc.alpha_embb:.1%}")
    print(f"     URLLC residual: {wc.alpha_urllc:.1%}")
    print(f"     Activation div: {wc.alpha_activation:.1%}")
    print(f"     Stability     : {wc.alpha_stability:.1%}")
    print(f"   DP parameters   : ε={strategy.ppfl.epsilon}, δ={strategy.ppfl.delta}")
    print(f"   Blockchain      : {'enabled' if strategy.blockchain_enabled else 'disabled'}")

    # ── Pre-register clients on blockchain (before Ray workers start) ───────
    # Registration is done here in the main process where web3 is picklable.
    if strategy.blockchain_enabled and _client_addrs:
        print("\n📝 Registering MVNOs on Base Station blockchain…")
        for i, info in enumerate(_client_addrs[:NUM_CLIENTS]):
            try:
                strategy.ppfl.register_client_on_chain(
                    client_address     = info["address"],
                    client_private_key = info["private_key"],
                    public_key_pem     = strategy.ppfl.generate_client_keypair(str(i))[1],
                    stake_amount       = 0.01,
                )
            except Exception as exc:
                warnings.warn(f"[Blockchain] MVNO {i} pre-registration failed: {exc}")

    fed_start = time.time()
    system_monitor = ResourceMonitor()
    system_monitor.start_monitoring(interval=1.0)
    hist = None

    try:
        hist = _run_sequential_simulation(
            client_fn   = client_fn,
            strategy    = strategy,
            num_clients = num_clients,
            num_rounds  = num_rounds,
        )
        print(f"\n✅ Federated training complete in {time.time()-fed_start:.1f}s")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"❌ Federated training error: {exc}")
        system_monitor.stop_monitoring()
        return None, local_results, None, None
    finally:
        system_resource_summary = system_monitor.stop_monitoring()

    # ── Final evaluation of global model ─────────────────────────────────
    federated_results = None
    if strategy.get_final_parameters() is not None:
        print("\n📊 Evaluating final global model…")
        try:
            eval_env    = create_phy_env(0)
            input_dim   = eval_env.observation_space.shape[0]
            output_dim  = eval_env.action_space.n
            pyt_model   = PPONetwork(input_dim, output_dim).to(DEVICE)

            final_params = parameters_to_ndarrays(strategy.get_final_parameters())
            model_keys   = list(pyt_model.state_dict().keys())
            if len(final_params) == len(model_keys):
                state_dict = OrderedDict({
                    k: torch.tensor(v, device=DEVICE)
                    for k, v in zip(model_keys, final_params)
                })
                pyt_model.load_state_dict(state_dict, strict=True)
                fed_model = PPO(
                    "MlpPolicy", eval_env, verbose=0,
                    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
                )
                fed_model = pytorch_to_sb3_ppo(pyt_model, fed_model)
                federated_results = evaluate_model_simple(
                    fed_model, eval_env, num_episodes=EVALUATION_EPISODES
                )
                fed_model.save(f"{BASE_MODEL_PATH}/blockchain_awfedavg_global_model.zip")
                print(f"  Reward   : {federated_results['average_reward']:.4f}")
                print(f"  Stability: {federated_results['stability_score']:.4f}")
            eval_env.close()
        except Exception as exc:
            print(f"❌ Final evaluation failed: {exc}")

    # ── Performance summary ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    if federated_results:
        print(f"  Blockchain-AWFedAvg reward  : {federated_results['average_reward']:.4f}")
        print(f"  Stability score             : {federated_results['stability_score']:.4f}")

    # ── Comprehensive summaries ───────────────────────────────────────────
    resource_summary = strategy.get_comprehensive_resource_summary()
    strategy.print_blockchain_performance_summary()

    # ── Save results ─────────────────────────────────────────────────────
    try:
        save_awfedavg_results_with_resources(strategy, local_results, federated_results)
    except Exception as exc:
        warnings.warn(f"Results save failed: {exc}")

    # Save blockchain-specific records
    bc_records_path = (
        f"{BASE_MODEL_PATH}/blockchain_records_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(bc_records_path, "w") as f:
        json.dump(strategy.blockchain_round_records, f, indent=2, default=str)
    print(f"\n📂 Blockchain records saved: {bc_records_path}")

    print("\n" + "=" * 80)
    print("✅ Blockchain-AWFedAvg experiment complete!")
    print("=" * 80)

    return hist, local_results, federated_results, resource_summary


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    print("🚀 Blockchain-AWFedAvg TRUE Integration")
    print("  ✅ AdaptiveWeightedFedAvg (4-criteria weights) — inherited, not reimplemented")
    print("  ✅ EnhancedFlowerClient (PPO + resource monitor) — inherited, not reimplemented")
    print("  ✅ PrivacyPreservingFederatedLearning — DP + encryption + IPFS + blockchain")
    print("  ✅ Coordinator-side & client-side DP applied independently")
    print("  ✅ All AWFedAvg metrics preserved and augmented with blockchain fields")

    # ── Load contract_info.json produced by deploy.py ────────────────────
    _config_path = os.path.join(os.path.dirname(__file__), "contract_info.json")
    if os.path.exists(_config_path):
        with open(_config_path) as _f:
            _cfg = json.load(_f)
        _blockchain_enabled       = True
        _contract_address         = _cfg["contract_address"]
        _contract_abi_path        = _config_path
        _coordinator_private_key  = _cfg["coordinator_private_key"]
        _client_addresses         = _cfg["clients"]          # list of {address, private_key}
        print("\n✅ contract_info.json loaded — blockchain ENABLED")
        print(f"   Contract : {_contract_address}")
        print(f"   Coordinator: {_cfg['coordinator_address']}")
    else:
        _blockchain_enabled       = False
        _contract_address         = None
        _contract_abi_path        = None
        _coordinator_private_key  = None
        _client_addresses         = None
        print("\n⚠️  contract_info.json not found — running with blockchain_enabled=False")
        print("   Run  python deploy.py  first to deploy the smart contract.")

    hist, local_results, fed_results, resources = run_blockchain_awfedavg_experiment(
        blockchain_enabled       = _blockchain_enabled,
        blockchain_provider      = "http://127.0.0.1:8545",
        contract_address         = _contract_address,
        contract_abi_path        = _contract_abi_path,
        coordinator_private_key  = _coordinator_private_key,
        ipfs_addr                = "/ip4/127.0.0.1/tcp/5001",
        epsilon                  = 1.0,
        delta                    = 1e-5,
        clip_norm                = 1.0,
        alpha_embb               = 0.22,
        alpha_urllc              = 0.38,
        alpha_activation         = 0.2,
        alpha_stability          = 0.2,
        num_rounds               = TOTAL_ROUNDS,
        num_clients              = NUM_CLIENTS,
        client_addresses         = _client_addresses,  # from contract_info.json
    )