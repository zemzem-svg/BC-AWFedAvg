"""
Scalability, Baseline & Ablation Experiments
=============================================

Three complementary experiments needed for a strong publication:

  1. FedAvg Baseline
     Plain FedAvg with uniform weights (1/K) on the same environment.
     Quantifies the gain from AWFedAvg's adaptive weighting.

  2. Scalability Study  (K = 3, 5, 10, 20)
     Runs the full Blockchain-AWFedAvg system at increasing client counts.
     Measures: reward convergence, BC overhead, IPFS size, training time.

  3. Ablation Study
     Five progressive configurations on K=3, T=15:
       A — FL only            (no DP, no SecAgg, no blockchain)
       B — FL + DP            (Gaussian DP added, rest off)
       C — FL + SecAgg        (secure masking added, rest off)
       D — FL + Blockchain    (audit trail, no DP / no SecAgg)
       E — Full system        (DP + SecAgg + Blockchain)

     Lets reviewers see the individual contribution of each component.

Usage
-----
    python scalability_ablation.py

Results are saved to federated_models_awfedavg_15/
  ablation_results_<timestamp>.json
  scaling_results_<timestamp>.json
  fedavg_baseline_<timestamp>.json
"""

from __future__ import annotations

import os, json, time, warnings, copy
import numpy as np
import torch
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import (
    ndarrays_to_parameters, parameters_to_ndarrays,
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    GetParametersIns, Code, Status,
)

# ── Import shared infrastructure ─────────────────────────────────────────────
from adaptive_weighted_fedavg import (
    AdaptiveWeightedFedAvg,
    EnhancedFlowerClient,
    PPONetwork,
    ResourceMonitor,
    CLIENT_CONFIGS,
    NUM_CLIENTS,
    TOTAL_ROUNDS,
    LOCAL_EPOCHS,
    EVALUATION_EPISODES,
    DEVICE,
    BASE_MODEL_PATH,
    create_phy_env,
    evaluate_model_simple,
    pytorch_to_sb3_ppo,
)
from blockchain_awfedavg_true_integration import (
    BlockchainAdaptiveWeightedFedAvg,
    BlockchainEnhancedFlowerClient,
    _run_sequential_simulation,
    _DummyClientManager,
    create_blockchain_awfedavg_strategy,
    make_blockchain_client_fn,
    PPONETWORK_LAYER_KEYS,
    ndarrays_to_ordered_dict,
)
from stable_baselines3 import PPO

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save(data: dict, filename: str):
    os.makedirs(BASE_MODEL_PATH, exist_ok=True)
    path = os.path.join(BASE_MODEL_PATH, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  💾 Saved → {path}")
    return path


def _evaluate_final(strategy, num_episodes: int = EVALUATION_EPISODES) -> Optional[Dict]:
    """Evaluate the global model from a finished strategy."""
    if strategy.get_final_parameters() is None:
        return None
    try:
        env       = create_phy_env(0)
        input_dim = env.observation_space.shape[0]
        output_dim= env.action_space.n
        pyt_model = PPONetwork(input_dim, output_dim).to(DEVICE)
        arrays    = parameters_to_ndarrays(strategy.get_final_parameters())
        keys      = list(pyt_model.state_dict().keys())
        if len(arrays) == len(keys):
            sd = OrderedDict({k: torch.tensor(v, device=DEVICE) for k, v in zip(keys, arrays)})
            pyt_model.load_state_dict(sd, strict=True)
            sb3 = PPO("MlpPolicy", env, verbose=0,
                      policy_kwargs=dict(net_arch=dict(pi=[64,64], vf=[64,64])))
            sb3 = pytorch_to_sb3_ppo(pyt_model, sb3)
            results = evaluate_model_simple(sb3, env, num_episodes=num_episodes)
            env.close()
            return results
        env.close()
    except Exception as e:
        warnings.warn(f"Final evaluation failed: {e}")
    return None


def _per_round_rewards(strategy) -> List[float]:
    """Extract global_performance per round from round_metrics."""
    return [m.get("global_performance", float("nan"))
            for m in strategy.round_metrics]


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED CLIENT CONFIGS for K > 3
# Cycles through the base 3 profiles so every client has a valid config.
# ─────────────────────────────────────────────────────────────────────────────

def _build_client_configs(num_clients: int) -> List[Dict]:
    """
    Return num_clients configs by cycling through [Light, Medium, Heavy].
    For K=10: [L,M,H, L,M,H, L,M,H, L]
    """
    base = CLIENT_CONFIGS
    return [base[i % len(base)] for i in range(num_clients)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEDAVG BASELINE
# ─────────────────────────────────────────────────────────────────────────────

class _UniformFedAvg(fl.server.strategy.FedAvg):
    """
    Standard FedAvg with uniform weights (1/K).
    Uses the same EnhancedFlowerClient as AWFedAvg for a fair comparison.
    Overrides aggregate_fit only to record per-round global performance.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_metrics   : List[Dict] = []
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.final_parameters = aggregated
            # Compute simple average of client rewards for tracking
            rewards = [r.metrics.get("average_reward", float("nan"))
                       for _, r in results]
            avg_reward = float(np.nanmean(rewards)) if rewards else float("nan")
            self.round_metrics.append({
                "round": server_round,
                "global_performance": avg_reward,
                "weights": [1.0 / len(results)] * len(results),
            })
            print(f"  [FedAvg] Round {server_round} — avg reward = {avg_reward:.4f}")
        return aggregated, metrics or {}

    def get_final_parameters(self):
        return self.final_parameters


def _make_plain_client_fn(num_clients: int):
    """Client factory for FedAvg baseline — no blockchain, no secure aggregation."""
    def client_fn(context) -> fl.client.Client:
        client_id = int(context.node_id) % num_clients
        client = EnhancedFlowerClient(client_id)
        return client.to_client()
    return client_fn


def run_fedavg_baseline(
    num_clients: int = NUM_CLIENTS,
    num_rounds:  int = TOTAL_ROUNDS,
) -> Dict:
    """
    Run standard FedAvg (uniform 1/K weights) as a comparison baseline.

    Returns a summary dict with per-round rewards and final performance.
    """
    print("\n" + "="*70)
    print("BASELINE: Standard FedAvg (uniform weights 1/K)")
    print(f"  K={num_clients} clients,  T={num_rounds} rounds")
    print("="*70)

    strategy = _UniformFedAvg(
        min_fit_clients      = num_clients,
        min_evaluate_clients = num_clients,
        min_available_clients= num_clients,
    )
    client_fn = _make_plain_client_fn(num_clients)

    t0 = time.time()
    _run_sequential_simulation(client_fn, strategy, num_clients, num_rounds)
    elapsed = time.time() - t0

    final_results = _evaluate_final(strategy)
    round_rewards  = _per_round_rewards(strategy)

    summary = {
        "experiment"     : "fedavg_baseline",
        "timestamp"      : _timestamp(),
        "num_clients"    : num_clients,
        "num_rounds"     : num_rounds,
        "strategy"       : "FedAvg (uniform 1/K)",
        "per_round_rewards": round_rewards,
        "reward_round_1" : round_rewards[0]  if round_rewards else None,
        "reward_final"   : round_rewards[-1] if round_rewards else None,
        "final_eval"     : final_results,
        "total_time_s"   : elapsed,
    }

    _save(summary, f"fedavg_baseline_{_timestamp()}.json")
    print(f"\n  FedAvg final reward   : {summary['reward_final']}")
    print(f"  Total time            : {elapsed:.1f}s")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCALABILITY STUDY
# ─────────────────────────────────────────────────────────────────────────────

def run_scaling_experiment(
    client_counts: List[int]    = [3, 5, 10, 20],
    num_rounds:    int           = TOTAL_ROUNDS,
    blockchain_provider: str     = "http://127.0.0.1:8545",
    contract_address: Optional[str]  = None,
    contract_abi_path: Optional[str] = None,
    coordinator_private_key: Optional[str] = None,
    ipfs_addr: str               = "/ip4/127.0.0.1/tcp/5001",
    blockchain_enabled: bool     = True,
    epsilon: float               = 1.0,
    delta: float                 = 1e-5,
) -> Dict:
    """
    Run Blockchain-AWFedAvg for each K in client_counts.

    Measures per-K:
      - Reward convergence curve
      - Final reward
      - BC overhead per round (mean ± std)
      - IPFS upload size
      - Training wall-clock time
      - Communication volume (K × model_size × T)

    Returns a dict with one sub-dict per K.
    """
    print("\n" + "="*70)
    print("SCALABILITY STUDY: K ∈ " + str(client_counts))
    print(f"  T={num_rounds} rounds per configuration")
    print("="*70)

    all_results = {}

    for K in client_counts:
        print(f"\n{'─'*60}")
        print(f"  K = {K} clients")
        print(f"{'─'*60}")

        strategy = create_blockchain_awfedavg_strategy(
            blockchain_provider     = blockchain_provider,
            ipfs_addr               = ipfs_addr,
            contract_address        = contract_address,
            contract_abi_path       = contract_abi_path,
            coordinator_private_key = coordinator_private_key,
            blockchain_enabled      = blockchain_enabled,
            epsilon                 = epsilon,
            delta                   = delta,
            clip_norm               = 1.0,
            alpha_embb              = 0.3,
            alpha_urllc             = 0.3,
            alpha_activation        = 0.2,
            alpha_stability         = 0.15,
            alpha_reputation        = 0.05,
            min_fit_clients         = K,
            min_evaluate_clients    = K,
            min_available_clients   = K,
        )
        client_fn = make_blockchain_client_fn(
            ppfl_config        = strategy.ppfl_config,
            blockchain_enabled = blockchain_enabled,
            secure_aggregation = True,
            num_clients        = K,
        )

        t0 = time.time()
        _run_sequential_simulation(client_fn, strategy, K, num_rounds)
        elapsed = time.time() - t0

        # Per-round reward
        round_rewards = _per_round_rewards(strategy)

        # Blockchain overhead stats
        bc_overheads = [r.get("blockchain_overhead_s", 0.0)
                        for r in strategy.blockchain_round_records]
        ipfs_sizes   = [r.get("ipfs_upload_size_kb", 0.0)
                        for r in strategy.blockchain_round_records]
        eps_total    = strategy.blockchain_round_records[-1].get("eps_total", 0.0) \
                       if strategy.blockchain_round_records else 0.0

        # Communication: K clients × params_size × T rounds
        # Each client sends ~43 KB (float32 params)
        model_kb    = 44600 / 1024          # raw model size KB
        comm_total_mb = K * model_kb * num_rounds / 1024

        final_results = _evaluate_final(strategy)

        all_results[f"K{K}"] = {
            "num_clients"          : K,
            "num_rounds"           : num_rounds,
            "per_round_rewards"    : round_rewards,
            "reward_round_1"       : round_rewards[0]  if round_rewards else None,
            "reward_final"         : round_rewards[-1] if round_rewards else None,
            "bc_overhead_mean_ms"  : float(np.mean(bc_overheads))*1000 if bc_overheads else 0,
            "bc_overhead_std_ms"   : float(np.std(bc_overheads))*1000  if bc_overheads else 0,
            "ipfs_size_mean_kb"    : float(np.mean(ipfs_sizes))        if ipfs_sizes   else 0,
            "communication_total_mb": comm_total_mb,
            "eps_total"            : eps_total,
            "wall_time_s"          : elapsed,
            "final_eval"           : final_results,
        }

        print(f"  K={K}  final reward={round_rewards[-1] if round_rewards else 'N/A':.4f}"
              f"  BC={all_results[f'K{K}']['bc_overhead_mean_ms']:.1f}ms"
              f"  time={elapsed:.1f}s")

    scaling_summary = {
        "experiment"   : "scalability_study",
        "timestamp"    : _timestamp(),
        "client_counts": client_counts,
        "num_rounds"   : num_rounds,
        "results"      : all_results,
    }

    _save(scaling_summary, f"scaling_results_{_timestamp()}.json")
    return scaling_summary


# ─────────────────────────────────────────────────────────────────────────────
# 3. ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────

# Each ablation config toggles exactly one feature on top of the previous.
ABLATION_CONFIGS = [
    {
        "label"             : "A — FL only",
        "description"       : "AWFedAvg, no DP, no SecAgg, no Blockchain",
        "blockchain_enabled": False,
        "apply_dp"          : False,
        "secure_aggregation": False,
    },
    {
        "label"             : "B — FL + DP",
        "description"       : "AWFedAvg + Gaussian DP (ε=1.0)",
        "blockchain_enabled": False,
        "apply_dp"          : True,
        "secure_aggregation": False,
    },
    {
        "label"             : "C — FL + SecAgg",
        "description"       : "AWFedAvg + Secure Aggregation masks",
        "blockchain_enabled": False,
        "apply_dp"          : False,
        "secure_aggregation": True,
    },
    {
        "label"             : "D — FL + Blockchain",
        "description"       : "AWFedAvg + Blockchain audit trail (no DP, no SecAgg)",
        "blockchain_enabled": True,
        "apply_dp"          : False,
        "secure_aggregation": False,
    },
    {
        "label"             : "E — Full System",
        "description"       : "AWFedAvg + DP + SecAgg + Blockchain",
        "blockchain_enabled": True,
        "apply_dp"          : True,
        "secure_aggregation": True,
    },
]


def run_ablation_study(
    num_clients: int             = NUM_CLIENTS,
    num_rounds:  int             = TOTAL_ROUNDS,
    blockchain_provider: str     = "http://127.0.0.1:8545",
    contract_address: Optional[str]  = None,
    contract_abi_path: Optional[str] = None,
    coordinator_private_key: Optional[str] = None,
    ipfs_addr: str               = "/ip4/127.0.0.1/tcp/5001",
    epsilon: float               = 1.0,
    delta: float                 = 1e-5,
) -> Dict:
    """
    Run all 5 ablation configurations and record per-round rewards,
    final performance, and overhead for each.

    Returns a dict with one sub-dict per configuration label.
    """
    print("\n" + "="*70)
    print("ABLATION STUDY: 5 configurations (A→E)")
    print(f"  K={num_clients} clients,  T={num_rounds} rounds each")
    print("="*70)

    ablation_results = {}

    for cfg in ABLATION_CONFIGS:
        label = cfg["label"]
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"  {cfg['description']}")
        print(f"{'─'*60}")

        # Build strategy — DP is disabled by setting apply_coordinator_dp=False
        strategy = create_blockchain_awfedavg_strategy(
            blockchain_provider     = blockchain_provider,
            ipfs_addr               = ipfs_addr,
            contract_address        = contract_address,
            contract_abi_path       = contract_abi_path,
            coordinator_private_key = coordinator_private_key,
            blockchain_enabled      = cfg["blockchain_enabled"],
            apply_coordinator_dp    = cfg["apply_dp"],
            epsilon                 = epsilon,
            delta                   = delta,
            clip_norm               = 1.0,
            alpha_embb              = 0.3,
            alpha_urllc             = 0.3,
            alpha_activation        = 0.2,
            alpha_stability         = 0.15,
            alpha_reputation        = 0.05,
            min_fit_clients         = num_clients,
            min_evaluate_clients    = num_clients,
            min_available_clients   = num_clients,
        )
        client_fn = make_blockchain_client_fn(
            ppfl_config        = strategy.ppfl_config,
            blockchain_enabled = cfg["blockchain_enabled"],
            secure_aggregation = cfg["secure_aggregation"],
            num_clients        = num_clients,
        )

        t0 = time.time()
        _run_sequential_simulation(client_fn, strategy, num_clients, num_rounds)
        elapsed = time.time() - t0

        round_rewards = _per_round_rewards(strategy)

        # Blockchain overhead (0 if disabled)
        bc_overheads = [r.get("blockchain_overhead_s", 0.0)
                        for r in strategy.blockchain_round_records]
        eps_final    = strategy.blockchain_round_records[-1].get("eps_total", 0.0) \
                       if (cfg["apply_dp"] and strategy.blockchain_round_records) else 0.0

        # Overhead breakdown
        overhead_pct = 0.0
        total_train  = sum(r.get("training_time", 0.0)
                           for r in (strategy.round_metrics or []))
        if total_train > 0 and bc_overheads:
            overhead_pct = sum(bc_overheads) / total_train * 100

        final_results = _evaluate_final(strategy)

        ablation_results[label] = {
            "label"              : label,
            "description"        : cfg["description"],
            "blockchain_enabled" : cfg["blockchain_enabled"],
            "dp_enabled"         : cfg["apply_dp"],
            "secure_agg_enabled" : cfg["secure_aggregation"],
            "per_round_rewards"  : round_rewards,
            "reward_round_1"     : round_rewards[0]  if round_rewards else None,
            "reward_round_5"     : round_rewards[4]  if len(round_rewards) > 4 else None,
            "reward_final"       : round_rewards[-1] if round_rewards else None,
            "bc_overhead_mean_ms": float(np.mean(bc_overheads))*1000 if bc_overheads else 0,
            "bc_overhead_pct"    : overhead_pct,
            "eps_total"          : eps_final,
            "wall_time_s"        : elapsed,
            "final_eval"         : final_results,
        }

        r1  = round_rewards[0]  if round_rewards else float("nan")
        rN  = round_rewards[-1] if round_rewards else float("nan")
        imp = (rN - r1) / abs(r1) * 100 if r1 != 0 else 0
        print(f"  Reward R1={r1:.3f} → R{num_rounds}={rN:.3f}  ({imp:+.1f}%)")
        print(f"  Wall-clock: {elapsed:.1f}s  BC overhead: {ablation_results[label]['bc_overhead_mean_ms']:.1f}ms/round")

    # Summary table
    print("\n" + "="*70)
    print("ABLATION SUMMARY TABLE")
    print(f"{'Config':<30} {'R1':>8} {'R_final':>8} {'Δ%':>8} {'BC(ms)':>8} {'ε_total':>8}")
    print("─"*70)
    for label, res in ablation_results.items():
        r1  = res["reward_round_1"] or 0
        rN  = res["reward_final"]   or 0
        imp = (rN - r1) / abs(r1) * 100 if r1 != 0 else 0
        bc  = res["bc_overhead_mean_ms"]
        eps = res["eps_total"]
        print(f"{label:<30} {r1:>8.3f} {rN:>8.3f} {imp:>8.1f} {bc:>8.1f} {eps:>8.3f}")

    ablation_summary = {
        "experiment"  : "ablation_study",
        "timestamp"   : _timestamp(),
        "num_clients" : num_clients,
        "num_rounds"  : num_rounds,
        "epsilon"     : epsilon,
        "delta"       : delta,
        "configs"     : ABLATION_CONFIGS,
        "results"     : ablation_results,
    }

    _save(ablation_summary, f"ablation_results_{_timestamp()}.json")
    return ablation_summary


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON HELPER: AWFedAvg vs FedAvg
# ─────────────────────────────────────────────────────────────────────────────

def compare_awfedavg_vs_fedavg(
    num_clients: int = NUM_CLIENTS,
    num_rounds:  int = TOTAL_ROUNDS,
    **blockchain_kwargs,
) -> Dict:
    """
    Run both FedAvg and full Blockchain-AWFedAvg side-by-side and
    report the gain attributable to adaptive weighting.
    """
    print("\n" + "="*70)
    print("COMPARISON: FedAvg (uniform) vs Blockchain-AWFedAvg (adaptive)")
    print("="*70)

    fedavg  = run_fedavg_baseline(num_clients, num_rounds)
    awfedavg_summary = run_ablation_study(
        num_clients=num_clients,
        num_rounds=num_rounds,
        **blockchain_kwargs,
    )
    full_result = awfedavg_summary["results"].get("E — Full System", {})

    fedavg_final  = fedavg["reward_final"]  or float("nan")
    awfedavg_final= full_result.get("reward_final") or float("nan")
    gain          = awfedavg_final - fedavg_final

    print("\n" + "="*70)
    print("COMPARISON RESULT")
    print(f"  FedAvg  final reward      : {fedavg_final:.4f}")
    print(f"  AWFedAvg (full) reward    : {awfedavg_final:.4f}")
    print(f"  Absolute gain             : {gain:+.4f}")
    if fedavg_final != 0:
        print(f"  Relative improvement      : {gain/abs(fedavg_final)*100:+.2f}%")
    print("="*70)

    comparison = {
        "experiment"      : "awfedavg_vs_fedavg",
        "timestamp"       : _timestamp(),
        "num_clients"     : num_clients,
        "num_rounds"      : num_rounds,
        "fedavg_final"    : fedavg_final,
        "awfedavg_final"  : awfedavg_final,
        "absolute_gain"   : gain,
        "relative_gain_pct": gain/abs(fedavg_final)*100 if fedavg_final != 0 else 0,
        "fedavg_rewards"  : fedavg["per_round_rewards"],
        "awfedavg_rewards": full_result.get("per_round_rewards", []),
    }
    _save(comparison, f"comparison_fedavg_vs_awfedavg_{_timestamp()}.json")
    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # ── Load contract info if available ──────────────────────────────────────
    _cfg_path = os.path.join(os.path.dirname(__file__), "contract_info.json")
    if os.path.exists(_cfg_path):
        with open(_cfg_path) as _f:
            _info = json.load(_f)
        _bc_kwargs = dict(
            blockchain_enabled      = True,
            blockchain_provider     = "http://127.0.0.1:8545",
            contract_address        = _info["contract_address"],
            contract_abi_path       = _cfg_path,
            coordinator_private_key = _info["coordinator_private_key"],
            ipfs_addr               = "/ip4/127.0.0.1/tcp/5001",
        )
        print("✅ contract_info.json found — blockchain ENABLED for all experiments")
    else:
        _bc_kwargs = dict(blockchain_enabled=False)
        print("⚠️  contract_info.json not found — blockchain DISABLED (run deploy.py first)")

    # ── 1. FedAvg baseline (K=3, T=15) ───────────────────────────────────────
    fedavg_results = run_fedavg_baseline(
        num_clients = NUM_CLIENTS,
        num_rounds  = TOTAL_ROUNDS,
    )

    # ── 2. Ablation study (K=3, T=15, 5 configs) ─────────────────────────────
    ablation_results = run_ablation_study(
        num_clients = NUM_CLIENTS,
        num_rounds  = TOTAL_ROUNDS,
        epsilon     = 1.0,
        delta       = 1e-5,
        **_bc_kwargs,
    )

    # ── 3. Scalability study (K=3,5,10,20, T=15) ─────────────────────────────
    scaling_results = run_scaling_experiment(
        client_counts = [3, 5, 10, 20],
        num_rounds    = TOTAL_ROUNDS,
        epsilon       = 1.0,
        delta         = 1e-5,
        **_bc_kwargs,
    )

    # ── Print final comparison table ──────────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    # FedAvg vs AWFedAvg (full system)
    fedavg_r  = fedavg_results["reward_final"] or float("nan")
    awfedavg_r= ablation_results["results"]["E — Full System"]["reward_final"] or float("nan")
    gain      = awfedavg_r - fedavg_r
    print(f"\n  FedAvg baseline       : {fedavg_r:.4f}")
    print(f"  AWFedAvg (full system): {awfedavg_r:.4f}")
    print(f"  Gain from AWFedAvg    : {gain:+.4f} ({gain/abs(fedavg_r)*100:+.1f}%)")

    # Scalability
    print("\n  Scalability (final reward vs K):")
    for key, res in scaling_results["results"].items():
        print(f"    {key}: reward={res['reward_final']:.4f}  "
              f"BC={res['bc_overhead_mean_ms']:.1f}ms  "
              f"time={res['wall_time_s']:.1f}s")

    print("\n✅ All experiments complete. Results saved to", BASE_MODEL_PATH)
