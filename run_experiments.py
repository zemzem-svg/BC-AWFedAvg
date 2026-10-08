"""
BC-AWFedAvg — canonical experiment driver.

This file is an experiment orchestrator only. It deliberately does NOT contain a
second implementation of BC-AWFedAvg. Every proposed-method run is delegated to
`blockchain_awfedavg.run_blockchain_awfedavg_experiment`, which is the canonical
implementation of the thesis protocol.

Canonical protocol:
    Phase 1: PPO local training -> QoS/learning metrics -> five-criterion weights
    Phase 2: client-update DP -> weighting -> pairwise masks -> server secure sum
    Governance: blockchain metadata/reputation
    Storage: publication-level DP -> encryption -> IPFS -> on-chain CID

Default thesis settings:
    K = 5
    T = 15
    seeds = {42, 101, 202, 303, 404, 505, 606, 707, 808, 909}
    alpha = (0.22, 0.38, 0.20, 0.15, 0.05)
    epsilon = 1.0
    delta = 1e-5
    clip norm = 1.0
    reputation beta = 0.85
    smoothing eta = 0.7
    isolation threshold theta_iso = 1/(2K)

Reputation model synchronized with the current thesis specification:
    (1) Reputation score used in weighting:
            s_rep,k^(t) = rho_k^(t) / sum_j rho_j^(t)

    (2) Leaky-integrator reputation dynamics:
            rho_k^(t) = beta*rho_k^(t-1)
                       + (1-beta)*g(E_k^(t), R_k^(t), S_k^(t))
        with beta = 0.85 and bounded g(.) in [0,1].

    (3) Reproducible implementation of the bounded target:
            g_k^(t) = (s_E,k^(t) + s_R,k^(t) + s_S,k^(t)) / 3
        The thesis specifies g(.) semantically as a bounded E/R/S scoring
        function, but does not state a unique closed-form expression. The
        arithmetic mean above is therefore documented as the implementation
        convention, not as a verbatim additional thesis equation.

    (4) Five-criterion preliminary weight:
            w_tilde_k^(t) = 0.22*s_E + 0.38*s_R + 0.20*s_A
                            + 0.15*s_S + 0.05*s_rep

    (5) Exponential smoothing and renormalization:
            w_k^(t) = [eta*w_tilde_k^(t) + (1-eta)*w_k^(t-1)]
                      / sum_j[eta*w_tilde_j^(t) + (1-eta)*w_j^(t-1)]
        with eta = 0.7 and w_k^(0) = 1/K.

    (6) On-chain scale:
        rho_k^(0) = 1/K in the normalized thesis model. The contract uses a
        0..1000 integer representation, hence initial reputation = 1000/K.
        The beta = 0.85 recurrence is executed on-chain on the bounded target.

    (7) Gradual isolation / Time-to-Isolation:
            theta_iso = 1/(2K)
            TTI_k = min { t : w_k^(t) < theta_iso }
        Crossing theta_iso records TTI only. It does not deactivate or exclude
        the client; isolation is gradual influence control.

Important experimental hygiene:
    * A blockchain-enabled seed should use a fresh Ganache/contract state if the
      run is intended to be statistically independent. Reusing a contract carries
      reputation history across seeds.
    * A nominal "33%" attack is represented by an integer attacker count. For K=5,
      33% therefore means 2/5 = 40% actual participants, matching the thesis setup.
    * This file does not fabricate baseline results or hard-code reported numbers.
      Baselines should come from their own explicit implementations.
    * `--offline` is a smoke/debug mode only; it is NOT a blockchain-equivalent
      BC-AWFedAvg experiment because persistent on-chain reputation is disabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Thesis configuration
# ---------------------------------------------------------------------------

SEEDS: List[int] = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
DEFAULT_K = 5
DEFAULT_T = 15
DEFAULT_EPSILON = 1.0
DEFAULT_DELTA = 1e-5
DEFAULT_CLIP_NORM = 1.0
DEFAULT_REPUTATION_BETA = 0.85
DEFAULT_SMOOTHING_ETA = 0.7
DEFAULT_ISOLATION_FACTOR = 0.5

ALPHAS = {
    "alpha_embb": 0.22,
    "alpha_urllc": 0.38,
    "alpha_activation": 0.20,
    "alpha_stability": 0.15,
    "alpha_reputation": 0.05,
}

THESIS_REPUTATION_SPEC: Dict[str, Any] = {
    "equation_reputation_score": "s_rep,k^(t) = rho_k^(t) / sum_j rho_j^(t)",
    "equation_initial_weight": "w_k^(0) = 1/K",
    "equation_reputation_normalization": "normalized rho_k^(0) = 1/K; on-chain rho is scaled by 1000",
    "equation_reputation_update": "rho_k^(t) = beta*rho_k^(t-1) + (1-beta)*g(E_k^(t),R_k^(t),S_k^(t))",
    "equation_operational_g": "g_k^(t) = (s_E,k^(t) + s_R,k^(t) + s_S,k^(t))/3",
    "equation_preliminary_weight": "w_tilde_k^(t) = 0.22*s_E + 0.38*s_R + 0.20*s_A + 0.15*s_S + 0.05*s_rep",
    "equation_smoothed_weight": "w_k^(t) = [eta*w_tilde_k^(t) + (1-eta)*w_k^(t-1)] / sum_j[eta*w_tilde_j^(t) + (1-eta)*w_j^(t-1)]",
    "equation_isolation_threshold": "theta_iso = 1/(2K)",
    "equation_tti": "TTI_k = min {t : w_k^(t) < theta_iso}",
    "beta": DEFAULT_REPUTATION_BETA,
    "eta": DEFAULT_SMOOTHING_ETA,
    "isolation_factor": DEFAULT_ISOLATION_FACTOR,
    "normalized_initial_reputation": "rho_k^(0) = 1/K",
    "on_chain_reputation_scale": 1000,
    "on_chain_initial_reputation": "1000/K",
    "reputation_target_range": [0, 1],
    "alpha": ALPHAS,
    "g_definition_status": "implementation convention; thesis does not specify a unique closed form for g(.)",
}

ATTACK_SCENARIOS: List[Dict[str, Any]] = [
    {"name": "No Attack", "attack_type": "none", "nominal_fraction": 0.0, "strength": 1.0},
    {"name": "Byzantine 20%", "attack_type": "byzantine", "nominal_fraction": 0.20, "strength": 1.0},
    {"name": "Byzantine 33%", "attack_type": "byzantine", "nominal_fraction": 1.0 / 3.0, "strength": 1.0},
    {"name": "Poisoning a=5", "attack_type": "poisoning", "nominal_fraction": 1.0, "count": 1, "strength": 5.0},
    {"name": "Poisoning a=10", "attack_type": "poisoning", "nominal_fraction": 1.0, "count": 1, "strength": 10.0},
    {"name": "Free-rider", "attack_type": "freerider", "nominal_fraction": 0.20, "strength": 1.0},
    {"name": "Collusion 33%", "attack_type": "collusion", "nominal_fraction": 1.0 / 3.0, "strength": 3.0},
    {"name": "Replay", "attack_type": "replay", "nominal_fraction": 0.20, "strength": 1.0},
    {"name": "Sybil", "attack_type": "sybil", "nominal_fraction": 0.20, "strength": 1.0},
]

PRIVACY_EPSILONS = [0.1, 0.5, 1.0, 5.0]
SCALABILITY_K = [3, 5, 10, 20, 50]

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dependency loading
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Seed Python/NumPy/PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_canonical_runner():
    """Import the single authoritative BC-AWFedAvg runner."""
    from blockchain_awfedavg import (
        create_blockchain_awfedavg_strategy,
        run_blockchain_awfedavg_experiment,
    )

    return create_blockchain_awfedavg_strategy, run_blockchain_awfedavg_experiment


def load_contract_info(path: Path) -> Dict[str, Any]:
    """Load deployment metadata produced by deploy.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Contract information not found: {path}. "
            "Run deploy.py first, or use --offline for a smoke test."
        )
    with path.open("r", encoding="utf-8") as fh:
        info = json.load(fh)

    required = ["contract_address", "clients"]
    missing = [key for key in required if not info.get(key)]
    if missing:
        raise ValueError(f"Invalid contract_info.json; missing: {missing}")

    # The contract's initial reputation is defined as 1000/K. Therefore a
    # blockchain experiment must use a deployment whose max-clients setting
    # matches the requested federation size K. A single deployment cannot be
    # silently reused across different K values without changing that scaling.
    constructor = info.get("constructor") or {}
    if "max_clients" in constructor:
        max_clients = int(constructor["max_clients"])
        if max_clients < 2:
            raise ValueError("contract_info.json has an invalid max_clients value.")

    return info


# ---------------------------------------------------------------------------
# Attack-count handling
# ---------------------------------------------------------------------------


def actual_attack_fraction(K: int, scenario: Mapping[str, Any]) -> Tuple[float, int]:
    """Return a realizable fraction and attacker count for a nominal scenario.

    The thesis uses integer attacker counts. In particular, a nominal one-third
    attack with K=5 is represented by two adversaries (2/5 = 40%).
    """
    if K < 1:
        raise ValueError("K must be positive")

    if "count" in scenario:
        n_attack = int(scenario["count"])
    else:
        nominal = float(scenario.get("nominal_fraction", 0.0))
        if nominal <= 0.0:
            n_attack = 0
        else:
            # Ceil converts a nominal fraction into the smallest integer attacker
            # count that meets/exceeds the requested fraction.
            n_attack = int(math.ceil(nominal * K - 1e-12))
            n_attack = max(1, n_attack)

    n_attack = min(n_attack, K)
    return n_attack / K, n_attack


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


def _fit_records(history: Any) -> List[Dict[str, Any]]:
    """Extract per-round fit metrics from the canonical history object."""
    if history is None:
        return []
    metrics = getattr(history, "metrics_distributed", {}) or {}
    records = metrics.get("fit", []) or []
    out: List[Dict[str, Any]] = []
    for rnd, payload in records:
        row = dict(payload or {})
        row["round"] = int(rnd)
        out.append(row)
    return out


def _mean_numeric(records: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def _last_numeric(records: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    for record in reversed(records):
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _extract_tti(strategy: Any) -> Dict[str, Optional[int]]:
    """Return first threshold-crossing round from the canonical strategy."""
    first = getattr(strategy, "first_isolation_round", {}) or {}
    return {str(int(cid)): int(rnd) for cid, rnd in first.items()}


def _extract_isolation_summary(strategy: Any) -> Dict[str, Any]:
    history = getattr(strategy, "isolation_history", []) or []
    first = _extract_tti(strategy)
    tti_values = [int(v) for v in first.values() if v is not None]
    return {
        "first_isolation_round": first,
        "mean_tti": float(np.mean(tti_values)) if tti_values else None,
        "max_tti": int(max(tti_values)) if tti_values else None,
        "isolation_history_rounds": len(history),
    }


@dataclass
class RunRecord:
    experiment: str
    seed: int
    num_clients: int
    num_rounds: int
    attack: str
    attack_fraction_actual: float
    attacker_count: int
    epsilon: float
    blockchain_enabled: bool
    secure_aggregation: bool
    reward_mean: Optional[float]
    reward_final: Optional[float]
    embb_outage_mean: Optional[float]
    urllc_residual_mean: Optional[float]
    blockchain_overhead_mean_s: Optional[float]
    ipfs_upload_mean_kb: Optional[float]
    epsilon_total: Optional[float]
    reputation_beta: float
    smoothing_eta: float
    isolation_threshold: float
    reputation_scale: float
    wall_clock_s: float
    mean_tti: Optional[float]
    max_tti: Optional[int]
    first_isolation_round: Dict[str, int]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# One canonical run
# ---------------------------------------------------------------------------


def run_one(
    *,
    experiment: str,
    seed: int,
    K: int,
    T: int,
    epsilon: float,
    contract_info: Optional[Mapping[str, Any]],
    attack_type: str = "none",
    attack_fraction: float = 0.0,
    attack_strength: float = 1.0,
    blockchain_enabled: bool = True,
) -> RunRecord:
    """Execute exactly one canonical BC-AWFedAvg run."""
    set_global_seed(seed)
    started = time.time()

    create_strategy, run_experiment = load_canonical_runner()

    if blockchain_enabled:
        if contract_info is None:
            raise ValueError("Blockchain-enabled execution requires contract_info.json")
        clients = list(contract_info.get("clients", []))
        if len(clients) < K:
            raise ValueError(
                f"contract_info.json exposes {len(clients)} clients, but K={K} is requested."
            )
        constructor = dict(contract_info.get("constructor") or {})
        max_clients = constructor.get("max_clients")
        min_clients = constructor.get("min_clients")
        if max_clients is not None and int(max_clients) != int(K):
            raise ValueError(
                f"This deployment was configured for max_clients={int(max_clients)}, "
                f"but the thesis protocol requires K={K} so initial reputation remains 1000/K. "
                "Deploy a separate contract for this K."
            )
        if min_clients is not None and int(min_clients) > int(K):
            raise ValueError(
                f"This deployment requires min_clients={int(min_clients)}, but K={K} is requested."
            )
        client_addresses = clients[:K]
        contract_address = str(contract_info["contract_address"])
        coordinator_private_key = contract_info.get("coordinator_private_key")
        contract_abi_path = "contract_info.json"
        provider = str(contract_info.get("rpc_url") or os.getenv("GANACHE_RPC", "http://127.0.0.1:8545"))
    else:
        client_addresses = None
        contract_address = None
        coordinator_private_key = None
        contract_abi_path = None
        provider = "http://127.0.0.1:8545"

    strategy = create_strategy(
        blockchain_provider=provider,
        contract_address=contract_address,
        contract_abi_path=contract_abi_path,
        coordinator_private_key=coordinator_private_key,
        epsilon=epsilon,
        delta=DEFAULT_DELTA,
        clip_norm=DEFAULT_CLIP_NORM,
        apply_coordinator_dp=True,
        compression=True,
        blockchain_enabled=blockchain_enabled,
        alpha_embb=ALPHAS["alpha_embb"],
        alpha_urllc=ALPHAS["alpha_urllc"],
        alpha_activation=ALPHAS["alpha_activation"],
        alpha_stability=ALPHAS["alpha_stability"],
        alpha_reputation=ALPHAS["alpha_reputation"],
        reputation_beta=DEFAULT_REPUTATION_BETA,
        reputation_scale=1000.0,
        isolation_threshold_factor=DEFAULT_ISOLATION_FACTOR,
        min_fit_clients=K,
        min_evaluate_clients=K,
        min_available_clients=K,
    )

    try:
        history, _, _, _ = run_experiment(
            strategy=strategy,
            num_rounds=T,
            num_clients=K,
            client_addresses=client_addresses,
            secure_aggregation=True,
            attack_type=attack_type,
            attack_fraction=attack_fraction,
            attack_strength=attack_strength,
        )

        records = _fit_records(history)
        iso = _extract_isolation_summary(strategy)
        epsilon_total = getattr(strategy, "final_privacy_epsilon", None)
        if epsilon_total is None:
            try:
                clients_obj = history  # keep extraction side-effect free
                del clients_obj
                # The canonical strategy does not expose a federation-wide epsilon
                # scalar in all versions; prefer the last fit record when present.
                epsilon_total = _last_numeric(records, "client_dp_accountant_epsilon")
            except Exception:
                epsilon_total = None

        return RunRecord(
            experiment=experiment,
            seed=seed,
            num_clients=K,
            num_rounds=T,
            attack=attack_type,
            attack_fraction_actual=float(attack_fraction),
            attacker_count=int(round(float(attack_fraction) * K)),
            epsilon=float(epsilon),
            blockchain_enabled=bool(blockchain_enabled),
            secure_aggregation=True,
            reward_mean=_mean_numeric(records, "average_reward"),
            reward_final=_last_numeric(records, "average_reward"),
            embb_outage_mean=_mean_numeric(records, "avg_embb_outage_counter"),
            urllc_residual_mean=_mean_numeric(records, "avg_residual_urllc_pkt"),
            blockchain_overhead_mean_s=_mean_numeric(records, "blockchain_overhead_s"),
            ipfs_upload_mean_kb=_mean_numeric(records, "ipfs_upload_size_kb"),
            epsilon_total=float(epsilon_total) if isinstance(epsilon_total, (int, float)) else None,
            reputation_beta=float(DEFAULT_REPUTATION_BETA),
            smoothing_eta=float(DEFAULT_SMOOTHING_ETA),
            isolation_threshold=float(DEFAULT_ISOLATION_FACTOR / max(K, 1)),
            reputation_scale=1000.0,
            wall_clock_s=time.time() - started,
            mean_tti=iso["mean_tti"],
            max_tti=iso["max_tti"],
            first_isolation_round=iso["first_isolation_round"],
        )

    except Exception as exc:
        return RunRecord(
            experiment=experiment,
            seed=seed,
            num_clients=K,
            num_rounds=T,
            attack=attack_type,
            attack_fraction_actual=float(attack_fraction),
            attacker_count=int(round(float(attack_fraction) * K)),
            epsilon=float(epsilon),
            blockchain_enabled=bool(blockchain_enabled),
            secure_aggregation=True,
            reward_mean=None,
            reward_final=None,
            embb_outage_mean=None,
            urllc_residual_mean=None,
            blockchain_overhead_mean_s=None,
            ipfs_upload_mean_kb=None,
            epsilon_total=None,
            reputation_beta=float(DEFAULT_REPUTATION_BETA),
            smoothing_eta=float(DEFAULT_SMOOTHING_ETA),
            isolation_threshold=float(DEFAULT_ISOLATION_FACTOR / max(K, 1)),
            reputation_scale=1000.0,
            wall_clock_s=time.time() - started,
            mean_tti=None,
            max_tti=None,
            first_isolation_round={},
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Multi-seed execution
# ---------------------------------------------------------------------------


def run_multi_seed(
    *,
    experiment: str,
    K: int,
    T: int,
    seeds: Sequence[int],
    epsilon: float,
    contract_info: Optional[Mapping[str, Any]],
    attack_type: str = "none",
    attack_fraction: float = 0.0,
    attack_strength: float = 1.0,
    blockchain_enabled: bool = True,
) -> List[RunRecord]:
    records: List[RunRecord] = []

    for seed in seeds:
        print(
            f"\n▶ {experiment} | seed={seed} | K={K} | T={T} | "
            f"attack={attack_type} | ε={epsilon}"
        )
        result = run_one(
            experiment=experiment,
            seed=seed,
            K=K,
            T=T,
            epsilon=epsilon,
            contract_info=contract_info,
            attack_type=attack_type,
            attack_fraction=attack_fraction,
            attack_strength=attack_strength,
            blockchain_enabled=blockchain_enabled,
        )
        records.append(result)

        if result.error:
            print(f"  ❌ {result.error}")
        else:
            print(
                f"  reward={result.reward_final if result.reward_final is not None else float('nan'):.4f} "
                f"TTI={result.mean_tti if result.mean_tti is not None else '—'}"
            )

    return records


# ---------------------------------------------------------------------------
# Experiment groups
# ---------------------------------------------------------------------------


def run_clean(
    *, K: int, T: int, seeds: Sequence[int], contract_info: Optional[Mapping[str, Any]],
    blockchain_enabled: bool,
) -> List[RunRecord]:
    return run_multi_seed(
        experiment="clean_full_system",
        K=K,
        T=T,
        seeds=seeds,
        epsilon=DEFAULT_EPSILON,
        contract_info=contract_info,
        blockchain_enabled=blockchain_enabled,
    )


def run_attacks(
    *, K: int, T: int, seeds: Sequence[int], contract_info: Optional[Mapping[str, Any]],
    blockchain_enabled: bool,
) -> List[RunRecord]:
    rows: List[RunRecord] = []
    for scenario in ATTACK_SCENARIOS:
        fraction, count = actual_attack_fraction(K, scenario)
        print(
            f"\n{'=' * 72}\n"
            f"{scenario['name']} | nominal={scenario.get('nominal_fraction', 0):.3f} "
            f"actual={count}/{K}={fraction:.3f}\n"
            f"{'=' * 72}"
        )
        rows.extend(
            run_multi_seed(
                experiment=scenario["name"],
                K=K,
                T=T,
                seeds=seeds,
                epsilon=DEFAULT_EPSILON,
                contract_info=contract_info,
                attack_type=scenario["attack_type"],
                attack_fraction=fraction,
                attack_strength=float(scenario["strength"]),
                blockchain_enabled=blockchain_enabled,
            )
        )
    return rows


def run_privacy(
    *, K: int, T: int, seeds: Sequence[int], contract_info: Optional[Mapping[str, Any]],
    blockchain_enabled: bool,
) -> List[RunRecord]:
    rows: List[RunRecord] = []
    for epsilon in PRIVACY_EPSILONS:
        rows.extend(
            run_multi_seed(
                experiment=f"privacy_epsilon_{epsilon}",
                K=K,
                T=T,
                seeds=seeds,
                epsilon=epsilon,
                contract_info=contract_info,
                blockchain_enabled=blockchain_enabled,
            )
        )
    return rows


def run_scalability(
    *, T: int, seeds: Sequence[int], contract_info: Optional[Mapping[str, Any]],
    blockchain_enabled: bool,
) -> List[RunRecord]:
    rows: List[RunRecord] = []
    for K in SCALABILITY_K:
        rows.extend(
            run_multi_seed(
                experiment=f"scalability_K_{K}",
                K=K,
                T=T,
                seeds=seeds,
                epsilon=DEFAULT_EPSILON,
                contract_info=contract_info,
                blockchain_enabled=blockchain_enabled,
            )
        )
    return rows


def validate_thesis_reputation_configuration(K: int) -> None:
    """Fail fast if runtime reputation parameters diverge from the thesis."""
    if not math.isclose(sum(ALPHAS.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The five aggregation coefficients must sum to 1.")
    if DEFAULT_REPUTATION_BETA != 0.85:
        raise ValueError("Thesis reputation decay beta must be 0.85.")
    if DEFAULT_SMOOTHING_ETA != 0.7:
        raise ValueError("Thesis weight smoothing eta must be 0.7.")
    expected = 1.0 / (2.0 * K)
    actual = DEFAULT_ISOLATION_FACTOR / max(K, 1)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Isolation threshold must equal 1/(2K).")


def save_thesis_protocol_spec(path: Path, K: int, T: int) -> None:
    """Write equations and runtime parameters used by the experiment."""
    validate_thesis_reputation_configuration(K)
    spec = dict(THESIS_REPUTATION_SPEC)
    spec.update({
        "K": int(K),
        "T": int(T),
        "epsilon": float(DEFAULT_EPSILON),
        "delta": float(DEFAULT_DELTA),
        "clip_norm_C": float(DEFAULT_CLIP_NORM),
        "seeds": list(SEEDS),
        "isolation_threshold_value": float(DEFAULT_ISOLATION_FACTOR / max(K, 1)),
        "five_criterion_alphas": dict(ALPHAS),
    })
    with path.open("w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)


# ---------------------------------------------------------------------------
# Statistics / persistence
# ---------------------------------------------------------------------------


def aggregate_records(records: Sequence[RunRecord]) -> List[Dict[str, Any]]:
    """Aggregate independent seed runs by experiment."""
    groups: Dict[str, List[RunRecord]] = {}
    for record in records:
        groups.setdefault(record.experiment, []).append(record)

    out: List[Dict[str, Any]] = []
    for name, group in groups.items():
        valid = [r for r in group if r.error is None and r.reward_final is not None]
        rewards = [float(r.reward_final) for r in valid if r.reward_final is not None]
        tti = [float(r.mean_tti) for r in valid if r.mean_tti is not None]

        row = {
            "experiment": name,
            "num_clients": group[0].num_clients,
            "num_rounds": group[0].num_rounds,
            "n_runs": len(group),
            "n_success": len(valid),
            "n_errors": len(group) - len(valid),
            "reward_mean": float(np.mean(rewards)) if rewards else None,
            "reward_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0 if rewards else None,
            "reward_ci95": (
                1.96 * float(np.std(rewards, ddof=1)) / math.sqrt(len(rewards))
                if len(rewards) > 1 else 0.0 if rewards else None
            ),
            "tti_mean": float(np.mean(tti)) if tti else None,
            "tti_std": float(np.std(tti, ddof=1)) if len(tti) > 1 else 0.0 if tti else None,
            "embb_outage_mean": _mean_of_values(group, "embb_outage_mean"),
            "urllc_residual_mean": _mean_of_values(group, "urllc_residual_mean"),
            "blockchain_overhead_mean_s": _mean_of_values(group, "blockchain_overhead_mean_s"),
            "ipfs_upload_mean_kb": _mean_of_values(group, "ipfs_upload_mean_kb"),
            "epsilon_total_mean": _mean_of_values(group, "epsilon_total"),
        }
        out.append(row)

    return sorted(out, key=lambda row: row["experiment"])


def _mean_of_values(records: Sequence[RunRecord], field_name: str) -> Optional[float]:
    vals: List[float] = []
    for record in records:
        value = getattr(record, field_name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            vals.append(float(value))
    return float(np.mean(vals)) if vals else None


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def save_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[Mapping[str, Any]], title: str) -> None:
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")
    for row in rows:
        reward = row.get("reward_mean")
        ci = row.get("reward_ci95")
        tti = row.get("tti_mean")
        status = f"{row['n_success']}/{row['n_runs']}"
        print(
            f"{row['experiment']:<30} "
            f"K={row['num_clients']:<2} T={row['num_rounds']:<2} "
            f"n={status:<5} "
            f"reward={reward if reward is not None else float('nan'):.4f} "
            f"CI={ci if ci is not None else float('nan'):.4f} "
            f"TTI={tti if tti is not None else '—'}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical BC-AWFedAvg experiment driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["clean", "attacks", "privacy", "scalability", "all"],
        default="all",
        help="Experiment group to run.",
    )
    parser.add_argument("--clients", type=int, default=DEFAULT_K, help="Number of MVNO clients for clean/attack/privacy runs.")
    parser.add_argument("--rounds", type=int, default=DEFAULT_T, help="Federated rounds.")
    parser.add_argument("--seed", type=int, default=None, help="Run a single seed instead of the ten thesis seeds.")
    parser.add_argument("--contract-info", type=Path, default=Path("contract_info.json"), help="Deployment metadata generated by deploy.py.")
    parser.add_argument("--offline", action="store_true", help="Disable blockchain/IPFS. Smoke/debug only; not thesis-equivalent.")
    parser.add_argument("--fast", action="store_true", help="Smoke test: one round and first two seeds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.clients < 2:
        raise SystemExit("K must be at least 2 for the secure-aggregation protocol.")
    if args.rounds < 1:
        raise SystemExit("T must be positive.")

    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    rounds = 1 if args.fast else int(args.rounds)
    if args.fast and args.seed is None:
        seeds = seeds[:2]

    blockchain_enabled = not args.offline
    contract_info = None if args.offline else load_contract_info(args.contract_info)
    validate_thesis_reputation_configuration(args.clients)

    print("\n" + "=" * 96)
    print("BC-AWFedAvg — CANONICAL EXPERIMENT DRIVER")
    print("=" * 96)
    print(f"K={args.clients} | T={rounds} | seeds={seeds}")
    print(f"alpha=(0.22, 0.38, 0.20, 0.15, 0.05)")
    print(f"epsilon={DEFAULT_EPSILON} | delta={DEFAULT_DELTA} | clip={DEFAULT_CLIP_NORM}")
    print(f"reputation_beta={DEFAULT_REPUTATION_BETA} | eta={DEFAULT_SMOOTHING_ETA}")
    print(f"isolation_threshold=1/(2K) = {DEFAULT_ISOLATION_FACTOR / args.clients:.4f}")
    print(f"blockchain={'enabled' if blockchain_enabled else 'DISABLED (offline smoke only)'}")
    print(f"results={RESULTS_DIR.resolve()}")

    started = time.time()
    all_records: List[RunRecord] = []

    if args.mode in ("clean", "all"):
        all_records.extend(
            run_clean(
                K=args.clients,
                T=rounds,
                seeds=seeds,
                contract_info=contract_info,
                blockchain_enabled=blockchain_enabled,
            )
        )

    if args.mode in ("attacks", "all"):
        all_records.extend(
            run_attacks(
                K=args.clients,
                T=rounds,
                seeds=seeds,
                contract_info=contract_info,
                blockchain_enabled=blockchain_enabled,
            )
        )

    if args.mode in ("privacy", "all"):
        all_records.extend(
            run_privacy(
                K=args.clients,
                T=rounds,
                seeds=seeds,
                contract_info=contract_info,
                blockchain_enabled=blockchain_enabled,
            )
        )

    if args.mode in ("scalability", "all"):
        # K is intentionally fixed by the thesis scalability design, not by
        # --clients. A deployment must expose enough blockchain accounts.
        all_records.extend(
            run_scalability(
                T=rounds,
                seeds=seeds,
                contract_info=contract_info,
                blockchain_enabled=blockchain_enabled,
            )
        )

    raw_rows = [asdict(record) for record in all_records]
    aggregate_rows = aggregate_records(all_records)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_thesis_protocol_spec(RESULTS_DIR / "thesis_protocol_spec.json", args.clients, rounds)
    save_json(raw_rows, RESULTS_DIR / f"canonical_runs_{timestamp}.json")
    save_csv(raw_rows, RESULTS_DIR / f"canonical_runs_{timestamp}.csv")
    save_json(aggregate_rows, RESULTS_DIR / f"canonical_summary_{timestamp}.json")
    save_csv(aggregate_rows, RESULTS_DIR / f"canonical_summary_{timestamp}.csv")

    print_summary(aggregate_rows, "BC-AWFedAvg canonical summary")
    print(f"\n✅ Finished in {time.time() - started:.1f}s")
    print("Note: independent blockchain-enabled seeds should use isolated Ganache/contract state.")
    print("Note: baseline comparison is intentionally not fabricated by this driver.")


if __name__ == "__main__":
    main()
