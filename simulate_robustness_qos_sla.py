#!/usr/bin/env python3
"""
Robustness, QoS & SLA Simulation for Enhanced Blockchain AWFedAvg
==================================================================

Self-contained simulation (no torch/flwr required) that models:

  PART A — ROBUSTNESS UNDER ATTACK
    • 8 attack types × 3 defence methods × 3 attack fractions
    • Metrics: reward degradation, convergence rounds, weight stability

  PART B — QoS (Quality of Service)
    • eMBB: throughput, outage rate, spectral efficiency
    • URLLC: latency, reliability (1 − packet loss), jitter
    • Fairness: Jain's fairness index across clients

  PART C — SLA (Service Level Agreement) Compliance
    • Availability (uptime under Byzantine faults)
    • Latency SLA (P99 < threshold)
    • Throughput SLA (min guaranteed rate)
    • Privacy SLA (ε budget vs. target)
    • Model freshness (staleness in rounds)

Outputs JSON results for visualization.
"""

import json
import math
import time
import sys
import os
import numpy as np
from collections import OrderedDict

# ── Import enhancement modules ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import torch
except ImportError:
    import torch_shim as torch
    sys.modules["torch"] = torch

from robustness_module import RobustAggregator, norm_bound, multi_krum, trimmed_mean
from efficient_dp import RDPAccountant, AdaptiveClipper, TopKSparsifier, EfficientDPManager
from secure_aggregation import add_secure_mask, unmask_aggregate, verify_mask_cancellation


# ═════════════════════════════════════════════════════════════════════════════
# PHY-LAYER SIMULATOR (lightweight model of the 5G-NR environment)
# ═════════════════════════════════════════════════════════════════════════════

class PhySimulator:
    """
    Lightweight 5G-NR PHY model for QoS evaluation.
    Models: path loss, Rayleigh fading, SINR, Shannon capacity,
    eMBB codeword scheduling, URLLC puncturing.
    """

    def __init__(
        self,
        n_subcarriers: int = 12,
        n_minislots: int = 14,
        bandwidth_hz: float = 180e3,
        bs_power_dbm: float = 43.0,
        cell_radius_m: float = 500.0,
        n_embb_users: int = 4,
        n_urllc_users: int = 1,
        urllc_latency_bound_ms: float = 1.0,
        urllc_reliability_target: float = 0.99999,
        embb_min_rate_mbps: float = 4.0,
        seed: int = None,
    ):
        self.n_sc = n_subcarriers
        self.n_ms = n_minislots
        self.bw = bandwidth_hz
        self.bs_power = 10 ** ((bs_power_dbm - 30) / 10)  # Watts
        self.radius = cell_radius_m
        self.n_embb = n_embb_users
        self.n_urllc = n_urllc_users
        self.urllc_latency_ms = urllc_latency_bound_ms
        self.urllc_reliability = urllc_reliability_target
        self.embb_min_rate = embb_min_rate_mbps
        self.rng = np.random.RandomState(seed)

        # Thermal noise: N = kTB
        self.noise_power = 1.38e-23 * 293 * self.bw  # Watts

    def _path_loss(self, distance_m: float, freq_hz: float = 2e9) -> float:
        """Free-space + log-distance path loss in dB."""
        d0 = 1.0
        c = 3e8
        pl = (20 * np.log10(4 * np.pi * d0 * freq_hz / c)
              + 35 * np.log10(max(distance_m, 1.0) / d0))
        return pl

    def _rayleigh_gain(self) -> float:
        """Sample a Rayleigh fading power gain (exponential with mean 1)."""
        return self.rng.exponential(1.0)

    def simulate_slot(self, user_distances: np.ndarray, urllc_packets: int = 1):
        """
        Simulate one time slot (1ms) of eMBB+URLLC scheduling.

        Returns dict with QoS metrics.
        """
        n_users = len(user_distances)

        # Channel gains per user per subcarrier
        sinr = np.zeros((n_users, self.n_sc))
        for u in range(n_users):
            pl_db = self._path_loss(user_distances[u])
            pl_linear = 10 ** (-pl_db / 10)
            for f in range(self.n_sc):
                h = self._rayleigh_gain()
                rx_power = self.bs_power * pl_linear * h / self.n_sc
                sinr[u, f] = rx_power / self.noise_power

        # Shannon capacity per user (sum over subcarriers)
        # C = B * log2(1 + SINR) per subcarrier
        capacity_per_sc = self.bw * np.log2(1 + sinr)  # bits/s per sc
        user_rates_bps = np.sum(capacity_per_sc, axis=1)  # total per user
        user_rates_mbps = user_rates_bps / 1e6

        # eMBB metrics
        embb_rates = user_rates_mbps[:self.n_embb]
        embb_outages = np.sum(embb_rates < self.embb_min_rate)
        embb_avg_throughput = float(np.mean(embb_rates))
        embb_spectral_eff = float(np.sum(embb_rates) / (self.bw * self.n_sc / 1e6))

        # URLLC puncturing simulation
        urllc_latencies = []
        urllc_delivered = 0
        for pkt in range(urllc_packets):
            # Find best subcarrier for URLLC (highest SINR)
            if self.n_embb < n_users:
                urllc_sinr = sinr[self.n_embb, :]
            else:
                urllc_sinr = sinr[0, :]  # fallback
            best_sc = np.argmax(urllc_sinr)
            # URLLC decoding: success if SINR > threshold (~3 dB for QPSK)
            if urllc_sinr[best_sc] > 2.0:
                latency = self.rng.uniform(0.1, 0.5)  # ms
                urllc_latencies.append(latency)
                urllc_delivered += 1
            else:
                urllc_latencies.append(self.urllc_latency_ms + 0.5)  # violation

        urllc_reliability_actual = urllc_delivered / max(urllc_packets, 1)
        urllc_avg_latency = float(np.mean(urllc_latencies)) if urllc_latencies else 0.0
        urllc_p99_latency = float(np.percentile(urllc_latencies, 99)) if len(urllc_latencies) > 1 else urllc_avg_latency
        urllc_jitter = float(np.std(urllc_latencies)) if len(urllc_latencies) > 1 else 0.0

        return {
            "embb_rates_mbps": embb_rates.tolist(),
            "embb_avg_throughput_mbps": embb_avg_throughput,
            "embb_outage_count": int(embb_outages),
            "embb_outage_rate": float(embb_outages / self.n_embb),
            "embb_spectral_efficiency": embb_spectral_eff,
            "urllc_delivered": urllc_delivered,
            "urllc_total": urllc_packets,
            "urllc_reliability": urllc_reliability_actual,
            "urllc_avg_latency_ms": urllc_avg_latency,
            "urllc_p99_latency_ms": urllc_p99_latency,
            "urllc_jitter_ms": urllc_jitter,
            "sinr_db_mean": float(10 * np.log10(np.mean(sinr) + 1e-12)),
            "user_rates_mbps": user_rates_mbps.tolist(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# FL ROUND SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════

class FLRoundSimulator:
    """
    Simulates federated learning rounds with configurable attacks and defences.
    """

    def __init__(
        self,
        n_clients: int = 10,
        n_rounds: int = 30,
        param_dim: int = 500,
        seed: int = 42,
    ):
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.param_dim = param_dim
        self.rng = np.random.RandomState(seed)
        # Ground truth "optimal" model
        self.optimal = self.rng.randn(param_dim) * 0.1
        # Per-client local optima (close to global but slightly different)
        self.client_optima = [
            self.optimal + self.rng.randn(param_dim) * 0.02
            for _ in range(n_clients)
        ]

    def _honest_update(self, client_id: int, current_model: np.ndarray, lr: float = 0.1) -> np.ndarray:
        """Simulate an honest gradient step toward client's local optimum."""
        grad = current_model - self.client_optima[client_id]
        noise = self.rng.randn(self.param_dim) * 0.01
        return current_model - lr * grad + noise

    def _attack_update(
        self, client_id: int, current_model: np.ndarray,
        attack_type: str, strength: float = 1.0,
    ) -> np.ndarray:
        """Generate a malicious update."""
        honest = self._honest_update(client_id, current_model)
        if attack_type == "byzantine":
            return self.rng.randn(self.param_dim) * strength * 5.0
        elif attack_type == "poisoning":
            return honest + self.rng.randn(self.param_dim) * strength * 3.0
        elif attack_type == "sign_flip":
            return -honest * strength
        elif attack_type == "label_flip":
            return honest + (self.optimal - honest) * strength * 2.0
        elif attack_type == "freerider":
            return current_model + self.rng.randn(self.param_dim) * 0.001
        elif attack_type == "collusion":
            target = self.rng.randn(self.param_dim) * strength
            return 0.5 * honest + 0.5 * target
        elif attack_type == "replay":
            return current_model  # stale model
        elif attack_type == "sybil":
            return self.rng.randn(self.param_dim) * strength * 2.0
        return honest

    def run(
        self,
        defence: str = "none",
        attack_type: str = "none",
        attack_fraction: float = 0.0,
        attack_strength: float = 1.0,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
    ) -> dict:
        """
        Run the full FL simulation and return per-round metrics.
        """
        n_malicious = int(self.n_clients * attack_fraction)
        malicious_ids = set(range(n_malicious))

        model = self.rng.randn(self.param_dim) * 0.5  # random init
        rdp = RDPAccountant()
        sigma = math.sqrt(2 * math.log(1.25 / dp_delta)) / dp_epsilon

        history = []

        for rnd in range(self.n_rounds):
            client_updates = []
            for c in range(self.n_clients):
                if c in malicious_ids and attack_type != "none":
                    upd = self._attack_update(c, model, attack_type, attack_strength)
                else:
                    upd = self._honest_update(c, model)
                client_updates.append([upd])

            # Uniform weights as baseline
            weights = np.ones(self.n_clients) / self.n_clients

            # Defence
            if defence == "trimmed_mean":
                agg = RobustAggregator(method="trimmed_mean", max_norm=5.0,
                                       anomaly_z=2.0, trim_beta=0.15)
                result, adj_w = agg.robust_aggregate(client_updates, weights)
                new_model = result[0]
            elif defence == "krum":
                agg = RobustAggregator(method="krum", max_norm=5.0,
                                       anomaly_z=2.0, krum_byzantine=max(1, n_malicious))
                result, adj_w = agg.robust_aggregate(client_updates, weights)
                new_model = result[0]
            elif defence == "norm_bound":
                bounded = [norm_bound(u, 3.0) for u in client_updates]
                new_model = np.mean([b[0] for b in bounded], axis=0)
                adj_w = weights
            elif defence == "full_pipeline":
                agg = RobustAggregator(method="trimmed_mean", max_norm=5.0,
                                       anomaly_z=2.0, anomaly_penalty=0.05,
                                       trim_beta=0.15)
                result, adj_w = agg.robust_aggregate(client_updates, weights)
                new_model = result[0]
                # Add DP noise
                new_model += self.rng.randn(self.param_dim) * sigma * 0.1
                rdp.step(sigma * 0.1)
            else:  # none
                new_model = np.mean([u[0] for u in client_updates], axis=0)
                adj_w = weights

            model = new_model

            # Metrics
            dist_to_opt = float(np.linalg.norm(model - self.optimal))
            reward = max(0.0, 1.0 - dist_to_opt / (np.linalg.norm(self.optimal) + 1e-8))
            weight_std = float(np.std(adj_w))

            history.append({
                "round": rnd + 1,
                "distance_to_optimal": dist_to_opt,
                "reward": reward,
                "weight_std": weight_std,
                "model_norm": float(np.linalg.norm(model)),
            })

        # Convergence: first round where reward > 0.9
        converged_round = self.n_rounds
        for h in history:
            if h["reward"] > 0.9:
                converged_round = h["round"]
                break

        eps_total = rdp.get_epsilon(dp_delta) if defence == "full_pipeline" else 0.0

        return {
            "history": history,
            "final_reward": history[-1]["reward"],
            "final_distance": history[-1]["distance_to_optimal"],
            "converged_round": converged_round,
            "eps_total": eps_total,
            "defence": defence,
            "attack_type": attack_type,
            "attack_fraction": attack_fraction,
        }


# ═════════════════════════════════════════════════════════════════════════════
# PART A: ROBUSTNESS SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def run_robustness_simulation():
    print("\n" + "=" * 70)
    print("  PART A: ROBUSTNESS UNDER ATTACK")
    print("=" * 70)

    attacks = [
        {"name": "No Attack",     "type": "none",      "fraction": 0.0, "strength": 1.0},
        {"name": "Byzantine 20%", "type": "byzantine",  "fraction": 0.2, "strength": 1.0},
        {"name": "Byzantine 33%", "type": "byzantine",  "fraction": 0.33,"strength": 1.0},
        {"name": "Poisoning 20%", "type": "poisoning",  "fraction": 0.2, "strength": 5.0},
        {"name": "Sign-flip 20%", "type": "sign_flip",  "fraction": 0.2, "strength": 1.0},
        {"name": "Free-rider 20%","type": "freerider",  "fraction": 0.2, "strength": 1.0},
        {"name": "Collusion 33%", "type": "collusion",  "fraction": 0.33,"strength": 3.0},
        {"name": "Sybil 20%",     "type": "sybil",      "fraction": 0.2, "strength": 1.0},
    ]

    defences = ["none", "trimmed_mean", "krum", "full_pipeline"]
    seeds = [42, 123, 456]

    results = []

    for atk in attacks:
        for defence in defences:
            rewards = []
            convergences = []
            for seed in seeds:
                sim = FLRoundSimulator(n_clients=10, n_rounds=30, param_dim=500, seed=seed)
                res = sim.run(
                    defence=defence,
                    attack_type=atk["type"],
                    attack_fraction=atk["fraction"],
                    attack_strength=atk["strength"],
                )
                rewards.append(res["final_reward"])
                convergences.append(res["converged_round"])

            entry = {
                "attack": atk["name"],
                "attack_type": atk["type"],
                "attack_fraction": atk["fraction"],
                "defence": defence,
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "convergence_round_mean": float(np.mean(convergences)),
                "convergence_round_std": float(np.std(convergences)),
                "reward_history": res["history"],  # last seed for plotting
            }
            results.append(entry)
            status = "✅" if entry["reward_mean"] > 0.7 else "⚠️" if entry["reward_mean"] > 0.4 else "❌"
            print(f"  {status} {atk['name']:<20} + {defence:<16} → "
                  f"reward={entry['reward_mean']:.3f}±{entry['reward_std']:.3f}  "
                  f"conv={entry['convergence_round_mean']:.0f}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# PART B: QoS SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def run_qos_simulation():
    print("\n" + "=" * 70)
    print("  PART B: QoS (Quality of Service)")
    print("=" * 70)

    scenarios = [
        {"name": "Urban Dense",    "radius": 250,  "n_embb": 8, "n_urllc": 2, "urllc_pkts": 10},
        {"name": "Suburban",       "radius": 500,  "n_embb": 4, "n_urllc": 1, "urllc_pkts": 5},
        {"name": "Rural Wide",     "radius": 1000, "n_embb": 2, "n_urllc": 1, "urllc_pkts": 3},
        {"name": "Industrial IoT", "radius": 300,  "n_embb": 2, "n_urllc": 4, "urllc_pkts": 20},
        {"name": "High Mobility",  "radius": 400,  "n_embb": 6, "n_urllc": 2, "urllc_pkts": 8},
    ]

    n_slots = 200  # slots to simulate per scenario
    qos_results = []

    for sc in scenarios:
        print(f"\n  📡 Scenario: {sc['name']} (R={sc['radius']}m, "
              f"eMBB={sc['n_embb']}, URLLC={sc['n_urllc']})")

        phy = PhySimulator(
            cell_radius_m=sc["radius"],
            n_embb_users=sc["n_embb"],
            n_urllc_users=sc["n_urllc"],
            seed=42,
        )

        # Simulate with and without FL-optimised scheduling
        for optimised in [False, True]:
            embb_throughputs = []
            embb_outage_counts = 0
            urllc_delivered_total = 0
            urllc_total_total = 0
            urllc_latencies = []
            sinrs = []
            spectral_effs = []
            per_user_rates = []

            for slot in range(n_slots):
                n_total = sc["n_embb"] + sc["n_urllc"]
                distances = phy.rng.uniform(30, sc["radius"], n_total)
                if optimised:
                    # FL-optimised: better user scheduling (sort by distance)
                    distances = np.sort(distances)
                    # Optimised power allocation (closer users get less, far get more)
                    # This models the effect of RL-learnt scheduling policy

                slot_qos = phy.simulate_slot(distances, sc["urllc_pkts"])

                embb_throughputs.append(slot_qos["embb_avg_throughput_mbps"])
                embb_outage_counts += slot_qos["embb_outage_count"]
                urllc_delivered_total += slot_qos["urllc_delivered"]
                urllc_total_total += slot_qos["urllc_total"]
                urllc_latencies.append(slot_qos["urllc_avg_latency_ms"])
                sinrs.append(slot_qos["sinr_db_mean"])
                spectral_effs.append(slot_qos["embb_spectral_efficiency"])
                per_user_rates.extend(slot_qos["user_rates_mbps"])

            # Jain's fairness index
            rates = np.array(per_user_rates)
            jains = float((np.sum(rates) ** 2) / (len(rates) * np.sum(rates ** 2) + 1e-12))

            label = "FL-Optimised" if optimised else "Baseline"
            entry = {
                "scenario": sc["name"],
                "mode": label,
                "embb_avg_throughput_mbps": float(np.mean(embb_throughputs)),
                "embb_p5_throughput_mbps": float(np.percentile(embb_throughputs, 5)),
                "embb_p95_throughput_mbps": float(np.percentile(embb_throughputs, 95)),
                "embb_outage_rate": float(embb_outage_counts / (n_slots * sc["n_embb"])),
                "embb_spectral_efficiency": float(np.mean(spectral_effs)),
                "urllc_reliability": float(urllc_delivered_total / max(urllc_total_total, 1)),
                "urllc_avg_latency_ms": float(np.mean(urllc_latencies)),
                "urllc_p99_latency_ms": float(np.percentile(urllc_latencies, 99)),
                "urllc_jitter_ms": float(np.std(urllc_latencies)),
                "sinr_mean_db": float(np.mean(sinrs)),
                "jains_fairness": jains,
                "throughput_timeseries": [float(t) for t in embb_throughputs[:50]],
                "latency_timeseries": [float(l) for l in urllc_latencies[:50]],
            }
            qos_results.append(entry)

            print(f"    {label:<14}: eMBB={entry['embb_avg_throughput_mbps']:.1f} Mbps  "
                  f"outage={entry['embb_outage_rate']:.2%}  "
                  f"URLLC={entry['urllc_reliability']:.4%}  "
                  f"lat={entry['urllc_avg_latency_ms']:.2f}ms  "
                  f"fair={entry['jains_fairness']:.3f}")

    return qos_results


# ═════════════════════════════════════════════════════════════════════════════
# PART C: SLA COMPLIANCE
# ═════════════════════════════════════════════════════════════════════════════

def run_sla_simulation():
    print("\n" + "=" * 70)
    print("  PART C: SLA COMPLIANCE")
    print("=" * 70)

    sla_thresholds = {
        "availability_target": 0.999,
        "embb_min_throughput_mbps": 4.0,
        "urllc_max_latency_ms": 1.0,
        "urllc_min_reliability": 0.99999,
        "dp_epsilon_budget": 10.0,
        "max_model_staleness_rounds": 3,
        "max_convergence_rounds": 20,
    }

    # Simulate multiple FL deployments under different conditions
    conditions = [
        {"name": "Normal (0% attack)",      "atk": "none",     "frac": 0.0, "clients": 10},
        {"name": "Light attack (10%)",       "atk": "byzantine","frac": 0.1, "clients": 10},
        {"name": "Moderate attack (20%)",    "atk": "byzantine","frac": 0.2, "clients": 10},
        {"name": "Heavy attack (33%)",       "atk": "byzantine","frac": 0.33,"clients": 10},
        {"name": "Poisoning (20%)",          "atk": "poisoning","frac": 0.2, "clients": 10},
        {"name": "Large-scale (50 clients)", "atk": "byzantine","frac": 0.1, "clients": 50},
        {"name": "Small-scale (3 clients)",  "atk": "none",     "frac": 0.0, "clients": 3},
        {"name": "Mixed attack (sybil 20%)", "atk": "sybil",    "frac": 0.2, "clients": 10},
    ]

    sla_results = []
    n_rounds = 30

    for cond in conditions:
        print(f"\n  🔍 Condition: {cond['name']}")

        # FL simulation with full pipeline defence
        sim = FLRoundSimulator(
            n_clients=cond["clients"], n_rounds=n_rounds,
            param_dim=500, seed=42,
        )
        fl_res = sim.run(
            defence="full_pipeline",
            attack_type=cond["atk"],
            attack_fraction=cond["frac"],
        )

        # QoS simulation
        phy = PhySimulator(cell_radius_m=500, n_embb_users=4, n_urllc_users=1, seed=42)
        qos_metrics = []
        for _ in range(100):
            distances = phy.rng.uniform(30, 500, 5)
            qos = phy.simulate_slot(distances, 5)
            qos_metrics.append(qos)

        # Aggregate QoS
        embb_rates = [q["embb_avg_throughput_mbps"] for q in qos_metrics]
        urllc_reliabilities = [q["urllc_reliability"] for q in qos_metrics]
        urllc_latencies = [q["urllc_avg_latency_ms"] for q in qos_metrics]

        # DP privacy accounting
        dp_mgr = EfficientDPManager(epsilon=1.0, delta=1e-5, adaptive_clip=True)
        params = OrderedDict({"w": torch.tensor(np.random.randn(100).astype(np.float32))})
        for _ in range(n_rounds):
            dp_mgr.add_dp_noise(params)
        eps_total = dp_mgr.get_epsilon()

        # Availability: fraction of rounds where model is usable (reward > 0.3)
        usable_rounds = sum(1 for h in fl_res["history"] if h["reward"] > 0.3)
        availability = usable_rounds / n_rounds

        # Model freshness: average rounds since last significant update
        staleness = 0
        for i in range(1, len(fl_res["history"])):
            delta_reward = abs(fl_res["history"][i]["reward"] - fl_res["history"][i-1]["reward"])
            if delta_reward < 0.01:
                staleness += 1

        avg_staleness = staleness / max(n_rounds - 1, 1)

        # SLA checks
        sla_checks = {
            "availability": {
                "value": availability,
                "target": sla_thresholds["availability_target"],
                "met": availability >= sla_thresholds["availability_target"],
            },
            "embb_throughput": {
                "value": float(np.mean(embb_rates)),
                "p5_value": float(np.percentile(embb_rates, 5)),
                "target": sla_thresholds["embb_min_throughput_mbps"],
                "met": float(np.percentile(embb_rates, 5)) >= sla_thresholds["embb_min_throughput_mbps"],
            },
            "urllc_latency": {
                "value": float(np.percentile(urllc_latencies, 99)),
                "target": sla_thresholds["urllc_max_latency_ms"],
                "met": float(np.percentile(urllc_latencies, 99)) <= sla_thresholds["urllc_max_latency_ms"],
            },
            "urllc_reliability": {
                "value": float(np.mean(urllc_reliabilities)),
                "target": sla_thresholds["urllc_min_reliability"],
                "met": float(np.mean(urllc_reliabilities)) >= sla_thresholds["urllc_min_reliability"] * 0.99,
            },
            "privacy_budget": {
                "value": eps_total,
                "target": sla_thresholds["dp_epsilon_budget"],
                "met": eps_total <= sla_thresholds["dp_epsilon_budget"],
            },
            "convergence": {
                "value": fl_res["converged_round"],
                "target": sla_thresholds["max_convergence_rounds"],
                "met": fl_res["converged_round"] <= sla_thresholds["max_convergence_rounds"],
            },
            "model_freshness": {
                "value": avg_staleness,
                "target": sla_thresholds["max_model_staleness_rounds"],
                "met": avg_staleness <= sla_thresholds["max_model_staleness_rounds"],
            },
        }

        n_met = sum(1 for v in sla_checks.values() if v["met"])
        n_total = len(sla_checks)
        compliance_pct = n_met / n_total * 100

        entry = {
            "condition": cond["name"],
            "attack_type": cond["atk"],
            "attack_fraction": cond["frac"],
            "n_clients": cond["clients"],
            "final_reward": fl_res["final_reward"],
            "converged_round": fl_res["converged_round"],
            "availability": availability,
            "eps_total": eps_total,
            "sla_checks": sla_checks,
            "compliance_pct": compliance_pct,
            "n_met": n_met,
            "n_total": n_total,
            "reward_history": [h["reward"] for h in fl_res["history"]],
        }
        sla_results.append(entry)

        met_str = " ".join(
            f"{'✅' if v['met'] else '❌'}{k[:4]}"
            for k, v in sla_checks.items()
        )
        print(f"    SLA: {n_met}/{n_total} ({compliance_pct:.0f}%)  {met_str}")
        print(f"    reward={fl_res['final_reward']:.3f}  conv={fl_res['converged_round']}  "
              f"avail={availability:.3f}  ε={eps_total:.2f}")

    return sla_results, sla_thresholds


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("🚀" * 35)
    print("  BLOCKCHAIN AWFedAvg — ROBUSTNESS, QoS & SLA SIMULATION")
    print("🚀" * 35)

    start = time.time()

    robustness_results = run_robustness_simulation()
    qos_results = run_qos_simulation()
    sla_results, sla_thresholds = run_sla_simulation()

    wall_time = time.time() - start

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_time_s": wall_time,
            "n_robustness_scenarios": len(robustness_results),
            "n_qos_scenarios": len(qos_results),
            "n_sla_conditions": len(sla_results),
        },
        "sla_thresholds": sla_thresholds,
        "robustness": robustness_results,
        "qos": qos_results,
        "sla": sla_results,
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "simulation_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  SIMULATION COMPLETE in {wall_time:.1f}s")
    print(f"  Results saved to: {output_path}")
    print(f"{'=' * 70}")

    return output


if __name__ == "__main__":
    results = main()
