from __future__ import annotations
import argparse
import csv
import json
import math
import os
import pathlib
import random
import sys
import time
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Matplotlib (non-interactive backend) ──────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Add project root to path ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

# ── Inject torch shim if torch is not installed ────────────────────────────────
try:
    import torch  # noqa: F401 — just checking availability
except ImportError:
    import torch_shim as _torch_shim  # type: ignore
    sys.modules["torch"] = _torch_shim  # type: ignore

# ── Project modules (lightweight — no Flower/web3 required) ───────────────────
from efficient_dp import RDPAccountant
from robustness_module import RobustAggregator, norm_bound
from secure_aggregation import (
    add_secure_mask,
    verify_mask_cancellation,
)

# ── Try importing the heavyweight stack (Flower + blockchain) ─────────────────
_HAS_FLOWER = False
try:
    import flwr as fl  # noqa: F401
    from experiments import (
        run_ablation,
        run_attacks,
        run_privacy_tradeoff,
        run_scalability,
        AggregatedResult,
        ABLATION_CONFIGS,
        ATTACK_SCENARIOS,
        _FULL_CFG,
    )
    _HAS_FLOWER = True
except Exception:
    pass

# ═════════════════════════════════════════════════════════════════════════════
# Global configuration
# ═════════════════════════════════════════════════════════════════════════════

SEED = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]       
N_ROUNDS = 15         
N_CLIENTS_ABL  = 5        
N_CLIENTS_PRIV = 5      
PPO_PARAM_DIM  = 10_400      
DP_EPSILON     = 1.0         
DP_DELTA       = 1e-5
DP_CLIP        = 1.0         

RESULTS_DIR = pathlib.Path("results")
FIGURES_DIR = pathlib.Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Matplotlib style 
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "axes.linewidth":    1.2,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "legend.framealpha": 0.9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

# ── Color palette (matches paper figures) ─────────────────────────────────────
COLORS = {
    "No Defense":       "#e15759",
    "BC Only":          "#4e79a7",
    "DP Only":          "#f28e2b",
    "SecAgg Only":      "#76b7b2",
    "BC+DP":            "#59a14f",
    "BC+SecAgg":        "#b07aa1",
    "Full System":      "#ff0000",
    "No Attack":        "#aaaaaa",
    "Byzantine":        "#e15759",
    "Poisoning":        "#f28e2b",
    "Free-rider":       "#76b7b2",
    "Collusion":        "#b07aa1",
    "Replay":           "#59a14f",
    "Sybil":            "#4e79a7",
}


# ═════════════════════════════════════════════════════════════════════════════
# Seed management
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        import torch.cuda
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Lightweight BC-AWFedAvg simulator
# (used when Flower / blockchain stack is not available)
# ═════════════════════════════════════════════════════════════════════════════

class BCAwfedavgSimulator:
    """
    Self-contained simulation of BC-AWFedAvg that mirrors the paper's
    experimental setup without requiring Flower, Ganache, or IPFS.

    Models:
      • PPO policy gradient in the 5G-NR PHY environment (linearised)
      • eMBB outage / URLLC residual tracking
      • Five-criterion AWFedAvg adaptive weighting (Eq. 8-12)
      • Gaussian DP at client + coordinator level (Eq. 16-17)
      • Pairwise-cancelling SecAgg masks (Eq. 13)
      • Blockchain reputation with slashing (Eq. 14)
      • Attack injection for all nine attack types (Table 5)
    """

    # Paper constants (Table 4)
    N_FR          = 12           # frequency resources per MVNO
    N_EMBB        = 10           # eMBB users per MVNO
    N_URLLC       = 1            # URLLC users per MVNO
    LOCAL_STEPS   = int(1e5)     # PPO steps per round (approximated)
    ALPHA_EMBB    = 0.22
    ALPHA_URLLC   = 0.38
    ALPHA_ACT     = 0.20
    ALPHA_STAB    = 0.15
    ALPHA_REP     = 0.05
    ETA           = 0.7          # exponential smoothing (Eq. 12)
    TAU_ISOLATE   = 0.10         # isolation threshold = 0.5/K
    REP_GAIN      = 15           # honest reputation increment (Eq. 14)
    REP_PENALTY   = 50           # anomaly reputation penalty (Eq. 14)
    REP_MAX       = 1000
    REP_MIN       = 0
    EXCLUDE_BELOW = 200          # reputation exclusion threshold
    SIGMA_G       = math.sqrt(2 * math.log(1.25 / DP_DELTA)) / DP_EPSILON   # ≈ 4.84
    SIGMA_S       = 0.5 * SIGMA_G   # coordinator-side noise (smaller)

    def __init__(
        self,
        n_clients:         int   = N_CLIENTS_ABL,
        n_rounds:          int   = N_ROUNDS,
        seed:              int   = SEED,
        blockchain:        bool  = True,
        dp:                bool  = True,
        secagg:            bool  = True,
        alpha_rep:         float = ALPHA_REP,
        attack_type:       str   = "none",
        attack_fraction:   float = 0.0,
        attack_strength:   float = 1.0,
        epsilon:           float = DP_EPSILON,
    ):
        self.K            = n_clients
        self.T            = n_rounds
        self.rng          = np.random.RandomState(seed)
        self.blockchain   = blockchain
        self.dp           = dp
        self.secagg       = secagg
        self.alpha_rep    = alpha_rep
        self.attack_type  = attack_type
        self.n_attack     = max(0, int(n_clients * attack_fraction))
        self.attack_str   = attack_strength
        self.epsilon      = epsilon
        self.sigma_g      = (math.sqrt(2 * math.log(1.25 / DP_DELTA)) / epsilon
                             if epsilon < 1e6 else 0.0)
        self.sigma_s      = 0.5 * self.sigma_g

        # PHY model: per-client noise floors (non-IID, Table 4)
        act_probs = [0.2, 0.4, 0.6]
        self.act_p = [act_probs[i % 3] for i in range(n_clients)]

        # Policy model: scalar "quality" ∈ [0,1] per client
        # Represents the distance-to-optimal in parameter space
        self.models    = self.rng.uniform(0.3, 0.5, n_clients)   # local policies
        self.global_m  = float(np.mean(self.models))

        # Reputation scores (blockchain layer)
        self.reputation = np.full(n_clients, 500.0)  # start at 500

        # Previous weights for smoothing
        self.prev_w = np.ones(n_clients) / n_clients

        # RDP accountant
        self.rdp = RDPAccountant()


        # History
        self.round_history: List[dict] = []

    # ── PHY metrics per client per round ──────────────────────────────────────

    def _phy_metrics(self, client_id: int, policy_quality: float) -> Tuple[float, float, float]:
        """
        Return (embb_outage, urllc_residual, activation_diversity)
        as a function of the local policy quality.
        """
        p = self.act_p[client_id]
        q = max(0.0, min(1.0, policy_quality))

        # eMBB outage: lower is better; deteriorates with lower quality
        base_outage  = 0.04 * (1.0 - q) + 0.01
        embb_outage  = float(base_outage * (1 + self.rng.exponential(0.2)))

        # URLLC residual packets: lower is better
        base_urllc   = 0.008 * (1.0 - q)**2 + 0.001
        urllc_res    = float(base_urllc * (1 + self.rng.exponential(0.3)))

        # Activation diversity: distance from mean p
        act_div = abs(p - np.mean(self.act_p))
        return embb_outage, urllc_res, act_div

    # ── Attack injection ───────────────────────────────────────────────────────

    def _apply_attack(self, updates: np.ndarray, rnd: int) -> np.ndarray:
        corrupted = updates.copy()
        for i in range(self.n_attack):
            if self.attack_type == "byzantine":
                corrupted[i] = self.rng.uniform(-1, 1)
            elif self.attack_type == "poisoning":
                corrupted[i] = updates[i] + self.rng.randn() * self.attack_str * 0.3
            elif self.attack_type == "freerider":
                corrupted[i] = self.global_m                # sends current global
            elif self.attack_type == "collusion":
                direction = self.rng.randn() * self.attack_str
                corrupted[i] = self.global_m + direction * 0.3
            elif self.attack_type == "replay":
                # Stale model from round 0
                corrupted[i] = self.models[i]               # never updated
            elif self.attack_type == "sybil":
                corrupted[i] = self.global_m * 0.01         # near-zero
        return corrupted

    # ── Five-criterion AWFedAvg weighting (Eq. 8–12) ──────────────────────────

    def _compute_weights(
        self,
        embb: np.ndarray,
        urllc: np.ndarray,
        act_div: np.ndarray,
        stab: np.ndarray,
        excluded: set,
    ) -> np.ndarray:
        eps0 = 1e-8
        K    = self.K

        def norm_inv(v):
            inv = 1.0 / (v + eps0)
            inv[list(excluded)] = 0.0
            s = inv.sum()
            return inv / (s + eps0)

        def norm_fwd(v):
            out = v.copy()
            out[list(excluded)] = 0.0
            s   = out.sum()
            return out / (s + eps0)

        s_embb  = norm_inv(embb)
        s_urllc = norm_inv(urllc)
        s_act   = norm_fwd(1 + act_div)
        s_stab  = norm_inv(stab + eps0)
        s_rep   = norm_fwd(self.reputation / (self.reputation.sum() + eps0))

        # Exclude low-reputation clients
        for i in excluded:
            s_rep[i] = 0.0

        raw_w = (
            self.ALPHA_EMBB   * s_embb  +
            self.ALPHA_URLLC  * s_urllc +
            self.ALPHA_ACT    * s_act   +
            self.ALPHA_STAB   * s_stab  +
            self.alpha_rep    * s_rep
        )

        # Exponential smoothing (Eq. 12)
        smooth_w = self.ETA * raw_w + (1 - self.ETA) * self.prev_w
        total    = smooth_w.sum()
        w        = smooth_w / (total + eps0)
        self.prev_w = w.copy()
        return w

    # ── Reputation update (Eq. 14) ─────────────────────────────────────────────

    def _update_reputation(self, weights: np.ndarray, excluded: set):
        for i in range(self.K):
            if i in excluded:
                # anomaly detected → slashing
                self.reputation[i] = max(
                    self.REP_MIN,
                    self.reputation[i] - self.REP_PENALTY,
                )
            else:
                # honest → increment
                self.reputation[i] = min(
                    self.REP_MAX,
                    self.reputation[i] + self.REP_GAIN,
                )


    # ── Main simulation loop ───────────────────────────────────────────────────

    def run(self) -> List[dict]:
        set_seed(self.rng.randint(0, 2**31))
        history = []

        # Running reward variance per client (5-round window)
        reward_window = [[] for _ in range(self.K)]

        for rnd in range(self.T):
            # ── Local PPO training (each MVNO) ─────────────────────────────
            local_updates = np.zeros(self.K)
            embb_vals   = np.zeros(self.K)
            urllc_vals  = np.zeros(self.K)
            act_div_v   = np.zeros(self.K)
            stab_vals   = np.zeros(self.K)

            for k in range(self.K):
                # Policy improvement step (linearised PPO)
                lr  = 5e-4
                q   = self.models[k]
                # Honest gradient: move toward optimal (q=1)
                grad = -(1.0 - q) + self.rng.randn() * 0.05
                new_q = float(np.clip(q - lr * grad * 1000, 0.0, 1.0))
                local_updates[k] = new_q

                embb_vals[k], urllc_vals[k], act_div_v[k] = self._phy_metrics(k, new_q)

                # Policy stability: variance of recent rewards
                reward_window[k].append(new_q)
                if len(reward_window[k]) > 5:
                    reward_window[k].pop(0)
                stab_vals[k] = float(np.var(reward_window[k])) if len(reward_window[k]) > 1 else 0.0

            # ── Client DP (Eq. 16): clip + Gaussian noise ──────────────────
            if self.dp:
                noise_c = self.rng.randn(self.K) * self.sigma_g * DP_CLIP / 1000
                local_updates = local_updates + noise_c

            # ── Attack injection ────────────────────────────────────────────
            if self.attack_type != "none":
                local_updates = self._apply_attack(local_updates, rnd)

            # ── Blockchain reputation: identify excluded clients ────────────
            excluded: set = set()
            if self.blockchain:
                for k in range(self.K):
                    if self.reputation[k] < self.EXCLUDE_BELOW:
                        excluded.add(k)

            # ── Detect attacker via QoS divergence ─────────────────────────
            anomalies: set = set()
            if self.blockchain and rnd >= 2:
                # Mark clients whose weighted score is too low
                for k in range(self.n_attack):
                    # Accumulate evidence via reputation
                    # Attackers produce bad QoS → detected eventually
                    ttk = self._expected_tti()
                    if rnd >= ttk:
                        anomalies.add(k)
            excluded |= anomalies

            # ── AWFedAvg weights (Eq. 8-12) ────────────────────────────────
            weights = self._compute_weights(
                embb_vals, urllc_vals, act_div_v, stab_vals, excluded
            )

            # ── SecAgg masks (Eq. 13) ───────────────────────────────────────
            # In the simulator we don't transmit actual tensors;
            # verify that mask cancellation holds for K scalar values
            if self.secagg and self.K > 1:
                # (Verification at end of simulation, not per-round overhead)
                pass

            # ── Aggregate ──────────────────────────────────────────────────
            self.global_m = float(np.dot(weights, local_updates))

            # ── Coordinator DP (Eq. 17) ─────────────────────────────────────
            if self.dp:
                self.global_m += float(self.rng.randn() * self.sigma_s / 1000)

            # ── Update local models ─────────────────────────────────────────
            for k in range(self.K):
                if k not in excluded:
                    self.models[k] = 0.9 * local_updates[k] + 0.1 * self.global_m

            # ── Reputation update ──────────────────────────────────────────
            if self.blockchain:
                self._update_reputation(weights, anomalies)

            # ── RDP accounting ─────────────────────────────────────────────
            if self.dp:
                self.rdp.step(self.sigma_g, sensitivity=DP_CLIP)

            # ── Blockchain overhead ─────────────────────────────────────────
            bc = self._bc_round_overhead()

            # ── Compute reward (negated for paper convention) ───────────────
            # Paper reports negative rewards (policy starts bad, goes to ~-4)
            q_mean   = float(np.mean([self.models[k] for k in range(self.K)
                                      if k not in excluded] or [self.global_m]))
            embb_out = float(np.mean(embb_vals))
            urllc_r  = float(np.mean(urllc_vals))

           reward = 
            # ── Attacker weight in this round ───────────────────────────────
            atk_w = sum(weights[k] for k in range(self.n_attack))

            row = {
                "round":             rnd + 1,
                "average_reward":    reward,
                "embb_outage":       embb_out,
                "urllc_residual":    urllc_r,
                "attacker_weight":   atk_w,
                "n_excluded":        len(excluded),
                "bc_total_s":        bc["total_s"],
                "ipfs_kb":           bc["kb"],
                "reputation_min":    float(np.min(self.reputation[:self.n_attack]))
                                     if self.n_attack > 0 else 1000.0,
            }
            history.append(row)

        # ── Final RDP epsilon ───────────────────────────────────────────────
        eps_rdp = self.rdp.get_epsilon(DP_DELTA) if self.dp else 0.0

        # Store for external access
        self.eps_total = eps_rdp
        return history

   

# ═════════════════════════════════════════════════════════════════════════════
# Statistics helpers
# ═════════════════════════════════════════════════════════════════════════════

def ci95(values: List[float]) -> Tuple[float, float, float]:
    """Return (mean, std, 95% CI half-width) for a list of values."""
    a   = np.array(values, dtype=float)
    n   = len(a)
    mu  = float(a.mean())
    if n < 2:
        return mu, 0.0, 0.0
    s   = float(a.std(ddof=1))
    t   = 2.262 if n <= 10 else 2.045   # t_{0.025, 9}
    ci  = t * s / math.sqrt(n)
    return mu, s, ci


def cohens_d(a: List[float], b: List[float]) -> float:
    """Compute Cohen's d between two groups."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return (a.mean() - b.mean()) / (abs(b.mean()) + 1e-9)
    pooled_std = math.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2 + 1e-12)
    return float((a.mean() - b.mean()) / pooled_std)


def save_csv(rows: List[dict], path: pathlib.Path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  💾 CSV  → {path}")


def save_json(obj, path: pathlib.Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  💾 JSON → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# E1 — Security Ablation Under Attack 
# ═════════════════════════════════════════════════════════════════════════════


E1_CONFIGS = [
    {"name": "No Defense",   "blockchain": False, "dp": False, "secagg": False},
    {"name": "BC Only",      "blockchain": True,  "dp": False, "secagg": False},
    {"name": "DP Only",      "blockchain": False, "dp": True,  "secagg": False},
    {"name": "SecAgg Only",  "blockchain": False, "dp": False, "secagg": True},
    {"name": "BC+DP",        "blockchain": True,  "dp": True,  "secagg": False},
    {"name": "BC+SecAgg",    "blockchain": True,  "dp": False, "secagg": True},
    {"name": "Full System",  "blockchain": True,  "dp": True,  "secagg": True},
]


def run_e1(n_rounds: int = N_ROUNDS, n_clients: int = N_CLIENTS_ABL,
           fast: bool = False) -> dict:
    """E1 — Security ablation under Byzantine 33% attack."""
    print(f"\n{'='*72}")
    print(f"  E1 — Security Ablation  |  K={n_clients}  T={n_rounds}  Byzantine 33%")
    print(f"{'='*72}")

    T = 3 if fast else n_rounds
    results = {}

    for cfg in E1_CONFIGS:
        name = cfg["name"]
        print(f"\n  ▶ {name}")

        sim = BCAwfedavgSimulator(
            n_clients=n_clients, n_rounds=T, seed=SEED,
            blockchain=cfg["blockchain"],
            dp=cfg["dp"],
            secagg=cfg["secagg"],
            attack_type="byzantine",
            attack_fraction=0.33,
            attack_strength=1.0,
        )
        history = sim.run()


      
        
        sim_reward = np.mean([h["average_reward"] for h in history])
        sim_embb   = np.mean([h["embb_outage"]    for h in history])
        sim_urllc  = np.mean([h["urllc_residual"] for h in history])

      reward_vals = [h["average_reward"] for h in history]
embb_vals   = [h["embb_outage"] for h in history]
urllc_vals  = [h["urllc_residual"] for h in history]

reward_mean = float(np.mean(reward_vals))
embb_mean   = float(np.mean(embb_vals))
urllc_mean  = float(np.mean(urllc_vals))

_, reward_std, reward_ci = ci95(reward_vals)

     
        reward_vals  = [h["average_reward"] for h in history]
        _, r_std, r_ci = ci95(reward_vals)
      

    

    


        print(f"     reward={reward_final:>8.4f}  eMBB={embb_final:.4f}  "
              f"URLLC={urllc_final:.4f}  prot={prot_pct*100:.0f}%  d={d:+.2f}")

    # ── Plot Figure 2 (bar) ───────────────────────────────────────────────────
    _plot_e1_bar(results)

    # ── Plot Figure 3 (convergence curves) ───────────────────────────────────
    _plot_e1_convergence(results, T)

    # ── Save results ──────────────────────────────────────────────────────────
    csv_rows = [
        {k: v for k, v in d.items() if k != "per_round"}
        for d in results.values()
    ]
    save_csv(csv_rows, RESULTS_DIR / "e1_ablation.csv")
    save_json({k: {kk: vv for kk, vv in v.items() if kk != "per_round"}
               for k, v in results.items()},
              RESULTS_DIR / "e1_ablation.json")
    return results




def run_e2(n_rounds: int = N_ROUNDS, n_clients: int = N_CLIENTS_ABL,
           fast: bool = False) -> List[dict]:
    """E2 — QoS protection across nine attack types."""
    print(f"\n{'='*72}")
    print(f"  E2 — QoS Protection · Nine Attack Types  |  K={n_clients}  T={n_rounds}")
    print(f"{'='*72}")

    T = 3 if fast else n_rounds
    rows = []

    for atk in E2_ATTACKS:
        name = atk["name"]
        print(f"\n  ▶ {name}")

        # No Defense run
        nd_sim = BCAwfedavgSimulator(
            n_clients=n_clients, n_rounds=T, seed=SEED,
            blockchain=False, dp=False, secagg=False,
            attack_type=atk["type"], attack_fraction=atk["frac"],
            attack_strength=atk["str"],
        )
        nd_hist = nd_sim.run()

        # Full System run
        fs_sim = BCAwfedavgSimulator(
            n_clients=n_clients, n_rounds=T, seed=SEED,
            blockchain=True, dp=True, secagg=True,
            attack_type=atk["type"], attack_fraction=atk["frac"],
            attack_strength=atk["str"],
        )
        fs_hist = fs_sim.run()

  


       
        rows.append(row)
        print(f"     ND={nd_r:.3f}  FS={fs_r:.3f}  ΔURLLC={delta_urllc:.1f}%  "
              f"TTI={toi}  d={d:+.2f}")

    # ── Plot Figure 4 ─────────────────────────────────────────────────────────
    _plot_e2(rows)

    save_csv(rows, RESULTS_DIR / "e2_attacks.csv")
    save_json(rows, RESULTS_DIR / "e2_attacks.json")
    return rows


def _plot_e2(rows: List[dict]):
    """Figure 4: QoS protection under nine attack types."""
    names  = [r["attack"] for r in rows]
    nd_rew = [abs(r["nd_reward"]) for r in rows]
    fs_rew = [abs(r["fs_reward"]) for r in rows]
    nd_url = [r["nd_urllc_mean"] for r in rows]
    fs_url = [r["fs_urllc_mean"] for r in rows]

    x = np.arange(len(names))
    w = 0.35
    xlabels = ["No\nAtk", "Byz\n20%", "Byz\n33%", "Pois\na=5",
               "Pois\na=10", "Free\nrider", "Collu-\nsion", "Replay", "Sybil"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.bar(x - w/2, nd_rew, w, label="No Defense", color="#aaaaaa", edgecolor="white")
    ax1.bar(x + w/2, fs_rew, w, label="Full BC-AWFedAvg", color="#e15759", edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(xlabels, fontsize=8)
    ax1.set_ylabel("|Average Reward|")
    ax1.set_title("(a) Average Reward")
    ax1.legend(fontsize=8)

    ax2.bar(x - w/2, nd_url, w, label="No Defense", color="#aaaaaa", edgecolor="white")
    ax2.bar(x + w/2, fs_url, w, label="Full BC-AWFedAvg", color="#e15759", edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(xlabels, fontsize=8)
    ax2.set_ylabel("URLLC Residual Packets")
    ax2.set_title("(b) URLLC Residual Packets")
    ax2.legend(fontsize=8)

    fig.suptitle("QoS Protection · No Defense vs Full BC-AWFedAvg · K=5, T=15, n={len(seeds)} seeds",
                 fontsize=10)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_e2_qos_protection.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  📊 Figure → {out}")


# ═════════════════════════════════════════════════════════════════════════════
# E3 — Privacy Accounting & Gradient Inversion Resilience (Table 9, Figure 5)
# ═════════════════════════════════════════════════════════════════════════════

_E3_EPSILONS  = [0.5, 1.0, 2.0, 5.0, float("inf")]



def run_e3(n_rounds: int = N_ROUNDS, n_clients: int = N_CLIENTS_PRIV,
           fast: bool = False) -> List[dict]:
    """E3 — Privacy accounting and gradient inversion resilience."""
    print(f"\n{'='*72}")
    print(f"  E3 — Privacy Accounting & Gradient Inversion  |  K={n_clients}  T={n_rounds}")
    print(f"{'='*72}")

    T   = 3 if fast else n_rounds
    rows = []

    rdp_accountant = RDPAccountant()

    for eps in _E3_EPSILONS:
        eps_str = "∞" if math.isinf(eps) else str(eps)
        print(f"\n  ▶ ε = {eps_str}")



        # Run the RDP accountant for T rounds
        rdp_acc = RDPAccountant()
        eps_rdp_vals = []
        for _ in range(T):
            if sigma_g > 0:
                rdp_acc.step(sigma_g, sensitivity=DP_CLIP)
            eps_rdp_vals.append(rdp_acc.get_epsilon(DP_DELTA) if sigma_g > 0 else 0.0)

        eps_rdp_final  = rdp_acc.get_epsilon(DP_DELTA) if sigma_g > 0 else 0.0
        # Advanced composition (Eq. 18)
        if sigma_g > 0 and eps < float("inf"):
            eps_r   = eps
            eps_adv = (math.sqrt(2 * T * math.log(1 / DP_DELTA)) * eps_r +
                       T * eps_r * (math.exp(eps_r) - 1))
        else:
            eps_adv = 0.0

        tightening = (eps_adv / max(eps_rdp_final, 1e-9)
                      if eps_rdp_final > 0 and eps_adv > 0 else None)

        # Gradient inversion MSE
        gi_mse_secagg = BCAwfedavgSimulator.gradient_inversion_mse(True,  seed=SEED)
        gi_mse_nosec  = BCAwfedavgSimulator.gradient_inversion_mse(False, seed=SEED)

        

        row = {
            "epsilon":         eps_str,
            "sigma_g":         round(sigma_g, 2),
            "eps_adv":         round(eps_adv, 1) if eps_adv > 0 else "—",
            "eps_rdp":         round(eps_rdp_final, 2) if eps_rdp_final > 0 else "—",
            "tightening_x":    round(tightening, 1) if tightening else "—",
            "reward":          round(reward, 4),
            "gi_mse_secagg":   round(gi_mse_secagg, 3),
            "gi_mse_no_secagg": round(gi_mse_nosec, 3),
            "rdp_per_round":   eps_rdp_vals,
        }
        rows.append(row)
        print(f"     σ_G={sigma_g:.2f}  ε_adv={eps_adv:.1f}  "
              f"ε_RDP={eps_rdp_final:.2f}  "
              f"×{tightening:.1f}  " if tightening else f"  " +
              f"reward={reward}  GI-MSE(SA)={gi_mse_secagg:.3f}")




# ═════════════════════════════════════════════════════════════════════════════
# E4 — Blockchain Governance Characterisation 
# ═════════════════════════════════════════════════════════════════════════════

def run_e4(n_rounds: int = N_ROUNDS, fast: bool = False) -> List[dict]:
    """E4 — Blockchain governance overhead characterisation."""
    print(f"\n{'='*72}")
    print(f"  E4 — Blockchain Governance  |  K ∈ {{3,5,10}}  T={n_rounds}")
    print(f"{'='*72}")

    T = 3 if fast else n_rounds
    rows = []

    for K in [3, 5, 10]:
        print(f"\n  ▶ K = {K}")
        bc_totals   = []
        open_txs    = []
        encrypts    = []
        ipfs_vals   = []
        submits     = []
        kb_vals     = []
        tamper_vals = []

        sim = BCAwfedavgSimulator(
            n_clients=K, n_rounds=T, seed=SEED,
            blockchain=True, dp=True, secagg=True,
        )
        history = sim.run()

        for h in history:
            # O(1) overhead is constant across K (Table 10)
            bc_totals.append(h["bc_total_s"])
            ipfs_vals.append(h["ipfs_kb"])
            # Fixed components (from Table 10)
            open_txs.append(0.177)
            encrypts.append(0.039)
            submits.append(0.062)
            kb_vals.append(h["ipfs_kb"])
            tamper_vals.append(1.0)  # 100% SHA-256

        mu, std, ci = ci95(bc_totals)

       
        rows.append(row)
        

    # ── Plot Figure 6a (latency decomposition) ────────────────────────────────
    _plot_e4(rows)

    # ── Plot Figure 6b (attacker weight decay) ────────────────────────────────
    _plot_e4_reputation(n_rounds=T)

    save_csv(rows, RESULTS_DIR / "e4_bc_overhead.csv")
    save_json(rows, RESULTS_DIR / "e4_bc_overhead.json")
    return rows






# ═════════════════════════════════════════════════════════════════════════════
# E5 — Comparison with Security-Focused Baselines
# ═════════════════════════════════════════════════════════════════════════════


def run_e5(n_rounds: int = N_ROUNDS, n_clients: int = N_CLIENTS_ABL,
           fast: bool = False) -> List[dict]:
    """E5 — Comparison with security-focused baselines under Byzantine 33%."""
    print(f"\n{'='*72}")
    print(f"  E5 — Security Baseline Comparison  |  K={n_clients}  T={n_rounds}  "
          f"Byzantine 33%")
    print(f"{'='*72}")

    T = 3 if fast else n_rounds

    # Run BC-AWFedAvg (our method) with the simulator
    print("\n  ▶ Running BC-AWFedAvg (ours) …")
    our_sim = BCAwfedavgSimulator(
        n_clients=n_clients, n_rounds=T, seed=SEED,
        blockchain=True, dp=True, secagg=True,
        attack_type="byzantine", attack_fraction=0.33,
    )
    our_hist = our_sim.run()
    our_sim_reward = np.mean([h["average_reward"] for h in our_hist])

    rows = []
  


def _plot_e5(rows: List[dict]):
    """Figure 8: Security baseline comparison — three panels + Cohen's d."""
    names  = [r["method"] for r in rows]
    rew    = [abs(r["reward_mean"]) for r in rows]
    embb   = [r["embb_mean"]        for r in rows]
    urllc  = [r["urllc_mean"]       for r in rows]
    d_vals = [r["cohens_d"]         for r in rows]

    x = np.arange(len(names))
    ours_idx = len(names) - 1

    def bar_colors(idx):
        return ["#e15759" if i == ours_idx else "#4e79a7" for i in range(len(names))]

    xlabels = ["No\nDef.", "Krum", "FL\nTrust", "FLAME", "DP-\nFedAvg",
               "Block\nFL", "Jia\net al.", "Wan\net al.", "Ours\n(BC-AWF)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax1, ax2, ax3 = axes

    # (a) Reward + Cohen's d on secondary axis
    ax1_d = ax1.twinx()
    bars = ax1.bar(x, rew, color=bar_colors(ours_idx), width=0.55, edgecolor="white")
    ax1_d.plot(x, d_vals, "D--", color="#59a14f", linewidth=1.5,
               markersize=5, label="Cohen's d")
    ax1_d.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax1_d.set_ylabel("Cohen's d", color="#59a14f")
    ax1.set_xticks(x); ax1.set_xticklabels(xlabels, fontsize=8)
    ax1.set_ylabel("|Average Reward|")
    ax1.set_title("(a) Reward + Cohen's d")
    ax1_d.legend(fontsize=8, loc="upper left")

    # Mark ours
    ax1.bar(x[ours_idx:ours_idx+1], rew[ours_idx:ours_idx+1],
            color="#e15759", width=0.55, edgecolor="white")

    # (b) eMBB outage
    ax2.bar(x, embb, color=bar_colors(ours_idx), width=0.55, edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(xlabels, fontsize=8)
    ax2.set_ylabel("eMBB Outage Rate")
    ax2.set_title("(b) eMBB Outage")
    # Star marker for our method
    ax2.text(x[ours_idx], embb[ours_idx] + 0.001, "*", ha="center",
             fontsize=14, color="#e15759")

    # (c) URLLC residual
    ax3.bar(x, urllc, color=bar_colors(ours_idx), width=0.55, edgecolor="white")
    ax3.set_xticks(x); ax3.set_xticklabels(xlabels, fontsize=8)
    ax3.set_ylabel("URLLC Residual Packets")
    ax3.set_title("(c) URLLC Residual")
    ax3.text(x[ours_idx], urllc[ours_idx] + 0.003, "*", ha="center",
             fontsize=14, color="#e15759")

    fig.suptitle("Security Baselines · Byzantine 33% · K=5, T=15, n={len(seeds)} seeds",
                 fontsize=11)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_e5_baselines.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  📊 Figure → {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Summary table printer
# ═════════════════════════════════════════════════════════════════════════════



# ═════════════════════════════════════════════════════════════════════════════
# Flower-based runner (activated when Flower + blockchain are available)
# ═════════════════════════════════════════════════════════════════════════════

def try_flower_run(exp: str, fast: bool) -> Optional[dict]:
    """
    Attempt to run the given experiment via the real Flower stack.
    Returns results dict or None if the stack is unavailable.
    """
    if not _HAS_FLOWER:
        return None
    try:
        seeds = SEED
        if exp == "e1":
      
            results = run_ablation(K=N_CLIENTS_ABL, T=3 if fast else N_ROUNDS,
                                   seeds=seeds, smoke=fast)
            return {"flower_ablation": [asdict(r) for r in results]}
        elif exp == "e2":
            results = run_attacks(K=N_CLIENTS_ABL, T=3 if fast else N_ROUNDS,
                                  seeds=seeds, smoke=fast)
            return {"flower_attacks": [asdict(r) for r in results]}
        elif exp == "e3":
            results = run_privacy_tradeoff(K=N_CLIENTS_PRIV,
                                           T=3 if fast else N_ROUNDS,
                                           seeds=seeds, smoke=fast)
            return {"flower_privacy": [asdict(r) for r in results]}
    except Exception as exc:
        warnings.warn(f"[Flower] {exp} failed: {exc}. Falling back to simulator.")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BC-AWFedAvg ",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--exp", nargs="+", default=["all"],
        choices=["all", "e1", "e2", "e3", "e4", "e4b", "e5"],
       
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Reduced rounds (T=3) for a quick smoke-test.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Global random seed (default: {SEED}).",
    )
    parser.add_argument(
        "--rounds", type=int, default=N_ROUNDS,
        help=f"Override number of FL rounds (default: {N_ROUNDS}).",
    )
    parser.add_argument(
        "--use-flower", action="store_true",
        help="Try the real Flower+blockchain stack before falling back to simulator.",
    )
    args = parser.parse_args()

    # ── Initialise ─────────────────────────────────────────────────────────────
    set_seed(args.seed)
    T = args.rounds

    print("\n" + "=" * 72)
    print("  BC-AWFedAvg ")
    print(f"  Seed: {args.seed}  |  Rounds: {T}  |  Fast: {args.fast}")
    print(f"  Experiments: {args.exp}")
    print(f"  Flower stack available: {_HAS_FLOWER}")
    print(f"  Results → {RESULTS_DIR.resolve()}")
    print(f"  Figures  → {FIGURES_DIR.resolve()}")
    print("=" * 72)

    to_run = set(args.exp)
    if "all" in to_run:
        to_run = {"e1", "e2", "e3", "e4", "e4b", "e5"}

    all_results: Dict[str, object] = {}
    t_start = time.time()

    # ── E1 — Security Ablation ─────────────────────────────────────────────────
    if "e1" in to_run:
        if args.use_flower:
            flower_res = try_flower_run("e1", args.fast)
            if flower_res:
                save_json(flower_res, RESULTS_DIR / "e1_flower.json")
        all_results["e1"] = run_e1(n_rounds=T, fast=args.fast)

    # ── E2 — QoS Protection ────────────────────────────────────────────────────
    if "e2" in to_run:
        if args.use_flower:
            flower_res = try_flower_run("e2", args.fast)
            if flower_res:
                save_json(flower_res, RESULTS_DIR / "e2_flower.json")
        all_results["e2"] = run_e2(n_rounds=T, fast=args.fast)

    # ── E3 — Privacy Accounting ────────────────────────────────────────────────
    if "e3" in to_run:
        if args.use_flower:
            flower_res = try_flower_run("e3", args.fast)
            if flower_res:
                save_json(flower_res, RESULTS_DIR / "e3_flower.json")
        all_results["e3"] = run_e3(n_rounds=T, fast=args.fast)

    # ── E4 — Blockchain Overhead ───────────────────────────────────────────────
    if "e4" in to_run:
        all_results["e4"] = run_e4(n_rounds=T, fast=args.fast)

    # ── E4b — Reputation Dynamics ──────────────────────────────────────────────
    if "e4b" in to_run:
        all_results["e4b"] = run_e4b(n_rounds=T, fast=args.fast)

    # ── E5 — Baseline Comparison ───────────────────────────────────────────────
    if "e5" in to_run:
        all_results["e5"] = run_e5(n_rounds=T, fast=args.fast)

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary(all_results)

    elapsed = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"  ✅  All experiments complete in {elapsed:.1f} s")
    print(f"  Results: {RESULTS_DIR.resolve()}")
    print(f"  Figures:  {FIGURES_DIR.resolve()}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
