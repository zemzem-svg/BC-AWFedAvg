"""
experiments.py  —  Complete Experimental Protocol
==================================================

1. run_ablation()         — 5-config ablation study
2. run_scalability()      — K in {3,5,10,20,50}
3. run_privacy_tradeoff() — epsilon in {0.1,0.5,1.0,5.0}
4. run_attacks()          — 9 attack scenarios
All experiments: 10 seeds, mean±std, 95% CI, Cohen's d

Usage
-----
  python experiments.py                         # full protocol
  python experiments.py --mode ablation
  python experiments.py --mode scalability
  python experiments.py --mode privacy
  python experiments.py --mode attacks
  python experiments.py --smoke                 # 1 round, 2 seeds
  python experiments.py --clients 10 --rounds 20 --dp_eps 1.0 --attack byzantine --attack_frac 0.2 --seed 42
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, time, warnings
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import numpy as np

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    name: str; seed: int; num_clients: int; num_rounds: int
    final_reward: float = 0.0;  avg_reward: float = 0.0
    reward_std: float = 0.0;    bc_latency_s: float = 0.0
    ipfs_kb: float = 0.0;       eps_total: float = 0.0
    wall_clock_s: float = 0.0;  rounds_completed: int = 0
    embb_outage: float = 0.0;   urllc_residual: float = 0.0
    total_data_kb: float = 0.0; error: Optional[str] = None
    per_round: List[Dict] = field(default_factory=list)


@dataclass
class AggregatedResult:
    name: str; num_clients: int; num_rounds: int; n_runs: int = 0
    reward_mean: float = 0.0;  reward_std: float = 0.0;  reward_ci95: float = 0.0
    bc_mean: float = 0.0;      bc_std: float = 0.0
    ipfs_mean: float = 0.0;    eps_total: float = 0.0
    wall_mean: float = 0.0;    wall_std: float = 0.0
    embb_mean: float = 0.0;    urllc_mean: float = 0.0
    data_kb_mean: float = 0.0; cohens_d: float = 0.0
    error_runs: int = 0;       runs: List[RunResult] = field(default_factory=list)

    @staticmethod
    def from_runs(name: str, runs: List[RunResult]) -> "AggregatedResult":
        good = [r for r in runs if r.error is None]
        n = len(good)
        res = AggregatedResult(name=name, num_clients=runs[0].num_clients,
                               num_rounds=runs[0].num_rounds, n_runs=len(runs),
                               error_runs=len(runs)-n, runs=runs)
        if n == 0:
            return res
        def _agg(vals):
            a = np.array(vals, dtype=float)
            m = float(a.mean()); s = float(a.std(ddof=1)) if n > 1 else 0.0
            t = 2.262 if n < 10 else 2.045
            ci = t * s / math.sqrt(n) if n > 1 else 0.0
            return m, s, ci
        res.reward_mean, res.reward_std, res.reward_ci95 = _agg([r.final_reward for r in good])
        res.bc_mean,   res.bc_std,  _ = _agg([r.bc_latency_s   for r in good])
        res.ipfs_mean, _, _           = _agg([r.ipfs_kb         for r in good])
        res.wall_mean, res.wall_std,_ = _agg([r.wall_clock_s    for r in good])
        res.embb_mean,  _, _          = _agg([r.embb_outage     for r in good])
        res.urllc_mean, _, _          = _agg([r.urllc_residual  for r in good])
        res.data_kb_mean, _, _        = _agg([r.total_data_kb   for r in good])
        res.eps_total = good[-1].eps_total
        return res

    def compute_cohens_d(self, baseline: "AggregatedResult"):
        pooled = math.sqrt((self.reward_std**2 + baseline.reward_std**2) / 2 + 1e-9)
        self.cohens_d = (self.reward_mean - baseline.reward_mean) / pooled

    def summary_line(self) -> str:
        s = f"({self.error_runs}err)" if self.error_runs else "✅"
        return (f"{s} {self.name:<32} K={self.num_clients:>2} T={self.num_rounds:>2} n={self.n_runs}"
                f" | rwd={self.reward_mean:>7.4f}±{self.reward_std:.4f} CI±{self.reward_ci95:.4f}"
                f" | bc={self.bc_mean:>5.2f}s ipfs={self.ipfs_mean:>6.1f}KB"
                f" ε={self.eps_total:.4f} d={self.cohens_d:>+.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _import_project():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from blockchain_awfedavg_true_integration import run_blockchain_awfedavg_experiment
    return {"run_experiment": run_blockchain_awfedavg_experiment}


def _load_contract_info() -> dict:
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contract_info.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _extract_run(hist, wall, name, K, T, seed) -> RunResult:
    r = RunResult(name=name, seed=seed, num_clients=K, num_rounds=T, wall_clock_s=wall)
    per_round = []
    if hist and hasattr(hist, "metrics_distributed"):
        for rnd, metrics in hist.metrics_distributed.get("fit", []):
            row = {"round": rnd}; row.update(metrics); per_round.append(row)
    r.per_round = per_round; r.rounds_completed = len(per_round)
    def _mean(key): return float(np.mean([p[key] for p in per_round if key in p])) if per_round else 0.0
    r.avg_reward    = _mean("average_reward");  r.final_reward   = float([p.get("average_reward",0) for p in per_round][-1]) if per_round else 0.0
    r.bc_latency_s  = _mean("blockchain_overhead_s");  r.ipfs_kb = _mean("ipfs_upload_size_kb")
    r.eps_total     = float([p.get("eps_total",0) for p in per_round][-1]) if per_round else 0.0
    r.embb_outage   = _mean("avg_embb_outage_counter")
    r.urllc_residual= _mean("avg_residual_urllc_pkt")
    r.total_data_kb = sum(p.get("ipfs_upload_size_kb",0) for p in per_round)
    rewards = [p.get("average_reward",0) for p in per_round]
    r.reward_std = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
    return r


def _run_one(name, seed, K, T, cfg, mods, info) -> RunResult:
    set_global_seed(seed)
    t0 = time.time()
    try:
        hist, _, _, _ = mods["run_experiment"](
            blockchain_enabled      = cfg.get("blockchain_enabled", False),
            blockchain_provider     = "http://127.0.0.1:8545",
            contract_address        = info.get("contract_address"),
            contract_abi_path       = "contract_info.json" if info else None,
            coordinator_private_key = info.get("coordinator_private_key"),
            ipfs_addr               = "/ip4/127.0.0.1/tcp/5001",
            epsilon                 = cfg.get("epsilon", 1.0),
            delta                   = 1e-5, clip_norm=1.0,
            alpha_embb              = cfg.get("alpha_embb", 0.25),
            alpha_urllc             = cfg.get("alpha_urllc", 0.25),
            alpha_activation        = cfg.get("alpha_activation", 0.25),
            alpha_stability         = cfg.get("alpha_stability", 0.25),
            alpha_reputation        = cfg.get("alpha_reputation", 0.0),
            num_rounds              = T, num_clients=K,
            secure_aggregation      = cfg.get("secure_aggregation", False),
            attack_type             = cfg.get("attack_type", "none"),
            attack_fraction         = cfg.get("attack_fraction", 0.0),
            attack_strength         = cfg.get("attack_strength", 1.0),
        )
        return _extract_run(hist, time.time()-t0, name, K, T, seed)
    except Exception as exc:
        warnings.warn(f"[{name}] seed={seed}: {exc}")
        return RunResult(name=name, seed=seed, num_clients=K, num_rounds=T,
                         wall_clock_s=time.time()-t0, error=str(exc))


def _multi_seed(name, K, T, cfg, mods, info, seeds) -> AggregatedResult:
    runs = []
    for s in seeds:
        print(f"     seed={s} ...", end=" ", flush=True)
        r = _run_one(name, s, K, T, cfg, mods, info)
        print(f"reward={r.final_reward:.4f}" if not r.error else f"ERR:{r.error[:35]}")
        runs.append(r)
    return AggregatedResult.from_runs(name, runs)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configs
# ─────────────────────────────────────────────────────────────────────────────

_FULL_CFG = dict(blockchain_enabled=True, epsilon=1.0, secure_aggregation=True,
                 alpha_reputation=0.05, alpha_embb=0.22, alpha_urllc=0.38,
                 alpha_activation=0.20, alpha_stability=0.15)

ABLATION_CONFIGS = [
    {"name": "AWFedAvg Only",     "blockchain_enabled": False, "epsilon": 1.0, "secure_aggregation": False, "alpha_reputation": 0.0, "alpha_embb": 0.22, "alpha_urllc": 0.38, "alpha_activation": 0.2, "alpha_stability": 0.2},
    {"name": "AWFedAvg + DP",              "blockchain_enabled": False, "epsilon": 1.0,   "secure_aggregation": False, "alpha_reputation": 0.0, "alpha_embb": 0.22, "alpha_urllc": 0.38, "alpha_activation": 0.2, "alpha_stability": 0.2},
    {"name": "AWFedAvg + Blockchain",      "blockchain_enabled": True,  "epsilon": 1.0, "secure_aggregation": False, "alpha_reputation": 0.05,"alpha_embb": 0.22, "alpha_urllc": 0.38, "alpha_activation": 0.20, "alpha_stability": 0.2},
    {"name": "AWFedAvg + SecAgg",          "blockchain_enabled": False, "epsilon": 1.0, "secure_aggregation": True,  "alpha_reputation": 0.0, "alpha_embb": 0.22, "alpha_urllc": 0.38, "alpha_activation": 0.2, "alpha_stability": 0.2},
    {"name": "Full System (Proposed)","blockchain_enabled": True, "epsilon": 1.0,   "secure_aggregation": True,  "alpha_reputation": 0.05,"alpha_embb": 0.22, "alpha_urllc": 0.38, "alpha_activation": 0.20, "alpha_stability": 0.2},
]

SCALABILITY_CONFIGS = [
    {"num_clients": 3,  "num_rounds": 10},
    {"num_clients": 5,  "num_rounds": 20},
    {"num_clients": 10, "num_rounds": 30},
    {"num_clients": 20, "num_rounds": 30},
    {"num_clients": 50, "num_rounds": 20},
]

EPSILON_VALUES = [0.1, 0.5, 1.0, 5.0]

ATTACK_SCENARIOS = [
    {"name": "No Attack (baseline)",  "attack_type": "none",      "attack_fraction": 0.00, "attack_strength": 1.0},
    {"name": "Byzantine 20%",         "attack_type": "byzantine", "attack_fraction": 0.20, "attack_strength": 1.0},
    {"name": "Byzantine 33%",         "attack_type": "byzantine", "attack_fraction": 0.33, "attack_strength": 1.0},
    {"name": "Poisoning α=5  10%",    "attack_type": "poisoning", "attack_fraction": 0.10, "attack_strength": 5.0},
    {"name": "Poisoning α=10 10%",    "attack_type": "poisoning", "attack_fraction": 0.10, "attack_strength":10.0},
    {"name": "Free-rider 20%",        "attack_type": "freerider", "attack_fraction": 0.20, "attack_strength": 1.0},
    {"name": "Collusion 33%",         "attack_type": "collusion", "attack_fraction": 0.33, "attack_strength": 3.0},
    {"name": "Replay 20%",            "attack_type": "replay",    "attack_fraction": 0.20, "attack_strength": 1.0},
    {"name": "Sybil 20%",             "attack_type": "sybil",     "attack_fraction": 0.20, "attack_strength": 1.0},
]


# ─────────────────────────────────────────────────────────────────────────────
# Attack injection — applied inside BlockchainEnhancedFlowerClient.fit()
# ─────────────────────────────────────────────────────────────────────────────

def apply_attack_to_params(param_list: list, client_id: int,
                            config: dict, round_history: list) -> list:
    """
    Corrupt param_list according to attack config.  Called at end of fit()
    when config['attack_type'] != 'none'.
    """
    attack_type     = config.get("attack_type",     "none")
    attack_fraction = config.get("attack_fraction", 0.0)
    attack_strength = config.get("attack_strength", 1.0)
    num_clients     = config.get("num_clients",     3)
    server_round    = config.get("server_round",    0)

    n_attackers = max(1, int(num_clients * attack_fraction))
    if client_id >= n_attackers or attack_type == "none":
        return param_list

    if attack_type == "byzantine":
        total_norm = sum(float(np.linalg.norm(p)) for p in param_list) + 1e-9
        scale = total_norm / len(param_list)
        return [np.random.randn(*p.shape).astype(p.dtype) * scale for p in param_list]
    elif attack_type == "poisoning":
        return [p * attack_strength for p in param_list]
    elif attack_type == "freerider":
        return [np.zeros_like(p) for p in param_list]
    elif attack_type == "collusion":
        rng = np.random.RandomState(seed=server_round * 1000)
        total_norm = sum(float(np.linalg.norm(p)) for p in param_list) + 1e-9
        scale = total_norm / len(param_list) * attack_strength
        return [rng.randn(*p.shape).astype(p.dtype) * scale for p in param_list]
    elif attack_type == "replay":
        return list(round_history[0]) if round_history else param_list
    elif attack_type == "sybil":
        return [p * 0.01 for p in param_list]
    return param_list


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runners
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(K=3, T=10, seeds=SEEDS, smoke=False) -> List[AggregatedResult]:
    mods, info = _import_project(), _load_contract_info()
    T_run = 1 if smoke else T;  seeds_run = seeds[:2] if smoke else seeds
    print(f"\n{'='*72}\nABLATION STUDY  K={K} T={T_run} n_seeds={len(seeds_run)}\n{'='*72}")
    results = []; baseline = None
    for cfg in ABLATION_CONFIGS:
        print(f"\n▶  {cfg['name']}")
        agg = _multi_seed(cfg["name"], K, T_run, cfg, mods, info, seeds_run)
        if baseline is None: baseline = agg
        else: agg.compute_cohens_d(baseline)
        results.append(agg); print(f"   {agg.summary_line()}")
    return results


def run_scalability(seeds=SEEDS, smoke=False) -> List[AggregatedResult]:
    mods, info = _import_project(), _load_contract_info()
    seeds_run = seeds[:2] if smoke else seeds
    print(f"\n{'='*72}\nSCALABILITY  (Full System)\n{'='*72}")
    results = []; baseline = None
    for sc in SCALABILITY_CONFIGS:
        K, T = sc["num_clients"], sc["num_rounds"]
        T_run = 1 if smoke else T; name = f"K={K:>2} T={T_run}"
        print(f"\n▶  {name}")
        agg = _multi_seed(name, K, T_run, _FULL_CFG, mods, info, seeds_run)
        if baseline is None: baseline = agg
        else: agg.compute_cohens_d(baseline)
        results.append(agg); print(f"   {agg.summary_line()}")
    return results


def run_privacy_tradeoff(K=3, T=10, seeds=SEEDS, smoke=False) -> List[AggregatedResult]:
    mods, info = _import_project(), _load_contract_info()
    T_run = 1 if smoke else T; seeds_run = seeds[:2] if smoke else seeds
    print(f"\n{'='*72}\nPRIVACY TRADE-OFF  ε sweep  K={K} T={T_run}\n{'='*72}")
    results = []; baseline = None
    for eps in EPSILON_VALUES:
        cfg = {**_FULL_CFG, "epsilon": eps}; name = f"Full System ε={eps}"
        print(f"\n▶  {name}")
        agg = _multi_seed(name, K, T_run, cfg, mods, info, seeds_run)
        if baseline is None: baseline = agg
        else: agg.compute_cohens_d(baseline)
        results.append(agg); print(f"   {agg.summary_line()}")
    return results


def run_attacks(K=5, T=10, seeds=SEEDS, smoke=False) -> List[AggregatedResult]:
    mods, info = _import_project(), _load_contract_info()
    T_run = 1 if smoke else T; seeds_run = seeds[:2] if smoke else seeds
    print(f"\n{'='*72}\nATTACK ROBUSTNESS  K={K} T={T_run}\n{'='*72}")
    results = []; baseline = None
    for atk in ATTACK_SCENARIOS:
        cfg = {**_FULL_CFG, "attack_type": atk["attack_type"],
               "attack_fraction": atk["attack_fraction"],
               "attack_strength": atk["attack_strength"]}
        name = atk["name"]; print(f"\n▶  {name}")
        agg = _multi_seed(name, K, T_run, cfg, mods, info, seeds_run)
        if baseline is None: baseline = agg
        else: agg.compute_cohens_d(baseline)
        results.append(agg); print(f"   {agg.summary_line()}")
    if results:
        b_r = results[0].reward_mean
        print(f"\n{'─'*72}\nREWARD DEGRADATION vs No Attack")
        for agg in results[1:]:
            d = (agg.reward_mean - b_r) / (abs(b_r) + 1e-9) * 100
            print(f"  {agg.name:<34} Δ={d:>+7.2f}%  d={agg.cohens_d:>+.3f}")
    return results


def save_results(results, filename):
    with open(filename, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    print(f"\n💾 Saved → {filename}")


def print_table(results, title):
    print(f"\n{'='*80}\n  {title}\n{'='*80}")
    print(f"{'Config':<34} K  T  n | {'Reward±std':>16}  CI±  | BC(s) IPFS(KB) ε      d")
    print("─" * 80)
    for r in results:
        m = "✅" if not r.error_runs else f"⚠️{r.error_runs}"
        print(f"{m} {r.name:<32} {r.num_clients:>2} {r.num_rounds:>2} {r.n_runs:>2}"
              f" | {r.reward_mean:>7.4f}±{r.reward_std:>6.4f} ±{r.reward_ci95:.4f}"
              f" | {r.bc_mean:>5.2f} {r.ipfs_mean:>7.1f} {r.eps_total:>6.4f} {r.cohens_d:>+.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ablation","scalability","privacy","attacks","all"], default="all")
    parser.add_argument("--smoke",           action="store_true")
    parser.add_argument("--clients",         type=int,   default=None)
    parser.add_argument("--rounds",          type=int,   default=None)
    parser.add_argument("--dp_eps",          type=float, default=1.0)
    parser.add_argument("--attack",          type=str,   default="none")
    parser.add_argument("--attack_frac",     type=float, default=0.2)
    parser.add_argument("--attack_strength", type=float, default=1.0)
    parser.add_argument("--seed",            type=int,   default=None)
    args = parser.parse_args()

    # Single-run CLI mode
    if args.seed is not None or args.attack != "none":
        mods, info = _import_project(), _load_contract_info()
        seeds = [args.seed] if args.seed else [42]
        K = args.clients or 3; T = args.rounds or 10
        cfg = {**_FULL_CFG, "epsilon": args.dp_eps,
               "attack_type": args.attack, "attack_fraction": args.attack_frac,
               "attack_strength": args.attack_strength}
        print(f"\n🔬 Single run: K={K} T={T} ε={args.dp_eps} attack={args.attack}")
        agg = _multi_seed(f"custom_{args.attack}", K, T, cfg, mods, info, seeds)
        print_table([agg], "RESULT"); save_results([agg], "single_run_result.json")
        return

    all_results = []
    if args.mode in ("ablation","all"):
        res = run_ablation(K=args.clients or 3, T=args.rounds or 10, seeds=SEEDS, smoke=args.smoke)
        all_results.extend(res); print_table(res, "ABLATION"); save_results(res, "ablation_results.json")
    if args.mode in ("scalability","all"):
        res = run_scalability(seeds=SEEDS, smoke=args.smoke)
        all_results.extend(res); print_table(res, "SCALABILITY"); save_results(res, "scalability_results.json")
    if args.mode in ("privacy","all"):
        res = run_privacy_tradeoff(K=args.clients or 3, T=args.rounds or 10, seeds=SEEDS, smoke=args.smoke)
        all_results.extend(res); print_table(res, "PRIVACY TRADE-OFF"); save_results(res, "privacy_results.json")
    if args.mode in ("attacks","all"):
        res = run_attacks(K=args.clients or 5, T=args.rounds or 10, seeds=SEEDS, smoke=args.smoke)
        all_results.extend(res); print_table(res, "ATTACK ROBUSTNESS"); save_results(res, "attack_results.json")
    if args.mode == "all":
        save_results(all_results, "all_experiment_results.json")
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
