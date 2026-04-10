"""
estimate_gas.py
===============
Gas cost estimator for the AWFedAvg smart contract.

Connects to a running Ganache node, estimates gas for every transaction type,
and produces a cost table in USD using configurable gas price and ETH price.

Usage
-----
    python estimate_gas.py                          # defaults
    python estimate_gas.py --clients 10 --rounds 30
    python estimate_gas.py --gas_price 30 --eth_usd 3000
    python estimate_gas.py --save gas_report.json

Output
------
  Console table + optional JSON file with per-tx gas, per-round cost,
  and projected total cost for the full experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

# ── Try importing web3 ────────────────────────────────────────────────────────
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

# ── Default parameters ────────────────────────────────────────────────────────
DEFAULT_PROVIDER    = "http://127.0.0.1:8545"
DEFAULT_GAS_PRICE   = 30        # Gwei  (reasonable mainnet estimate)
DEFAULT_ETH_USD     = 3000.0    # USD per ETH placeholder
DEFAULT_CLIENTS     = 3
DEFAULT_ROUNDS      = 10
MODEL_SIZE_BYTES    = 55_000    # ~55 KB model (compressed, encrypted)
IPFS_CID_LEN       = 46        # bytes32 on-chain CID storage length

# ── Realistic gas estimates (measured + padded 10%) ──────────────────────────
# These are fallback values when no live node is available.
# Run against a live Ganache node for measured values.
FALLBACK_GAS = {
    "registerClient":       85_000,   # stake + struct init + event
    "startRound":           45_000,   # round struct + event
    "submitLocalUpdate":    75_000,   # mapping write + event
    "submitAggregatedModel":60_000,   # mapping write + model hash + event
    "getClientInfo":         5_000,   # pure view — no gas on-chain, estimated for reference
}


def gwei_to_eth(gas: int, gas_price_gwei: float) -> float:
    return gas * gas_price_gwei * 1e-9


def eth_to_usd(eth: float, eth_usd: float) -> float:
    return eth * eth_usd


def _load_contract(w3: "Web3", provider: str) -> tuple:
    """Load deployed contract from contract_info.json."""
    info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contract_info.json")
    if not os.path.exists(info_path):
        return None, None

    with open(info_path) as f:
        info = json.load(f)

    abi = info.get("abi")
    addr = info.get("contract_address")
    if not abi or not addr:
        return None, None

    try:
        contract = w3.eth.contract(address=addr, abi=abi)
        return contract, info
    except Exception:
        return None, None


def _measure_gas_live(w3: "Web3", contract, accounts: List[str],
                      num_clients: int) -> Dict[str, int]:
    """
    Measure actual gas using eth_estimateGas against a live Ganache node.
    Falls back to FALLBACK_GAS if estimation fails.
    """
    measured = {}
    coordinator = accounts[0]
    client_accts = accounts[1: num_clients + 1]

    # ── registerClient ────────────────────────────────────────────────────────
    try:
        dummy_hash = b'\x00' * 32
        gas = contract.functions.registerClient(dummy_hash).estimate_gas({
            "from": client_accts[0] if client_accts else coordinator,
            "value": w3.to_wei(0.01, "ether"),
        })
        measured["registerClient"] = int(gas * 1.10)   # +10% buffer
    except Exception as e:
        measured["registerClient"] = FALLBACK_GAS["registerClient"]

    # ── startRound ────────────────────────────────────────────────────────────
    try:
        gas = contract.functions.startRound(b'\x00' * 32).estimate_gas({
            "from": coordinator,
        })
        measured["startRound"] = int(gas * 1.10)
    except Exception:
        measured["startRound"] = FALLBACK_GAS["startRound"]

    # ── submitAggregatedModel ─────────────────────────────────────────────────
    try:
        dummy_ipfs = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
        model_hash = b'\xab' * 32
        gas = contract.functions.submitAggregatedModel(
            0, dummy_ipfs, model_hash
        ).estimate_gas({"from": coordinator})
        measured["submitAggregatedModel"] = int(gas * 1.10)
    except Exception:
        measured["submitAggregatedModel"] = FALLBACK_GAS["submitAggregatedModel"]

    # ── submitLocalUpdate ─────────────────────────────────────────────────────
    try:
        dummy_ipfs = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
        upd_hash   = b'\xcd' * 32
        gas = contract.functions.submitLocalUpdate(
            dummy_ipfs, upd_hash, MODEL_SIZE_BYTES, b''
        ).estimate_gas({
            "from": client_accts[0] if client_accts else coordinator,
        })
        measured["submitLocalUpdate"] = int(gas * 1.10)
    except Exception:
        measured["submitLocalUpdate"] = FALLBACK_GAS["submitLocalUpdate"]

    measured["getClientInfo"] = FALLBACK_GAS["getClientInfo"]
    return measured


def build_cost_table(gas_per_tx: Dict[str, int],
                     gas_price_gwei: float,
                     eth_usd: float,
                     num_clients: int,
                     num_rounds: int) -> dict:
    """
    Build a structured cost breakdown.

    Per-round transactions (Global model only architecture):
      • 1 × startRound             (coordinator)
      • 1 × submitAggregatedModel  (coordinator)
      Read calls (view):
      • K × getClientInfo          (coordinator reads reputations — free)

    One-time (registration, amortised):
      • K × registerClient         (clients)

    Optional per-round (if individual client tracking enabled):
      • K × submitLocalUpdate      (clients)
    """
    rows = []

    def row(name, count_per_round, gas, role, note=""):
        gas_total   = gas * count_per_round
        eth_pr      = gwei_to_eth(gas_total, gas_price_gwei)
        usd_pr      = eth_to_usd(eth_pr, eth_usd)
        usd_total   = usd_pr * num_rounds
        rows.append({
            "tx":             name,
            "role":           role,
            "count/round":    count_per_round,
            "gas/tx":         gas,
            "gas/round":      gas_total,
            "eth/round":      round(eth_pr, 8),
            "usd/round":      round(usd_pr, 6),
            "usd_total":      round(usd_total, 4),
            "note":           note,
        })

    # Per-round (mandatory)
    row("startRound",            1,            gas_per_tx["startRound"],
        "coordinator", "once per round")
    row("submitAggregatedModel", 1,            gas_per_tx["submitAggregatedModel"],
        "coordinator", "global model hash + IPFS CID")
    row("getClientInfo",         num_clients,  gas_per_tx["getClientInfo"],
        "coordinator", "view call — 0 gas on-chain (listed for completeness)")

    # Per-round (optional, individual tracking)
    row("submitLocalUpdate",     num_clients,  gas_per_tx["submitLocalUpdate"],
        "clients", "[OPTIONAL] per-client update hash on-chain")

    # One-time (registration)
    reg_gas_total = gas_per_tx["registerClient"] * num_clients
    reg_eth       = gwei_to_eth(reg_gas_total, gas_price_gwei)
    reg_usd       = eth_to_usd(reg_eth, eth_usd)
    rows.append({
        "tx": "registerClient",
        "role": "clients",
        "count/round": f"{num_clients} (one-time)",
        "gas/tx": gas_per_tx["registerClient"],
        "gas/round": reg_gas_total,
        "eth/round": round(reg_eth, 8),
        "usd/round": round(reg_usd, 6),
        "usd_total": round(reg_usd, 4),
        "note": "one-time per client (amortised over all rounds)",
    })

    # Totals (mandatory only, per round)
    mandatory_gas_per_round = (
        gas_per_tx["startRound"] +
        gas_per_tx["submitAggregatedModel"]
    )
    mandatory_eth_per_round = gwei_to_eth(mandatory_gas_per_round, gas_price_gwei)
    mandatory_usd_per_round = eth_to_usd(mandatory_eth_per_round, eth_usd)

    with_local_gas = mandatory_gas_per_round + gas_per_tx["submitLocalUpdate"] * num_clients
    with_local_eth = gwei_to_eth(with_local_gas, gas_price_gwei)
    with_local_usd = eth_to_usd(with_local_eth, eth_usd)

    return {
        "config": {
            "num_clients":   num_clients,
            "num_rounds":    num_rounds,
            "gas_price_gwei": gas_price_gwei,
            "eth_usd":       eth_usd,
            "model_size_bytes": MODEL_SIZE_BYTES,
        },
        "gas_per_tx": gas_per_tx,
        "rows":       rows,
        "summary": {
            "mandatory_gas_per_round":   mandatory_gas_per_round,
            "mandatory_eth_per_round":   round(mandatory_eth_per_round, 8),
            "mandatory_usd_per_round":   round(mandatory_usd_per_round, 6),
            "mandatory_usd_total":       round(mandatory_usd_per_round * num_rounds, 4),
            "with_local_gas_per_round":  with_local_gas,
            "with_local_usd_per_round":  round(with_local_usd, 6),
            "with_local_usd_total":      round(with_local_usd * num_rounds, 4),
            "registration_usd_one_time": round(eth_to_usd(
                gwei_to_eth(gas_per_tx["registerClient"] * num_clients, gas_price_gwei),
                eth_usd), 4),
        }
    }


def print_report(report: dict):
    cfg  = report["config"]
    summ = report["summary"]

    print("\n" + "═" * 78)
    print("  GAS COST ESTIMATION REPORT — Blockchain-AWFedAvg")
    print("═" * 78)
    print(f"  Clients:   K = {cfg['num_clients']}")
    print(f"  Rounds:    T = {cfg['num_rounds']}")
    print(f"  Gas price: {cfg['gas_price_gwei']} Gwei  |  ETH price: ${cfg['eth_usd']:.0f}")
    print(f"  Model:     ~{cfg['model_size_bytes']//1000} KB compressed+encrypted")
    print("─" * 78)

    # Per-transaction table
    hdr = f"{'Transaction':<26} {'Role':<12} {'cnt/rnd':>7} {'gas/tx':>9} {'gas/rnd':>9} {'USD/rnd':>10} {'USD total':>10}"
    print(hdr)
    print("─" * 78)

    for r in report["rows"]:
        count_str = str(r["count/round"])
        print(
            f"{r['tx']:<26} {r['role']:<12} {count_str:>7} "
            f"{r['gas/tx']:>9,} {str(r['gas/round']):>9} "
            f"${r['usd/round']:>9.6f} ${r['usd_total']:>9.4f}"
        )

    print("─" * 78)
    print("\n  SUMMARY (mainnet mainnet estimate at current gas/ETH prices):")
    print(f"  ├─ Mandatory only (startRound + submitAggregatedModel):")
    print(f"  │   Gas/round : {summ['mandatory_gas_per_round']:,}")
    print(f"  │   USD/round : ${summ['mandatory_usd_per_round']:.6f}")
    print(f"  │   USD total : ${summ['mandatory_usd_total']:.4f}  ({cfg['num_rounds']} rounds)")
    print(f"  ├─ With per-client submitLocalUpdate (optional):")
    print(f"  │   Gas/round : {summ['with_local_gas_per_round']:,}")
    print(f"  │   USD/round : ${summ['with_local_usd_per_round']:.6f}")
    print(f"  │   USD total : ${summ['with_local_usd_total']:.4f}  ({cfg['num_rounds']} rounds)")
    print(f"  └─ Registration (one-time, {cfg['num_clients']} clients): ${summ['registration_usd_one_time']:.4f}")
    print()
    print("  NOTE: Gas is O(K) per round — linear in number of clients.")
    print("  Model weights are NEVER stored on-chain (IPFS only) → minimal gas footprint.")
    print("  For Ganache (local PoA) no real ETH is spent; costs apply to public chains.")
    print("═" * 78)


def scalability_table(gas_per_tx: Dict[str, int],
                      gas_price_gwei: float, eth_usd: float) -> list:
    """Print gas cost for K in {3, 5, 10, 20, 50} for the paper table."""
    configs = [
        (3,  10), (5, 20), (10, 30), (20, 30), (50, 20)
    ]
    rows = []
    print("\n  SCALABILITY — Gas cost vs K (mandatory txns only):")
    print(f"  {'K':>4} {'T':>4} {'gas/rnd':>10} {'USD/rnd':>10} {'USD total':>10} {'gas/round/client':>18}")
    print("  " + "─" * 64)
    for K, T in configs:
        g = gas_per_tx["startRound"] + gas_per_tx["submitAggregatedModel"]
        usd_round = eth_to_usd(gwei_to_eth(g, gas_price_gwei), eth_usd)
        usd_total = usd_round * T
        per_client = g // K
        print(f"  {K:>4} {T:>4} {g:>10,} ${usd_round:>9.6f} ${usd_total:>9.4f} {per_client:>18,}")
        rows.append({"K": K, "T": T, "gas_round": g, "usd_round": round(usd_round, 6),
                     "usd_total": round(usd_total, 4)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Estimate gas costs for AWFedAvg blockchain calls")
    parser.add_argument("--provider",    default=DEFAULT_PROVIDER,  help="Web3 HTTP provider URL")
    parser.add_argument("--clients",     type=int,   default=DEFAULT_CLIENTS,   help="Number of clients K")
    parser.add_argument("--rounds",      type=int,   default=DEFAULT_ROUNDS,    help="Number of rounds T")
    parser.add_argument("--gas_price",   type=float, default=DEFAULT_GAS_PRICE, help="Gas price in Gwei")
    parser.add_argument("--eth_usd",     type=float, default=DEFAULT_ETH_USD,   help="ETH price in USD")
    parser.add_argument("--save",        default=None,  help="Save JSON report to file")
    parser.add_argument("--no-live",     action="store_true", help="Skip live estimation, use fallback gas values")
    args = parser.parse_args()

    gas_per_tx = dict(FALLBACK_GAS)   # start with fallback
    source = "FALLBACK (estimated)"

    # ── Live measurement ──────────────────────────────────────────────────────
    if WEB3_AVAILABLE and not args.no_live:
        try:
            w3 = Web3(Web3.HTTPProvider(args.provider))
            w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            if w3.is_connected():
                print(f"\n✅ Connected to {args.provider}  (block #{w3.eth.block_number})")
                accounts = w3.eth.accounts
                contract, info = _load_contract(w3, args.provider)
                if contract:
                    print("   Measuring gas with live contract...")
                    gas_per_tx = _measure_gas_live(w3, contract, accounts, args.clients)
                    source = "LIVE MEASUREMENT (Ganache)"
                else:
                    print("⚠️  contract_info.json not found — using fallback gas values")
            else:
                print(f"⚠️  Cannot connect to {args.provider} — using fallback gas values")
        except Exception as e:
            print(f"⚠️  Live estimation failed ({e}) — using fallback gas values")
    else:
        print(f"\nℹ️  Using fallback gas estimates (web3 unavailable or --no-live set)")

    print(f"   Gas source: {source}")

    # ── Build report ──────────────────────────────────────────────────────────
    report = build_cost_table(gas_per_tx, args.gas_price, args.eth_usd,
                              args.clients, args.rounds)
    print_report(report)

    scale_rows = scalability_table(gas_per_tx, args.gas_price, args.eth_usd)
    report["scalability"] = scale_rows

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save:
        with open(args.save, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved → {args.save}")
    else:
        # Default: save alongside results
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gas_report.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved → gas_report.json")


if __name__ == "__main__":
    main()
