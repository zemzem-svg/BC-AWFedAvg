#!/usr/bin/env python3
"""
Deploy the BC-AWFedAvg governance contract to a local Ganache/Ethereum node.

This script is intentionally limited to deployment and configuration generation.
It does not start the federated-learning experiment.

Expected contract:
    contracts/FederatedLearningContract.sol

Constructor:
    (minClientsPerRound, maxClientsPerRound, roundTimeout)

The generated contract_info.json contains public addresses and ABI metadata.
Private keys are NOT written by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from solcx import compile_standard, get_installed_solc_versions, install_solc, set_solc_version
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: py-solc-x. Install with: pip install py-solc-x"
    ) from exc

try:
    from web3 import Web3
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: web3. Install with: pip install web3"
    ) from exc


DEFAULT_RPC = os.getenv("GANACHE_RPC", "http://127.0.0.1:8545")
DEFAULT_SOLC = "0.8.20"
DEFAULT_MIN_CLIENTS = int(os.getenv("MIN_CLIENTS_PER_ROUND", "3"))
DEFAULT_MAX_CLIENTS = int(os.getenv("MAX_CLIENTS_PER_ROUND", str(DEFAULT_MIN_CLIENTS)))
DEFAULT_ROUND_TIMEOUT = int(os.getenv("ROUND_TIMEOUT", "600"))
DEFAULT_NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", str(DEFAULT_MAX_CLIENTS)))
DEFAULT_OUTPUT = Path("contract_info.json")
DEFAULT_CONTRACT = Path("contracts/FederatedLearningContract.sol")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy BC-AWFedAvg FederatedLearningContract."
    )
    parser.add_argument("--rpc", default=DEFAULT_RPC, help="Ethereum JSON-RPC endpoint.")
    parser.add_argument("--solc", default=DEFAULT_SOLC, help="Solidity compiler version.")
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
        help="Solidity source file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Deployment metadata JSON output.",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=DEFAULT_MIN_CLIENTS,
        help="Minimum contributors required per round.",
    )
    parser.add_argument(
        "--max-clients",
        type=int,
        default=DEFAULT_MAX_CLIENTS,
        help="Maximum contributors allowed per round.",
    )
    parser.add_argument(
        "--round-timeout",
        type=int,
        default=DEFAULT_ROUND_TIMEOUT,
        help="Round timeout in seconds.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=DEFAULT_NUM_CLIENTS,
        help="Number of client addresses to include in contract_info.json.",
    )
    parser.add_argument(
        "--no-install-solc",
        action="store_true",
        help="Do not attempt to install the requested solc version.",
    )
    return parser.parse_args()


def ensure_solc(version: str, allow_install: bool) -> None:
    wanted = version if version.startswith("v") else f"v{version}"
    installed = {str(v) for v in get_installed_solc_versions()}
    if version in installed or wanted in installed:
        set_solc_version(version)
        return

    if not allow_install:
        raise RuntimeError(
            f"solc {version} is not installed. Install it with py-solc-x or omit "
            "--no-install-solc."
        )

    print(f"[deploy] Installing solc {version}...")
    install_solc(version)
    set_solc_version(version)


def read_contract_source(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Contract source not found: {path}")
    source = path.read_text(encoding="utf-8")
    if "contract FederatedLearningContract" not in source:
        raise ValueError(
            f"{path} does not contain FederatedLearningContract."
        )
    return source


def compile_contract(source: str, solc_version: str) -> Tuple[List[Dict[str, Any]], str]:
    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {
                "FederatedLearningContract.sol": {
                    "content": source,
                }
            },
            "settings": {
                "optimizer": {
                    "enabled": True,
                    "runs": 200,
                },
                "outputSelection": {
                    "*": {
                        "*": [
                            "abi",
                            "evm.bytecode.object",
                        ]
                    }
                },
            },
        },
        solc_version=solc_version,
    )

    contracts = compiled.get("contracts", {}).get("FederatedLearningContract.sol", {})
    artifact = contracts.get("FederatedLearningContract")
    if not artifact:
        raise RuntimeError("Compiled contract artifact not found.")

    abi = artifact.get("abi")
    bytecode = artifact.get("evm", {}).get("bytecode", {}).get("object", "")
    if not isinstance(abi, list) or not bytecode:
        raise RuntimeError("Compilation produced an empty ABI or bytecode.")

    required = {
        "startRound",
        "recordContribution",
        "recordContributionsBatch",
        "updateReputation",
        "updateReputationsBatch",
        "submitAggregatedModel",
        "getClientInfo",
        "getModelInfo",
    }
    functions = {
        item.get("name")
        for item in abi
        if item.get("type") == "function"
    }
    missing = sorted(required - functions)
    if missing:
        raise RuntimeError(
            "Compiled ABI is missing required BC-AWFedAvg functions: "
            + ", ".join(missing)
        )

    return abi, bytecode


def connect_web3(rpc: str) -> Web3:
    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 20}))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to Ethereum node at {rpc}")
    return w3


def get_accounts(w3: Web3, num_clients: int) -> Tuple[str, List[str]]:
    accounts = list(w3.eth.accounts)
    required = 1 + num_clients
    if len(accounts) < required:
        raise RuntimeError(
            f"Ganache exposes {len(accounts)} accounts, but {required} are required "
            f"(1 coordinator + {num_clients} clients)."
        )

    coordinator = Web3.to_checksum_address(accounts[0])
    clients = [Web3.to_checksum_address(a) for a in accounts[1 : num_clients + 1]]

    return coordinator, clients


def deploy_contract(
    w3: Web3,
    abi: List[Dict[str, Any]],
    bytecode: str,
    coordinator: str,
    min_clients: int,
    max_clients: int,
    round_timeout: int,
) -> Tuple[str, str]:
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    tx_hash = contract.constructor(
        int(min_clients),
        int(max_clients),
        int(round_timeout),
    ).transact({"from": coordinator})

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    if not receipt or not receipt.contractAddress:
        raise RuntimeError("Deployment transaction did not produce a contract address.")

    address = Web3.to_checksum_address(receipt.contractAddress)
    code = w3.eth.get_code(address)
    if not code or code == b"\x00":
        raise RuntimeError("Deployed contract has no bytecode at its address.")

    return address, tx_hash.hex()


def write_artifact(
    output: Path,
    *,
    w3: Web3,
    contract_path: Path,
    abi: List[Dict[str, Any]],
    contract_address: str,
    deployment_tx: str,
    coordinator: str,
    client_addresses: List[str],
    min_clients: int,
    max_clients: int,
    round_timeout: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    public_clients = [
        {
            "id": i,
            "address": address,
        }
        for i, address in enumerate(client_addresses)
    ]

    data: Dict[str, Any] = {
        "contract_address": contract_address,
        "contract_name": "FederatedLearningContract",
        "contract_source": str(contract_path.as_posix()),
        "abi": abi,
        "contract_abi": abi,
        "rpc_url": w3.provider.endpoint_uri,
        "chain_id": int(w3.eth.chain_id),
        "coordinator_address": coordinator,
        "coordinator_private_key": None,
        "clients": public_clients,
        "transaction_mode": "unlocked_ganache_accounts",
        "constructor": {
            "min_clients": int(min_clients),
            "max_clients": int(max_clients),
            "round_timeout": int(round_timeout),
        },
        "bc_awfedavg": {
            "alpha_embb": 0.22,
            "alpha_urllc": 0.38,
            "alpha_activation": 0.20,
            "alpha_stability": 0.15,
            "alpha_reputation": 0.05,
            "epsilon": 1.0,
            "delta": 1e-5,
            "clip_norm": 1.0,
            "total_rounds": 15,
            "reputation_scale": 1000,
            "initial_reputation": "1000/K",
            "reputation_decay_beta": 0.85,
            "reputation_update": "rho_t = 0.85*rho_(t-1) + 0.15*g_t",
            "reputation_target": "g_t = mean(normalized eMBB, uRLLC, stability scores)",
            "isolation_threshold": "theta_iso = 1/(2K)",
            "isolation_mode": "gradual influence control; crossing theta_iso does not deactivate a client",
            "min_stake_ether": 0.01,
        },
        "deployment": {
            "transaction_hash": deployment_tx,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    output.write_text(
        json.dumps(data, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    if args.min_clients <= 0:
        raise ValueError("--min-clients must be > 0.")
    if args.max_clients < args.min_clients:
        raise ValueError("--max-clients must be >= --min-clients.")
    if args.round_timeout <= 0:
        raise ValueError("--round-timeout must be > 0.")
    if args.num_clients < args.max_clients:
        raise ValueError(
            "--num-clients must be >= --max-clients so the generated client list "
            "can satisfy the round configuration."
        )

    contract_path = args.contract.resolve()
    source = read_contract_source(contract_path)

    ensure_solc(
        args.solc,
        allow_install=not args.no_install_solc,
    )
    abi, bytecode = compile_contract(source, args.solc)

    w3 = connect_web3(args.rpc)
    coordinator, client_addresses = get_accounts(w3, args.num_clients)

    print("=" * 72)
    print("BC-AWFedAvg CONTRACT DEPLOYMENT")
    print("=" * 72)
    print(f"RPC              : {args.rpc}")
    print(f"Chain ID         : {w3.eth.chain_id}")
    print(f"Coordinator      : {coordinator}")
    print(f"Client count     : {len(client_addresses)}")
    print(f"Min contributors : {args.min_clients}")
    print(f"Max contributors : {args.max_clients}")
    print(f"Round timeout    : {args.round_timeout}s")
    print(f"Solidity         : {args.solc}")
    print(f"Contract source  : {contract_path}")
    print("=" * 72)

    address, tx_hash = deploy_contract(
        w3,
        abi,
        bytecode,
        coordinator,
        args.min_clients,
        args.max_clients,
        args.round_timeout,
    )

    write_artifact(
        args.output.resolve(),
        w3=w3,
        contract_path=contract_path,
        abi=abi,
        contract_address=address,
        deployment_tx=tx_hash,
        coordinator=coordinator,
        client_addresses=client_addresses,
        min_clients=args.min_clients,
        max_clients=args.max_clients,
        round_timeout=args.round_timeout,
    )

    deployed = w3.eth.contract(address=address, abi=abi)
    on_chain_coordinator = Web3.to_checksum_address(
        deployed.functions.coordinator().call()
    )
    if on_chain_coordinator.lower() != coordinator.lower():
        raise RuntimeError(
            "Deployment verification failed: on-chain coordinator does not match deployer."
        )

    print("\n✅ Deployment successful")
    print(f"Contract address : {address}")
    print(f"Transaction      : {tx_hash}")
    print(f"Config written   : {args.output.resolve()}")
    print("\nNext:")
    print("  1. Keep contract_info.json local; do not commit private keys.")
    print("  2. Start IPFS/Ganache as configured.")
    print("  3. Run blockchain_awfedavg.py.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"❌ Deployment failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
