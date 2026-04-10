"""
deploy.py
=========
One-shot deployment script. Run this ONCE before starting the experiment.

Prerequisites
-------------
1. Ganache running:
       ganache --accounts 10 --deterministic --networkId 1337 --port 8545
2. IPFS daemon running:
       ipfs daemon
3. Dependencies installed:
       pip install -r requirements.txt

Usage
-----
    python deploy.py

Output
------
contract_info.json  — written to the project root. Contains:
    • contract_address      – deployed contract address
    • contract_abi          – full ABI (for web3 calls)
    • coordinator_address   – Ganache account[0]
    • coordinator_private_key
    • clients               – list of {address, private_key} for accounts[1..3]
"""

import sys
import json
import time
import os

# ── Third-party ──────────────────────────────────────────────────────────────
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
except ImportError:
    sys.exit("❌  web3 not installed. Run: pip install -r requirements.txt")

try:
    from solcx import compile_source, install_solc, get_installed_solc_versions
except ImportError:
    sys.exit("❌  py-solc-x not installed. Run: pip install -r requirements.txt")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GANACHE_URL   = "http://127.0.0.1:8545"
SOLC_VERSION  = "0.8.20"
CONTRACT_FILE = os.path.join(os.path.dirname(__file__), "contracts", "FederatedLearningContract.sol")
OUTPUT_FILE   = os.path.join(os.path.dirname(__file__), "contract_info.json")
IPFS_ADDR     = "/ip4/127.0.0.1/tcp/5001"

# Contract constructor args (must match NUM_CLIENTS in adaptive_weighted_fedavg.py)
MIN_CLIENTS   = 3
MAX_CLIENTS   = 3
ROUND_TIMEOUT = 7200   # seconds — 2 hours, plenty for simulation

# Ganache accounts are fetched live from the node in get_ganache_accounts().
# This works regardless of whether --deterministic was used.
GANACHE_ACCOUNTS = []   # populated at runtime by get_ganache_accounts()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _raw_tx(signed):
    """Return raw signed transaction bytes — works with both web3.py 5.x and 6.x."""
    return getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)


def connect_ganache() -> Web3:
    print(f"🔌 Connecting to Ganache at {GANACHE_URL} …")
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    for attempt in range(10):
        if w3.is_connected():
            print(f"   ✅ Connected  (chainId={w3.eth.chain_id})")
            return w3
        print(f"   ⏳ Waiting for Ganache… ({attempt+1}/10)")
        time.sleep(2)
    sys.exit(
        "❌  Cannot connect to Ganache.\n"
        "    Start it with:\n"
        "        ganache --accounts 10 --networkId 1337 --port 8545"
    )


def get_ganache_accounts(w3: Web3) -> list:
    """
    Fetch Ganache accounts.  Private keys are NOT required — all transactions
    use .transact() which Ganache handles automatically for unlocked accounts.
    Returns a list of {address, private_key} dicts (private_key may be None).
    """
    global GANACHE_ACCOUNTS

    addresses = w3.eth.accounts[:4]
    if len(addresses) < 4:
        sys.exit(f"❌  Need at least 4 Ganache accounts, found {len(addresses)}.")

    accounts = [{"address": addr, "private_key": None} for addr in addresses]

    # Verify coordinator has ETH
    bal = w3.eth.get_balance(accounts[0]["address"])
    if bal == 0:
        sys.exit(f"❌  Coordinator {accounts[0]['address']} has 0 ETH. "
                 "Is Ganache running?")

    GANACHE_ACCOUNTS = accounts
    print(f"   💳 Coordinator : {accounts[0]['address']}  "
          f"({w3.from_wei(bal, 'ether'):.1f} ETH)")
    for i, a in enumerate(accounts[1:], 1):
        b = w3.eth.get_balance(a["address"])
        print(f"   💳 Client {i-1}     : {a['address']}  "
              f"({w3.from_wei(b, 'ether'):.1f} ETH)")
    return accounts


def ensure_solc():
    print(f"\n🔧 Checking Solidity compiler (solc {SOLC_VERSION}) …")
    installed = [str(v) for v in get_installed_solc_versions()]
    if SOLC_VERSION not in installed:
        print(f"   ⬇️  Installing solc {SOLC_VERSION} (one-time download) …")
        install_solc(SOLC_VERSION)
    print(f"   ✅ solc {SOLC_VERSION} ready")


def compile_contract() -> dict:
    print(f"\n📝 Compiling {os.path.basename(CONTRACT_FILE)} …")
    with open(CONTRACT_FILE) as f:
        source = f.read()

    compiled = compile_source(
        source,
        output_values=["abi", "bin"],
        solc_version=SOLC_VERSION,
    )
    key = "<stdin>:FederatedLearningContract"
    if key not in compiled:
        # py-solc-x may use filename as key
        key = [k for k in compiled if "FederatedLearningContract" in k][0]

    iface = compiled[key]
    print(f"   ✅ Compiled  (bytecode {len(iface['bin'])//2} bytes)")
    return iface


def deploy_contract(w3: Web3, iface: dict) -> str:
    if not GANACHE_ACCOUNTS:
        sys.exit("❌  GANACHE_ACCOUNTS is empty — call get_ganache_accounts(w3) first.")
    coordinator = GANACHE_ACCOUNTS[0]
    from_addr   = coordinator["address"]

    print(f"\n🚀 Deploying contract …")
    print(f"   Coordinator : {from_addr}")
    print(f"   minClients={MIN_CLIENTS}, maxClients={MAX_CLIENTS}, roundTimeout={ROUND_TIMEOUT}s")

    Contract = w3.eth.contract(abi=iface["abi"], bytecode=iface["bin"])

    # Ganache auto-unlocks all its accounts, so we can use send_transaction
    # directly without signing — no private key needed for deployment.
    tx_hash = Contract.constructor(MIN_CLIENTS, MAX_CLIENTS, ROUND_TIMEOUT).transact({
        "from": from_addr,
        "gas":  3_000_000,
    })
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    address = receipt.contractAddress
    print(f"   ✅ Contract deployed at {address}  (tx={tx_hash.hex()[:16]}…)")
    return address


def check_ipfs() -> bool:
    print(f"\n📡 Checking IPFS daemon at {IPFS_ADDR} …")
    try:
        import ipfshttpclient
        client = ipfshttpclient.connect(IPFS_ADDR)
        node_id = client.id()["ID"]
        print(f"   ✅ IPFS connected  (peer={node_id[:20]}…)")
        return True
    except Exception as e:
        print(f"   ⚠️  IPFS not reachable: {e}")
        print(     "   Start it with:  ipfs daemon")
        print(     "   The experiment will skip IPFS uploads until the daemon is running.")
        return False


def write_config(contract_address: str, abi: list):
    coordinator = GANACHE_ACCOUNTS[0]
    clients     = [
        {"address": a["address"], "private_key": a["private_key"]}
        for a in GANACHE_ACCOUNTS[1:]          # accounts 1, 2, 3 → clients 0, 1, 2
    ]

    config = {
        "contract_address":         contract_address,
        "contract_abi":             abi,           # stored inline for convenience
        "coordinator_address":      coordinator["address"],
        "coordinator_private_key":  coordinator["private_key"],
        "clients":                  clients,
        "blockchain_provider":      GANACHE_URL,
        "ipfs_addr":                IPFS_ADDR,
        "deployed_at":              time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n💾 Config saved to {OUTPUT_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# PATCH contract_info.json to include ABI in the format privacy_blockchain_fl
# expects (it calls json.load(contract_abi_path) and passes abi to w3.eth.contract)
# privacy_blockchain_fl.py loads the JSON file and expects a top-level 'abi' key
# ─────────────────────────────────────────────────────────────────────────────
# Nothing to patch — we write the ABI under "contract_abi" (for humans) AND
# under "abi" so privacy_blockchain_fl.py's existing code works directly.
# See the write_config function: we add both keys.

def write_config_dual_key(contract_address: str, abi: list):
    """Write config with 'abi' key for privacy_blockchain_fl.py."""
    coordinator = GANACHE_ACCOUNTS[0]
    clients     = [
        {"address": a["address"], "private_key": a.get("private_key")}
        for a in GANACHE_ACCOUNTS[1:]         # accounts 1-3 → FL clients 0-2
    ]

    config = {
        # privacy_blockchain_fl.py does: json.load(contract_abi_path) → passes to w3.eth.contract
        # It expects the top-level object to be the ABI list, OR a dict with 'abi' key.
        # We store the full config here and privacy_blockchain_fl already handles dict with 'abi'.
        "abi":                      abi,
        "contract_address":         contract_address,
        "coordinator_address":      coordinator["address"],
        "coordinator_private_key":  coordinator["private_key"],
        "clients":                  clients,
        "blockchain_provider":      GANACHE_URL,
        "ipfs_addr":                IPFS_ADDR,
        "deployed_at":              time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n💾 contract_info.json written to: {OUTPUT_FILE}")
    print(f"   Contract  : {contract_address}")
    print(f"   Coordinator: {coordinator['address']}")
    for i, c in enumerate(clients):
        print(f"   Client {i}  : {c['address']}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Blockchain-AWFedAvg  —  Deploy Script")
    print("=" * 60)

    # 1. Connect to Ganache and load accounts
    w3 = connect_ganache()
    print("")
    get_ganache_accounts(w3)

    # 2. Ensure Solidity compiler is available
    ensure_solc()

    # 3. Compile contract
    iface = compile_contract()

    # 4. Deploy
    address = deploy_contract(w3, iface)

    # 5. Check IPFS (non-fatal)
    check_ipfs()

    # 6. Write contract_info.json
    write_config_dual_key(address, iface["abi"])

    print("\n" + "=" * 60)
    print("✅  Deployment complete!")
    print("")
    print("Next step — run the experiment:")
    print("    python blockchain_awfedavg_true_integration.py")
    print("=" * 60)
