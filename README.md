# Blockchain-Enabled Adaptive Weighted FedAvg

Privacy-preserving federated reinforcement learning for wireless network resource allocation (eMBB/URLLC scheduling), combining **AdaptiveWeightedFedAvg** with a **blockchain + IPFS** audit and privacy layer.

---

## Project Structure

```
blockchain_awfedavg/
│
├── blockchain_awfedavg_true_integration.py   ← MAIN ENTRY POINT
├── adaptive_weighted_fedavg.py               ← AWFedAvg strategy + Flower client
├── privacy_blockchain_fl.py                  ← DP + encryption + IPFS + blockchain
├── gym_phy_env.py                            ← Gymnasium wrapper for Phy environment
├── phy_env_class.py                          ← Core wireless network simulator
│
├── phy/                                      ← Internal package (auto-discovered)
│   ├── common/
│   │   ├── common_dict.py                    ← Node/traffic type definitions
│   │   └── common_method.py                  ← Fading, noise, geometry helpers
│   └── scenario/
│       ├── resources.py                      ← 5G-NR resource block definitions
│       ├── nodes.py                          ← Base station & user node classes
│       ├── cells.py                          ← Cell class
│       ├── cluster.py                        ← Multi-cell cluster + channel model
│       └── waterfilling.py                   ← Water-filling resource allocation
│
├── contracts/
│   └── FederatedLearningContract.sol         ← Solidity smart contract
│
└── requirements.txt
```

**Files NOT included** (removed as superseded):
- `blockchain_awfedavg_integration.py` — old stub that never used real AWFedAvg
- `complete_blockchain_fl_example.py`  — old example with commented-out imports

---

## Architecture

```
BlockchainAdaptiveWeightedFedAvg
        │  inherits
        ▼
AdaptiveWeightedFedAvg          (4-criteria adaptive weights: eMBB, URLLC,
        │                        activation diversity, stability)
        │  aggregate_fit() adds:
        │   1. start_round_on_chain()
        │   2. super().aggregate_fit()  ← full AWFedAvg unchanged
        │   3. coordinator DP → encrypt → IPFS → submit_aggregated_model_on_chain()
        ▼

BlockchainEnhancedFlowerClient
        │  inherits
        ▼
EnhancedFlowerClient            (PPO training, ResourceMonitor, evaluate_model_simple)
        │  fit() adds:
        │   1. super().fit()  ← full client training unchanged
        │   2. client DP → clip_gradients → encrypt → IPFS → submit_local_update_on_chain()
        ▼
```

DP-noised parameters go to IPFS only. Flower receives the original parameters so AWFedAvg's adaptive weighting stays accurate.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start infrastructure (for blockchain + IPFS)

```bash
# Local Ethereum node
ganache-cli --accounts 10 --deterministic --networkId 1337

# IPFS daemon
ipfs daemon &
```

### 3. Deploy smart contract

```bash
npm install --save-dev hardhat
npx hardhat init
cp contracts/FederatedLearningContract.sol contracts/
npx hardhat run scripts/deploy.js --network localhost
```

### 4. Run experiment

**Without blockchain** (local testing, DP still active):
```python
from blockchain_awfedavg_true_integration import run_blockchain_awfedavg_experiment

hist, local, fed, resources = run_blockchain_awfedavg_experiment(
    blockchain_enabled=False,
    epsilon=1.0, delta=1e-5, clip_norm=1.0,
)
```

**With blockchain**:
```python
hist, local, fed, resources = run_blockchain_awfedavg_experiment(
    blockchain_enabled=True,
    blockchain_provider="http://localhost:8545",
    contract_address="0x...",
    contract_abi_path="contract_info.json",
    coordinator_private_key="0x...",
    epsilon=1.0, delta=1e-5, clip_norm=1.0,
    alpha_embb=0.3, alpha_urllc=0.3,
    alpha_activation=0.2, alpha_stability=0.2,
)
```

Or use the factory directly for full control:
```python
from blockchain_awfedavg_true_integration import (
    create_blockchain_awfedavg_strategy,
    make_blockchain_client_fn,
)

strategy = create_blockchain_awfedavg_strategy(
    blockchain_enabled=True,
    contract_address="0x...",
    coordinator_private_key="0x...",
)
client_fn = make_blockchain_client_fn(strategy.ppfl)
```

---

## AWFedAvg Weight Criteria

| Criterion | Default α | Meaning |
|---|---|---|
| eMBB outage | 0.30 | Clients with fewer eMBB outages get higher weight |
| URLLC residual | 0.30 | Clients with lower undelivered URLLC packets get higher weight |
| Activation diversity | 0.20 | Clients with more diverse traffic loads contribute more |
| Performance stability | 0.20 | Clients with stable reward history get higher weight |

Weights are smoothed across rounds (factor 0.7) to prevent oscillation.

---

## Privacy Guarantees

- **Client-side**: gradient clipping + Gaussian noise `(ε, δ)-DP` applied before IPFS upload
- **Coordinator-side**: additional DP noise on the aggregated global model before publishing
- **Encryption**: Fernet symmetric encryption with per-upload keys
- **Blockchain**: SHA-256 model hash registered on-chain for tamper detection

---

## Key Configuration

| Parameter | Default | Description |
|---|---|---|
| `epsilon` | 1.0 | DP privacy budget (lower = more private) |
| `delta` | 1e-5 | DP failure probability |
| `clip_norm` | 1.0 | L2 gradient clipping bound |
| `NUM_CLIENTS` | 3 | Total simulated FL clients |
| `TOTAL_ROUNDS` | 10 | Federated learning rounds |
| `LOCAL_EPOCHS` | 100 | PPO timestep multiplier per local round |
