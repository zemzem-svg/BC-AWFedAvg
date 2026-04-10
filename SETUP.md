# How to Run on Your PC

## Prerequisites

| Software | Version | Purpose |
|---|---|---|
| Python | 3.9 – 3.11 | Core runtime |
| Node.js | 18+ | Smart contract deployment |
| Git | any | Clone the project |
| IPFS (Kubo) | 0.18+ | Distributed model storage |
| Ganache | 7+ | Local Ethereum blockchain |

---

## Step 1: Clone & Enter the Project

```bash
git clone <your-repo-url> blockchain_awfedavg
cd blockchain_awfedavg
```

Or if you downloaded the zip:

```bash
unzip blockchain_awfedavg_enhanced.zip
cd blockchain_awfedavg
```

---

## Step 2: Create a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows PowerShell
```

---

## Step 3: Install Python Dependencies

**CPU-only (recommended to start):**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**With CUDA GPU support (optional, faster training):**

```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import flwr, torch, numpy, web3, stable_baselines3; print('All packages OK')"
```

---

## Step 4: Install Blockchain Infrastructure

### 4a. Ganache (local Ethereum node)

```bash
npm install -g ganache
```

### 4b. IPFS (Kubo)

Download from https://docs.ipfs.tech/install/command-line/ then:

```bash
ipfs init --profile=lowpower
```

### 4c. Solidity Compiler (for contract deployment)

```bash
pip install py-solc-x
python -c "from solcx import install_solc; install_solc('0.8.19')"
```

---

## Step 5: Choose What to Run

The project has **4 run modes** from simplest to full:

### Mode A: Simulation Only (no blockchain, no IPFS — easiest)

This runs the FL training with the PHY environment, AWFedAvg weighting, and DP noise but skips all blockchain/IPFS infrastructure. Perfect for testing on a laptop.

```bash
python -c "
from blockchain_awfedavg_true_integration import run_blockchain_awfedavg_experiment

hist, local, fed, resources = run_blockchain_awfedavg_experiment(
    blockchain_enabled=False,
    epsilon=1.0,
    delta=1e-5,
    clip_norm=1.0,
)
"
```

### Mode B: Enhanced Simulation (robustness + RDP + sparsification)

Uses all the new enhancement modules, still no blockchain needed:

```bash
python -c "
from enhanced_integration import run_enhanced_experiment

history, strategy, resources = run_enhanced_experiment(
    num_clients=5,
    num_rounds=15,
    blockchain_enabled=False,
    robust_method='trimmed_mean',
    use_rdp=True,
    adaptive_clip=True,
    use_sparsification=True,
    compression_ratio=0.1,
    use_secure_agg=True,
    epsilon=1.0,
    delta=1e-5,
)

strategy.print_enhanced_summary()
"
```

### Mode C: Robustness / QoS / SLA Simulation (lightweight, fast)

Runs the full simulation from the previous analysis — takes about 5 seconds:

```bash
python simulate_robustness_qos_sla.py
```

Output saved to `simulation_results.json`.

### Mode D: Full Blockchain Experiment (requires Ganache + IPFS)

**Terminal 1 — Start Ganache:**

```bash
ganache --accounts 15 --deterministic --networkId 1337 --port 8545 --gasLimit 12000000
```

**Terminal 2 — Start IPFS:**

```bash
ipfs daemon
```

**Terminal 3 — Deploy contract & run:**

```bash
python deploy.py
python -c "
from blockchain_awfedavg_true_integration import run_blockchain_awfedavg_experiment

hist, local, fed, resources = run_blockchain_awfedavg_experiment(
    blockchain_enabled=True,
    blockchain_provider='http://localhost:8545',
    contract_address='<address from deploy.py output>',
    contract_abi_path='contract_info.json',
    coordinator_private_key='<private key from Ganache output>',
    epsilon=1.0,
    delta=1e-5,
)
"
```

### Mode E: Automated Full Pipeline (one command)

The included shell script handles everything automatically:

```bash
chmod +x run_experiment.sh

# Full protocol (ablation + scalability + privacy + attacks)
./run_experiment.sh

# Quick smoke test
./run_experiment.sh --smoke

# Specific experiment mode
./run_experiment.sh --mode ablation
./run_experiment.sh --mode scalability
./run_experiment.sh --mode privacy
./run_experiment.sh --mode attacks

# Custom single run
./run_experiment.sh --clients 10 --rounds 30 --dp_eps 1.0 \
                    --attack byzantine --attack_frac 0.2 --seed 42

# Skip infra if already running
./run_experiment.sh --no-ganache --no-ipfs --mode ablation
```

Results are saved to `results/` and logs to `logs/`.

---

## Step 6: Run the Enhancement Tests

Verify all robustness and efficiency modules work correctly:

```bash
python test_enhancements.py
```

Expected output:

```
  RESULTS: 23/23 tests passed
  ✅ ALL ENHANCEMENTS VALIDATED SUCCESSFULLY
```

---

## Step 7: Generate Plots

```bash
# Jupyter notebook
jupyter notebook plot_results.ipynb

# Or standalone script
python plot_results.py
```

---

## Project File Map

```
blockchain_awfedavg/
│
├── CORE FL + PHY
│   ├── adaptive_weighted_fedavg.py        ← AWFedAvg strategy + Flower client
│   ├── blockchain_awfedavg_true_integration.py ← Main entry point
│   ├── gym_phy_env.py                     ← Gymnasium wrapper
│   ├── phy_env_class.py                   ← 5G-NR wireless simulator
│   └── phy/                               ← Channel models, waterfilling, nodes
│
├── BLOCKCHAIN + PRIVACY
│   ├── privacy_blockchain_fl.py           ← DP + encryption + IPFS + blockchain
│   ├── secure_aggregation.py              ← Pairwise mask cancellation (UPDATED)
│   ├── contracts/FederatedLearningContract.sol  ← Solidity smart contract
│   ├── deploy.py                          ← Contract deployment script
│   └── estimate_gas.py                    ← Gas cost estimator
│
├── ENHANCEMENTS (NEW)
│   ├── robustness_module.py               ← Trimmed mean, Krum, cosine anomaly
│   ├── efficient_dp.py                    ← Rényi DP, adaptive clip, Top-K
│   ├── enhanced_integration.py            ← Drop-in enhanced strategy + client
│   └── torch_shim.py                      ← Lightweight torch compatibility
│
├── EXPERIMENTS + SIMULATION
│   ├── experiments.py                     ← Ablation, scalability, attacks
│   ├── scalability_ablation.py            ← Scalability experiments
│   ├── simulate_robustness_qos_sla.py     ← Robustness/QoS/SLA simulation
│   └── test_enhancements.py               ← 23-test validation suite
│
├── RESULTS + PLOTS
│   ├── plot_results.py                    ← Matplotlib figure generation
│   ├── plot_results.ipynb                 ← Jupyter notebook for plots
│   ├── simulation_results.json            ← Robustness/QoS/SLA results
│   └── *_results.json                     ← Experiment result files
│
├── CONFIG
│   ├── requirements.txt                   ← Python dependencies
│   ├── requirements-dev.txt               ← Dev/test dependencies
│   ├── run_experiment.sh                  ← Automated experiment runner
│   └── contract_info.json                 ← Deployed contract address + ABI
│
└── README.md
```

---

## Quick Reference: Common Commands

| What | Command |
|---|---|
| Run without blockchain | `python -c "from blockchain_awfedavg_true_integration import run_blockchain_awfedavg_experiment; run_blockchain_awfedavg_experiment(blockchain_enabled=False)"` |
| Run enhanced (no blockchain) | `python -c "from enhanced_integration import run_enhanced_experiment; run_enhanced_experiment(blockchain_enabled=False)"` |
| Run robustness simulation | `python simulate_robustness_qos_sla.py` |
| Validate enhancements | `python test_enhancements.py` |
| Full automated experiment | `./run_experiment.sh` |
| Smoke test | `./run_experiment.sh --smoke` |
| Attack experiment | `./run_experiment.sh --mode attacks` |
| Generate plots | `python plot_results.py` |
| Estimate gas costs | `python estimate_gas.py --clients 10 --rounds 30` |
| Deploy contract | `python deploy.py` |

---

## Troubleshooting

**"No module named torch"** — Install PyTorch: `pip install torch`

**"Failed to connect to blockchain"** — Either start Ganache (`ganache --deterministic`) or use `blockchain_enabled=False`

**"IPFS connection failed"** — Start the daemon (`ipfs daemon`) or the system will continue without IPFS (DP and encryption still work)

**"web3 version conflict"** — Pin exact versions: `pip install web3==6.15.1 eth-account==0.9.0 hexbytes==0.3.1`

**"ray/flwr simulation error"** — Flower simulation requires `ray`. Install with: `pip install "flwr[simulation]"`

**Windows users** — Use WSL2 (Windows Subsystem for Linux) for best compatibility. Ganache and IPFS run natively on Windows but the shell script requires bash.

**macOS Apple Silicon** — Use `pip install torch` (ARM-native wheels are available since torch 2.0). No special flags needed.
