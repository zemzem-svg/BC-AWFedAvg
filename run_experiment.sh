#!/usr/bin/env bash
# =============================================================================
# run_experiment.sh
# =============================================================================
# Unified entry point for all AWFedAvg experiments.
# Handles environment setup, Ganache + IPFS startup, deployment,
# and experiment execution — all from a single command.
#
# USAGE
# -----
#   chmod +x run_experiment.sh
#
#   # Full protocol (ablation + scalability + privacy + attacks)
#   ./run_experiment.sh
#
#   # Specific mode
#   ./run_experiment.sh --mode ablation
#   ./run_experiment.sh --mode scalability
#   ./run_experiment.sh --mode privacy
#   ./run_experiment.sh --mode attacks
#
#   # Smoke test (1 round, 2 seeds — fast CI check)
#   ./run_experiment.sh --smoke
#
#   # Single parameterised run
#   ./run_experiment.sh --clients 10 --rounds 30 --dp_eps 1.0 \
#                       --attack byzantine --attack_frac 0.2   \
#                       --seed 42
#
#   # Skip infra startup if already running
#   ./run_experiment.sh --no-ganache --no-ipfs --mode ablation
#
# OUTPUT
# ------
#   results/ablation_results.json
#   results/scalability_results.json
#   results/privacy_results.json
#   results/attack_results.json
#   results/all_experiment_results.json
#   logs/run_<timestamp>.log
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE="all"
SMOKE=false
CLIENTS=""
ROUNDS=""
DP_EPS="1.0"
ATTACK="none"
ATTACK_FRAC="0.2"
ATTACK_STRENGTH="1.0"
SEED=""
START_GANACHE=true
START_IPFS=true
DEPLOY=true

GANACHE_PORT=8545
IPFS_PORT=5001
GANACHE_PID=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET}  $*" | tee -a "$LOG_FILE"; }
ok()      { echo -e "${GREEN}[OK]${RESET}    $*" | tee -a "$LOG_FILE"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}   $*" | tee -a "$LOG_FILE"; }
error()   { echo -e "${RED}[ERROR]${RESET}  $*" | tee -a "$LOG_FILE"; exit 1; }
section() { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${RESET}"; \
            echo -e "${BOLD}${CYAN}  $*${RESET}";                                          \
            echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}\n" | tee -a "$LOG_FILE"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)            MODE="$2"; shift 2 ;;
        --smoke)           SMOKE=true; shift ;;
        --clients)         CLIENTS="$2"; shift 2 ;;
        --rounds)          ROUNDS="$2"; shift 2 ;;
        --dp_eps)          DP_EPS="$2"; shift 2 ;;
        --attack)          ATTACK="$2"; shift 2 ;;
        --attack_frac)     ATTACK_FRAC="$2"; shift 2 ;;
        --attack_strength) ATTACK_STRENGTH="$2"; shift 2 ;;
        --seed)            SEED="$2"; shift 2 ;;
        --no-ganache)      START_GANACHE=false; shift ;;
        --no-ipfs)         START_IPFS=false; shift ;;
        --no-deploy)       DEPLOY=false; shift ;;
        --help|-h)
            sed -n '/^# USAGE/,/^# ==/p' "$0" | head -40; exit 0 ;;
        *) warn "Unknown flag: $1"; shift ;;
    esac
done

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
echo "Log file: $LOG_FILE"

# ── Banner ────────────────────────────────────────────────────────────────────
section "Blockchain-AWFedAvg Experiment Runner"
info "Mode:        $MODE"
info "Smoke:       $SMOKE"
info "Timestamp:   $TIMESTAMP"
[[ -n "$CLIENTS" ]] && info "Clients:     $CLIENTS"
[[ -n "$ROUNDS"  ]] && info "Rounds:      $ROUNDS"
info "DP epsilon:  $DP_EPS"
info "Attack:      $ATTACK (frac=$ATTACK_FRAC, strength=$ATTACK_STRENGTH)"
[[ -n "$SEED"    ]] && info "Seed:        $SEED"

# ── Dependency checks ─────────────────────────────────────────────────────────
section "Checking dependencies"

command -v python3 &>/dev/null  || error "python3 not found"
command -v node    &>/dev/null  || error "node not found (required for deploy.py)"

# Ganache
if $START_GANACHE; then
    if command -v ganache &>/dev/null; then
        GANACHE_CMD="ganache"
    elif command -v ganache-cli &>/dev/null; then
        GANACHE_CMD="ganache-cli"
    else
        error "ganache not found. Install with: npm install -g ganache"
    fi
    ok "Ganache: $GANACHE_CMD"
fi

# IPFS
if $START_IPFS; then
    command -v ipfs &>/dev/null || error "ipfs not found. Install kubo: https://docs.ipfs.tech/install/command-line/"
    ok "IPFS daemon: $(ipfs version --number 2>/dev/null || echo 'unknown')"
fi

# Python packages
python3 -c "import flwr, torch, numpy, web3, ipfshttpclient" 2>/dev/null \
    || warn "Some Python packages may be missing. Run: pip install -r requirements.txt"

ok "Dependencies OK"

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
    info "Cleaning up background processes..."
    [[ -n "$GANACHE_PID" ]] && kill "$GANACHE_PID" 2>/dev/null && info "Ganache stopped"
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

# ── Start Ganache ─────────────────────────────────────────────────────────────
if $START_GANACHE; then
    section "Starting Ganache"
    if lsof -i :"$GANACHE_PORT" &>/dev/null; then
        warn "Port $GANACHE_PORT already in use — skipping Ganache start (assuming already running)"
    else
        $GANACHE_CMD \
            --accounts 15 \
            --deterministic \
            --networkId 1337 \
            --port "$GANACHE_PORT" \
            --gasLimit 12000000 \
            --gasPrice 20000000000 \
            >> "$LOG_DIR/ganache_${TIMESTAMP}.log" 2>&1 &
        GANACHE_PID=$!
        info "Ganache PID: $GANACHE_PID"
        sleep 3

        # Verify connection
        if python3 -c "from web3 import Web3; w=Web3(Web3.HTTPProvider('http://127.0.0.1:$GANACHE_PORT')); assert w.is_connected()" 2>/dev/null; then
            ok "Ganache running on port $GANACHE_PORT"
        else
            error "Ganache failed to start. Check logs/ganache_${TIMESTAMP}.log"
        fi
    fi
fi

# ── Start IPFS ────────────────────────────────────────────────────────────────
if $START_IPFS; then
    section "Starting IPFS daemon"
    if ipfs swarm peers &>/dev/null 2>&1; then
        warn "IPFS daemon already running — skipping start"
    else
        # Init if needed
        [ -d "$HOME/.ipfs" ] || ipfs init --profile=lowpower >> "$LOG_DIR/ipfs_init.log" 2>&1

        ipfs daemon >> "$LOG_DIR/ipfs_${TIMESTAMP}.log" 2>&1 &
        IPFS_PID=$!
        info "IPFS PID: $IPFS_PID"

        # Wait for IPFS to be ready (up to 15 s)
        for i in $(seq 1 15); do
            if ipfs id &>/dev/null 2>&1; then
                ok "IPFS daemon ready"
                break
            fi
            sleep 1
            [[ $i -eq 15 ]] && warn "IPFS may not be fully ready — continuing anyway"
        done
    fi
fi

# ── Deploy smart contract ─────────────────────────────────────────────────────
if $DEPLOY; then
    section "Deploying smart contract"
    cd "$SCRIPT_DIR"

    if [[ -f contract_info.json ]]; then
        warn "contract_info.json already exists — skipping re-deploy (use --no-deploy to suppress this)"
        ok "Using existing deployment: $(python3 -c "import json; print(json.load(open('contract_info.json')).get('contract_address','?'))" 2>/dev/null)"
    else
        python3 deploy.py 2>&1 | tee -a "$LOG_FILE"
        if [[ -f contract_info.json ]]; then
            CONTRACT_ADDR="$(python3 -c "import json; print(json.load(open('contract_info.json'))['contract_address'])")"
            ok "Contract deployed at: $CONTRACT_ADDR"
        else
            error "deploy.py did not produce contract_info.json"
        fi
    fi
fi

# ── Gas estimation (informational) ────────────────────────────────────────────
section "Gas estimation"
if [[ -f "$SCRIPT_DIR/estimate_gas.py" ]]; then
    python3 "$SCRIPT_DIR/estimate_gas.py" \
        --clients "${CLIENTS:-3}" \
        --rounds  "${ROUNDS:-10}" \
        2>&1 | tee -a "$LOG_FILE" || warn "Gas estimation failed — continuing"
else
    warn "estimate_gas.py not found — skipping"
fi

# ── Build Python command ──────────────────────────────────────────────────────
section "Running experiments"

PY_ARGS=("--mode" "$MODE")
$SMOKE                    && PY_ARGS+=("--smoke")
[[ -n "$CLIENTS" ]]       && PY_ARGS+=("--clients" "$CLIENTS")
[[ -n "$ROUNDS"  ]]       && PY_ARGS+=("--rounds"  "$ROUNDS")
[[ -n "$SEED"    ]]       && PY_ARGS+=("--seed"     "$SEED")
[[ "$ATTACK" != "none" ]] && PY_ARGS+=("--attack" "$ATTACK" \
                                        "--attack_frac" "$ATTACK_FRAC" \
                                        "--attack_strength" "$ATTACK_STRENGTH")
[[ "$DP_EPS" != "1.0" ]]  && PY_ARGS+=("--dp_eps" "$DP_EPS")

info "python3 experiments.py ${PY_ARGS[*]}"

cd "$SCRIPT_DIR"
python3 experiments.py "${PY_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

# ── Collect results ───────────────────────────────────────────────────────────
section "Collecting results"

for f in ablation_results.json scalability_results.json \
          privacy_results.json attack_results.json \
          all_experiment_results.json single_run_result.json; do
    [[ -f "$SCRIPT_DIR/$f" ]] && mv "$SCRIPT_DIR/$f" "$RESULTS_DIR/$f" \
        && ok "→ results/$f"
done

# ── Summary ───────────────────────────────────────────────────────────────────
section "Done ✅"
info "Results in:  $RESULTS_DIR"
info "Log file:    $LOG_FILE"
echo ""
echo -e "${BOLD}${GREEN}Experiment run complete.${RESET}"
echo -e "To generate figures:  ${CYAN}jupyter notebook plot_results.ipynb${RESET}"
echo -e "To estimate gas:      ${CYAN}python3 estimate_gas.py --clients 10 --rounds 30${RESET}"
