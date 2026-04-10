import os
import sys
import logging

import warnings
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, Parameters, ndarrays_to_parameters, Context
from flwr.client import Client
import torch
import torch.nn as nn
from collections import OrderedDict
from stable_baselines3 import PPO
import random
import json
from datetime import datetime
import time
import glob
import psutil  # For system resource monitoring
import threading
import GPUtil  # For GPU monitoring (if available)
from memory_profiler import profile  # For detailed memory profiling

# Suppress Ray's verbose output
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_BACKEND_LOG_LEVEL"] = "error"

# Configure logging to reduce verbosity
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message=r".*As shared layers in the mlp_extractor are removed since SB3 v1\.8\.0.*",
    category=UserWarning,
)
# Suppress GPUtil deprecation warning
warnings.filterwarnings(
    "ignore",
    message=r".*Use shutil\.which instead of find_executable.*",
    category=DeprecationWarning,
)
# Suppress stable_baselines3 GPU warning for PPO (user wants GPU for speed)
warnings.filterwarnings(
    "ignore",
    message=r".*You are trying to run PPO on the GPU.*",
    category=UserWarning,
)

# Import environment
from gym_phy_env import PhyEnv as BasePhyEnv

# ──────────────────────────────────────────────────────────────────────────────
# RESOURCE MONITORING CLASS
# ──────────────────────────────────────────────────────────────────────────────

class ResourceMonitor:
    """Comprehensive resource monitoring for federated learning."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.resource_data = []
        self.start_time = None
        self.gpu_available = False
        
        # Check GPU availability
        try:
            import GPUtil
            self.gpu_available = len(GPUtil.getGPUs()) > 0
        except ImportError:
            self.gpu_available = False
    
    def start_monitoring(self, interval=0.1):
        """Start continuous resource monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.resource_data = []
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring and return collected data."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        return self._summarize_resources()
    
    def _monitor_loop(self, interval):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # CPU and Memory
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                
                # System-wide resources
                system_cpu = psutil.cpu_percent()
                system_memory = psutil.virtual_memory().percent
                
                resource_point = {
                    'timestamp': timestamp,
                    'elapsed_time': timestamp - self.start_time,
                    'process_cpu_percent': cpu_percent,
                    'process_memory_mb': memory_info.rss / (1024 * 1024),
                    'process_memory_percent': memory_percent,
                    'system_cpu_percent': system_cpu,
                    'system_memory_percent': system_memory,
                    'threads_count': self.process.num_threads(),
                }
                
                # GPU monitoring if available
                if self.gpu_available:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU
                            resource_point.update({
                                'gpu_utilization': gpu.load * 100,
                                'gpu_memory_used': gpu.memoryUsed,
                                'gpu_memory_total': gpu.memoryTotal,
                                'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                                'gpu_temperature': gpu.temperature,
                            })
                    except Exception:
                        pass
                
                self.resource_data.append(resource_point)
                time.sleep(interval)
                
            except Exception as e:
                print(f"Warning: Resource monitoring error: {e}")
                break
    
    def _summarize_resources(self):
        """Summarize collected resource data."""
        if not self.resource_data:
            return {}
        
        # Convert to numpy arrays for easier computation
        cpu_usage = [d['process_cpu_percent'] for d in self.resource_data]
        memory_mb = [d['process_memory_mb'] for d in self.resource_data]
        memory_percent = [d['process_memory_percent'] for d in self.resource_data]
        system_cpu = [d['system_cpu_percent'] for d in self.resource_data]
        system_memory = [d['system_memory_percent'] for d in self.resource_data]
        
        summary = {
            'duration': self.resource_data[-1]['elapsed_time'],
            'data_points': len(self.resource_data),
            'process_cpu': {
                'mean': float(np.mean(cpu_usage)),
                'max': float(np.max(cpu_usage)),
                'min': float(np.min(cpu_usage)),
                'std': float(np.std(cpu_usage)),
            },
            'process_memory_mb': {
                'mean': float(np.mean(memory_mb)),
                'max': float(np.max(memory_mb)),
                'min': float(np.min(memory_mb)),
                'peak': float(np.max(memory_mb)),
            },
            'process_memory_percent': {
                'mean': float(np.mean(memory_percent)),
                'max': float(np.max(memory_percent)),
                'min': float(np.min(memory_percent)),
            },
            'system_cpu': {
                'mean': float(np.mean(system_cpu)),
                'max': float(np.max(system_cpu)),
            },
            'system_memory': {
                'mean': float(np.mean(system_memory)),
                'max': float(np.max(system_memory)),
            },
            'threads_count': self.resource_data[-1]['threads_count'],
        }
        
        # Add GPU summary if available
        if self.gpu_available and 'gpu_utilization' in self.resource_data[0]:
            gpu_util = [d.get('gpu_utilization', 0) for d in self.resource_data]
            gpu_memory = [d.get('gpu_memory_percent', 0) for d in self.resource_data]
            
            summary['gpu'] = {
                'utilization': {
                    'mean': float(np.mean(gpu_util)),
                    'max': float(np.max(gpu_util)),
                },
                'memory_percent': {
                    'mean': float(np.mean(gpu_memory)),
                    'max': float(np.max(gpu_memory)),
                },
            }
        
        return summary

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

NUM_CLIENTS = 3
CLIENTS_PER_ROUND = 3
TOTAL_ROUNDS = 15
LOCAL_EPOCHS = 100
EVALUATION_EPISODES = 100
# Auto-detect GPU; fall back to CPU if CUDA is unavailable.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"🖥️  GPU detected: {torch.cuda.get_device_name(0)} — using CUDA")
else:
    print("💻 No GPU detected — using CPU")
BASE_MODEL_PATH = "./federated_models_awfedavg_15"
os.makedirs(BASE_MODEL_PATH, exist_ok=True)

# Client configurations
CLIENT_CONFIGS = [
    {"activation": 0.2, "seed": 42, "q_norm": 0, "noise_level": 0, "name": "Light"},
    {"activation": 0.4, "seed": 42, "q_norm": 0, "noise_level": 0, "name": "Medium"},
    {"activation": 0.6, "seed": 42, "q_norm": 0, "noise_level": 0, "name": "Heavy"},
    #    {"activation": 0.4, "seed": 42, "q_norm": 0.5, "noise_level": 0.1, "name": "Medium"},
    # {"activation": 0.5, "seed": 42, "q_norm": 0.5, "noise_level": 0.1, "name": "Heavy"},
    #  {"activation": 0.6, "seed": 42, "q_norm": 0.002378, "noise_level": 0.1, "name": "Light"},
    # {"activation": 0.7, "seed": 42, "q_norm": 0.002378, "noise_level": 0.1, "name": "Medium"},
    # {"activation": 0.8, "seed": 42, "q_norm": 0.002378, "noise_level": 0.1, "name": "Heavy"},
    #    {"activation": 0.9, "seed": 42, "q_norm": 0.002378, "noise_level": 0.1, "name": "Medium"},
    # {"activation": 0.95, "seed": 42, "q_norm": 0.002378, "noise_level": 0.1, "name": "Heavy"},
]

def get_client_hyperparams(client_id):
    """Get optimized hyperparameters for each client based on their load."""
    configs = {
        0: {"lr": 2e-4, "batch_size": 64, "n_epochs": 12},
        1: {"lr": 2e-4, "batch_size": 64, "n_epochs": 12},
        2: {"lr": 2e-4, "batch_size": 64, "n_epochs": 12},
    }
    return configs.get(client_id, configs[0])

# ──────────────────────────────────────────────────────────────────────────────
# ENHANCED ENVIRONMENT WRAPPER (keeping original implementation)
# ──────────────────────────────────────────────────────────────────────────────

class SingleEnvPhyEnv(BasePhyEnv):
    """Enhanced PhyEnv with better metrics collection and resource management."""

    def __init__(self, render_mode=None, seed=None):
        super().__init__(render_mode=render_mode, seed=seed)
        self._episode_done_flag = False
        self._embb_outage_counters = []
        self._urllc_delay_counters = []
        self._residual_urllc_pkts = []

    def reset(self, *, seed=None, options=None):
        self._episode_done_flag = False
        obs, info = super().reset(seed=seed, options=options)
        return obs , info

    def step(self, action):
        step_result = super().step(action)
        # Handle both old Gym API (4 values) and new Gymnasium API (5 values)
        if len(step_result) == 4:
            next_state, reward, done, info = step_result
            terminated = done
            truncated = False
        else:
            next_state, reward, terminated, truncated, info = step_result
        
        done = terminated or truncated

        if done and not self._episode_done_flag:
            embb_cnt = info.get("embb_outage_counter", 0.0)
            urllc_cnt = info.get("urllc_delay_counter", 0.0)
            res_pkt = info.get("residual_urllc_pkt", 0.0)

            self._embb_outage_counters.append(embb_cnt)
            self._urllc_delay_counters.append(urllc_cnt)
            self._residual_urllc_pkts.append(res_pkt)
            self._episode_done_flag = True

        return next_state, reward, terminated, truncated, info

    @property
    def embb_outage_counters(self):
        return self._embb_outage_counters

    @property
    def urllc_delay_counters(self):
        return self._urllc_delay_counters

    @property
    def residual_urllc_pkts(self):
        return self._residual_urllc_pkts

    def clear_metrics(self):
        """Clear accumulated metrics for fresh evaluation."""
        self._embb_outage_counters.clear()
        self._urllc_delay_counters.clear()
        self._residual_urllc_pkts.clear()

def create_phy_env(client_id):
    """Create environment with improved configuration and noise injection."""
    config = CLIENT_CONFIGS[client_id % len(CLIENT_CONFIGS)]
    env = SingleEnvPhyEnv(render_mode=None, seed=config["seed"])
    
    # Apply client-specific configuration
    env.phy.pkt_arrival = np.array([config["activation"]])
    env.phy.q_norm = config["q_norm"]
    
    # Add noise for diversity (if supported by environment)
    if hasattr(env.phy, 'noise_level'):
        env.phy.noise_level = config["noise_level"]
    
    return env

# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION AND NETWORK FUNCTIONS (keeping original implementation)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model_simple(model, env, num_episodes=1000):
    """Simplified evaluation without statistical analysis."""
    # Clear previous metrics
    env.clear_metrics()
    
    # Collect metrics
    rewards_per_episode = []
    embb_outages = []
    residual_urllc = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards_per_episode.append(episode_reward)
        
        # Collect final metrics
        if env.embb_outage_counters:
            embb_outages.append(env.embb_outage_counters[-1] / 120.0)
        if env.residual_urllc_pkts:
            residual_urllc.append(env.residual_urllc_pkts[-1])
    
    # Calculate basic metrics only
    average_reward = float(np.mean(rewards_per_episode))
    avg_embb_outage_counter = float(np.mean(embb_outages)) if embb_outages else 0.0
    avg_residual_urllc_pkt = float(np.mean(residual_urllc)) if residual_urllc else 0.0
    
    # Simple stability score (inverse of reward variance)
    reward_variance = np.var(rewards_per_episode)
    stability_score = 1.0 / (1.0 + reward_variance)
    
    return {
        "average_reward": average_reward,
        "avg_embb_outage_counter": avg_embb_outage_counter,
        "avg_residual_urllc_pkt": avg_residual_urllc_pkt,
        "stability_score": float(stability_score),
        "per_episode_metrics": {
            "rewards": rewards_per_episode,
            "embb_outages": embb_outages,
            "residual_urllc": residual_urllc,
        }
    }

class PPONetwork(nn.Module):
    """Enhanced PyTorch network with better initialization."""

    def __init__(self, input_dim, output_dim):
        super(PPONetwork, self).__init__()
        self.input_dim = input_dim
        self.shared_dim = 64
        self.output_dim = output_dim

        # Initialize networks with proper weight initialization
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, self.shared_dim),
            nn.ReLU(),
            nn.Linear(self.shared_dim, self.shared_dim),
            nn.ReLU(),
        )

        self.value_net = nn.Sequential(
            nn.Linear(input_dim, self.shared_dim),
            nn.ReLU(),
            nn.Linear(self.shared_dim, self.shared_dim),
            nn.ReLU(),
        )

        self.action_net = nn.Linear(self.shared_dim, output_dim)
        self.value_head = nn.Linear(self.shared_dim, 1)
        
        # Apply proper weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        policy_features = self.policy_net(x)
        value_features = self.value_net(x)
        policy_logits = self.action_net(policy_features)
        values = self.value_head(value_features)
        return policy_logits, values

def sb3_ppo_to_pytorch(sb3_model):
    """Convert SB3 PPO to PyTorch with enhanced error handling."""
    try:
        input_dim = sb3_model.policy.observation_space.shape[0]
        output_dim = sb3_model.policy.action_space.n
        pytorch_model = PPONetwork(input_dim, output_dim).to(DEVICE)
        sb3_state_dict = sb3_model.policy.state_dict()

        # Enhanced mapping with better error handling
        mapping = {}
        for sb3_key, sb3_tensor in sb3_state_dict.items():
            # Policy network mappings
            if "mlp_extractor.policy_net.0" in sb3_key:
                if "weight" in sb3_key and sb3_tensor.shape == pytorch_model.policy_net[0].weight.shape:
                    mapping[sb3_key] = "policy_net.0.weight"
                elif "bias" in sb3_key and sb3_tensor.shape == pytorch_model.policy_net[0].bias.shape:
                    mapping[sb3_key] = "policy_net.0.bias"
            elif "mlp_extractor.policy_net.2" in sb3_key:
                if "weight" in sb3_key and sb3_tensor.shape == pytorch_model.policy_net[2].weight.shape:
                    mapping[sb3_key] = "policy_net.2.weight"
                elif "bias" in sb3_key and sb3_tensor.shape == pytorch_model.policy_net[2].bias.shape:
                    mapping[sb3_key] = "policy_net.2.bias"
            # Value network mappings
            elif "mlp_extractor.value_net.0" in sb3_key:
                if "weight" in sb3_key and sb3_tensor.shape == pytorch_model.value_net[0].weight.shape:
                    mapping[sb3_key] = "value_net.0.weight"
                elif "bias" in sb3_key and sb3_tensor.shape == pytorch_model.value_net[0].bias.shape:
                    mapping[sb3_key] = "value_net.0.bias"
            elif "mlp_extractor.value_net.2" in sb3_key:
                if "weight" in sb3_key and sb3_tensor.shape == pytorch_model.value_net[2].weight.shape:
                    mapping[sb3_key] = "value_net.2.weight"
                elif "bias" in sb3_key and sb3_tensor.shape == pytorch_model.value_net[2].bias.shape:
                    mapping[sb3_key] = "value_net.2.bias"
            # Output heads
            elif "action_net" in sb3_key:
                if "weight" in sb3_key and sb3_tensor.shape == pytorch_model.action_net.weight.shape:
                    mapping[sb3_key] = "action_net.weight"
                elif "bias" in sb3_key and sb3_tensor.shape == pytorch_model.action_net.bias.shape:
                    mapping[sb3_key] = "action_net.bias"
            elif "value_net" in sb3_key and "mlp_extractor" not in sb3_key:
                if "weight" in sb3_key and sb3_tensor.shape == pytorch_model.value_head.weight.shape:
                    mapping[sb3_key] = "value_head.weight"
                elif "bias" in sb3_key and sb3_tensor.shape == pytorch_model.value_head.bias.shape:
                    mapping[sb3_key] = "value_head.bias"

        # Copy mapped parameters
        pytorch_state_dict = pytorch_model.state_dict()
        for sb3_key, pytorch_key in mapping.items():
            pytorch_state_dict[pytorch_key].copy_(sb3_state_dict[sb3_key])

        return pytorch_model
        
    except Exception as e:
        print(f"Error in SB3 to PyTorch conversion: {e}")
        raise

def pytorch_to_sb3_ppo(pytorch_model, sb3_model):
    """Convert PyTorch to SB3 PPO with enhanced error handling."""
    try:
        pytorch_state_dict = pytorch_model.state_dict()
        sb3_state_dict = sb3_model.policy.state_dict()

        # Reverse mapping
        reverse_mapping = {
            "policy_net.0.weight": "mlp_extractor.policy_net.0.weight",
            "policy_net.0.bias": "mlp_extractor.policy_net.0.bias",
            "policy_net.2.weight": "mlp_extractor.policy_net.2.weight",
            "policy_net.2.bias": "mlp_extractor.policy_net.2.bias",
            "value_net.0.weight": "mlp_extractor.value_net.0.weight",
            "value_net.0.bias": "mlp_extractor.value_net.0.bias",
            "value_net.2.weight": "mlp_extractor.value_net.2.weight",
            "value_net.2.bias": "mlp_extractor.value_net.2.bias",
            "action_net.weight": "action_net.weight",
            "action_net.bias": "action_net.bias",
            "value_head.weight": "value_net.weight",
            "value_head.bias": "value_net.bias",
        }

        # Copy parameters with shape verification
        for pytorch_key, sb3_key in reverse_mapping.items():
            if pytorch_key in pytorch_state_dict and sb3_key in sb3_state_dict:
                if pytorch_state_dict[pytorch_key].shape == sb3_state_dict[sb3_key].shape:
                    sb3_state_dict[sb3_key].copy_(pytorch_state_dict[pytorch_key])

        sb3_model.policy.load_state_dict(sb3_state_dict, strict=False)
        return sb3_model
        
    except Exception as e:
        print(f"Error in PyTorch to SB3 conversion: {e}")
        raise

# ──────────────────────────────────────────────────────────────────────────────
# ADAPTIVE WEIGHT CALCULATION MODULE (keeping original implementation)
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveWeightCalculator:
    """Advanced weight calculation module for AWFedAvg."""
    
    def __init__(self, alpha_embb=0.22, alpha_urllc=0.38, alpha_activation=0.2,
                 alpha_stability=0.15, alpha_reputation=0.05):
        """
        5-criteria adaptive weight calculator.

        alpha_reputation : weight given to the blockchain reputation score (0-1000 scale).
                           Default 0.05 — small but non-zero so on-chain governance has
                           measurable impact without overwhelming learning-based criteria.
                           Set to 0.0 to reproduce the original 4-criteria AWFedAvg baseline.
        """
        self.alpha_embb       = alpha_embb
        self.alpha_urllc      = alpha_urllc
        self.alpha_activation = alpha_activation
        self.alpha_stability  = alpha_stability
        self.alpha_reputation = alpha_reputation
        # Alphas are used exactly as provided — no normalisation.

        # In-memory reputation cache: client_id → float in [0, 1]
        # Updated from blockchain after each round via update_reputation_from_chain().
        self.reputation_scores: dict = {}

        # Historical data for adaptive weighting
        self.performance_history = {}
        self.weight_history = []

    def update_reputation_from_chain(self, reputations: dict):
        """
        Inject on-chain reputation scores into the calculator.

        Parameters
        ----------
        reputations : {client_id: reputation_int}  where reputation_int ∈ [0, 1000]
                      (matches the contract's 0-1000 scale).
        """
        for cid, rep in reputations.items():
            self.reputation_scores[cid] = float(rep) / 1000.0   # normalise to [0, 1]
        
    def calculate_adaptive_weights(self, client_metrics, client_configs, round_num):
        """Calculate adaptive weights based on multiple criteria."""
        client_ids = list(client_metrics.keys())
        n_clients = len(client_ids)
        
        if n_clients == 0:
            return []
        
        # Initialize components
        embb_scores = np.zeros(n_clients)
        urllc_scores = np.zeros(n_clients)
        activation_scores = np.zeros(n_clients)
        stability_scores = np.zeros(n_clients)
        
        # Extract metrics
        embb_outages = []
        urllc_residuals = []
        activations = []
        rewards = []
        
        for i, client_id in enumerate(client_ids):
            metrics = client_metrics[client_id]
            config = client_configs[client_id % len(client_configs)]
            
            embb_outages.append(metrics.get('avg_embb_outage_counter', 0.0))
            urllc_residuals.append(metrics.get('avg_residual_urllc_pkt', 0.0))
            activations.append(config['activation'])
            rewards.append(metrics.get('average_reward', 0.0))
        
        # Convert to numpy arrays
        embb_outages = np.array(embb_outages)
        urllc_residuals = np.array(urllc_residuals)
        activations = np.array(activations)
        rewards = np.array(rewards)
        
        # 1. eMBB Outage Component (lower is better)
        if np.std(embb_outages) > 1e-8:
            embb_scores = 1.0 / (embb_outages + 1e-8)
            embb_scores = embb_scores / np.sum(embb_scores)
        else:
            embb_scores = np.ones(n_clients) / n_clients
        
        # 2. URLLC Residual Component (lower is better)
        if np.std(urllc_residuals) > 1e-8:
            urllc_scores = 1.0 / (urllc_residuals + 1e-8)
            urllc_scores = urllc_scores / np.sum(urllc_scores)
        else:
            urllc_scores = np.ones(n_clients) / n_clients
        
        # 3. Activation Level Component
        activation_diversity = np.abs(activations - np.mean(activations))
        if np.std(activation_diversity) > 1e-8:
            activation_scores = (1.0 + activation_diversity) / np.sum(1.0 + activation_diversity)
        else:
            activation_scores = np.ones(n_clients) / n_clients
        
        # 4. Stability Component
        for i, client_id in enumerate(client_ids):
            if client_id not in self.performance_history:
                self.performance_history[client_id] = []
            
            self.performance_history[client_id].append(rewards[i])
            
            # Calculate stability score (inverse of variance)
            if len(self.performance_history[client_id]) > 1:
                recent_performance = self.performance_history[client_id][-5:]  # Last 5 rounds
                stability_var = np.var(recent_performance)
                stability_scores[i] = 1.0 / (1.0 + stability_var)
            else:
                stability_scores[i] = 1.0  # Maximum stability for first round
        
        # Normalize stability scores
        if np.sum(stability_scores) > 0:
            stability_scores = stability_scores / np.sum(stability_scores)
        else:
            stability_scores = np.ones(n_clients) / n_clients
        
        # 5. Reputation Component (from blockchain, normalised to [0,1])
        reputation_scores = np.zeros(n_clients)
        for i, client_id in enumerate(client_ids):
            rep = self.reputation_scores.get(client_id, 0.5)   # neutral default
            reputation_scores[i] = float(rep)
        if np.sum(reputation_scores) > 0:
            reputation_scores = reputation_scores / np.sum(reputation_scores)
        else:
            reputation_scores = np.ones(n_clients) / n_clients

        # 6. Combine all five components
        combined_scores = (
            self.alpha_embb        * embb_scores +
            self.alpha_urllc       * urllc_scores +
            self.alpha_activation  * activation_scores +
            self.alpha_stability   * stability_scores +
            self.alpha_reputation  * reputation_scores
        )
        
        # Ensure weights are positive and normalized
        weights = np.maximum(combined_scores, 1e-8)
        weights = weights / np.sum(weights)
        
        # Apply smoothing to prevent extreme weight changes
        if len(self.weight_history) > 0:
            prev_weights = self.weight_history[-1]
            smoothing_factor = 0.7
            weights = smoothing_factor * weights + (1 - smoothing_factor) * prev_weights
            weights = weights / np.sum(weights)
        
        # Store current weights
        self.weight_history.append(weights.copy())
        
        return weights.tolist()

# ──────────────────────────────────────────────────────────────────────────────
# ENHANCED ADAPTIVE WEIGHTED FEDERATED AVERAGING STRATEGY WITH RESOURCE TRACKING
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveWeightedFedAvg(fl.server.strategy.FedAvg):
    """Enhanced Adaptive Weighted FedAvg (AWFedAvg) strategy with resource tracking."""
    
    def __init__(self, alpha_embb=0.22, alpha_urllc=0.38, alpha_activation=0.2, alpha_stability=0.15, alpha_reputation=0.05, **kwargs):
        super().__init__(**kwargs)
        self.weight_calculator = AdaptiveWeightCalculator(
            alpha_embb=alpha_embb,
            alpha_urllc=alpha_urllc,
            alpha_reputation=alpha_reputation,
            alpha_activation=alpha_activation,
            alpha_stability=alpha_stability
        )
        self.round_metrics = []
        self.final_parameters = None
        self.performance_history = []
        
        # Enhanced tracking with resource monitoring
        self.resource_monitor = ResourceMonitor()
        self.round_resources = []
        self.total_training_time = 0
        self.total_communication_time = 0
        self.total_aggregation_time = 0
        
    def aggregate_fit(self, server_round, results, failures):
        """Enhanced aggregation with comprehensive time and resource tracking."""
        
        # Start comprehensive monitoring
        round_start_time = time.time()
        self.resource_monitor.start_monitoring(interval=0.1)
        
        print(f"\n--- AWFedAvg Round {server_round} Resource Tracking ---")
        print(f"Starting resource monitoring...")
        
        # Phase 1: Client metrics collection
        collection_start_time = time.time()
        
        client_metrics = {}
        client_performances = []
        client_ids = []
        client_training_times = []
        client_communication_sizes = []
        
        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get("client_id", "unknown")
            performance = fit_res.metrics.get("average_reward", 0.0)
            stability = fit_res.metrics.get("stability_score", 0.5)
            training_time = fit_res.metrics.get("training_time", 0.0)
            
            # Calculate parameter size for communication cost
            params_size = sum(p.nbytes for p in parameters_to_ndarrays(fit_res.parameters))
            client_communication_sizes.append(params_size)
            client_training_times.append(training_time)
            
            client_metrics[client_id] = {
                "avg_embb_outage_counter": fit_res.metrics.get("avg_embb_outage_counter", 0.0),
                "avg_residual_urllc_pkt": fit_res.metrics.get("avg_residual_urllc_pkt", 0.0),
                "average_reward": performance,
                "stability_score": stability,
                "training_time": training_time,
                "communication_size_bytes": params_size,
            }
            client_performances.append(performance)
            client_ids.append(client_id)
            
            print(f"Client {client_id}: Reward={performance:.4f}, Training={training_time:.2f}s, "
                  f"Params={params_size/(1024*1024):.2f}MB")
        
        collection_end_time = time.time()
        collection_duration = collection_end_time - collection_start_time
        
        # Phase 2: Adaptive weight calculation
        weight_calc_start_time = time.time()
        
        adaptive_weights = self.weight_calculator.calculate_adaptive_weights(
            client_metrics, CLIENT_CONFIGS, server_round
        )
        
        weight_calc_end_time = time.time()
        weight_calc_duration = weight_calc_end_time - weight_calc_start_time
        
        print(f"Adaptive weights: {[f'{w:.3f}' for w in adaptive_weights]}")
        print(f"Weight calculation time: {weight_calc_duration:.3f}s")
        
        # Display weight breakdown
        for i, (client_id, weight) in enumerate(zip(client_ids, adaptive_weights)):
            config = CLIENT_CONFIGS[client_id % len(CLIENT_CONFIGS)]
            print(f"  Client {client_id} ({config['name']}): weight={weight:.3f} "
                  f"(activation={config['activation']})")
        
        # Phase 3: Parameter aggregation
        aggregation_start_time = time.time()
        
        aggregated_parameters = self.adaptive_weighted_average(results, adaptive_weights)
        
        aggregation_end_time = time.time()
        aggregation_duration = aggregation_end_time - aggregation_start_time
        
        # Phase 4: Final metrics calculation
        metrics_calc_start_time = time.time()
        
        # Handle empty results case
        if not client_performances:
            print("⚠️  No successful client results - skipping aggregation")
            return None, {}
        
        avg_global_reward = float(np.mean(client_performances))
        self.performance_history.append(avg_global_reward)
        
        # Communication statistics
        total_comm_size = sum(client_communication_sizes)
        avg_training_time = np.mean(client_training_times) if client_training_times else 0.0
        max_training_time = np.max(client_training_times) if client_training_times else 0.0
        
        metrics_calc_end_time = time.time()
        metrics_calc_duration = metrics_calc_end_time - metrics_calc_start_time
        
        # Stop resource monitoring and get summary
        round_end_time = time.time()
        total_round_duration = round_end_time - round_start_time
        
        resource_summary = self.resource_monitor.stop_monitoring()
        
        # Update cumulative times
        self.total_training_time += max_training_time  # Bottleneck training time
        self.total_communication_time += collection_duration + (total_comm_size / (1024*1024*10))  # Assume 10MB/s
        self.total_aggregation_time += aggregation_duration
        
        # Detailed timing breakdown
        timing_breakdown = {
            "total_round_time": total_round_duration,
            "collection_time": collection_duration,
            "weight_calculation_time": weight_calc_duration,
            "aggregation_time": aggregation_duration,
            "metrics_calculation_time": metrics_calc_duration,
            "client_training_times": {
                "average": avg_training_time,
                "maximum": max_training_time,
                "individual": dict(zip(client_ids, client_training_times))
            }
        }
        
        # Communication analysis
        communication_analysis = {
            "total_parameters_size_mb": total_comm_size / (1024 * 1024),
            "average_params_size_mb": np.mean(client_communication_sizes) / (1024 * 1024),
            "total_communication_overhead": collection_duration,
            "estimated_network_time": total_comm_size / (1024*1024*10),  # 10MB/s assumption
        }
        
        print(f"\n📊 Round {server_round} Resource Summary:")
        print(f"  Total round time: {total_round_duration:.2f}s")
        print(f"  Peak memory usage: {resource_summary.get('process_memory_mb', {}).get('peak', 0):.1f}MB")
        print(f"  Average CPU usage: {resource_summary.get('process_cpu', {}).get('mean', 0):.1f}%")
        print(f"  Communication size: {total_comm_size/(1024*1024):.1f}MB")
        
        if 'gpu' in resource_summary:
            print(f"  GPU utilization: {resource_summary['gpu']['utilization']['mean']:.1f}%")
            print(f"  GPU memory: {resource_summary['gpu']['memory_percent']['mean']:.1f}%")
        
        # Store comprehensive round metrics
        round_resource_data = {
            "round": server_round,
            "timing": timing_breakdown,
            "communication": communication_analysis,
            "resources": resource_summary,
            "client_metrics": client_metrics,
            "adaptive_weights": adaptive_weights,
            "global_performance": avg_global_reward,
        }
        
        self.round_resources.append(round_resource_data)
        
        # Aggregated metrics for return
        aggregated_metrics = {
            "average_reward": avg_global_reward,
            "round_duration": total_round_duration,
            "adaptive_weights": adaptive_weights,
            "peak_memory_mb": resource_summary.get('process_memory_mb', {}).get('peak', 0),
            "avg_cpu_percent": resource_summary.get('process_cpu', {}).get('mean', 0),
            "communication_mb": total_comm_size / (1024 * 1024),
        }
        
        self.final_parameters = aggregated_parameters
        self.round_metrics.append({
            "round": server_round,
            "client_metrics": client_metrics,
            "aggregated_metrics": aggregated_metrics,
            "adaptive_weights": adaptive_weights,
            "resource_data": round_resource_data,
        })
        
        return aggregated_parameters, aggregated_metrics
    
    def adaptive_weighted_average(self, results, weights):
        """Perform adaptive weighted averaging with the calculated weights."""
        if not results or not weights:
            return None
            
        # Extract parameters from first result to get structure
        first_params = parameters_to_ndarrays(results[0][1].parameters)
        
        # Initialize weighted sum
        weighted_params = [np.zeros_like(param) for param in first_params]
        
        # Accumulate parameters with adaptive weights
        for (_, fit_res), weight in zip(results, weights):
            params = parameters_to_ndarrays(fit_res.parameters)
            for i, param in enumerate(params):
                weighted_params[i] += weight * param
        
        return ndarrays_to_parameters(weighted_params)
    
    def get_final_parameters(self):
        return self.final_parameters
    
    def get_comprehensive_resource_summary(self):
        """Get comprehensive resource usage summary across all rounds."""
        if not self.round_resources:
            return {}
        
        # Aggregate timing data
        total_times = [r['timing']['total_round_time'] for r in self.round_resources]
        training_times = [r['timing']['client_training_times']['maximum'] for r in self.round_resources]
        aggregation_times = [r['timing']['aggregation_time'] for r in self.round_resources]
        communication_sizes = [r['communication']['total_parameters_size_mb'] for r in self.round_resources]
        
        # Aggregate resource data
        peak_memories = []
        avg_cpus = []
        gpu_utils = []
        gpu_memories = []
        
        for r in self.round_resources:
            resources = r['resources']
            peak_memories.append(resources.get('process_memory_mb', {}).get('peak', 0))
            avg_cpus.append(resources.get('process_cpu', {}).get('mean', 0))
            
            if 'gpu' in resources:
                gpu_utils.append(resources['gpu']['utilization']['mean'])
                gpu_memories.append(resources['gpu']['memory_percent']['mean'])
        
        summary = {
            'total_experiment_time': sum(total_times),
            'total_training_time': self.total_training_time,
            'total_communication_time': self.total_communication_time,
            'total_aggregation_time': self.total_aggregation_time,
            'rounds_completed': len(self.round_resources),
            
            'timing_statistics': {
                'average_round_time': np.mean(total_times),
                'max_round_time': np.max(total_times),
                'min_round_time': np.min(total_times),
                'round_time_std': np.std(total_times),
                'average_training_time': np.mean(training_times),
                'max_training_time': np.max(training_times),
                'average_aggregation_time': np.mean(aggregation_times),
            },
            
            'resource_statistics': {
                'peak_memory_usage_mb': np.max(peak_memories),
                'average_memory_usage_mb': np.mean(peak_memories),
                'peak_cpu_usage_percent': np.max(avg_cpus),
                'average_cpu_usage_percent': np.mean(avg_cpus),
            },
            
            'communication_statistics': {
                'total_data_transferred_mb': sum(communication_sizes),
                'average_round_data_mb': np.mean(communication_sizes),
                'max_round_data_mb': np.max(communication_sizes),
                'communication_efficiency': self.total_communication_time / sum(total_times),
            }
        }
        
        # Add GPU statistics if available
        if gpu_utils:
            summary['gpu_statistics'] = {
                'peak_gpu_utilization': np.max(gpu_utils),
                'average_gpu_utilization': np.mean(gpu_utils),
                'peak_gpu_memory_percent': np.max(gpu_memories),
                'average_gpu_memory_percent': np.mean(gpu_memories),
            }
        
        return summary

# ──────────────────────────────────────────────────────────────────────────────
# ENHANCED FLOWER CLIENT WITH RESOURCE TRACKING
# ──────────────────────────────────────────────────────────────────────────────

class EnhancedFlowerClient(fl.client.NumPyClient):
    """Enhanced Flower client with resource tracking and adaptive hyperparameters."""
    
    def __init__(self, client_id):
        self.client_id = client_id
        self.env = None
        self.model = None
        self.pytorch_model = None
        self.hyperparams = get_client_hyperparams(client_id)
        self.performance_history = []
        
        # Resource tracking
        self.client_monitor = ResourceMonitor()
        self.training_resources = []
        
    def to_client(self):
        return super().to_client()
        
    def _get_env(self):
        if self.env is None:
            self.env = create_phy_env(self.client_id)
        return self.env
    
    def _get_model(self):
        if self.model is None:
            env = self._get_env()
            config = CLIENT_CONFIGS[self.client_id % len(CLIENT_CONFIGS)]
            
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=self.hyperparams["lr"],
                n_steps=1024,
                batch_size=self.hyperparams["batch_size"],
                n_epochs=self.hyperparams["n_epochs"],
                gamma=0.99,
                seed=config["seed"],
                device=DEVICE,                                    # ← GPU if available
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
            )
            
            # Initialize PyTorch model
            input_dim = env.observation_space.shape[0]
            output_dim = env.action_space.n
            self.pytorch_model = PPONetwork(input_dim, output_dim).to(DEVICE)
        
        return self.model
    
    def get_parameters(self, config):
        model = self._get_model()
        return [val.cpu().numpy() for _, val in self.pytorch_model.state_dict().items()]

    def set_parameters(self, parameters):
        model = self._get_model()
        
        if isinstance(parameters, Parameters):
            param_arrays = parameters_to_ndarrays(parameters)
        elif hasattr(parameters, "tensors"):
            param_arrays = parameters_to_ndarrays(parameters)
        elif isinstance(parameters, (list, tuple)):
            param_arrays = list(parameters)
        else:
            raise TypeError(f"Unsupported parameters type: {type(parameters)}")

        model_keys = list(self.pytorch_model.state_dict().keys())
        if len(param_arrays) != len(model_keys):
            raise ValueError(f"Parameter count mismatch: expected {len(model_keys)}, got {len(param_arrays)}")

        state_dict = OrderedDict({
            k: torch.tensor(v, device=DEVICE)
            for k, v in zip(model_keys, param_arrays)
        })
        self.pytorch_model.load_state_dict(state_dict, strict=True)
        self.model = pytorch_to_sb3_ppo(self.pytorch_model, self.model)

    def fit(self, parameters, config):
        """Enhanced fit method with comprehensive resource tracking."""
        model = self._get_model()
        env = self._get_env()
        
        # Start resource monitoring for this training session
        training_start_time = time.time()
        self.client_monitor.start_monitoring(interval=0.05)  # More frequent monitoring during training
        
        print(f"MVNO  {self.client_id}: Starting PPO training with resource monitoring...")
        
        try:
            # Load global parameters
            param_load_start = time.time()
            self.set_parameters(parameters)
            param_load_time = time.time() - param_load_start
            
            # Adaptive training based on recent performance
            base_epochs = config.get("local_epochs", LOCAL_EPOCHS)
            if len(self.performance_history) > 2:
                recent_trend = np.mean(np.diff(self.performance_history[-3:]))
                if recent_trend < 0:
                    training_epochs = int(base_epochs * 1.2)
                else:
                    training_epochs = base_epochs
            else:
                training_epochs = base_epochs

            # Model training phase
            model_training_start = time.time()
            total_timesteps = training_epochs * 1024
            self.model.learn(total_timesteps=total_timesteps, progress_bar=False)
            model_training_time = time.time() - model_training_start
            
            # Model conversion phase
            conversion_start = time.time()
            self.pytorch_model = sb3_ppo_to_pytorch(self.model)
            conversion_time = time.time() - conversion_start
            
            # Evaluation phase
            eval_start = time.time()
            metrics = evaluate_model_simple(self.model, env, num_episodes=10)
            eval_time = time.time() - eval_start
            
            self.performance_history.append(metrics["average_reward"])
            
            # Stop monitoring and collect resource data
            training_end_time = time.time()
            total_training_time = training_end_time - training_start_time
            
            resource_summary = self.client_monitor.stop_monitoring()
            
            # Detailed timing breakdown
            timing_breakdown = {
                "total_training_time": total_training_time,
                "parameter_loading_time": param_load_time,
                "model_training_time": model_training_time,
                "model_conversion_time": conversion_time,
                "evaluation_time": eval_time,
                "training_epochs": training_epochs,
                "total_timesteps": total_timesteps,
            }
            
            # Store training resource data
            training_resource_data = {
                "client_id": self.client_id,
                "timing": timing_breakdown,
                "resources": resource_summary,
                "performance_metrics": metrics,
            }
            
            self.training_resources.append(training_resource_data)
            
            print(f"MVNO  {self.client_id}: Training completed in {total_training_time:.2f}s")
            print(f"  - Model training: {model_training_time:.2f}s")
            print(f"  - Peak memory: {resource_summary.get('process_memory_mb', {}).get('peak', 0):.1f}MB")
            print(f"  - Avg CPU: {resource_summary.get('process_cpu', {}).get('mean', 0):.1f}%")
            
            # Enhanced metrics for server
            enhanced_metrics = {
                "client_id": self.client_id,
                "avg_embb_outage_counter": metrics["avg_embb_outage_counter"],
                "avg_residual_urllc_pkt": metrics["avg_residual_urllc_pkt"],
                "average_reward": metrics["average_reward"],
                "stability_score": metrics["stability_score"],
                "training_epochs": training_epochs,
                "training_time": total_training_time,
                "peak_memory_mb": resource_summary.get('process_memory_mb', {}).get('peak', 0),
                "avg_cpu_percent": resource_summary.get('process_cpu', {}).get('mean', 0),
                "model_training_time": model_training_time,
                "evaluation_time": eval_time,
            }
            
            if 'gpu' in resource_summary:
                enhanced_metrics.update({
                    "avg_gpu_utilization": resource_summary['gpu']['utilization']['mean'],
                    "peak_gpu_memory_percent": resource_summary['gpu']['memory_percent']['max'],
                })
            
            return (
                self.get_parameters(config),
                total_timesteps,      # actual training samples, not obs dimension
                enhanced_metrics,
            )
            
        except Exception as e:
            # Stop monitoring in case of error
            self.client_monitor.stop_monitoring()
            print(f"Error in MVNO {self.client_id} training: {e}")
            raise

    def evaluate(self, parameters, config):
        """Enhanced evaluation with resource tracking."""
        model = self._get_model()
        env = self._get_env()
        
        eval_start_time = time.time()
        self.client_monitor.start_monitoring(interval=0.1)
        
        try:
            self.set_parameters(parameters)
            metrics = evaluate_model_simple(self.model, env, num_episodes=EVALUATION_EPISODES)
            
            eval_end_time = time.time()
            eval_duration = eval_end_time - eval_start_time
            
            resource_summary = self.client_monitor.stop_monitoring()
            
            enhanced_metrics = {
                "client_id": self.client_id,
                "avg_embb_outage_counter": metrics["avg_embb_outage_counter"],
                "avg_residual_urllc_pkt": metrics["avg_residual_urllc_pkt"],
                "average_reward": metrics["average_reward"],
                "stability_score": metrics["stability_score"],
                "evaluation_time": eval_duration,
                "peak_memory_mb": resource_summary.get('process_memory_mb', {}).get('peak', 0),
                "avg_cpu_percent": resource_summary.get('process_cpu', {}).get('mean', 0),
            }
            
            return (
                float(metrics["average_reward"]),
                len(env.reset()[0]),
                enhanced_metrics,
            )
            
        except Exception as e:
            self.client_monitor.stop_monitoring()
            print(f"Error in client {self.client_id} evaluation: {e}")
            raise
    
    def get_resource_summary(self):
        """Get comprehensive resource usage summary for this client."""
        if not self.training_resources:
            return {}
        
        # Aggregate across all training sessions
        total_times = [r['timing']['total_training_time'] for r in self.training_resources]
        training_times = [r['timing']['model_training_time'] for r in self.training_resources]
        peak_memories = [r['resources'].get('process_memory_mb', {}).get('peak', 0) for r in self.training_resources]
        avg_cpus = [r['resources'].get('process_cpu', {}).get('mean', 0) for r in self.training_resources]
        
        return {
            'client_id': self.client_id,
            'total_training_sessions': len(self.training_resources),
            'total_training_time': sum(total_times),
            'average_session_time': np.mean(total_times),
            'peak_memory_usage_mb': np.max(peak_memories),
            'average_cpu_usage_percent': np.mean(avg_cpus),
            'peak_cpu_usage_percent': np.max(avg_cpus),
            'training_efficiency': np.mean(training_times) / np.mean(total_times),
        }
    
    def __del__(self):
        """Clean up resources."""
        if self.env:
            self.env.close()

# ──────────────────────────────────────────────────────────────────────────────
# CLIENT FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def client_fn(context: Context):
    """Create a Flower client with improved resource management."""
    # Extract client ID from context
    # node_id is a large number, convert it to simple index (0, 1, 2, ...)
    client_id = int(context.node_id) % NUM_CLIENTS
    client = EnhancedFlowerClient(client_id)
    return client.to_client()

# ──────────────────────────────────────────────────────────────────────────────
# ENHANCED RESULTS SAVING WITH RESOURCE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def save_awfedavg_results_with_resources(strategy, local_results, federated_results):
    """Save comprehensive AWFedAvg results with detailed resource analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get comprehensive resource summary
    resource_summary = strategy.get_comprehensive_resource_summary()
    
    # Prepare comprehensive results
    results_data = {
        "timestamp": timestamp,
        "strategy": "Adaptive Weighted FedAvg (AWFedAvg) with Resource Tracking",
        "experiment_config": {
            "num_clients": NUM_CLIENTS,
            "total_rounds": TOTAL_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "evaluation_episodes": EVALUATION_EPISODES,
            "client_configs": CLIENT_CONFIGS,
        },
        "awfedavg_config": {
            "alpha_embb": strategy.weight_calculator.alpha_embb,
            "alpha_urllc": strategy.weight_calculator.alpha_urllc,
            "alpha_activation": strategy.weight_calculator.alpha_activation,
            "alpha_stability": strategy.weight_calculator.alpha_stability,
        },
        
        # Comprehensive resource analysis
        "resource_analysis": resource_summary,
        
        # Round-by-round resource tracking
        "round_by_round_resources": strategy.round_resources,
        
        # Weight analysis
        "weight_analysis": {
            "weight_evolution": strategy.weight_calculator.weight_history,
        },
        
        # Performance tracking
        "training_history": strategy.round_metrics,
        "local_results": local_results,
        "federated_results": federated_results,
    }
    
    # Calculate efficiency metrics
    if resource_summary:
        efficiency_metrics = {
            "time_per_round": resource_summary.get('timing_statistics', {}).get('average_round_time', 0),
            "memory_efficiency": resource_summary.get('resource_statistics', {}).get('average_memory_usage_mb', 0) / 1024,  # GB
            "communication_efficiency": resource_summary.get('communication_statistics', {}).get('communication_efficiency', 0),
            "training_to_total_ratio": resource_summary.get('total_training_time', 0) / resource_summary.get('total_experiment_time', 1),
            "aggregation_overhead": resource_summary.get('total_aggregation_time', 0) / resource_summary.get('total_experiment_time', 1),
        }
        results_data["efficiency_metrics"] = efficiency_metrics
    
    # Calculate summary statistics
    if strategy.performance_history:
        final_performance = strategy.performance_history[-1]
        best_performance = max(strategy.performance_history)
        
        # Safe performance trend calculation
        try:
            if len(strategy.performance_history) >= 2:
                performance_trend = np.polyfit(range(len(strategy.performance_history)), 
                                             strategy.performance_history, 1)[0]
            else:
                performance_trend = 0.0
        except (np.linalg.LinAlgError, ValueError):
            if len(strategy.performance_history) >= 2:
                performance_trend = (strategy.performance_history[-1] - strategy.performance_history[0]) / len(strategy.performance_history)
            else:
                performance_trend = 0.0
        
        results_data["summary_stats"] = {
            "final_global_performance": final_performance,
            "best_global_performance": best_performance,
            "performance_trend": float(performance_trend),
            "performance_improvement": float(final_performance - strategy.performance_history[0]) if len(strategy.performance_history) > 1 else 0.0,
        }
    
    # Save to JSON
    results_file = f"{BASE_MODEL_PATH}/awfedavg_resource_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Save a separate CSV for easy analysis
    csv_file = f"{BASE_MODEL_PATH}/awfedavg_round_resources_{timestamp}.csv"
    save_resource_csv(strategy.round_resources, csv_file)
    
    print(f"AWFedAvg results with resource tracking saved to: {results_file}")
    print(f"Round-by-round resource CSV saved to: {csv_file}")
    
    return results_file, csv_file

def save_resource_csv(round_resources, csv_file):
    """Save round-by-round resource data as CSV for easy analysis."""
    try:
        import pandas as pd
        
        # Prepare data for CSV
        csv_data = []
        for round_data in round_resources:
            round_num = round_data['round']
            timing = round_data['timing']
            communication = round_data['communication']
            resources = round_data['resources']
            
            row = {
                'round': round_num,
                'total_round_time': timing['total_round_time'],
                'collection_time': timing['collection_time'],
                'weight_calculation_time': timing['weight_calculation_time'],
                'aggregation_time': timing['aggregation_time'],
                'max_client_training_time': timing['client_training_times']['maximum'],
                'avg_client_training_time': timing['client_training_times']['average'],
                'communication_size_mb': communication['total_parameters_size_mb'],
                'peak_memory_mb': resources.get('process_memory_mb', {}).get('peak', 0),
                'avg_memory_mb': resources.get('process_memory_mb', {}).get('mean', 0),
                'avg_cpu_percent': resources.get('process_cpu', {}).get('mean', 0),
                'max_cpu_percent': resources.get('process_cpu', {}).get('max', 0),
                'global_performance': round_data['global_performance'],
            }
            
            # Add GPU data if available
            if 'gpu' in resources:
                row.update({
                    'avg_gpu_utilization': resources['gpu']['utilization']['mean'],
                    'max_gpu_utilization': resources['gpu']['utilization']['max'],
                    'avg_gpu_memory_percent': resources['gpu']['memory_percent']['mean'],
                    'max_gpu_memory_percent': resources['gpu']['memory_percent']['max'],
                })
            
            csv_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        
    except ImportError:
        # Fallback to manual CSV writing
        import csv
        
        if not round_resources:
            return
        
        # Get all possible keys from the first round
        sample_round = round_resources[0]
        fieldnames = ['round', 'total_round_time', 'collection_time', 'weight_calculation_time', 
                     'aggregation_time', 'max_client_training_time', 'avg_client_training_time',
                     'communication_size_mb', 'peak_memory_mb', 'avg_memory_mb', 'avg_cpu_percent',
                     'max_cpu_percent', 'global_performance']
        
        # Add GPU fields if available
        if 'gpu' in sample_round['resources']:
            fieldnames.extend(['avg_gpu_utilization', 'max_gpu_utilization', 
                             'avg_gpu_memory_percent', 'max_gpu_memory_percent'])
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for round_data in round_resources:
                row = {
                    'round': round_data['round'],
                    'total_round_time': round_data['timing']['total_round_time'],
                    'collection_time': round_data['timing']['collection_time'],
                    'weight_calculation_time': round_data['timing']['weight_calculation_time'],
                    'aggregation_time': round_data['timing']['aggregation_time'],
                    'max_client_training_time': round_data['timing']['client_training_times']['maximum'],
                    'avg_client_training_time': round_data['timing']['client_training_times']['average'],
                    'communication_size_mb': round_data['communication']['total_parameters_size_mb'],
                    'peak_memory_mb': round_data['resources'].get('process_memory_mb', {}).get('peak', 0),
                    'avg_memory_mb': round_data['resources'].get('process_memory_mb', {}).get('mean', 0),
                    'avg_cpu_percent': round_data['resources'].get('process_cpu', {}).get('mean', 0),
                    'max_cpu_percent': round_data['resources'].get('process_cpu', {}).get('max', 0),
                    'global_performance': round_data['global_performance'],
                }
                
                # Add GPU data if available
                if 'gpu' in round_data['resources']:
                    gpu_data = round_data['resources']['gpu']
                    row.update({
                        'avg_gpu_utilization': gpu_data['utilization']['mean'],
                        'max_gpu_utilization': gpu_data['utilization']['max'],
                        'avg_gpu_memory_percent': gpu_data['memory_percent']['mean'],
                        'max_gpu_memory_percent': gpu_data['memory_percent']['max'],
                    })
                
                writer.writerow(row)

def load_local_baseline_results():
    """Load local baseline results from the most recent file."""
    local_files = glob.glob(f"{BASE_MODEL_PATH}/local_baseline_results_*.json")
    
    if not local_files:
        print("No local baseline results found - will run without baseline comparison.")
        return None
    
    # Get the most recent file
    latest_file = max(local_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        local_results = results.get('local_results', {})
        # Convert string keys to int for consistency
        local_results = {int(k): v for k, v in local_results.items()}
        
        print(f"✅ Loaded local baseline results from: {os.path.basename(latest_file)}")
        return local_results
        
    except Exception as e:
        print(f"⚠️ Error loading local results: {e}")
        print("Continuing without baseline comparison.")
        return None

def print_resource_summary(resource_summary):
    """Print a comprehensive resource usage summary."""
    print(f"\n" + "="*80)
    print("COMPREHENSIVE RESOURCE USAGE SUMMARY")
    print("="*80)
    
    if not resource_summary:
        print("No resource data available")
        return
    
    # Overall timing
    print(f"\n⏱️  TIMING ANALYSIS")
    print("-" * 50)
    total_time = resource_summary.get('total_experiment_time', 0)
    training_time = resource_summary.get('total_training_time', 0)
    comm_time = resource_summary.get('total_communication_time', 0)
    agg_time = resource_summary.get('total_aggregation_time', 0)
    
    print(f"Total experiment time:     {total_time:.2f}s ({total_time/60:.1f}min)")
    print(f"Total training time:       {training_time:.2f}s ({(training_time/total_time)*100:.1f}%)")
    print(f"Total communication time:  {comm_time:.2f}s ({(comm_time/total_time)*100:.1f}%)")
    print(f"Total aggregation time:    {agg_time:.2f}s ({(agg_time/total_time)*100:.1f}%)")
    
    timing_stats = resource_summary.get('timing_statistics', {})
    if timing_stats:
        print(f"Average round time:        {timing_stats.get('average_round_time', 0):.2f}s")
        print(f"Max round time:            {timing_stats.get('max_round_time', 0):.2f}s")
        print(f"Round time std dev:        {timing_stats.get('round_time_std', 0):.2f}s")
    
    # Resource usage
    print(f"\n💾 RESOURCE USAGE")
    print("-" * 50)
    resource_stats = resource_summary.get('resource_statistics', {})
    if resource_stats:
        peak_mem = resource_stats.get('peak_memory_usage_mb', 0)
        avg_mem = resource_stats.get('average_memory_usage_mb', 0)
        peak_cpu = resource_stats.get('peak_cpu_usage_percent', 0)
        avg_cpu = resource_stats.get('average_cpu_usage_percent', 0)
        
        print(f"Peak memory usage:         {peak_mem:.1f}MB ({peak_mem/1024:.2f}GB)")
        print(f"Average memory usage:      {avg_mem:.1f}MB ({avg_mem/1024:.2f}GB)")
        print(f"Peak CPU usage:            {peak_cpu:.1f}%")
        print(f"Average CPU usage:         {avg_cpu:.1f}%")
    
    # GPU usage (if available)
    gpu_stats = resource_summary.get('gpu_statistics', {})
    if gpu_stats:
        print(f"\n🖥️  GPU USAGE")
        print("-" * 50)
        print(f"Peak GPU utilization:      {gpu_stats.get('peak_gpu_utilization', 0):.1f}%")
        print(f"Average GPU utilization:   {gpu_stats.get('average_gpu_utilization', 0):.1f}%")
        print(f"Peak GPU memory:           {gpu_stats.get('peak_gpu_memory_percent', 0):.1f}%")
        print(f"Average GPU memory:        {gpu_stats.get('average_gpu_memory_percent', 0):.1f}%")
    
    # Communication analysis
    print(f"\n📡 COMMUNICATION ANALYSIS")
    print("-" * 50)
    comm_stats = resource_summary.get('communication_statistics', {})
    if comm_stats:
        total_data = comm_stats.get('total_data_transferred_mb', 0)
        avg_round_data = comm_stats.get('average_round_data_mb', 0)
        max_round_data = comm_stats.get('max_round_data_mb', 0)
        comm_eff = comm_stats.get('communication_efficiency', 0)
        
        print(f"Total data transferred:    {total_data:.1f}MB ({total_data/1024:.2f}GB)")
        print(f"Average per round:         {avg_round_data:.1f}MB")
        print(f"Maximum per round:         {max_round_data:.1f}MB")
        print(f"Communication efficiency:  {comm_eff*100:.1f}%")
        
        # Estimate network requirements
        if total_time > 0:
            avg_bandwidth = (total_data * 8) / total_time  # Mbps
            print(f"Average bandwidth usage:   {avg_bandwidth:.2f}Mbps")
    
    # Efficiency metrics
    rounds = resource_summary.get('rounds_completed', 0)
    if rounds > 0:
        print(f"\n📊 EFFICIENCY METRICS")
        print("-" * 50)
        print(f"Rounds completed:          {rounds}")
        print(f"Time per round:            {total_time/rounds:.2f}s")
        print(f"Training efficiency:       {(training_time/total_time)*100:.1f}%")
        print(f"Communication overhead:    {(comm_time/total_time)*100:.1f}%")
        print(f"Aggregation overhead:      {(agg_time/total_time)*100:.1f}%")
        
        if peak_mem > 0:
            print(f"Memory efficiency:         {avg_mem/peak_mem*100:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN AWFedAvg EXPERIMENT WITH COMPREHENSIVE RESOURCE TRACKING
# ──────────────────────────────────────────────────────────────────────────────

def run_awfedavg_experiment_with_resources():
    """Run the AWFedAvg federated learning experiment with comprehensive resource tracking."""
    print("="*80)
    print("ADAPTIVE WEIGHTED FEDAVG WITH COMPREHENSIVE RESOURCE TRACKING")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Strategy: AWFedAvg with Resource Monitoring")
    print(f"  - Clients: {NUM_CLIENTS}")
    print(f"  - Rounds: {TOTAL_ROUNDS}")
    print(f"  - Local Epochs: {LOCAL_EPOCHS}")
    print(f"  - Evaluation Episodes: {EVALUATION_EPISODES}")
    print(f"  - Device: {DEVICE}")
    print(f"  - Resource Monitoring: Enabled (CPU, Memory, Network, GPU)")
    print(f"  - Weighting: eMBB (20%) + URLLC (40%) + Activation (20%) + Stability (20%)")
    print("="*80)
    
    # Initialize system monitoring
    system_monitor = ResourceMonitor()
    experiment_start_time = time.time()
    system_monitor.start_monitoring(interval=0.5)  # Less frequent for system overview
    
    # Load local baseline results for comparison
    print("\n🔄 Loading Local Baseline Results...")
    local_results = load_local_baseline_results()
    
    if local_results:
        print("Local baseline summary:")
        for client_id, results in local_results.items():
            config = CLIENT_CONFIGS[client_id % len(CLIENT_CONFIGS)]
            print(f"  Client {client_id} ({config['name']}): Reward={results['average_reward']:.4f}")
    else:
        print("No local baseline available - running AWFedAvg without comparison.")
    
    # AWFedAvg Federated Learning
    print(f"\n🌐 Starting AWFedAvg Federated Learning with Resource Tracking...")
    print("-" * 50)
    
    # AWFedAvg strategy with balanced weighting and resource tracking
    strategy = AdaptiveWeightedFedAvg(
        alpha_embb=0.22,        # 20% weight for eMBB outage minimization
        alpha_urllc=0.38,       # 40% weight for URLLC residual minimization
        alpha_activation=0.2,   # 20% weight for activation diversity
        alpha_stability=0.2,    # 20% weight for performance stability
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )
    
    print("Weight components:")
    print(f"  - eMBB Outage: {strategy.weight_calculator.alpha_embb:.1%}")
    print(f"  - URLLC Residual: {strategy.weight_calculator.alpha_urllc:.1%}")
    print(f"  - Activation Diversity: {strategy.weight_calculator.alpha_activation:.1%}")
    print(f"  - Performance Stability: {strategy.weight_calculator.alpha_stability:.1%}")
    print(f"  - Resource Monitoring: CPU, Memory, Network, GPU (if available)")
    
    # Start federated learning simulation
    print("\nStarting AWFedAvg federated learning simulation with full resource tracking...")
    
    federated_start_time = time.time()
    
    try:
        # Suppress Flower's start_simulation deprecation warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*start_simulation.*is deprecated.*")
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            
            hist = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=NUM_CLIENTS,
                config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
                strategy=strategy,
                client_resources={"num_cpus": 1, "num_gpus": 1},  # Use GPU for acceleration
            )
        
        federated_end_time = time.time()
        total_federated_time = federated_end_time - federated_start_time
        
        print(f"✅ AWFedAvg federated learning completed successfully!")
        print(f"   Total rounds: {len(hist.losses_distributed)}")
        print(f"   Total federated training time: {total_federated_time:.2f} seconds")
        
        # Get comprehensive resource summary
        resource_summary = strategy.get_comprehensive_resource_summary()
        
        print(f"   Communication time: {resource_summary.get('total_communication_time', 0):.2f}s")
        print(f"   Aggregation time: {resource_summary.get('total_aggregation_time', 0):.2f}s")
        print(f"   Peak memory usage: {resource_summary.get('resource_statistics', {}).get('peak_memory_usage_mb', 0):.1f}MB")
        
    except Exception as e:
        print(f"❌ Error during AWFedAvg federated learning: {e}")
        system_monitor.stop_monitoring()
        return None, local_results, None,None
    
    finally:
        # Stop system monitoring
        experiment_end_time = time.time()
        system_resource_summary = system_monitor.stop_monitoring()
    
    # Final Evaluation
    print(f"\n📊 Final Evaluation with Resource Analysis")
    print("-" * 50)
    
    # Evaluate the final federated model
    federated_results = None
    if strategy.get_final_parameters() is not None:
        print("Evaluating final AWFedAvg model...")
        
        eval_start_time = time.time()
        eval_monitor = ResourceMonitor()
        eval_monitor.start_monitoring(interval=0.1)
        
        try:
            # Create a representative environment (using client 0's config)
            eval_env = create_phy_env(0)
            
            # Create and load federated model
            fed_model = PPO(
                "MlpPolicy",
                eval_env,
                verbose=0,
                device=DEVICE,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
            )
            
            # Convert parameters and load into model
            # Create PyTorch model structure
            input_dim = eval_env.observation_space.shape[0]
            output_dim = eval_env.action_space.n
            pytorch_model = PPONetwork(input_dim, output_dim).to(DEVICE)
            
            # Load federated parameters
            final_params = parameters_to_ndarrays(strategy.get_final_parameters())
            model_keys = list(pytorch_model.state_dict().keys())
            
            if len(final_params) == len(model_keys):
                state_dict = OrderedDict({
                    k: torch.tensor(v, device=DEVICE)
                    for k, v in zip(model_keys, final_params)
                })
                pytorch_model.load_state_dict(state_dict, strict=True)
                fed_model = pytorch_to_sb3_ppo(pytorch_model, fed_model)
                
                # Evaluation with resource monitoring
                federated_results = evaluate_model_simple(
                    fed_model, eval_env, num_episodes=EVALUATION_EPISODES
                )
                
                eval_end_time = time.time()
                eval_duration = eval_end_time - eval_start_time
                eval_resources = eval_monitor.stop_monitoring()
                
                print(f"AWFedAvg Model Results:")
                print(f"  Reward: {federated_results['average_reward']:.4f}")
                print(f"  Stability: {federated_results['stability_score']:.4f}")
                print(f"  eMBB Outage: {federated_results['avg_embb_outage_counter']:.6f}")
                print(f"  URLLC Residual: {federated_results['avg_residual_urllc_pkt']:.6f}")
                print(f"  Evaluation time: {eval_duration:.2f}s")
                print(f"  Evaluation memory: {eval_resources.get('process_memory_mb', {}).get('peak', 0):.1f}MB")
                
                # Add evaluation resource data to results
                federated_results['evaluation_resources'] = eval_resources
                federated_results['evaluation_time'] = eval_duration
                
                # Save federated model
                fed_model_path = f"{BASE_MODEL_PATH}/awfedavg_global_model.zip"
                fed_model.save(fed_model_path)
                print(f"AWFedAvg model saved: {fed_model_path}")
                
            else:
                print(f"❌ Parameter mismatch: expected {len(model_keys)}, got {len(final_params)}")
                
        except Exception as e:
            print(f"❌ Error evaluating AWFedAvg model: {e}")
            eval_monitor.stop_monitoring()
        
        finally:
            if 'eval_env' in locals():
                eval_env.close()
    else:
        print("❌ No final parameters available from AWFedAvg training")
    
    # Comprehensive Analysis and Results
    print(f"\n📈 Performance and Resource Analysis")
    print("-" * 50)
    
    # Extract performance data
    if local_results:
        local_rewards = [results["average_reward"] for results in local_results.values()]
    else:
        local_rewards = []
    
    if federated_results and "average_reward" in federated_results:
        fed_reward = federated_results["average_reward"]
    else:
        fed_reward = None
    
    # Performance comparison
    if local_rewards:
        for i, reward in enumerate(local_rewards):
            config = CLIENT_CONFIGS[i % len(CLIENT_CONFIGS)]
            print(f"Client {i} ({config['name']:<6}): Reward = {reward:.4f}")
        
        avg_local_reward = np.mean(local_rewards)
        print(f"{'Average Local':<20}: Reward = {avg_local_reward:.4f}")
    else:
        print("No local baseline results available for comparison")
    
    if fed_reward is not None:
        print(f"{'AWFedAvg Model':<20}: Reward = {fed_reward:.4f}")
        
        if local_rewards:
            avg_local_reward = np.mean(local_rewards)
            improvement = ((fed_reward - avg_local_reward) / avg_local_reward) * 100
            print(f"\nAWFedAvg improvement: {improvement:+.2f}% over average local performance")
        else:
            print("\nAWFedAvg performance recorded (no baseline comparison available)")
    else:
        print("No federated results available")
    
    # Comprehensive resource summary
    resource_summary = strategy.get_comprehensive_resource_summary()
    print_resource_summary(resource_summary)
    
    # Save comprehensive results with resource analysis
    results_file, csv_file = save_awfedavg_results_with_resources(strategy, local_results, federated_results)
    
    # Final Summary
    print(f"\n" + "="*80)
    print("AWFedAvg EXPERIMENT WITH RESOURCE TRACKING - FINAL SUMMARY")
    print("="*80)
    
    print(f"Strategy: Adaptive Weighted FedAvg with Resource Monitoring")
    print(f"Rounds completed: {len(strategy.round_metrics)}")
    
    if strategy.performance_history:
        print(f"Final performance: {strategy.performance_history[-1]:.4f}")
        print(f"Best performance: {max(strategy.performance_history):.4f}")
        
        # Performance trend
        if len(strategy.performance_history) >= 2:
            trend = (strategy.performance_history[-1] - strategy.performance_history[0]) / len(strategy.performance_history)
            print(f"Performance trend: {trend:+.4f} per round")
    
    # Resource efficiency summary
    if resource_summary:
        total_time = resource_summary.get('total_experiment_time', 0)
        peak_memory = resource_summary.get('resource_statistics', {}).get('peak_memory_usage_mb', 0)
        total_data = resource_summary.get('communication_statistics', {}).get('total_data_transferred_mb', 0)
        
        print(f"\nResource Efficiency:")
        print(f"  Experiment duration: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"  Peak memory usage: {peak_memory:.1f}MB ({peak_memory/1024:.2f}GB)")
        print(f"  Total data transfer: {total_data:.1f}MB ({total_data/1024:.2f}GB)")
        print(f"  Average time per round: {total_time/len(strategy.round_metrics):.1f}s")
        
        # Efficiency ratios
        training_time = resource_summary.get('total_training_time', 0)
        comm_time = resource_summary.get('total_communication_time', 0)
        
        if total_time > 0:
            print(f"  Training efficiency: {(training_time/total_time)*100:.1f}%")
            print(f"  Communication overhead: {(comm_time/total_time)*100:.1f}%")
    
    # Weight evolution summary
    weight_evolution = strategy.weight_calculator.weight_history
    if weight_evolution:
        print(f"\nWeight Evolution Over {len(weight_evolution)} Rounds:")
        print("-" * 50)
        
        # Show initial and final weights
        initial_weights = weight_evolution[0]
        final_weights = weight_evolution[-1]
        
        print(f"{'Client':<8} {'Initial':<10} {'Final':<10} {'Change':<10} {'Config'}")
        print("-" * 50)
        
        for i, (init_w, final_w) in enumerate(zip(initial_weights, final_weights)):
            change = final_w - init_w
            config = CLIENT_CONFIGS[i % len(CLIENT_CONFIGS)]
            print(f"Client {i:<2} {init_w:<10.3f} {final_w:<10.3f} {change:+8.3f}   "
                  f"{config['name']} (act={config['activation']})")
        
        # Weight stability analysis
        weight_variances = []
        for client_idx in range(len(initial_weights)):
            client_weights = [round_weights[client_idx] for round_weights in weight_evolution]
            weight_variances.append(np.var(client_weights))
        
        avg_weight_stability = np.mean(weight_variances)
        print(f"\nWeight stability (lower variance = more stable): {avg_weight_stability:.6f}")
    
    print(f"\n📁 Files Generated:")
    print(f"  - Detailed results: {os.path.basename(results_file)}")
    print(f"  - Round-by-round CSV: {os.path.basename(csv_file)}")
    print(f"  - AWFedAvg model: awfedavg_global_model.zip")
    
    print("="*80)
    print("✅ AWFedAvg experiment with comprehensive resource tracking completed!")
    print(f"📂 All results saved to: {BASE_MODEL_PATH}/")
    print("📊 Use the CSV file for detailed analysis and visualization")
    print("="*80)
    
    return hist, local_results, federated_results, resource_summary

# ──────────────────────────────────────────────────────────────────────────────
# EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Check for required packages
    required_packages = ['psutil', 'memory-profiler']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️  Missing required packages for resource monitoring:")
        for pkg in missing_packages:
            print(f"    pip install {pkg}")
        print("Install these packages for full resource tracking functionality.")
        print("The script will continue with limited monitoring capabilities.\n")
    
    # Check for optional GPU monitoring
    try:
        import GPUtil
        print("✅ GPU monitoring available")
    except ImportError:
        print("ℹ️  GPU monitoring not available (install GPUtil for GPU tracking)")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    print("🚀 AWFedAvg with Comprehensive Resource Tracking")
    print("\nThis enhanced script provides:")
    print("  ✅ Adaptive weight calculation with 4 criteria")
    print("  ✅ Real-time resource monitoring (CPU, Memory, GPU)")
    print("  ✅ Round-by-round performance tracking")
    print("  ✅ Communication overhead analysis")
    print("  ✅ Training efficiency metrics")
    print("  ✅ Comprehensive CSV export for analysis")
    print("  ✅ Resource usage visualization data")
    print("  ✅ Time breakdown for each phase")
    
    # Check if local baseline results exist
    local_files = glob.glob(f"{BASE_MODEL_PATH}/local_baseline_results_*.json")
    if not local_files:
        print("\n⚠️  WARNING: No local baseline results found!")
        print("   The script will run without baseline comparison.")
        print("   You can run local baseline training later for comparison.")
    
    # Run AWFedAvg experiment with comprehensive resource tracking
    print("\n🎯 Running AWFedAvg Experiment with Resource Tracking...")
    
    try:
        history, local_results, federated_results, resource_summary = run_awfedavg_experiment_with_resources()
        
        print("\n🎉 AWFedAvg experiment with resource tracking completed successfully!")
        print(f"📊 Resource tracking captured:")
        if resource_summary:
            rounds = resource_summary.get('rounds_completed', 0)
            total_time = resource_summary.get('total_experiment_time', 0)
            peak_mem = resource_summary.get('resource_statistics', {}).get('peak_memory_usage_mb', 0)
            total_data = resource_summary.get('communication_statistics', {}).get('total_data_transferred_mb', 0)
            
            print(f"    - {rounds} rounds of detailed monitoring")
            print(f"    - {total_time:.1f}s total experiment time")
            print(f"    - {peak_mem:.1f}MB peak memory usage")
            print(f"    - {total_data:.1f}MB total data transfer")
        
        print(f"\n📂 Generated files in '{BASE_MODEL_PATH}/':")
        print("    - awfedavg_resource_results_[timestamp].json (detailed results)")
        print("    - awfedavg_round_resources_[timestamp].csv (round-by-round data)")
        print("    - awfedavg_global_model.zip (trained model)")
        
        print(f"\n💡 Next steps:")
        print("    - Analyze the CSV file for resource trends")
        print("    - Use the detailed JSON for comprehensive analysis")
        print("    - Compare resource usage across different rounds")
        print("    - Visualize communication patterns and efficiency")
        
    except Exception as e:
        print(f"❌ Error during AWFedAvg experiment: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 AWFedAvg Resource Tracking Experiment Complete!")