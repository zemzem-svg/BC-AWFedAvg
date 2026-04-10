import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys

from phy_env_class import Phy, std_param
# from gym_phy_env import PhyEnv as BasePhyEnv
# from phy_env_class import Phy  


class PhyEnv(gym.Env):
    """
    Gymnasium environment wrapper for the Phy class.

    This class adapts the Phy environment to the Gymnasium API, allowing it to be used
    with RL libraries like Stable-Baselines3.
    """
    metadata = {"render_modes": ["human", None], "render_fps": 4}

    def __init__(self, render_mode=None, seed=None):
        super(PhyEnv, self).__init__()

        # Set render mode
        self.render_mode = render_mode

        # Set up default simulation parameters
        activation = 0.5  # Default URLLC activation probability
        freqs, slots, minislots, tolerable_latency, outage_prob, urllc_pkt, \
            downlink_users, uplink_users, target_rate = std_param(activation)

        # Create the Phy environment
        self.phy = Phy(
            freqs=freqs,
            slots=slots,
            minislots=minislots,
            tolerable_latency=tolerable_latency,
            outage_prob=outage_prob,
            pkt_arrival=urllc_pkt,
            downlink_users=downlink_users,
            uplink_users=uplink_users,
            target_rate=target_rate,
            rl_ver='bernoulli',
            vision_ahead=0,
            cw_tot_number=None,
            cw_class_prob=None,
            q_norm=0.0,
            ra_algorithm='random',
            seed=seed,
            env_name='phy_gym',
            reward_weight=0.5,
            render=(render_mode is not None),
            render_path=None,
            random_queue=False
        )

        # Initialize the environment
        self.phy.env_init()

        # Get the state shape for the observation space
        state_shape = self.phy.env_get_state(only_shape=True)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.phy.action_space))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=state_shape, dtype=np.float32
        )

        # Set random seed if provided
        if seed is not None:
            self.seed(seed)

        # Episode-level counters (reset in reset())
        self.urllc_successes = 0
        self.embb_outages = 0
        self.total_steps = 0
        self.max_queue_length = 0

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to start a new episode.

        Returns:
            observation (np.ndarray): The initial observation
            info (dict): Additional information
        """
        # Reset the seed if provided
        if seed is not None:
            self.seed(seed)

        # Reset episode counters
        self.urllc_successes = 0
        self.embb_outages = 0
        self.total_steps = 0
        self.max_queue_length = 0

        # Reset the Phy environment
        initial_state = self.phy.env_of()

        # Optionally render
        if self.render_mode == "human":
            self.render()

        return initial_state, {}

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): The action to take

        Returns:
            observation (np.ndarray): The new observation
            reward (float): The reward for taking this action
            done (bool): Whether the episode is terminated
            info (dict): Additional information including metrics
        """
        # Store previous state for step-by-step metrics
        prev_queue_len = len(self.phy.urllc_queue[0]) if self.phy.urllc_queue else 0
        prev_embb_outages = (
            np.count_nonzero(self.phy.cw_fun == -1) if hasattr(self.phy, 'cw_fun') else 0
        )

        # Take a step in the Phy environment
        next_state, reward, done = self.phy.env_step(action)

        # Calculate step-by-step metrics
        current_queue_len = len(self.phy.urllc_queue[0]) if self.phy.urllc_queue else 0
        current_embb_outages = (
            np.count_nonzero(self.phy.cw_fun == -1) if hasattr(self.phy, 'cw_fun') else 0
        )

        # Detect URLLC success (queue length decreased)
        urllc_success = 1 if (prev_queue_len > current_queue_len) else 0

        # Detect new eMBB outage (outage count increased)
        new_embb_outage = 1 if (current_embb_outages > prev_embb_outages) else 0

        # Update episode counters
        self.urllc_successes += urllc_success
        self.embb_outages += new_embb_outage
        self.total_steps += 1

        # Basic step info
        info = {
            "urllc_success": urllc_success,
            "embb_outage": new_embb_outage,
            "queue_length": current_queue_len,
            "total_embb_outages": current_embb_outages,
            "step_number": self.phy.step_number,
            "episode_number": self.phy.episode_number
        }

        # If episode is done, extract comprehensive metrics from Phy class
        if done:
            final_metrics = {
                # Episode summary rates
                "urllc_success_rate": self.urllc_successes / max(1, self.total_steps),
                "embb_outage_rate": self.embb_outages / max(1, self.total_steps),

                # Metrics from Phy class properties
                "urllc_delay_counter": getattr(self.phy, "urllc_delay_counter", 0),
                "embb_outage_counter": getattr(self.phy, "embb_outage_counter", 0),
                "residual_urllc_pkt": getattr(self.phy, "residual_urllc_pkt", 0),
                "urllc_success_counter": getattr(self.phy, "urllc_success_counter", 0),

                # Additional derived metrics
                "total_reward": np.sum(self.phy.reward[: self.phy.step_number + 1]) 
                                 if hasattr(self.phy, 'reward') else 0.0,
                "average_reward": np.mean(self.phy.reward[: self.phy.step_number + 1]) 
                                  if hasattr(self.phy, 'reward') and self.phy.step_number >= 0 else 0.0,

                # Queue statistics
                "final_queue_length": current_queue_len,
                "max_queue_length": getattr(self, 'max_queue_length', current_queue_len),

                # Latency statistics
                "latency_violation": bool(getattr(self.phy, "urllc_delay_counter", 0)),
                "final_latency_left": (
                    self.phy.urllc_latency_left[0]
                    if hasattr(self.phy, 'urllc_latency_left') and len(self.phy.urllc_latency_left) > 0
                    else 0
                ),

                # Resource utilization
                "embb_capacity_final": getattr(self.phy, "embb_capacity", 0),
                "puncturing_efficiency": (
                    self.urllc_successes / max(1, np.sum(self.phy.action > 0))
                    if hasattr(self.phy, 'action') else 0
                ),
            }
            info.update(final_metrics)

        # Track maximum queue length during episode
        if not hasattr(self, 'max_queue_length'):
            self.max_queue_length = current_queue_len
        else:
            self.max_queue_length = max(self.max_queue_length, current_queue_len)

        # Optionally render
        if self.render_mode == "human":
            self.render()

        return next_state, reward, done, info

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode == "human":
            self.phy.env_render()

    def close(self):
        """
        Clean up resources when environment is no longer needed.
        """
        pass

    def sample_action(self):
        """
        Sample a random action from the action space.
        Returns:
            int: A random valid action
        """
        return self.action_space.sample()

    def get_possible_actions(self):
        """
        Get all possible actions for the current state.
        Returns:
            list: List of possible actions
        """
        if hasattr(self.phy, 'urllc_state') and self.phy.urllc_state[0] > 0:
            return list(range(len(self.phy.action_space)))
        return [0]

    @property
    def current_state(self):
        """Get the current state of the environment."""
        return self.phy.env_get_state()

    @property
    def episode_metrics(self):
        """Get current episode metrics."""
        return {
            "urllc_successes": self.urllc_successes,
            "embb_outages": self.embb_outages,
            "total_steps": self.total_steps,
            "urllc_success_rate": self.urllc_successes / max(1, self.total_steps),
            "embb_outage_rate": self.embb_outages / max(1, self.total_steps)
        }


class SingleEnvPhyEnv(PhyEnv):
    """
    Subclass of PhyEnv that, at the moment 'done == True', appends the final
    (embb_outage_counter, urllc_delay_counter, residual_urllc_pkt) exactly once
    into three lists. These lists grow by one entry each episode.
    """

    def __init__(self, render_mode=None, seed=None):
        super().__init__(render_mode=render_mode, seed=seed)
        self._episode_done_flag = False
        self._embb_outage_counters = []
        self._urllc_delay_counters = []
        self._residual_urllc_pkts = []

    def reset(self, *, seed=None, options=None):
        """
        Clear the “done” flag so that we can append metrics in the next episode.
        Do NOT clear the three outer lists—they accumulate across multiple episodes.
        """
        self._episode_done_flag = False
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        """
        Wrap PhyEnv.step(action). Once done == True for the very first time
        this episode, read the final counters from 'info' and append them.
        """
        next_state, reward, done, info = super().step(action)

        if done and not self._episode_done_flag:
            embb_cnt = info.get("embb_outage_counter", 0.0)
            urllc_cnt = info.get("urllc_delay_counter", 0.0)
            res_pkt = info.get("residual_urllc_pkt", 0.0)

            self._embb_outage_counters.append(embb_cnt)
            self._urllc_delay_counters.append(urllc_cnt)
            self._residual_urllc_pkts.append(res_pkt)

            self._episode_done_flag = True

        return next_state, reward, done, info

    @property
    def embb_outage_counters(self):
        """List of final eMBB outage counters (one entry per completed episode)."""
        return self._embb_outage_counters

    @property
    def urllc_delay_counters(self):
        """List of final URLLC delay counters (one entry per completed episode)."""
        return self._urllc_delay_counters

    @property
    def residual_urllc_pkts(self):
        """List of final residual URLLC packets (one entry per completed episode)."""
        return self._residual_urllc_pkts
