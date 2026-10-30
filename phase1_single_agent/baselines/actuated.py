"""
baselines/actuated.py
Actuated Rule-Based Controller Baseline.
Maintains green until the active green approach has an empty queue or reaches
maximum green (45s), then switches to the waiting conflicting approach.
"""

import sys
import os
import argparse
from typing import Tuple, Optional, Any
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.sumo_env import SumoTrafficEnv

class ActuatedController:
    """
    Sensor/queue-based dynamic extension baseline controller.
    Compatible with Stable-Baselines3 policy interface.
    """

    def __init__(
        self,
        min_green: float = 10.0,
        max_green: float = 45.0,
        max_green_norm_factor: float = 60.0
    ):
        self.min_green = min_green
        self.max_green = max_green
        self.max_green_norm_factor = max_green_norm_factor

    def reset(self):
        pass

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Any] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = True
    ) -> Tuple[int, Optional[Any]]:
        """
        Determines action from normalized observation vector:
          - obs[0..5]: Normalized queue lengths (0,1: East; 2,3: South; 4: West; 5: North)
          - obs[12]: 1.0 if current phase is NS, 0.0 otherwise
          - obs[13]: 1.0 if current phase is EW, 0.0 otherwise
          - obs[14]: Elapsed phase time / max_green_norm_factor
        """
        # Extract approach queues
        ew_queue = float(observation[0] + observation[1] + observation[4])
        ns_queue = float(observation[2] + observation[3] + observation[5])

        # Current phase and elapsed time
        is_ns_green = observation[12] > 0.5
        elapsed_sec = float(observation[14]) * self.max_green_norm_factor

        if is_ns_green:
            # Active green is North-South
            # If NS queue is clear and EW has waiting vehicles, or max_green reached:
            if elapsed_sec >= self.min_green:
                if (ns_queue < 0.02 and ew_queue > 0.01) or (elapsed_sec >= self.max_green and ew_queue > 0.01):
                    action = 1  # Switch to East-West
                else:
                    action = 0  # Maintain North-South Green
            else:
                action = 0  # Enforce minimum green
        else:
            # Active green is East-West
            # If EW queue is clear and NS has waiting vehicles, or max_green reached:
            if elapsed_sec >= self.min_green:
                if (ew_queue < 0.02 and ns_queue > 0.01) or (elapsed_sec >= self.max_green and ns_queue > 0.01):
                    action = 0  # Switch to North-South
                else:
                    action = 1  # Maintain East-West Green
            else:
                action = 1  # Enforce minimum green

        return action, state

def run_actuated(
    max_sim_time: int = 3600,
    seed: int = 42,
    use_gui: bool = False
):
    print(f"[Actuated] Running simulation for {max_sim_time}s (seed={seed})...")
    env = SumoTrafficEnv(max_sim_time=max_sim_time, use_gui=use_gui, seed=seed)
    controller = ActuatedController()

    obs, _ = env.reset(seed=seed)
    controller.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action, _ = controller.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    env.close()

    print("[Actuated] Finished!")
    print(f"  Total Steps:      {steps}")
    print(f"  Total Reward:     {total_reward:.2f}")
    print(f"  Avg Waiting Time: {info['avg_waiting_time']:.2f} s")
    print(f"  Avg Queue Length: {info['avg_queue_length']:.2f} veh")
    print(f"  Throughput:       {info['throughput']:.0f} veh")
    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Actuated Rule-Based Baseline")
    parser.add_argument("--sim-time", type=int, default=3600, help="Simulation duration (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gui", action="store_true", help="Launch sumo-gui")
    args = parser.parse_args()

    run_actuated(max_sim_time=args.sim_time, seed=args.seed, use_gui=args.gui)
