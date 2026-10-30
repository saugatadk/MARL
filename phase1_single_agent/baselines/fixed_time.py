"""
baselines/fixed_time.py
Fixed-Time Classical Baseline Controller.
Implements a static 60s cycle:
  - Phase 0: North-South Green (27s)
  - Phase 1: North-South Yellow (3s)
  - Phase 2: East-West Green (27s)
  - Phase 3: East-West Yellow (3s)
"""

import sys
import os
import argparse
from typing import Tuple, Optional, Any
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.sumo_env import SumoTrafficEnv

class FixedTimeController:
    """
    Fixed-Time Controller compatible with Stable-Baselines3 policy interface.
    Cycle duration: 60s (30s NS stage: 27s green + 3s yellow; 30s EW stage: 27s green + 3s yellow).
    """

    def __init__(self, cycle_time: int = 60, green_split: float = 0.5, delta_time: int = 5):
        self.cycle_time = cycle_time
        self.green_split = green_split
        self.delta_time = delta_time
        self.step_counter = 0

    def reset(self):
        self.step_counter = 0

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Any] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = True
    ) -> Tuple[int, Optional[Any]]:
        """
        Determines action based on static elapsed cycle time.
        With delta_time = 5s, a 60s cycle contains 12 steps:
          - Steps 0..5 (0s to 30s): Stage NS (Action 0)
          - Steps 6..11 (30s to 60s): Stage EW (Action 1)
        """
        elapsed_in_cycle = (self.step_counter * self.delta_time) % self.cycle_time
        ns_stage_duration = self.cycle_time * self.green_split

        if elapsed_in_cycle < ns_stage_duration:
            action = 0  # Phase 0: NS Green
        else:
            action = 1  # Phase 2: EW Green

        self.step_counter += 1
        return action, state

def run_fixed_time(
    max_sim_time: int = 3600,
    seed: int = 42,
    use_gui: bool = False
):
    print(f"[FixedTime] Running simulation for {max_sim_time}s (seed={seed})...")
    env = SumoTrafficEnv(max_sim_time=max_sim_time, use_gui=use_gui, seed=seed)
    controller = FixedTimeController()

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

    print("[FixedTime] Finished!")
    print(f"  Total Steps:      {steps}")
    print(f"  Total Reward:     {total_reward:.2f}")
    print(f"  Avg Waiting Time: {info['avg_waiting_time']:.2f} s")
    print(f"  Avg Queue Length: {info['avg_queue_length']:.2f} veh")
    print(f"  Throughput:       {info['throughput']:.0f} veh")
    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fixed-Time Controller Baseline")
    parser.add_argument("--sim-time", type=int, default=3600, help="Simulation duration (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gui", action="store_true", help="Launch sumo-gui")
    args = parser.parse_args()

    run_fixed_time(max_sim_time=args.sim_time, seed=args.seed, use_gui=args.gui)
