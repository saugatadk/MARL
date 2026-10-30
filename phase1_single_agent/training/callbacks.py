"""
training/callbacks.py
Custom Stable-Baselines3 callbacks for TensorBoard logging and model checkpointing.
Logs:
  - rollout/ep_rew_mean (standard SB3)
  - custom/avg_waiting_time
  - custom/avg_queue_length
  - custom/throughput
"""

import os
from stable_baselines3.common.callbacks import BaseCallback

class TrafficMetricsCallback(BaseCallback):
    """
    Records domain-specific traffic metrics to TensorBoard upon episode completion.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if an episode terminated in any environment in the vectorized wrapper
        for info in self.locals.get("infos", []):
            if "avg_waiting_time" in info:
                # Log step values or episode summary values
                self.logger.record("custom/step_wait_time", info.get("step_wait_time", 0.0))
                self.logger.record("custom/step_queue_len", info.get("step_queue_len", 0.0))

            # When episode ends
            if "episode" in info or self.locals.get("dones", [False])[0]:
                self.episode_count += 1
                if "avg_waiting_time" in info:
                    self.logger.record("custom/avg_waiting_time", info["avg_waiting_time"])
                    self.logger.record("custom/avg_queue_length", info["avg_queue_length"])
                    self.logger.record("custom/throughput", info["throughput"])

                if self.verbose > 0:
                    print(f"[Callback] Episode {self.episode_count} complete: "
                          f"AvgWait={info.get('avg_waiting_time', 0.0):.2f}s, "
                          f"AvgQueue={info.get('avg_queue_length', 0.0):.2f}veh, "
                          f"Throughput={info.get('throughput', 0.0):.0f}veh")
        return True
