"""
env/sumo_env.py
Custom Gymnasium environment wrapping SUMO/TraCI for Single-Agent RL
Traffic Signal Control at IOE Pashchimanchal Campus, Pokhara.
"""

import os
import sys
from typing import Optional, Dict, Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Ensure SUMO tools can be imported
if "SUMO_HOME" in os.environ:
    tools_dir = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_dir not in sys.path:
        sys.path.append(tools_dir)
import traci

from env.observation_extractor import ObservationExtractor

class SumoTrafficEnv(gym.Env):
    """
    Gymnasium environment for adaptive traffic signal control at
    IOE Pashchimanchal Campus (Western Regional Campus), Lamachaur, Pokhara.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        net_file: Optional[str] = None,
        route_file: Optional[str] = None,
        cfg_file: Optional[str] = None,
        tls_id: str = "3700742618",
        use_gui: bool = False,
        sim_step_length: float = 1.0,
        delta_time: int = 5,
        yellow_duration: int = 3,
        min_green: int = 10,
        max_green: int = 60,
        max_sim_time: int = 3600,
        gui_delay_ms: int = 50,
        reward_scale: float = 0.01,
        alpha: float = 0.5,
        seed: Optional[int] = None
    ):
        super().__init__()

        # Paths resolution
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(base_dir, "data")
        self.cfg_file = cfg_file or os.path.join(self.data_dir, "simulation.sumocfg")
        self.net_file = net_file or os.path.join(self.data_dir, "paschimanchal.net.xml")
        self.route_file = route_file or os.path.join(self.data_dir, "routes.rou.xml")

        self.tls_id = tls_id
        self.use_gui = use_gui
        self.sim_step_length = sim_step_length
        self.delta_time = delta_time
        self.yellow_duration = yellow_duration
        self.min_green = min_green
        self.max_green = max_green
        self.max_sim_time = max_sim_time
        self.gui_delay_ms = gui_delay_ms
        self.reward_scale = reward_scale
        self.alpha = alpha

        # Phase indices:
        # 0: North-South Green
        # 1: North-South Yellow
        # 2: East-West Green
        # 3: East-West Yellow
        self.GREEN_PHASE_NS = 0
        self.YELLOW_PHASE_NS = 1
        self.GREEN_PHASE_EW = 2
        self.YELLOW_PHASE_EW = 3

        # State tracking
        self.current_green_phase: int = self.GREEN_PHASE_NS
        self.elapsed_phase_time: float = 0.0
        self.sim_time: float = 0.0
        self.is_connected: bool = False
        self._seed = seed

        # Helper extractor
        self.extractor = ObservationExtractor(
            tls_id=self.tls_id,
            max_green=float(self.max_green),
            alpha=self.alpha
        )

        # Action space: Discrete(2)
        # 0: Choose / keep North-South Green
        # 1: Choose / keep East-West Green
        self.action_space = spaces.Discrete(2)

        # Observation space: Box(low=0, high=1, shape=(K,), dtype=np.float32)
        # 6 queues + 6 wait times + 2 phase one-hot + 1 elapsed time = 15
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32
        )

    def _start_sumo(self):
        """Launches SUMO or SUMO-GUI via TraCI."""
        if self.is_connected:
            self._close_sumo()

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            sumo_binary,
            "-c", self.cfg_file,
            "--step-length", str(self.sim_step_length),
            "--no-warnings", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "300"
        ]

        if self._seed is not None:
            cmd.extend(["--seed", str(self._seed)])

        if self.use_gui:
            cmd.extend(["--start", "--delay", str(self.gui_delay_ms), "--quit-on-end"])

        traci.start(cmd)
        self.is_connected = True

        # Initialize extractor on the live simulation
        self.extractor.initialize()

        # Update observation space shape dynamically if lanes differ
        obs_dim = self.extractor.observation_dim
        if self.observation_space.shape[0] != obs_dim:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )

    def _close_sumo(self):
        """Safely terminates the TraCI connection."""
        if self.is_connected:
            try:
                traci.close()
            except Exception:
                pass
            self.is_connected = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed

        self._start_sumo()

        # Set initial phase to North-South Green
        self.current_green_phase = self.GREEN_PHASE_NS
        self.elapsed_phase_time = 0.0
        self.sim_time = 0.0
        traci.trafficlight.setPhase(self.tls_id, self.GREEN_PHASE_NS)

        # Warm-up simulation for 5 seconds to let vehicles enter
        for _ in range(5):
            traci.simulationStep()
            self.extractor.record_step_vehicles()
            self.sim_time += self.sim_step_length

        self.elapsed_phase_time = 5.0
        obs = self.extractor.get_observation(self.current_green_phase, self.elapsed_phase_time)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one RL decision step of delta_time = 5 seconds.
        Action:
          0 -> Wants North-South Green (Phase 0)
          1 -> Wants East-West Green (Phase 2)
        """
        target_green = self.GREEN_PHASE_NS if action == 0 else self.GREEN_PHASE_EW

        # Check if action requires a phase switch
        if target_green != self.current_green_phase:
            # Enforce minimum green constraint before switching
            if self.elapsed_phase_time >= self.min_green:
                # 1. Switch to corresponding yellow clearance phase
                yellow_phase = (
                    self.YELLOW_PHASE_NS
                    if self.current_green_phase == self.GREEN_PHASE_NS
                    else self.YELLOW_PHASE_EW
                )
                traci.trafficlight.setPhase(self.tls_id, yellow_phase)

                for _ in range(self.yellow_duration):
                    traci.simulationStep()
                    self.extractor.record_step_vehicles()
                    self.sim_time += self.sim_step_length

                # 2. Switch to new green phase and simulate remainder of delta_time
                traci.trafficlight.setPhase(self.tls_id, target_green)
                remaining_time = max(0, self.delta_time - self.yellow_duration)
                for _ in range(remaining_time):
                    traci.simulationStep()
                    self.extractor.record_step_vehicles()
                    self.sim_time += self.sim_step_length

                self.current_green_phase = target_green
                self.elapsed_phase_time = float(remaining_time)
            else:
                # Minimum green not reached; maintain current green for delta_time
                for _ in range(self.delta_time):
                    traci.simulationStep()
                    self.extractor.record_step_vehicles()
                    self.sim_time += self.sim_step_length
                self.elapsed_phase_time += float(self.delta_time)
        else:
            # Action matches current green phase; maintain green for delta_time
            traci.trafficlight.setPhase(self.tls_id, self.current_green_phase)
            for _ in range(self.delta_time):
                traci.simulationStep()
                self.extractor.record_step_vehicles()
                self.sim_time += self.sim_step_length
            self.elapsed_phase_time += float(self.delta_time)

        # Compute reward and metrics
        raw_reward, wait_sec, queue_len = self.extractor.compute_reward()
        scaled_reward = raw_reward * self.reward_scale

        # Observation
        obs = self.extractor.get_observation(self.current_green_phase, self.elapsed_phase_time)

        # Check termination / truncation
        terminated = False
        truncated = False

        if self.sim_time >= self.max_sim_time:
            truncated = True

        try:
            # Check if simulation has ended or no more vehicles exist
            min_expected = traci.simulation.getMinExpectedNumber()
            if min_expected <= 0 and self.sim_time > 100:
                terminated = True
        except Exception:
            terminated = True

        metrics = self.extractor.get_episode_metrics()
        info = {
            "step_wait_time": wait_sec,
            "step_queue_len": queue_len,
            "raw_reward": raw_reward,
            "avg_waiting_time": metrics["avg_waiting_time"],
            "avg_queue_length": metrics["avg_queue_length"],
            "throughput": metrics["throughput"]
        }

        return obs, scaled_reward, terminated, truncated, info

    def render(self):
        """Render mode placeholder (sumo-gui handles rendering natively)."""
        pass

    def close(self):
        self._close_sumo()
