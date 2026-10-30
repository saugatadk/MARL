"""
env/observation_extractor.py
TraCI observation and metrics extractor for IOE Pashchimanchal Campus traffic signal control.
Extracts normalized lane queue lengths, vehicle waiting times, phase states, and metrics.
"""

from typing import List, Dict, Tuple
import numpy as np
import traci

class ObservationExtractor:
    """
    Extracts and normalizes state observations and traffic metrics
    from SUMO TraCI for the target signalized intersection.
    """

    def __init__(
        self,
        tls_id: str = "3700742618",
        max_green: float = 60.0,
        max_wait_norm: float = 300.0,
        vehicle_space_m: float = 7.5,
        alpha: float = 0.5
    ):
        self.tls_id = tls_id
        self.max_green = max_green
        self.max_wait_norm = max_wait_norm
        self.vehicle_space_m = vehicle_space_m
        self.alpha = alpha

        # Internal cache of incoming lanes and their capacities
        self.incoming_lanes: List[str] = []
        self.lane_capacities: Dict[str, float] = {}

        # Episode metrics tracking
        self.total_waiting_time_accum: float = 0.0
        self.total_queue_accum: float = 0.0
        self.measurement_steps: int = 0
        self.departed_vehicles: set = set()
        self.arrived_vehicles: set = set()

    def initialize(self):
        """Initializes lane lists and static capacities once TraCI simulation is running."""
        controlled = traci.trafficlight.getControlledLanes(self.tls_id)
        # Deduplicate while preserving order
        self.incoming_lanes = list(dict.fromkeys(controlled))

        for lane_id in self.incoming_lanes:
            try:
                length = traci.lane.getLength(lane_id)
                cap = max(2.0, length / self.vehicle_space_m)
            except Exception:
                cap = 20.0
            self.lane_capacities[lane_id] = cap

        self.reset_metrics()

    def reset_metrics(self):
        """Resets tracking accumulators at episode start."""
        self.total_waiting_time_accum = 0.0
        self.total_queue_accum = 0.0
        self.measurement_steps = 0
        self.departed_vehicles.clear()
        self.arrived_vehicles.clear()

    @property
    def num_lanes(self) -> int:
        return len(self.incoming_lanes)

    @property
    def observation_dim(self) -> int:
        # num_lanes queues + num_lanes waiting times + 2 phase one-hot + 1 elapsed time
        return (self.num_lanes * 2) + 2 + 1

    def get_observation(self, current_phase: int, elapsed_phase_time: float) -> np.ndarray:
        """
        Constructs normalized observation vector:
          - Normalized queue length per lane [0, 1]
          - Normalized waiting time per lane [0, 1]
          - Phase one-hot [2] (Phase 0 vs Phase 2)
          - Normalized elapsed green time [0, 1]
        """
        queue_lengths = []
        waiting_times = []

        for lane_id in self.incoming_lanes:
            # Halting count: vehicles with speed < 0.1 m/s
            halting = traci.lane.getLastStepHaltingNumber(lane_id)
            cap = self.lane_capacities.get(lane_id, 20.0)
            norm_queue = min(1.0, float(halting) / cap)
            queue_lengths.append(norm_queue)

            # Total accumulated waiting time on the lane
            wait = traci.lane.getWaitingTime(lane_id)
            norm_wait = min(1.0, float(wait) / self.max_wait_norm)
            waiting_times.append(norm_wait)

        # One-hot encoding of current active green phase:
        # Phase 0: NS Green -> [1.0, 0.0]
        # Phase 2: EW Green -> [0.0, 1.0]
        if current_phase == 0:
            phase_one_hot = [1.0, 0.0]
        else:
            phase_one_hot = [0.0, 1.0]

        # Normalized elapsed time in current phase
        norm_elapsed = min(1.0, float(elapsed_phase_time) / self.max_green)

        obs = np.array(queue_lengths + waiting_times + phase_one_hot + [norm_elapsed], dtype=np.float32)
        return obs

    def compute_reward(self) -> Tuple[float, float, float]:
        """
        Computes standard reward:
          r_t = - sum_{l in L} ( w_l(t) + alpha * q_l(t) )
        Returns:
          (reward, total_wait, total_queue)
        """
        total_wait = 0.0
        total_queue = 0.0

        for lane_id in self.incoming_lanes:
            q = float(traci.lane.getLastStepHaltingNumber(lane_id))
            w = float(traci.lane.getWaitingTime(lane_id))
            total_wait += w
            total_queue += q

        raw_penalty = total_wait + (self.alpha * total_queue)
        reward = -raw_penalty

        # Accumulate metrics
        self.total_waiting_time_accum += total_wait
        self.total_queue_accum += total_queue
        self.measurement_steps += 1

        return reward, total_wait, total_queue

    def record_step_vehicles(self):
        """Tracks departed and arrived vehicles for throughput computation."""
        departed = traci.simulation.getDepartedIDList()
        arrived = traci.simulation.getArrivedIDList()
        self.departed_vehicles.update(departed)
        self.arrived_vehicles.update(arrived)

    def get_episode_metrics(self) -> Dict[str, float]:
        """Calculates summary evaluation metrics for the episode."""
        steps = max(1, self.measurement_steps)
        avg_wait = self.total_waiting_time_accum / steps
        avg_queue = self.total_queue_accum / steps
        throughput = len(self.arrived_vehicles)

        return {
            "avg_waiting_time": float(avg_wait),
            "avg_queue_length": float(avg_queue),
            "throughput": float(throughput)
        }
