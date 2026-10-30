#!/usr/bin/env python3
"""
scripts/run_visual_demo.py
Launches SUMO-GUI with real-time adaptive traffic signal control replay
demonstrating the trained RL agent (or baselines) at IOE Pashchimanchal Campus, Pokhara.
Delay is set to 50ms so viewers can comfortably observe vehicle queues and signal switching.
"""

import os
import sys
import argparse
import time

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from stable_baselines3 import DQN, PPO
from env.sumo_env import SumoTrafficEnv
from baselines.fixed_time import FixedTimeController
from baselines.actuated import ActuatedController

def run_visual_demo(
    controller_type: str = "dqn",
    model_path: str = None,
    sim_time: int = 3600,
    delay_ms: int = 50,
    seed: int = 42
):
    checkpoints_dir = os.path.join(BASE_DIR, "training", "checkpoints")

    if model_path is None:
        if controller_type.lower() == "dqn":
            model_path = os.path.join(checkpoints_dir, "dqn_paschimanchal.zip")
        elif controller_type.lower() == "ppo":
            model_path = os.path.join(checkpoints_dir, "ppo_paschimanchal.zip")

    print("==========================================================")
    print(" Live Visual Replay: IOE Pashchimanchal Campus Traffic Control")
    print(f" Controller Type:  {controller_type.upper()}")
    print(f" Model Checkpoint: {model_path if controller_type in ['dqn', 'ppo'] else 'Rule-Based'}")
    print(f" Delay per Step:   {delay_ms} ms")
    print(f" Target Junction:  Lamachaur Chowk (3700742618)")
    print("==========================================================")

    # Initialize environment in GUI mode
    env = SumoTrafficEnv(
        max_sim_time=sim_time,
        use_gui=True,
        gui_delay_ms=delay_ms,
        seed=seed
    )

    # Load appropriate controller
    if controller_type.lower() == "dqn":
        if os.path.exists(model_path):
            print(f"[run_visual_demo] Loading DQN policy from {model_path}...")
            controller = DQN.load(model_path)
        else:
            print(f"[run_visual_demo] Checkpoint not found at {model_path}. Creating un-trained DQN for visual preview.")
            controller = DQN("MlpPolicy", env, policy_kwargs=dict(net_arch=[128, 128]))
    elif controller_type.lower() == "ppo":
        if os.path.exists(model_path):
            print(f"[run_visual_demo] Loading PPO policy from {model_path}...")
            controller = PPO.load(model_path)
        else:
            print(f"[run_visual_demo] Checkpoint not found at {model_path}. Creating un-trained PPO for visual preview.")
            controller = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))
    elif controller_type.lower() == "fixed":
        controller = FixedTimeController()
    elif controller_type.lower() == "actuated":
        controller = ActuatedController()
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    obs, _ = env.reset(seed=seed)
    if hasattr(controller, "reset"):
        controller.reset()

    step_count = 0
    phase_names = {0: "North-South Green", 1: "North-South Yellow", 2: "East-West Green", 3: "East-West Yellow"}

    try:
        while True:
            action, _ = controller.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            step_count += 1

            if step_count % 10 == 0:
                cur_phase = env.current_green_phase
                print(f"[Step {step_count:4d}] "
                      f"Signal: {phase_names.get(cur_phase, str(cur_phase)):<18} | "
                      f"Wait Time: {info['step_wait_time']:5.1f}s | "
                      f"Queue: {info['step_queue_len']:3.0f} veh | "
                      f"Throughput: {info['throughput']:3.0f} veh")

            if terminated or truncated:
                print("\nSimulation horizon reached. Replay completed.")
                break

    except KeyboardInterrupt:
        print("\nVisual demo paused/terminated by user.")
    finally:
        env.close()

    print("==========================================================")
    print(" Live Replay Summary:")
    print(f"  Avg Waiting Time: {info.get('avg_waiting_time', 0.0):.2f} s")
    print(f"  Avg Queue Length: {info.get('avg_queue_length', 0.0):.2f} veh")
    print(f"  Total Throughput: {info.get('throughput', 0.0):.0f} veh")
    print("==========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live SUMO-GUI replay of traffic controllers")
    parser.add_argument("--controller", "-c", type=str, default="dqn",
                        choices=["dqn", "ppo", "fixed", "actuated"],
                        help="Controller to visualize")
    parser.add_argument("--model-path", "-m", type=str, default=None,
                        help="Path to trained model .zip")
    parser.add_argument("--sim-time", "-t", type=int, default=3600,
                        help="Simulation horizon in seconds")
    parser.add_argument("--delay", "-d", type=int, default=50,
                        help="Delay in milliseconds per simulation step (default: 50)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Simulation random seed")
    args = parser.parse_args()

    run_visual_demo(
        controller_type=args.controller,
        model_path=args.model_path,
        sim_time=args.sim_time,
        delay_ms=args.delay,
        seed=args.seed
    )
