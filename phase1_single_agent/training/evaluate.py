#!/usr/bin/env python3
"""
training/evaluate.py
Benchmarking & Evaluation script.
Evaluates Fixed-Time, Actuated, DQN, and PPO controllers across 5 random seeds
and outputs a comparative performance summary table in Markdown format.
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from stable_baselines3 import DQN, PPO
from env.sumo_env import SumoTrafficEnv
from baselines.fixed_time import FixedTimeController
from baselines.actuated import ActuatedController

def evaluate_controller(
    controller: Any,
    controller_name: str,
    seeds: List[int],
    sim_time: int = 3600
) -> Dict[str, List[float]]:
    """
    Evaluates a single controller over multiple random seeds.
    Returns dictionary with lists of metrics across seeds.
    """
    results = {
        "avg_waiting_time": [],
        "avg_queue_length": [],
        "throughput": []
    }

    print(f"\n--- Evaluating {controller_name} across {len(seeds)} seeds ({sim_time}s each) ---")

    for seed_idx, seed in enumerate(seeds):
        env = SumoTrafficEnv(max_sim_time=sim_time, seed=seed)
        obs, _ = env.reset(seed=seed)

        if hasattr(controller, "reset"):
            controller.reset()

        step_count = 0
        while True:
            action, _ = controller.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            step_count += 1

            if terminated or truncated:
                break

        env.close()

        wait_time = info.get("avg_waiting_time", 0.0)
        queue_len = info.get("avg_queue_length", 0.0)
        throughput = info.get("throughput", 0.0)

        results["avg_waiting_time"].append(wait_time)
        results["avg_queue_length"].append(queue_len)
        results["throughput"].append(throughput)

        print(f"  [Seed {seed} ({seed_idx+1}/{len(seeds)})]: "
              f"AvgWait={wait_time:.2f}s | AvgQueue={queue_len:.2f}veh | Throughput={throughput:.0f}veh")

    return results

def run_evaluation(
    sim_time: int = 3600,
    seeds: List[int] = [42, 43, 44, 45, 46],
    dqn_path: Optional[str] = None,
    ppo_path: Optional[str] = None,
    output_md: Optional[str] = None
):
    checkpoints_dir = os.path.join(BASE_DIR, "training", "checkpoints")
    dqn_path = dqn_path or os.path.join(checkpoints_dir, "dqn_paschimanchal.zip")
    ppo_path = ppo_path or os.path.join(checkpoints_dir, "ppo_paschimanchal.zip")

    controllers = {}

    # 1. Fixed-Time Baseline
    controllers["Fixed-Time"] = FixedTimeController()

    # 2. Actuated Baseline
    controllers["Actuated"] = ActuatedController()

    # 3. DQN Agent
    if os.path.exists(dqn_path):
        print(f"[evaluate] Loading DQN checkpoint: {dqn_path}")
        controllers["DQN"] = DQN.load(dqn_path)
    else:
        print(f"[evaluate] Note: DQN checkpoint not found at {dqn_path}. Running with random initialization or untrained policy.")
        dummy_env = SumoTrafficEnv(max_sim_time=100)
        controllers["DQN"] = DQN("MlpPolicy", dummy_env, policy_kwargs=dict(net_arch=[128, 128]))
        dummy_env.close()

    # 4. PPO Agent
    if os.path.exists(ppo_path):
        print(f"[evaluate] Loading PPO checkpoint: {ppo_path}")
        controllers["PPO"] = PPO.load(ppo_path)
    else:
        print(f"[evaluate] Note: PPO checkpoint not found at {ppo_path}. Running with random initialization or untrained policy.")
        dummy_env = SumoTrafficEnv(max_sim_time=100)
        controllers["PPO"] = PPO("MlpPolicy", dummy_env, policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))
        dummy_env.close()

    # Run evaluations
    all_metrics = {}
    for name, ctrl in controllers.items():
        all_metrics[name] = evaluate_controller(ctrl, name, seeds, sim_time=sim_time)

    # Compute summary stats (Mean ± Std)
    summary = {}
    for name, data in all_metrics.items():
        summary[name] = {
            "wait_mean": np.mean(data["avg_waiting_time"]),
            "wait_std": np.std(data["avg_waiting_time"]),
            "queue_mean": np.mean(data["avg_queue_length"]),
            "queue_std": np.std(data["avg_queue_length"]),
            "tp_mean": np.mean(data["throughput"]),
            "tp_std": np.std(data["throughput"])
        }

    # Best RL model comparison against Fixed-Time baseline
    ft_wait = summary["Fixed-Time"]["wait_mean"]
    ft_queue = summary["Fixed-Time"]["queue_mean"]
    ft_tp = summary["Fixed-Time"]["tp_mean"]

    # Choose best performing RL model
    rl_candidates = [m for m in ["DQN", "PPO"] if m in summary]
    best_rl = min(rl_candidates, key=lambda m: summary[m]["wait_mean"])

    best_wait = summary[best_rl]["wait_mean"]
    best_queue = summary[best_rl]["queue_mean"]
    best_tp = summary[best_rl]["tp_mean"]

    # Improvements: positive means better (lower wait/queue, higher throughput)
    wait_imp = ((ft_wait - best_wait) / max(0.001, ft_wait)) * 100.0
    queue_imp = ((ft_queue - best_queue) / max(0.001, ft_queue)) * 100.0
    tp_imp = ((best_tp - ft_tp) / max(0.001, ft_tp)) * 100.0

    # Build Markdown table
    md_lines = [
        "### Comparative Traffic Signal Control Benchmarking: IOE Pashchimanchal Campus",
        f"*Evaluated across {len(seeds)} random seeds ({seeds}) with simulation horizon = {sim_time}s.*",
        "",
        "| Metric | Fixed-Time | Actuated | DQN | PPO | Improvement (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    def fmt_cell(val_mean, val_std):
        return f"{val_mean:.2f} ± {val_std:.2f}"

    wait_row = (f"| **Avg Waiting Time (s)** | "
                f"{fmt_cell(summary['Fixed-Time']['wait_mean'], summary['Fixed-Time']['wait_std'])} | "
                f"{fmt_cell(summary['Actuated']['wait_mean'], summary['Actuated']['wait_std'])} | "
                f"{fmt_cell(summary['DQN']['wait_mean'], summary['DQN']['wait_std'])} | "
                f"{fmt_cell(summary['PPO']['wait_mean'], summary['PPO']['wait_std'])} | "
                f"**+{wait_imp:.1f}%** ({best_rl}) |")

    queue_row = (f"| **Avg Queue Length (veh)** | "
                 f"{fmt_cell(summary['Fixed-Time']['queue_mean'], summary['Fixed-Time']['queue_std'])} | "
                 f"{fmt_cell(summary['Actuated']['queue_mean'], summary['Actuated']['queue_std'])} | "
                 f"{fmt_cell(summary['DQN']['queue_mean'], summary['DQN']['queue_std'])} | "
                 f"{fmt_cell(summary['PPO']['queue_mean'], summary['PPO']['queue_std'])} | "
                 f"**+{queue_imp:.1f}%** ({best_rl}) |")

    tp_row = (f"| **Total Throughput (veh)** | "
              f"{fmt_cell(summary['Fixed-Time']['tp_mean'], summary['Fixed-Time']['tp_std'])} | "
              f"{fmt_cell(summary['Actuated']['tp_mean'], summary['Actuated']['tp_std'])} | "
              f"{fmt_cell(summary['DQN']['tp_mean'], summary['DQN']['tp_std'])} | "
              f"{fmt_cell(summary['PPO']['tp_mean'], summary['PPO']['tp_std'])} | "
              f"**+{tp_imp:.1f}%** ({best_rl}) |")

    md_lines.extend([wait_row, queue_row, tp_row, ""])
    md_content = "\n".join(md_lines)

    print("\n" + "="*70)
    print(md_content)
    print("="*70)

    if output_md:
        with open(output_md, "w") as f:
            f.write(md_content)
        print(f"[evaluate] Saved markdown report to {output_md}")

    return summary, md_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate traffic controllers across seeds")
    parser.add_argument("--sim-time", type=int, default=3600, help="Simulation duration (s)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Random seeds list")
    parser.add_argument("--dqn-path", type=str, default=None, help="Path to trained DQN model")
    parser.add_argument("--ppo-path", type=str, default=None, help="Path to trained PPO model")
    parser.add_argument("--output-md", type=str, default="evaluation_results.md", help="Output Markdown report path")
    args = parser.parse_args()

    run_evaluation(
        sim_time=args.sim_time,
        seeds=args.seeds,
        dqn_path=args.dqn_path,
        ppo_path=args.ppo_path,
        output_md=args.output_md
    )
