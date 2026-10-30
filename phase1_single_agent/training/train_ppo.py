#!/usr/bin/env python3
"""
training/train_ppo.py
Trains a Proximal Policy Optimization (PPO) agent for adaptive traffic signal control
at IOE Pashchimanchal Campus, Pokhara using Stable-Baselines3.
"""

import os
import sys
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from env.sumo_env import SumoTrafficEnv
from training.callbacks import TrafficMetricsCallback

def train_ppo(
    total_timesteps: int = 50000,
    learning_rate: float = 3e-4,
    n_steps: int = 720,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    seed: int = 42,
    save_dir: str = "checkpoints",
    log_dir: str = "logs"
):
    save_path = os.path.join(BASE_DIR, "training", save_dir)
    log_path = os.path.join(BASE_DIR, "training", log_dir, "ppo")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print("==========================================================")
    print(" Training Single-Agent PPO: IOE Pashchimanchal Campus")
    print(f" Total Timesteps:     {total_timesteps:,}")
    print(f" Learning Rate:       {learning_rate}")
    print(f" Network Arch:        pi=[128, 128], vf=[128, 128]")
    print(f" Horizon (n_steps):   {n_steps}")
    print(f" TensorBoard Logs:    {log_path}")
    print(f" Checkpoint Output:   {save_path}")
    print("==========================================================")

    # Instantiate environment wrapped in Monitor for episode stats
    raw_env = SumoTrafficEnv(max_sim_time=3600, delta_time=5, seed=seed)
    env = Monitor(raw_env, filename=os.path.join(log_path, "monitor.csv"))

    # Define PPO policy and value network architecture [128, 128]
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_path,
        seed=seed,
        verbose=1
    )

    # Callbacks
    metrics_cb = TrafficMetricsCallback(verbose=1)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1000, total_timesteps // 5),
        save_path=save_path,
        name_prefix="ppo_ckpt"
    )
    callbacks = CallbackList([metrics_cb, checkpoint_cb])

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    finally:
        # Save final model
        final_model_path = os.path.join(save_path, "ppo_paschimanchal.zip")
        model.save(final_model_path)
        print(f"[train_ppo] Final PPO model saved successfully to: {final_model_path}")
        env.close()

    return final_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on IOE Pashchimanchal Campus Traffic")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total environment timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log directory")
    args = parser.parse_args()

    train_ppo(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        seed=args.seed,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
