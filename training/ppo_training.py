"""PPO training script for the Crop Treatment Drone.

Trains a Proximal Policy Optimization agent using Stable Baselines3.
Logs training metrics to TensorBoard and saves the best model.

Usage:
    python -m training.ppo_training
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from training.utils import (
    LOG_DIR,
    MODELS_DIR,
    TensorBoardLogCallback,
    get_env,
    get_eval_callback,
    save_training_results_csv,
)

# ---- Optimised Hyperparameters ----
# Tuned for the 5x5 crop grid with proximity shaping.
# Higher entropy keeps exploration alive while the agent learns to
# navigate toward sick crops and spray them within 100 steps.
HYPERPARAMS = {
    "learning_rate": 2.5e-4,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.05,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
}

TOTAL_TIMESTEPS = 1_000_000
ALGO_NAME = "ppo"
N_ENVS = 8


def train(
    total_timesteps: int = TOTAL_TIMESTEPS,
    seed: int = 42,
) -> PPO:
    """Train a PPO agent on the crop treatment environment.

    Args:
        total_timesteps: Total training steps.
        seed: Random seed for reproducibility.

    Returns:
        The trained PPO model.
    """
    env = make_vec_env(get_env, n_envs=N_ENVS)
    log_path = str(LOG_DIR / ALGO_NAME)
    os.makedirs(log_path, exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
        tensorboard_log=log_path,
        **HYPERPARAMS,
    )

    eval_callback = get_eval_callback(ALGO_NAME)
    tb_callback = TensorBoardLogCallback()

    print(f"[PPO] Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, tb_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = str(MODELS_DIR / ALGO_NAME / "ppo_final")
    model.save(final_path)
    print(f"[PPO] Final model saved to {final_path}")

    # Evaluate and save results to CSV
    save_training_results_csv(model, ALGO_NAME, total_timesteps, HYPERPARAMS)

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        seed=args.seed,
    )
