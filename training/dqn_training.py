"""DQN training script for the Crop Treatment Drone.

Trains a Deep Q-Network agent using Stable Baselines3.
Logs training metrics to TensorBoard and saves the best model.

Usage:
    python -m training.dqn_training
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import DQN

from training.utils import (
    LOG_DIR,
    MODELS_DIR,
    TensorBoardLogCallback,
    get_env,
    get_eval_callback,
)

# ---- Hyperparameters ----
HYPERPARAMS = {
    "learning_rate": 1e-3,
    "buffer_size": 50_000,
    "learning_starts": 1000,
    "batch_size": 64,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 4,
    "target_update_interval": 1000,
    "exploration_fraction": 0.3,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
}

TOTAL_TIMESTEPS = 100_000
ALGO_NAME = "dqn"


def train(
    total_timesteps: int = TOTAL_TIMESTEPS,
    seed: int = 42,
) -> DQN:
    """Train a DQN agent on the crop treatment environment.

    Args:
        total_timesteps: Total training steps.
        seed: Random seed for reproducibility.

    Returns:
        The trained DQN model.
    """
    env = get_env()
    log_path = str(LOG_DIR / ALGO_NAME)
    os.makedirs(log_path, exist_ok=True)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
        tensorboard_log=log_path,
        **HYPERPARAMS,
    )

    eval_callback = get_eval_callback(ALGO_NAME)
    tb_callback = TensorBoardLogCallback()

    print(f"[DQN] Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, tb_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = str(MODELS_DIR / ALGO_NAME / "dqn_final")
    model.save(final_path)
    print(f"[DQN] Final model saved to {final_path}")

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        seed=args.seed,
    )
