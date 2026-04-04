"""Shared training utilities for logging, evaluation, and callbacks.

Provides helper functions used across all training scripts to avoid
code duplication.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from environment.unity_env_wrapper import make_env
from environment.config import EnvConfig

# Project root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "results" / "logs"


def get_env(config: Optional[EnvConfig] = None, **kwargs) -> Monitor:
    """Create a monitored environment instance.

    Args:
        config: Optional environment configuration.
        **kwargs: Ignored (accepts legacy parameters like use_dummy).

    Returns:
        A Monitor-wrapped Gymnasium environment.
    """
    env = make_env(config=config)
    return Monitor(env)


def get_eval_callback(
    algo_name: str,
    eval_env: Optional[Monitor] = None,
    n_eval_episodes: int = 5,
    eval_freq: int = 5000,
    **kwargs,
) -> EvalCallback:
    """Create an evaluation callback that saves the best model.

    Args:
        algo_name: Algorithm name (used for save directory).
        eval_env: Evaluation environment. Created if None.
        n_eval_episodes: Episodes per evaluation round.
        eval_freq: Steps between evaluations.
        **kwargs: Ignored (accepts legacy parameters like use_dummy).

    Returns:
        A configured EvalCallback instance.
    """
    if eval_env is None:
        eval_env = get_env()

    best_model_dir = str(MODELS_DIR / algo_name)
    os.makedirs(best_model_dir, exist_ok=True)

    return EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=str(LOG_DIR / algo_name),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )


def evaluate_trained_model(
    model,
    env=None,
    n_episodes: int = 10,
    **kwargs,
) -> dict:
    """Evaluate a trained model and return summary statistics.

    Args:
        model: A trained Stable Baselines3 model.
        env: Environment for evaluation. Created if None.
        n_episodes: Number of evaluation episodes.
        **kwargs: Ignored (accepts legacy parameters like use_dummy).

    Returns:
        Dictionary with mean_reward, std_reward, and per-episode details.
    """
    if env is None:
        env = get_env()

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_episodes, deterministic=True
    )

    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "n_episodes": n_episodes,
    }


class TensorBoardLogCallback(BaseCallback):
    """Custom callback that logs additional metrics to TensorBoard.

    Logs episode-level information such as crops treated, pesticide
    remaining, and episode length.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "crops_treated" in info:
                self.logger.record("env/crops_treated", info["crops_treated"])
            if "unhealthy_remaining" in info:
                self.logger.record("env/unhealthy_remaining", info["unhealthy_remaining"])
            if "pesticide_remaining" in info:
                self.logger.record("env/pesticide_remaining", info["pesticide_remaining"])
        return True
