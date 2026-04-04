"""Evaluate a trained model on the Crop Treatment Drone environment.

Loads a saved model and runs evaluation episodes, printing performance
metrics and optionally rendering the environment.

Usage:
    python scripts/evaluate_model.py --algo ppo
    python scripts/evaluate_model.py --algo dqn --episodes 20 --render
    python scripts/evaluate_model.py --algo reinforce --model-path models/reinforce/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import DQN, PPO, A2C

from environment.unity_env_wrapper import make_env
from environment.config import EnvConfig
from training.reinforce_training import PolicyNetwork

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Mapping from algorithm name to SB3 class
SB3_ALGOS = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}


def evaluate_sb3_model(
    algo_name: str,
    model_path: str | None,
    num_episodes: int,
    render: bool,
) -> None:
    """Evaluate a Stable Baselines3 model.

    Args:
        algo_name: One of 'dqn', 'ppo', 'a2c'.
        model_path: Path to the saved model. Auto-detected if None.
        num_episodes: Number of evaluation episodes.
        render: Whether to render each step.
    """
    algo_cls = SB3_ALGOS[algo_name]

    if model_path is None:
        # Try best_model first, then final
        best = MODELS_DIR / algo_name / "best_model.zip"
        final = MODELS_DIR / algo_name / f"{algo_name}_final.zip"
        if best.exists():
            model_path = str(best)
        elif final.exists():
            model_path = str(final)
        else:
            print(f"No saved model found for {algo_name}. Train one first.")
            return

    print(f"Loading {algo_name.upper()} model from: {model_path}")
    model = algo_cls.load(model_path)
    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)

    _run_evaluation(model, env, num_episodes, render, algo_name, sb3=True)
    env.close()


def evaluate_reinforce_model(
    model_path: str | None,
    num_episodes: int,
    render: bool,
) -> None:
    """Evaluate a REINFORCE (PyTorch) model.

    Args:
        model_path: Path to the saved .pt file. Auto-detected if None.
        num_episodes: Number of evaluation episodes.
        render: Whether to render each step.
    """
    config = EnvConfig()

    if model_path is None:
        best = MODELS_DIR / "reinforce" / "best_model.pt"
        final = MODELS_DIR / "reinforce" / "reinforce_final.pt"
        if best.exists():
            model_path = str(best)
        elif final.exists():
            model_path = str(final)
        else:
            print("No saved REINFORCE model found. Train one first.")
            return

    print(f"Loading REINFORCE model from: {model_path}")
    policy = PolicyNetwork(
        obs_size=config.observation_size,
        n_actions=config.num_actions,
    )
    policy.load_state_dict(torch.load(model_path, weights_only=True))
    policy.eval()

    render_mode = "human" if render else None
    env = make_env(config=config, render_mode=render_mode)
    _run_evaluation(policy, env, num_episodes, render, "reinforce", sb3=False)
    env.close()


def _run_evaluation(
    model,
    env,
    num_episodes: int,
    render: bool,
    algo_name: str,
    sb3: bool,
) -> None:
    """Shared evaluation loop.

    Args:
        model: Trained model (SB3 or PolicyNetwork).
        env: Gymnasium environment.
        num_episodes: Number of episodes to run.
        render: Whether to render.
        algo_name: Algorithm name for display.
        sb3: True if model is a Stable Baselines3 model.
    """
    all_rewards = []
    all_lengths = []
    all_treated = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if sb3:
                action, _ = model.predict(obs, deterministic=True)
            else:
                with torch.no_grad():
                    action, _ = model.select_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if render:
                env.render()
                time.sleep(0.05)

        all_rewards.append(total_reward)
        all_lengths.append(steps)
        all_treated.append(info.get("crops_treated", 0))

        print(
            f"  Episode {ep:>3d} | "
            f"Reward: {total_reward:>8.2f} | "
            f"Steps: {steps:>4d} | "
            f"Treated: {info.get('crops_treated', 'N/A')}"
        )

    print(f"\n{'=' * 60}")
    print(f"  {algo_name.upper()} Evaluation Summary ({num_episodes} episodes)")
    print(f"{'=' * 60}")
    print(f"  Mean reward:     {np.mean(all_rewards):>8.2f} +/- {np.std(all_rewards):.2f}")
    print(f"  Mean ep length:  {np.mean(all_lengths):>8.1f}")
    print(f"  Mean treated:    {np.mean(all_treated):>8.1f}")
    print(f"  Best reward:     {np.max(all_rewards):>8.2f}")
    print(f"  Worst reward:    {np.min(all_rewards):>8.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["dqn", "ppo", "a2c", "reinforce"],
        help="Algorithm to evaluate",
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.algo == "reinforce":
        evaluate_reinforce_model(args.model_path, args.episodes, args.render)
    else:
        evaluate_sb3_model(args.algo, args.model_path, args.episodes, args.render)
