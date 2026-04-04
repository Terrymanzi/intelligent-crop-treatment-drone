"""Main entry point — load the best trained model and run inference.

Automatically detects the best available model across all algorithms
(by checking for saved model files) and runs it in the environment,
printing performance metrics.

Usage:
    python main.py
    python main.py --algo ppo --episodes 10 --render
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import DQN, PPO, A2C

from environment.unity_env_wrapper import make_env
from environment.config import EnvConfig
from training.reinforce_training import PolicyNetwork

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

# Algorithm registry: name -> (SB3 class or None, file extension)
ALGO_REGISTRY = {
    "ppo": (PPO, ".zip"),
    "dqn": (DQN, ".zip"),
    "a2c": (A2C, ".zip"),
    "reinforce": (None, ".pt"),
}


def find_best_model() -> tuple[str, Path] | None:
    """Scan the models directory and return the first available best model.

    Checks algorithms in priority order: PPO, DQN, A2C, REINFORCE.

    Returns:
        Tuple of (algo_name, model_path) or None if no model is found.
    """
    for algo_name, (_, ext) in ALGO_REGISTRY.items():
        best = MODELS_DIR / algo_name / f"best_model{ext}"
        final = MODELS_DIR / algo_name / f"{algo_name}_final{ext}"
        if best.exists():
            return algo_name, best
        if final.exists():
            return algo_name, final
    return None


def load_model(algo_name: str, model_path: Path, config: EnvConfig):
    """Load a trained model from disk.

    Args:
        algo_name: Algorithm identifier.
        model_path: Path to the saved model file.
        config: Environment configuration (needed for REINFORCE).

    Returns:
        Loaded model and a boolean indicating whether it is SB3.
    """
    if algo_name == "reinforce":
        policy = PolicyNetwork(
            obs_size=config.observation_size,
            n_actions=config.num_actions,
        )
        policy.load_state_dict(torch.load(model_path, weights_only=True))
        policy.eval()
        return policy, False

    algo_cls = ALGO_REGISTRY[algo_name][0]
    model = algo_cls.load(str(model_path))
    return model, True


def run_inference(
    algo_name: str | None = None,
    num_episodes: int = 5,
    render: bool = False,
) -> None:
    """Load the best model and run inference episodes.

    Args:
        algo_name: Algorithm to use. Auto-detects if None.
        num_episodes: Number of inference episodes.
        render: Whether to render the environment.
    """
    config = EnvConfig()

    if algo_name is not None:
        # Try to find the specified algorithm's model
        ext = ALGO_REGISTRY[algo_name][1]
        best = MODELS_DIR / algo_name / f"best_model{ext}"
        final = MODELS_DIR / algo_name / f"{algo_name}_final{ext}"
        if best.exists():
            model_path = best
        elif final.exists():
            model_path = final
        else:
            print(f"No model found for '{algo_name}'. Train one first:")
            print(f"  python -m training.{algo_name}_training")
            return
    else:
        result = find_best_model()
        if result is None:
            print("No trained models found. Train a model first:")
            print("  python -m training.ppo_training")
            print("  python -m training.dqn_training")
            print("  python -m training.a2c_training")
            print("  python -m training.reinforce_training")
            return
        algo_name, model_path = result

    print(f"Loading {algo_name.upper()} model from: {model_path}")
    model, is_sb3 = load_model(algo_name, model_path, config)

    render_mode = "human" if render else None
    env = make_env(config=config, render_mode=render_mode)

    print(f"\nRunning {num_episodes} inference episodes...\n")

    all_rewards = []
    all_treated = []
    all_steps = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if is_sb3:
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
        all_treated.append(info.get("crops_treated", 0))
        all_steps.append(steps)

        print(
            f"  Episode {ep:>3d} | "
            f"Reward: {total_reward:>8.2f} | "
            f"Steps: {steps:>4d} | "
            f"Treated: {info.get('crops_treated', 'N/A')}/{info.get('total_unhealthy', 'N/A')}"
        )

    env.close()

    # Performance summary
    print(f"\n{'=' * 60}")
    print(f"  Performance Summary — {algo_name.upper()}")
    print(f"{'=' * 60}")
    print(f"  Episodes:         {num_episodes}")
    print(f"  Mean reward:      {np.mean(all_rewards):>8.2f} +/- {np.std(all_rewards):.2f}")
    print(f"  Mean steps:       {np.mean(all_steps):>8.1f}")
    print(f"  Mean crops treated: {np.mean(all_treated):>6.1f}")
    print(f"  Best episode:     {np.max(all_rewards):>8.2f}")
    print(f"  Worst episode:    {np.min(all_rewards):>8.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with the best trained crop treatment drone model"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=["dqn", "ppo", "a2c", "reinforce"],
        help="Algorithm to use (auto-detects if omitted)",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    run_inference(
        algo_name=args.algo,
        num_episodes=args.episodes,
        render=args.render,
    )
