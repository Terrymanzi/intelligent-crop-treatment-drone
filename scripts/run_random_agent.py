"""Run a random agent in the Crop Treatment Drone environment.

Executes random actions to verify the environment works correctly
and to establish a performance baseline.

Usage:
    python scripts/run_random_agent.py
    python scripts/run_random_agent.py --episodes 5 --render
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.env_wrapper import make_env
from environment.config import EnvConfig


def run_random_agent(
    num_episodes: int = 3,
    render: bool = False,
    seed: int = 42,
) -> None:
    """Run the environment with a random action policy.

    Args:
        num_episodes: Number of episodes to run.
        render: Whether to render the environment each step via Pygame.
        seed: Random seed for reproducibility.
    """
    config = EnvConfig()
    render_mode = "human" if render else None
    env = make_env(config=config, render_mode=render_mode)

    print("=" * 60)
    print("  Random Agent — Crop Treatment Drone Environment")
    print("=" * 60)
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space:      {env.action_space}")
    print(f"  Episodes:          {num_episodes}")
    print(f"  Rendering:         {'ON (Pygame)' if render else 'OFF'}")
    print("=" * 60)

    all_rewards = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False

        print(f"\n--- Episode {ep} ---")
        print(f"  Initial obs (first 6): {obs[:6]}")
        print(f"  Initial info: {info}")

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if render:
                env.render()
                # Small delay so human can follow the drone movement
                time.sleep(0.05)

            # Print every 20 steps to avoid flooding the console
            if step_count % 20 == 0 or terminated or truncated:
                action_name = config.action_names[action]
                print(
                    f"  Step {step_count:>4d} | "
                    f"Action: {action_name:<10s} | "
                    f"Reward: {reward:>7.2f} | "
                    f"Total: {total_reward:>8.2f} | "
                    f"Done: {terminated} | Trunc: {truncated}"
                )

        print(f"\n  Episode {ep} finished:")
        print(f"    Total reward:        {total_reward:.2f}")
        print(f"    Steps:               {step_count}")
        print(f"    Crops treated:       {info.get('crops_treated', 'N/A')}")
        print(f"    Unhealthy remaining: {info.get('unhealthy_remaining', 'N/A')}")
        print(f"    Pesticide remaining: {info.get('pesticide_remaining', 'N/A')}")
        all_rewards.append(total_reward)

    env.close()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Mean reward: {np.mean(all_rewards):.2f}")
    print(f"  Std reward:  {np.std(all_rewards):.2f}")
    print(f"  Min reward:  {np.min(all_rewards):.2f}")
    print(f"  Max reward:  {np.max(all_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random agent")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_random_agent(
        num_episodes=args.episodes,
        render=args.render,
        seed=args.seed,
    )
