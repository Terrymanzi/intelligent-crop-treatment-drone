"""REINFORCE (vanilla policy gradient) training script for the Crop Treatment Drone.

Implements the REINFORCE algorithm from scratch using PyTorch, since
Stable Baselines3 does not include a pure REINFORCE implementation.

Logs training metrics to TensorBoard and saves the best model.

Usage:
    python -m training.reinforce_training
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from environment.env_wrapper import make_env
from environment.config import EnvConfig
from training.utils import MODELS_DIR, LOG_DIR, RESULTS_DIR

# ---- Optimised Hyperparameters ----
# REINFORCE with larger network, entropy bonus for exploration,
# gradient clipping for stability, and more episodes for convergence.
# gamma=0.99 matches the on-policy SB3 scripts; 10k episodes gives
# enough samples for the vanilla PG estimator to converge.
HYPERPARAMS = {
    "learning_rate": 5e-4,
    "gamma": 0.99,
    "hidden_size": 256,
    "num_episodes": 10_000,
    "max_steps_per_episode": 100,
    "log_interval": 100,
    "save_interval": 1000,
    "entropy_coef": 0.05,
    "max_grad_norm": 0.5,
}

ALGO_NAME = "reinforce"


class PolicyNetwork(nn.Module):
    """Simple feedforward policy network for REINFORCE.

    Maps observations to action probabilities via a two-layer MLP
    with ReLU activations and a softmax output.
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action logits from an observation."""
        return self.network(x)

    def select_action(
        self, state: np.ndarray
    ) -> Tuple[int, torch.Tensor]:
        """Sample an action from the policy and return its log-probability.

        Args:
            state: Current observation as a numpy array.

        Returns:
            Tuple of (action index, log probability of the chosen action).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self.forward(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """Compute discounted returns for a full episode.

    Args:
        rewards: List of rewards collected during the episode.
        gamma: Discount factor.

    Returns:
        List of discounted returns (same length as rewards).
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train(
    num_episodes: int = HYPERPARAMS["num_episodes"],
    seed: int = 42,
) -> PolicyNetwork:
    """Train a REINFORCE agent on the crop treatment environment.

    Args:
        num_episodes: Total training episodes.
        seed: Random seed for reproducibility.

    Returns:
        The trained policy network.
    """
    import csv
    from datetime import datetime

    torch.manual_seed(seed)
    np.random.seed(seed)

    config = EnvConfig()
    env = make_env(config=config)

    policy = PolicyNetwork(
        obs_size=config.observation_size,
        n_actions=config.num_actions,
        hidden_size=HYPERPARAMS["hidden_size"],
    )
    optimizer = optim.Adam(policy.parameters(), lr=HYPERPARAMS["learning_rate"])

    # TensorBoard logging
    log_path = str(LOG_DIR / ALGO_NAME)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    # Model save directory
    save_dir = MODELS_DIR / ALGO_NAME
    os.makedirs(save_dir, exist_ok=True)

    best_avg_reward = -float("inf")
    reward_history: List[float] = []
    treated_history: List[int] = []
    length_history: List[int] = []

    print(f"[REINFORCE] Starting training for {num_episodes} episodes...")

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=seed + episode)
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        rewards: List[float] = []

        for _ in range(HYPERPARAMS["max_steps_per_episode"]):
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            logits = policy(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            obs, reward, terminated, truncated, info = env.step(action.item())
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)

            if terminated or truncated:
                break

        # Compute discounted returns
        returns = compute_returns(rewards, HYPERPARAMS["gamma"])
        returns_tensor = torch.FloatTensor(returns)

        # Normalize returns for training stability
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        # Policy gradient loss with entropy bonus
        policy_loss = []
        for log_prob, ent, G in zip(log_probs, entropies, returns_tensor):
            policy_loss.append(-log_prob * G - HYPERPARAMS["entropy_coef"] * ent)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.parameters(), HYPERPARAMS["max_grad_norm"]
        )
        optimizer.step()

        episode_reward = sum(rewards)
        reward_history.append(episode_reward)
        treated_history.append(info.get("crops_treated", 0))
        length_history.append(len(rewards))

        # TensorBoard logging
        writer.add_scalar("train/episode_reward", episode_reward, episode)
        writer.add_scalar("train/episode_length", len(rewards), episode)
        writer.add_scalar("train/loss", loss.item(), episode)
        writer.add_scalar("env/crops_treated", info.get("crops_treated", 0), episode)

        # Periodic logging
        if episode % HYPERPARAMS["log_interval"] == 0:
            avg_reward = np.mean(reward_history[-HYPERPARAMS["log_interval"]:])
            avg_treated = np.mean(treated_history[-HYPERPARAMS["log_interval"]:])
            print(
                f"  Episode {episode}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Treated: {avg_treated:.1f} | "
                f"Loss: {loss.item():.4f}"
            )
            writer.add_scalar("train/avg_reward", avg_reward, episode)

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_path = save_dir / "best_model.pt"
                torch.save(policy.state_dict(), best_path)

        # Periodic checkpoint
        if episode % HYPERPARAMS["save_interval"] == 0:
            ckpt_path = save_dir / f"reinforce_ep{episode}.pt"
            torch.save(policy.state_dict(), ckpt_path)

    # Save final model
    final_path = save_dir / "reinforce_final.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"[REINFORCE] Final model saved to {final_path}")

    # Evaluate final model and save CSV results
    policy.eval()
    eval_rewards = []
    eval_treated = []
    eval_lengths = []
    eval_env = make_env(config=config)

    for ep in range(20):
        obs, _ = eval_env.reset()
        ep_reward = 0.0
        steps = 0
        done = False
        while not done:
            with torch.no_grad():
                action, _ = policy.select_action(obs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
        eval_rewards.append(ep_reward)
        eval_treated.append(info.get("crops_treated", 0))
        eval_lengths.append(steps)
    eval_env.close()

    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    csv_path = RESULTS_DIR / "training_results.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "algorithm", "timestamp", "total_episodes",
                "mean_reward", "std_reward", "mean_crops_treated",
                "mean_episode_length", "best_reward",
                "treatment_accuracy_pct", "hyperparameters",
            ])
        total_unhealthy = int(config.num_crops * config.unhealthy_ratio)
        accuracy = (np.mean(eval_treated) / total_unhealthy * 100) if total_unhealthy > 0 else 0
        hp_str = "; ".join(f"{k}={v}" for k, v in HYPERPARAMS.items())
        w.writerow([
            ALGO_NAME,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            num_episodes,
            f"{np.mean(eval_rewards):.2f}",
            f"{np.std(eval_rewards):.2f}",
            f"{np.mean(eval_treated):.1f}",
            f"{np.mean(eval_lengths):.1f}",
            f"{np.max(eval_rewards):.2f}",
            f"{accuracy:.1f}",
            hp_str,
        ])
    print(f"[REINFORCE] Results appended to {csv_path}")

    writer.close()
    env.close()
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REINFORCE agent")
    parser.add_argument(
        "--episodes", type=int, default=HYPERPARAMS["num_episodes"]
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        seed=args.seed,
    )
