"""Hyperparameter sweep: 8 experiments × 4 algorithms = 32 runs.

Each experiment varies the most impactful hyperparameters for that algorithm.
Results are logged to a single CSV for easy comparison.

Usage:
    # Run everything (32 experiments)
    python experiments.py

    # Run one algorithm only
    python experiments.py --algo ppo
    python experiments.py --algo reinforce
    python experiments.py --algo a2c
    python experiments.py --algo dqn

    # Run a specific experiment
    python experiments.py --algo ppo --exp 3

    # Dry run — print configs without training
    python experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
LOG_DIR = Path("logs")
RESULTS_CSV = RESULTS_DIR / "sweep_results.csv"

# ---------------------------------------------------------------------------
# Experiment definitions — 8 per algorithm
# ---------------------------------------------------------------------------
# Design philosophy:
#   Exp 1: Baseline (current best settings)
#   Exp 2-3: Vary learning rate (the most impactful single knob)
#   Exp 4-5: Vary exploration (entropy / epsilon)
#   Exp 6: Vary network architecture
#   Exp 7: Vary rollout length / batch size
#   Exp 8: "Kitchen sink" — combine best guesses aggressively

PPO_EXPERIMENTS = [
    {
        "name": "ppo_01_baseline",
        "timesteps": 2_000_000,
        "n_envs": 8,
        "params": {
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
        },
    },
    {
        "name": "ppo_02_lr_high",
        "timesteps": 2_000_000,
        "n_envs": 8,
        "params": {
            "learning_rate": 5e-4,
            "n_steps": 512,
            "batch_size": 128,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.05,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    },
    {
        "name": "ppo_03_lr_low_long",
        "timesteps": 3_000_000,
        "n_envs": 8,
        "params": {
            "learning_rate": 1e-4,
            "n_steps": 512,
            "batch_size": 128,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.05,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    },
    {
        "name": "ppo_04_high_entropy",
        "timesteps": 2_000_000,
        "n_envs": 8,
        "params": {
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 128,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.1,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    },
    {
        "name": "ppo_05_low_entropy_exploit",
        "timesteps": 3_000_000,
        "n_envs": 8,
        "params": {
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 128,
            "n_epochs": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.15,
            "ent_coef": 0.02,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    },
    {
        "name": "ppo_06_big_network",
        "timesteps": 2_000_000,
        "n_envs": 8,
        "params": {
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
            "policy_kwargs": dict(net_arch=dict(pi=[512, 256], vf=[512, 256])),
        },
    },
    {
        "name": "ppo_07_short_rollout_16env",
        "timesteps": 2_000_000,
        "n_envs": 16,
        "params": {
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.05,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    },
    {
        "name": "ppo_08_aggressive",
        "timesteps": 5_000_000,
        "n_envs": 16,
        "params": {
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 5,
            "gamma": 0.995,
            "gae_lambda": 0.98,
            "clip_range": 0.15,
            "ent_coef": 0.03,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        },
    },
]

A2C_EXPERIMENTS = [
    {
        "name": "a2c_01_baseline",
        "timesteps": 2_000_000,
        "n_envs": 8,
        "params": {
            "learning_rate": 7e-4,
            "n_steps": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.05,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
        },
    },
    {
        "name": "a2c_02_lr_low",
        "timesteps": 2_000_000,
        "n_envs": 8,
        "params": {
            "learning_rate": 3e-4,
            "n_steps": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.05,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
        },
    },
    {
        "name": "a2c_03_lr_high",
        "timesteps": 2_000_000,
        "n_envs": 8,
        "params": {
            "learning_rate": 1e-3,
            "n_steps": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.05,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
        },
    },
    {
        "name": "a2c_04_high_entropy",
        "timesteps": 2_000_000,
        "n_envs": 8,
        "params": {
            "learning_rate": 7e-4,
            "n_steps": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.1,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
        },
    },
    {
        "name": "a2c_05_low_entropy",
        "timesteps": 3_000_000,
        "n_envs": 8,
        "params": {
            "learning_rate": 5e-4,
            "n_steps": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.02,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
        },
    },
    {
        "name": "a2c_06_short_rollout_16env",
        "timesteps": 2_000_000,
        "n_envs": 16,
        "params": {
            "learning_rate": 5e-4,
            "n_steps": 128,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.05,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
        },
    },
    {
        "name": "a2c_07_big_network",
        "timesteps": 2_000_000,
        "n_envs": 8,
        "params": {
            "learning_rate": 5e-4,
            "n_steps": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.05,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
            "policy_kwargs": dict(net_arch=dict(pi=[512, 256], vf=[512, 256])),
        },
    },
    {
        "name": "a2c_08_aggressive",
        "timesteps": 5_000_000,
        "n_envs": 16,
        "params": {
            "learning_rate": 5e-4,
            "n_steps": 128,
            "gamma": 0.995,
            "gae_lambda": 0.98,
            "ent_coef": 0.03,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
            "policy_kwargs": dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        },
    },
]

DQN_EXPERIMENTS = [
    {
        "name": "dqn_01_baseline",
        "timesteps": 2_000_000,
        "n_envs": 1,
        "params": {
            "learning_rate": 5e-4,
            "buffer_size": 100_000,
            "learning_starts": 10_000,
            "batch_size": 128,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 500,
            "exploration_fraction": 0.5,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.02,
        },
    },
    {
        "name": "dqn_02_lr_low_stable",
        "timesteps": 3_000_000,
        "n_envs": 1,
        "params": {
            "learning_rate": 1e-4,
            "buffer_size": 200_000,
            "learning_starts": 20_000,
            "batch_size": 256,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.5,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.02,
        },
    },
    {
        "name": "dqn_03_soft_update",
        "timesteps": 2_000_000,
        "n_envs": 1,
        "params": {
            "learning_rate": 3e-4,
            "buffer_size": 100_000,
            "learning_starts": 10_000,
            "batch_size": 128,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "target_update_interval": 1,
            "exploration_fraction": 0.4,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.02,
        },
    },
    {
        "name": "dqn_04_long_explore",
        "timesteps": 3_000_000,
        "n_envs": 1,
        "params": {
            "learning_rate": 5e-4,
            "buffer_size": 100_000,
            "learning_starts": 10_000,
            "batch_size": 128,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 500,
            "exploration_fraction": 0.7,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
        },
    },
    {
        "name": "dqn_05_frequent_train",
        "timesteps": 2_000_000,
        "n_envs": 1,
        "params": {
            "learning_rate": 3e-4,
            "buffer_size": 100_000,
            "learning_starts": 10_000,
            "batch_size": 64,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 2,
            "target_update_interval": 500,
            "exploration_fraction": 0.5,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.02,
        },
    },
    {
        "name": "dqn_06_big_network",
        "timesteps": 2_000_000,
        "n_envs": 1,
        "params": {
            "learning_rate": 3e-4,
            "buffer_size": 150_000,
            "learning_starts": 15_000,
            "batch_size": 128,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 500,
            "exploration_fraction": 0.5,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.02,
            "policy_kwargs": dict(net_arch=[512, 256]),
        },
    },
    {
        "name": "dqn_07_high_gamma",
        "timesteps": 3_000_000,
        "n_envs": 1,
        "params": {
            "learning_rate": 3e-4,
            "buffer_size": 200_000,
            "learning_starts": 10_000,
            "batch_size": 128,
            "tau": 1.0,
            "gamma": 0.995,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.5,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.02,
        },
    },
    {
        "name": "dqn_08_aggressive",
        "timesteps": 5_000_000,
        "n_envs": 1,
        "params": {
            "learning_rate": 1e-4,
            "buffer_size": 300_000,
            "learning_starts": 25_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "target_update_interval": 1,
            "exploration_fraction": 0.4,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.02,
            "policy_kwargs": dict(net_arch=[256, 256]),
        },
    },
]

REINFORCE_EXPERIMENTS = [
    {
        "name": "reinforce_01_baseline",
        "episodes": 15_000,
        "params": {
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "hidden_size": 256,
            "entropy_coef": 0.05,
            "max_grad_norm": 0.5,
        },
    },
    {
        "name": "reinforce_02_lr_low_long",
        "episodes": 30_000,
        "params": {
            "learning_rate": 2e-4,
            "gamma": 0.99,
            "hidden_size": 256,
            "entropy_coef": 0.05,
            "max_grad_norm": 0.5,
        },
    },
    {
        "name": "reinforce_03_lr_high",
        "episodes": 15_000,
        "params": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "hidden_size": 256,
            "entropy_coef": 0.05,
            "max_grad_norm": 0.5,
        },
    },
    {
        "name": "reinforce_04_high_entropy",
        "episodes": 20_000,
        "params": {
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "hidden_size": 256,
            "entropy_coef": 0.1,
            "max_grad_norm": 0.5,
        },
    },
    {
        "name": "reinforce_05_low_entropy_exploit",
        "episodes": 30_000,
        "params": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "hidden_size": 256,
            "entropy_coef": 0.02,
            "max_grad_norm": 0.5,
        },
    },
    {
        "name": "reinforce_06_big_network",
        "episodes": 20_000,
        "params": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "hidden_size": 512,
            "entropy_coef": 0.05,
            "max_grad_norm": 0.5,
        },
    },
    {
        "name": "reinforce_07_high_gamma",
        "episodes": 20_000,
        "params": {
            "learning_rate": 5e-4,
            "gamma": 0.995,
            "hidden_size": 256,
            "entropy_coef": 0.05,
            "max_grad_norm": 1.0,
        },
    },
    {
        "name": "reinforce_08_aggressive",
        "episodes": 50_000,
        "params": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "hidden_size": 256,
            "entropy_coef": 0.03,
            "max_grad_norm": 0.5,
        },
    },
]

ALL_EXPERIMENTS = {
    "ppo": PPO_EXPERIMENTS,
    "a2c": A2C_EXPERIMENTS,
    "dqn": DQN_EXPERIMENTS,
    "reinforce": REINFORCE_EXPERIMENTS,
}


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "experiment_name",
    "algorithm",
    "timestamp",
    "total_timesteps_or_episodes",
    "n_envs",
    "mean_reward",
    "std_reward",
    "mean_crops_treated",
    "mean_episode_length",
    "best_reward",
    "treatment_accuracy_pct",
    "wall_time_minutes",
    "hyperparameters",
]


def init_csv():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def append_csv(row: List[Any]):
    with open(RESULTS_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ---------------------------------------------------------------------------
# Evaluation (shared across SB3 algorithms)
# ---------------------------------------------------------------------------

def evaluate_sb3(model, n_episodes: int = 20) -> Dict[str, float]:
    """Evaluate a trained SB3 model and return summary stats."""
    from environment.config import EnvConfig
    from environment.env_wrapper import make_env

    config = EnvConfig()
    env = make_env(config=config)
    rewards, treated, lengths = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        treated.append(info.get("crops_treated", 0))
        lengths.append(info.get("step", 0))

    env.close()
    total_unhealthy = int(config.num_crops * config.unhealthy_ratio)
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_treated": np.mean(treated),
        "mean_length": np.mean(lengths),
        "best_reward": np.max(rewards),
        "accuracy": (np.mean(treated) / total_unhealthy * 100) if total_unhealthy > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Runners per algorithm
# ---------------------------------------------------------------------------

def run_ppo(exp: Dict) -> Dict[str, float]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from training.utils import get_env

    n_envs = exp.get("n_envs", 8)
    env = make_vec_env(get_env, n_envs=n_envs)

    log_path = str(LOG_DIR / exp["name"])
    os.makedirs(log_path, exist_ok=True)
    save_dir = MODELS_DIR / exp["name"]
    os.makedirs(save_dir, exist_ok=True)

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_path, **exp["params"])
    model.learn(total_timesteps=exp["timesteps"], progress_bar=True)
    model.save(str(save_dir / "model"))

    env.close()
    return evaluate_sb3(model)


def run_a2c(exp: Dict) -> Dict[str, float]:
    from stable_baselines3 import A2C
    from stable_baselines3.common.env_util import make_vec_env
    from training.utils import get_env

    n_envs = exp.get("n_envs", 8)
    env = make_vec_env(get_env, n_envs=n_envs)

    log_path = str(LOG_DIR / exp["name"])
    os.makedirs(log_path, exist_ok=True)
    save_dir = MODELS_DIR / exp["name"]
    os.makedirs(save_dir, exist_ok=True)

    model = A2C("MlpPolicy", env, verbose=0, tensorboard_log=log_path, **exp["params"])
    model.learn(total_timesteps=exp["timesteps"], progress_bar=True)
    model.save(str(save_dir / "model"))

    env.close()
    return evaluate_sb3(model)


def run_dqn(exp: Dict) -> Dict[str, float]:
    from stable_baselines3 import DQN
    from training.utils import get_env

    env = get_env()

    log_path = str(LOG_DIR / exp["name"])
    os.makedirs(log_path, exist_ok=True)
    save_dir = MODELS_DIR / exp["name"]
    os.makedirs(save_dir, exist_ok=True)

    model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=log_path, **exp["params"])
    model.learn(total_timesteps=exp["timesteps"], progress_bar=True)
    model.save(str(save_dir / "model"))

    env.close()
    return evaluate_sb3(model)


def run_reinforce(exp: Dict) -> Dict[str, float]:
    """Run REINFORCE with custom hyperparameters."""
    import torch
    import torch.optim as optim
    from torch.distributions import Categorical

    from environment.config import EnvConfig
    from environment.env_wrapper import make_env
    from training.reinforce_training import PolicyNetwork, compute_returns

    config = EnvConfig()
    env = make_env(config=config)

    p = exp["params"]
    policy = PolicyNetwork(
        obs_size=config.observation_size,
        n_actions=config.num_actions,
        hidden_size=p["hidden_size"],
    )
    optimizer = optim.Adam(policy.parameters(), lr=p["learning_rate"])

    num_episodes = exp["episodes"]
    max_steps = config.max_steps

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=42 + episode)
        log_probs, entropies, rewards = [], [], []

        for _ in range(max_steps):
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            logits = policy(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()

            obs, reward, terminated, truncated, info = env.step(action.item())
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            rewards.append(reward)

            if terminated or truncated:
                break

        returns = compute_returns(rewards, p["gamma"])
        returns_t = torch.FloatTensor(returns)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss_terms = [
            -lp * G - p["entropy_coef"] * ent
            for lp, ent, G in zip(log_probs, entropies, returns_t)
        ]
        optimizer.zero_grad()
        torch.stack(loss_terms).sum().backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), p["max_grad_norm"])
        optimizer.step()

        if episode % 2000 == 0:
            print(f"    {exp['name']} — episode {episode}/{num_episodes}")

    env.close()

    # Save model
    save_dir = MODELS_DIR / exp["name"]
    os.makedirs(save_dir, exist_ok=True)
    torch.save(policy.state_dict(), save_dir / "model.pt")

    # Evaluate
    policy.eval()
    eval_env = make_env(config=config)
    rewards_list, treated_list, lengths_list = [], [], []

    for _ in range(20):
        obs, _ = eval_env.reset()
        ep_reward, done = 0.0, False
        steps = 0
        while not done:
            with torch.no_grad():
                action, _ = policy.select_action(obs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
        rewards_list.append(ep_reward)
        treated_list.append(info.get("crops_treated", 0))
        lengths_list.append(steps)

    eval_env.close()
    total_unhealthy = int(config.num_crops * config.unhealthy_ratio)
    return {
        "mean_reward": np.mean(rewards_list),
        "std_reward": np.std(rewards_list),
        "mean_treated": np.mean(treated_list),
        "mean_length": np.mean(lengths_list),
        "best_reward": np.max(rewards_list),
        "accuracy": (np.mean(treated_list) / total_unhealthy * 100) if total_unhealthy > 0 else 0,
    }


RUNNERS = {
    "ppo": run_ppo,
    "a2c": run_a2c,
    "dqn": run_dqn,
    "reinforce": run_reinforce,
}


# ---------------------------------------------------------------------------
# Main sweep logic
# ---------------------------------------------------------------------------

def run_experiment(algo: str, exp: Dict):
    """Run a single experiment, log results to CSV."""
    name = exp["name"]
    runner = RUNNERS[algo]

    units = exp.get("timesteps", exp.get("episodes", "?"))
    n_envs = exp.get("n_envs", 1)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {algo.upper()} | {units} steps/episodes | {n_envs} envs")
    print(f"  params: {exp['params']}")
    print(f"{'='*60}")

    start = time.time()
    results = runner(exp)
    wall_min = (time.time() - start) / 60.0

    hp_str = "; ".join(f"{k}={v}" for k, v in exp["params"].items())
    row = [
        name,
        algo,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        units,
        n_envs,
        f"{results['mean_reward']:.2f}",
        f"{results['std_reward']:.2f}",
        f"{results['mean_treated']:.1f}",
        f"{results['mean_length']:.1f}",
        f"{results['best_reward']:.2f}",
        f"{results['accuracy']:.1f}",
        f"{wall_min:.1f}",
        hp_str,
    ]
    append_csv(row)

    print(f"\n  RESULT: reward={results['mean_reward']:.2f} | "
          f"treated={results['mean_treated']:.1f} | "
          f"accuracy={results['accuracy']:.1f}% | "
          f"time={wall_min:.1f}min")
    print(f"  {'>>> 80%+ ACCURACY!' if results['accuracy'] >= 80 else ''}")

    return results


def print_summary():
    """Print a sorted summary of all experiments from CSV."""
    if not RESULTS_CSV.exists():
        return
    with open(RESULTS_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    rows.sort(key=lambda r: -float(r.get("treatment_accuracy_pct", 0)))

    print(f"\n{'='*80}")
    print("  SWEEP RESULTS — sorted by treatment accuracy")
    print(f"{'='*80}")
    print(f"{'Experiment':<30} {'Algo':<10} {'Reward':>8} {'Acc%':>6} {'Time':>7}")
    print(f"{'-'*30} {'-'*10} {'-'*8} {'-'*6} {'-'*7}")

    for r in rows:
        print(
            f"{r['experiment_name']:<30} "
            f"{r['algorithm']:<10} "
            f"{float(r['mean_reward']):>8.1f} "
            f"{float(r['treatment_accuracy_pct']):>5.1f}% "
            f"{float(r['wall_time_minutes']):>6.1f}m"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep: 8 × 4 algorithms")
    parser.add_argument("--algo", type=str, choices=["ppo", "a2c", "dqn", "reinforce"],
                        help="Run only this algorithm (default: all)")
    parser.add_argument("--exp", type=int, help="Run only experiment N (1-8) for the chosen algo")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without training")
    parser.add_argument("--summary", action="store_true", help="Print results summary and exit")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    algos = [args.algo] if args.algo else ["ppo", "a2c", "dqn", "reinforce"]

    if args.dry_run:
        for algo in algos:
            exps = ALL_EXPERIMENTS[algo]
            for i, exp in enumerate(exps, 1):
                units = exp.get("timesteps", exp.get("episodes", "?"))
                print(f"[{i}/8] {exp['name']} — {units} steps/ep — {exp['params']}")
        return

    init_csv()

    for algo in algos:
        exps = ALL_EXPERIMENTS[algo]
        if args.exp:
            if 1 <= args.exp <= len(exps):
                exps = [exps[args.exp - 1]]
            else:
                print(f"Error: --exp must be 1-{len(exps)}")
                return

        print(f"\n{'#'*60}")
        print(f"  Starting {algo.upper()} sweep — {len(exps)} experiments")
        print(f"{'#'*60}")

        for exp in exps:
            try:
                run_experiment(algo, exp)
            except Exception as e:
                print(f"\n  ERROR in {exp['name']}: {e}")
                import traceback
                traceback.print_exc()
                append_csv([
                    exp["name"], algo,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    exp.get("timesteps", exp.get("episodes", "?")),
                    exp.get("n_envs", 1),
                    "ERROR", "", "", "", "", "", "",
                    str(e),
                ])

    print_summary()


if __name__ == "__main__":
    main()
