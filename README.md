# Intelligent Crop Treatment Drone using Reinforcement Learning

![Crop Treatment Drone on a 2D Farm Demo](./Drone-2D-Simulation.gif)

A reinforcement learning system where a drone agent navigates a 2D farm grid, identifies unhealthy (sick) crops, and applies pesticide treatment efficiently. The project implements four RL algorithms (DQN, PPO, A2C, REINFORCE) with a hyperparameter sweep framework and uses a Gymnasium-compatible Pygame-based simulation.

## Features

- **Pygame 2D Simulation** — Gymnasium-compatible 5x5 grid-based farm environment with real-time Pygame rendering
- **Multiple RL Algorithms** — DQN, PPO, A2C (via Stable Baselines3), and REINFORCE (custom PyTorch implementation)
- **Hyperparameter Sweep** — 8 experiments per algorithm (32 total) with automated CSV result logging via `experiments.py`
- **Proximity Reward Shaping** — Manhattan-distance shaping guides the drone toward sick crops
- **CSV Training Results** — Automated post-training evaluation saves metrics to `results/training_results.csv` and `results/sweep_results.csv`
- **TensorBoard Logging** — Full training metric visualization (rewards, episode length, crops treated)
- **Modular Architecture** — Clean separation between environment, training, and evaluation code

## Project Structure

```
├── environment/
│   ├── __init__.py
│   ├── env_wrapper.py              # Gymnasium wrapper + Pygame 2D simulation
│   └── config.py                   # Environment configuration (grid, rewards, etc.)
│
├── training/
│   ├── __init__.py
│   ├── dqn_training.py             # Deep Q-Network training
│   ├── ppo_training.py             # Proximal Policy Optimization training
│   ├── a2c_training.py             # Advantage Actor-Critic training
│   ├── reinforce_training.py       # Vanilla Policy Gradient (custom PyTorch)
│   └── utils.py                    # Shared utilities, callbacks, and CSV export
│
├── models/                         # Saved model checkpoints (per algorithm + per experiment)
│   ├── dqn/
│   ├── ppo/
│   ├── a2c/
│   ├── reinforce/
│   ├── ppo_01_baseline/ ... ppo_08_aggressive/
│   ├── a2c_01_baseline/ ... a2c_08_aggressive/
│   └── reinforce_01_baseline/ ...
│
├── results/
│   ├── training_results.csv        # Per-algorithm training results
│   ├── sweep_results.csv           # Hyperparameter sweep results across all experiments
│   ├── logs/                       # TensorBoard logs
│   └── plots/                      # Generated plots
│
├── logs/                           # TensorBoard logs for sweep experiments
│
├── scripts/
│   ├── run_random_agent.py         # Baseline random agent
│   └── evaluate_model.py           # Model evaluation script
│
├── experiments.py                  # Hyperparameter sweep runner (8 × 4 algorithms)
├── main.py                         # Run inference with best trained model
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Intelligent-Crop-Treatment-Drone

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `stable-baselines3` — PPO, A2C, DQN implementations
- `gymnasium` — RL environment API
- `torch` — Neural networks (REINFORCE + SB3 backend)
- `numpy` — Numerical computation
- `tensorboard` — Training visualization
- `matplotlib` — Plotting
- `pygame` — 2D environment rendering

## Usage

### 1. Run the Random Agent (Baseline)

Verify the environment works and establish a performance baseline:

```bash
python scripts/run_random_agent.py
python scripts/run_random_agent.py --episodes 5 --render
```

### 2. Train Individual Models

Train any of the four RL algorithms. All scripts use optimised hyperparameters by default and automatically save evaluation results to `results/training_results.csv`.

```bash
# PPO (recommended — best performance with vectorized environments)
python -m training.ppo_training

# A2C (fast training with parallel environments)
python -m training.a2c_training

# DQN (off-policy with experience replay)
python -m training.dqn_training

# REINFORCE (custom PyTorch vanilla policy gradient)
python -m training.reinforce_training --episodes 10000
```

All training scripts support these flags:

- `--timesteps` / `--episodes` — training duration
- `--seed` — random seed (default: 42)

### 3. Run Hyperparameter Sweep

Run systematic experiments across all algorithms (8 experiments each) to find optimal configurations:

```bash
# Run everything (32 experiments)
python experiments.py

# Run one algorithm only
python experiments.py --algo ppo
python experiments.py --algo reinforce
python experiments.py --algo a2c
python experiments.py --algo dqn

# Run a specific experiment (1-8)
python experiments.py --algo ppo --exp 3

# Dry run — print configs without training
python experiments.py --dry-run

# View results summary sorted by accuracy
python experiments.py --summary
```

Sweep results are saved to `results/sweep_results.csv`.

### 4. Monitor Training with TensorBoard

```bash
# For individual training runs
tensorboard --logdir results/logs/

# For sweep experiments
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

### 5. View Training Results

After training, results are automatically appended to CSV files with the following columns:

| Column                 | Description                                        |
| ---------------------- | -------------------------------------------------- |
| algorithm              | Algorithm name (ppo, dqn, a2c, reinforce)          |
| timestamp              | When training completed                            |
| total_timesteps        | Training duration                                  |
| mean_reward            | Average evaluation reward                          |
| std_reward             | Reward standard deviation                          |
| mean_crops_treated     | Average number of sick crops treated               |
| mean_episode_length    | Average steps per episode                          |
| best_reward            | Highest reward across evaluation episodes          |
| treatment_accuracy_pct | Percentage of unhealthy crops successfully treated |

### 6. Evaluate a Trained Model

```bash
python scripts/evaluate_model.py --algo ppo --episodes 10
python scripts/evaluate_model.py --algo reinforce --render
```

### 7. Run Inference with the Best Model

```bash
python main.py                          # auto-detects best model
python main.py --algo ppo --episodes 10
python main.py --render                 # with Pygame visualization
```

## Environment Details

### Farm Grid

A 5x5 grid (25 cells) where 50% of crops start as unhealthy. The drone must navigate to each sick crop and spray it before running out of steps (max 100) or pesticide (capacity 20).

### Observation Space (28-dimensional vector)

Observations are normalised to [0, 1].

| Index | Description                                            |
| ----- | ------------------------------------------------------ |
| 0     | Drone x position (normalised)                          |
| 1     | Drone y position (normalised)                          |
| 2     | Remaining pesticide (normalised)                       |
| 3-27  | Crop health states (0=healthy, 0.5=unhealthy, 1=treated) |

### Action Space (5 discrete actions)

| Action | Description             |
| ------ | ----------------------- |
| 0      | Move +x                 |
| 1      | Move -x                 |
| 2      | Move +y                 |
| 3      | Move -y                 |
| 4      | Spray pesticide         |

### Reward Structure

| Event                       | Reward                         |
| --------------------------- | ------------------------------ |
| Each step                   | -0.05 (encourages efficiency)  |
| Spray unhealthy crop        | +15.0                          |
| Spray healthy/treated crop  | -2.0                           |
| All unhealthy crops treated | +100.0 (completion bonus)      |
| Move closer to sick crop    | +0.5 per Manhattan unit closer |
| Move away from sick crop    | -0.3 per Manhattan unit farther|

## Default Hyperparameters

### PPO

- **Network**: 256x256 (policy & value)
- **Learning rate**: 2.5e-4, **Batch size**: 128, **n_steps**: 512
- **Epochs**: 10, **Gamma**: 0.99, **GAE lambda**: 0.95
- **Entropy coef**: 0.05, **Clip range**: 0.2
- **Parallel environments**: 8, **Timesteps**: 1M

### A2C

- **Network**: 256x256 (policy & value)
- **Learning rate**: 7e-4, **n_steps**: 256
- **Gamma**: 0.99, **GAE lambda**: 0.95
- **Entropy coef**: 0.05, **Normalised advantages**: True
- **Parallel environments**: 8, **Timesteps**: 1M

### DQN

- **Network**: 256x256
- **Learning rate**: 5e-4, **Batch size**: 128
- **Buffer size**: 100K, **Learning starts**: 10,000
- **Gamma**: 0.99, **Target update interval**: 500
- **Exploration**: 50% of training, final epsilon 0.02
- **Timesteps**: 1M

### REINFORCE

- **Network**: 256x256
- **Learning rate**: 5e-4, **Gamma**: 0.99
- **Entropy bonus**: 0.05, **Gradient clipping**: 0.5
- **Episodes**: 10,000

## Training Results

Full sweep results are logged to `results/sweep_results.csv`. All experiments use a 100-step episode limit and are evaluated over multiple episodes post-training.

### A2C (8 experiments — 2026-04-01)

| Experiment | Timesteps | Envs | Mean Reward | Std | Crops Treated | Best Reward | Accuracy % | Time (min) |
|---|---|---|---|---|---|---|---|---|
| a2c_01_baseline | 2,000,000 | 8 | 16.15 | 22.05 | 1.8 | 60.40 | 14.6 | 18.9 |
| a2c_02_lr_low | 2,000,000 | 8 | 21.42 | 22.13 | 2.6 | 51.70 | 21.7 | 19.3 |
| a2c_03_lr_high | 2,000,000 | 8 | 15.73 | 16.25 | 1.8 | 53.70 | 14.6 | 17.4 |
| a2c_04_high_entropy | 2,000,000 | 8 | **28.26** | 12.24 | 2.5 | 54.10 | 20.4 | 16.3 |
| a2c_05_low_entropy | 3,000,000 | 8 | 19.09 | 18.33 | 1.9 | 49.40 | 16.2 | 27.4 |
| a2c_06_short_rollout_16env | 2,000,000 | 16 | 24.00 | 13.74 | **3.0** | 47.20 | **24.6** | 14.2 |
| a2c_07_big_network | 2,000,000 | 8 | 15.33 | 18.96 | 1.8 | 45.70 | 15.0 | 20.3 |
| a2c_08_aggressive | 5,000,000 | 16 | 22.38 | 17.99 | 2.8 | **60.50** | 22.9 | 42.9 |

Key hyperparameters varied: learning rate (`0.0003`–`0.001`), entropy coef (`0.02`–`0.1`), n_envs (`8`/`16`), n_steps (`128`/`256`), network size.

### PPO (8 experiments — 2026-04-02)

| Experiment | Timesteps | Envs | Mean Reward | Std | Crops Treated | Best Reward | Accuracy % | Time (min) |
|---|---|---|---|---|---|---|---|---|
| ppo_01_baseline | 2,000,000 | 8 | 45.13 | 27.84 | 3.8 | 99.60 | 31.7 | 27.8 |
| ppo_02_lr_high | 2,000,000 | 8 | **62.03** | 22.89 | **5.0** | **123.60** | **41.7** | 39.1 |
| ppo_03_lr_low_long | 3,000,000 | 8 | 37.96 | 23.33 | 3.4 | 82.90 | 27.9 | 61.6 |
| ppo_04_high_entropy | 2,000,000 | 8 | 41.73 | 17.99 | 3.5 | 76.00 | 29.2 | 42.4 |
| ppo_05_low_entropy_exploit | 3,000,000 | 8 | 59.70 | 25.90 | 4.6 | 117.70 | 38.3 | 47.3 |
| ppo_06_big_network | 2,000,000 | 8 | 51.03 | 21.21 | 4.2 | 98.60 | 35.4 | 52.3 |
| ppo_07_short_rollout_16env | 2,000,000 | 16 | 31.02 | 14.80 | 3.0 | 72.30 | 25.4 | 34.8 |
| ppo_08_aggressive | 5,000,000 | 16 | 47.92 | 14.59 | 4.0 | 68.10 | 32.9 | 107.4 |

Key hyperparameters varied: learning rate (`0.0001`–`0.0005`), entropy coef (`0.02`–`0.1`), clip range (`0.15`/`0.2`), n_envs (`8`/`16`), network size.

### REINFORCE (8 experiments — 2026-04-03)

| Experiment | Episodes | Mean Reward | Std | Crops Treated | Best Reward | Accuracy % | Time (min) |
|---|---|---|---|---|---|---|---|
| reinforce_01_baseline | 15,000 | 43.74 | 29.57 | 5.2 | 117.20 | 43.8 | 65.7 |
| reinforce_02_lr_low_long | 30,000 | 49.48 | 24.29 | 5.7 | 95.50 | 47.1 | 135.0 |
| reinforce_03_lr_high | 15,000 | 44.95 | 22.10 | 4.8 | 94.20 | 39.6 | 41.4 |
| reinforce_04_high_entropy | 20,000 | 57.07 | 24.46 | **5.7** | 110.20 | **47.1** | 65.5 |
| reinforce_05_low_entropy_exploit | 30,000 | 47.97 | 21.02 | 4.8 | 87.70 | 40.0 | 75.7 |
| reinforce_06_big_network | 20,000 | 55.16 | 18.96 | 5.2 | 94.50 | 43.3 | 46.7 |
| reinforce_07_high_gamma | 20,000 | 52.14 | 21.24 | 5.4 | 101.50 | 45.0 | 42.1 |
| reinforce_08_aggressive | 50,000 | **62.47** | 25.33 | **6.2** | 115.20 | **52.1** | 104.5 |

Key hyperparameters varied: learning rate (`0.0002`–`0.001`), entropy coef (`0.02`–`0.1`), hidden size (`256`/`512`), gamma (`0.99`/`0.995`), episodes (`15K`–`50K`).

### DQN (7 experiments — 2026-04-05/06)

| Experiment | Timesteps | Mean Reward | Std | Crops Treated | Best Reward | Accuracy % | Time (min) |
|---|---|---|---|---|---|---|---|
| dqn_01_baseline | 2,000,000 | 3.52 | 7.83 | 0.5 | 19.50 | 4.2 | 35.9 |
| dqn_02_lr_low_stable | 3,000,000 | **28.92** | 12.63 | **1.9** | **49.30** | **15.4** | 54.7 |
| dqn_03_soft_update | 2,000,000 | -4.63 | 0.33 | 0.0 | -4.10 | 0.0 | 84.1 |
| dqn_04_long_explore | 3,000,000 | 12.82 | 11.69 | 0.7 | 33.80 | 5.8 | 52.9 |
| dqn_05_frequent_train | 2,000,000 | -4.24 | 3.20 | 0.1 | 9.70 | 0.4 | 126.4 |
| dqn_06_big_network | 2,000,000 | 6.52 | 10.26 | 0.6 | 34.70 | 5.0 | 124.2 |
| dqn_07_high_gamma | 3,000,000 | 7.67 | 9.38 | 0.7 | 20.30 | 5.8 | 53.9 |

Key hyperparameters varied: learning rate (`0.0001`–`0.0005`), buffer size (`100K`–`200K`), tau (`0.005`/`1.0`), exploration fraction (`0.4`–`0.7`), train frequency.

### Cross-Algorithm Summary

| Algorithm | Best Experiment | Best Mean Reward | Best Accuracy % | Best Crops Treated |
|---|---|---|---|---|
| PPO | ppo_02_lr_high | **62.03** | **41.7** | **5.0** |
| REINFORCE | reinforce_08_aggressive | 62.47 | 52.1 | 6.2 |
| A2C | a2c_04_high_entropy | 28.26 | 24.6 | 3.0 |
| DQN | dqn_02_lr_low_stable | 28.92 | 15.4 | 1.9 |

> REINFORCE achieves the highest accuracy and crops treated per episode, while PPO delivers the best mean reward with lower variance. DQN underperforms significantly on this task, likely due to the dense reward structure and multi-step decision horizon favouring on-policy methods.

## Future Improvements

- Complete Unity 3D environment with realistic farm terrain and drone physics
- Add continuous action space support for smoother drone movement
- Implement curriculum learning (progressively larger farms)
- Add multi-agent support for fleet coordination
- Integrate computer vision for crop health detection from drone camera
- Add SAC (Soft Actor-Critic) for continuous control variant
- Deploy trained models to physical drone hardware via ROS integration
