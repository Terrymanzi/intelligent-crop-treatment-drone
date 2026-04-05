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

## Future Improvements

- Complete Unity 3D environment with realistic farm terrain and drone physics
- Add continuous action space support for smoother drone movement
- Implement curriculum learning (progressively larger farms)
- Add multi-agent support for fleet coordination
- Integrate computer vision for crop health detection from drone camera
- Add SAC (Soft Actor-Critic) for continuous control variant
- Deploy trained models to physical drone hardware via ROS integration
