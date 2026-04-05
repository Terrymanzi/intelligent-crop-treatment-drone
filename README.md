# Intelligent Crop Treatment Drone using Reinforcement Learning

A reinforcement learning system where a drone agent navigates a 3D farm environment, identifies unhealthy crops, and applies pesticide treatment efficiently. The project uses Unity ML-Agents for simulation and Stable Baselines3 for training RL algorithms.

## Features

- **3D Simulation Environment** — Unity ML-Agents integration with a Gymnasium-compatible Python wrapper
- **Multiple RL Algorithms** — DQN, PPO, A2C, and REINFORCE implementations with configurable hyperparameters
- **Dummy Environment** — Pure-Python fallback that mirrors the Unity environment for development and debugging without Unity
- **TensorBoard Logging** — Full training metric visualization (rewards, episode length, crops treated)
- **Modular Architecture** — Clean separation between environment, training, and evaluation code

## Project Structure

```
├── unity_project/                  # Unity ML-Agents project (placeholder)
│   └── README.md
│
├── environment/
│   ├── __init__.py
│   ├── unity_env_wrapper.py        # Gymnasium wrapper + dummy fallback environment
│   └── config.py                   # Environment configuration
│
├── training/
│   ├── __init__.py
│   ├── dqn_training.py             # Deep Q-Network training
│   ├── ppo_training.py             # Proximal Policy Optimization training
│   ├── a2c_training.py             # Advantage Actor-Critic training
│   ├── reinforce_training.py       # Vanilla Policy Gradient (custom PyTorch)
│   └── utils.py                    # Shared training utilities and callbacks
│
├── models/                         # Saved model checkpoints
│   ├── dqn/
│   ├── ppo/
│   ├── a2c/
│   └── reinforce/
│
├── results/
│   ├── logs/                       # TensorBoard logs
│   └── plots/                      # Generated plots
│
├── scripts/
│   ├── run_random_agent.py         # Baseline random agent
│   └── evaluate_model.py           # Model evaluation script
│
├── main.py                         # Run inference with best trained model
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.9+
- (Optional) Unity 2021.3 LTS with ML-Agents package for 3D simulation

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

## Usage

### 1. Run the Random Agent (Baseline)

Verify the environment works by running a random agent:

```bash
python scripts/run_random_agent.py
python scripts/run_random_agent.py --episodes 5 --render
```

### 2. Train Models

Train any of the four RL algorithms:

```bash
# DQN
python -m training.dqn_training --timesteps 1000000

# PPO (my best ideal training strategy)
# increase up to least "2M timesteps"
# Use vectorized environments (huge improvement)
# Increase n_steps to "4096" (2x)
# Tune entropy for exploration to "0.02" (2x)
# increase batch size to 128 (2x)
python -m training.ppo_training --timesteps 2000000

# A2C
python -m training.a2c_training --timesteps 3000000

# REINFORCE
python -m training.reinforce_training --episodes 5000
```

All training scripts support these flags:

- `--timesteps` / `--episodes` — training duration
- `--seed` — random seed (default: 42)
- `--unity` — use the Unity environment instead of the dummy

### 3. Monitor Training with TensorBoard

```bash
tensorboard --logdir results/logs/
```

Then open http://localhost:6006 in your browser.

### 4. Evaluate a Trained Model

```bash
python scripts/evaluate_model.py --algo ppo --episodes 10
python scripts/evaluate_model.py --algo reinforce --render
```

### 5. Run Inference with the Best Model

```bash
python main.py                          # auto-detects best model
python main.py --algo ppo --episodes 10
python main.py --render                 # with environment visualization
```

## Environment Details

### Observation Space (29-dimensional vector)

| Index | Description                                            |
| ----- | ------------------------------------------------------ |
| 0-2   | Drone position (x, y, z)                               |
| 3     | Remaining pesticide                                    |
| 4-28  | Crop health states (0=healthy, 1=unhealthy, 2=treated) |

### Action Space (7 discrete actions)

| Action | Description             |
| ------ | ----------------------- |
| 0-1    | Move along x-axis (+/-) |
| 2-3    | Move along y-axis (+/-) |
| 4-5    | Move along z-axis (+/-) |
| 6      | Spray pesticide         |

### Reward Structure

| Event                       | Reward                       |
| --------------------------- | ---------------------------- |
| Each step                   | -0.1 (encourages efficiency) |
| Spray unhealthy crop        | +10.0                        |
| Spray healthy/treated crop  | -5.0                         |
| All unhealthy crops treated | +50.0 (bonus)                |

## Unity Integration

The project is designed to work without Unity using the built-in dummy environment. To connect to Unity:

1. Build the Unity project (see `unity_project/README.md`)
2. Use the `--unity` flag when training or evaluating
3. The wrapper in `environment/unity_env_wrapper.py` will connect automatically

## Future Improvements

- Complete Unity 3D environment with realistic farm terrain and drone physics
- Add continuous action space support for smoother drone movement
- Implement curriculum learning (progressively larger farms)
- Add multi-agent support for fleet coordination
- Integrate computer vision for crop health detection from drone camera
- Implement prioritized experience replay for DQN
- Add SAC (Soft Actor-Critic) for continuous control variant
- Deploy trained models to physical drone hardware via ROS integration
