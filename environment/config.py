"""Configuration for the Crop Treatment Drone environment.

Centralizes all environment parameters so they can be adjusted
without modifying wrapper or training code.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvConfig:
    """Environment configuration parameters.

    Attributes:
        grid_size: Size of the farm grid (grid_size x grid_size).
        num_crops: Total number of crop cells in the grid.
        unhealthy_ratio: Fraction of crops that start as unhealthy.
        max_steps: Maximum steps per episode before truncation.
        observation_size: Dimensionality of the observation vector.
        num_actions: Number of discrete actions available to the drone.
        pesticide_capacity: Maximum pesticide units the drone can carry.
        movement_penalty: Small negative reward per step to encourage efficiency.
        spray_reward: Reward for correctly spraying an unhealthy crop.
        spray_penalty: Penalty for spraying a healthy crop (waste).
        completion_bonus: Bonus reward for treating all unhealthy crops.
    """

    # Farm layout
    grid_size: int = 5
    num_crops: int = 25
    unhealthy_ratio: float = 0.3

    # Episode limits
    max_steps: int = 200

    # Observation and action space
    # Observation: drone_x, drone_y, drone_z, pesticide_remaining,
    #              + crop_health for each cell (0=healthy, 1=unhealthy, 2=treated)
    observation_size: int = 29  # 4 (drone state) + 25 (crop states)
    num_actions: int = 7  # 0-5: move (±x, ±y, ±z), 6: spray

    # Drone resources
    pesticide_capacity: int = 15

    # Reward shaping
    movement_penalty: float = -0.1
    spray_reward: float = 10.0
    spray_penalty: float = -5.0
    completion_bonus: float = 50.0

    # Action mapping for readability
    action_names: List[str] = field(default_factory=lambda: [
        "move_+x", "move_-x",
        "move_+y", "move_-y",
        "move_+z", "move_-z",
        "spray",
    ])
