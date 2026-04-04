"""Environment module for Intelligent Crop Treatment Drone.

Provides the Gymnasium-compatible CropTreatmentEnv with Pygame-based
2D simulation and rendering, along with configuration.
"""

from environment.unity_env_wrapper import CropTreatmentEnv
from environment.config import EnvConfig
