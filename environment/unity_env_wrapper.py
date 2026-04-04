"""Gymnasium-compatible Crop Treatment Drone environment with Pygame rendering.

This module provides:
  - CropTreatmentEnv:  A fully functional 2D grid-based farm simulation with
                       optional Pygame visualization.
  - DummyCropTreatmentEnv: Alias for CropTreatmentEnv (deprecated).

Both follow the Gymnasium API (reset, step, render, close) and are compatible
with Stable Baselines3.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.config import EnvConfig

logger = logging.getLogger(__name__)

# Pygame is optional — only needed when rendering
try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------

_CELL_SIZE = 100          # pixels per grid cell
_DRONE_RADIUS = 30        # drone circle radius in pixels
_INFO_BAR_HEIGHT = 60     # height of the status bar at the bottom
_FPS = 10                 # frames per second during human rendering

# Colours (R, G, B)
_COLOR_HEALTHY = (34, 139, 34)      # forest green
_COLOR_UNHEALTHY = (200, 30, 30)    # red
_COLOR_TREATED = (30, 100, 200)     # blue
_COLOR_DRONE = (255, 215, 0)        # gold / yellow
_COLOR_GRID_LINE = (60, 60, 60)     # dark grey
_COLOR_BG = (20, 20, 20)            # near-black background
_COLOR_TEXT = (240, 240, 240)        # white text


# ---------------------------------------------------------------------------
# Helper: create the environment
# ---------------------------------------------------------------------------

def make_env(
    config: Optional[EnvConfig] = None,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> gym.Env:
    """Factory function that returns the Crop Treatment environment.

    Args:
        config: Environment configuration. Uses defaults if None.
        render_mode: Optional render mode ('human' for Pygame window).
        **kwargs: Ignored (accepts legacy parameters like use_dummy).

    Returns:
        A Gymnasium-compatible environment instance.
    """
    config = config or EnvConfig()
    logger.info("Creating CropTreatmentEnv (Pygame simulation).")
    return CropTreatmentEnv(config, render_mode=render_mode)


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class CropTreatmentEnv(gym.Env):
    """2D grid-based crop treatment simulation with Pygame rendering.

    Farm layout:
        A ``grid_size x grid_size`` grid where each cell is a crop with state:
        0 = healthy, 1 = unhealthy, 2 = treated.

    Drone state:
        Position (x, y) on the grid, altitude z (clamped 0-3), and remaining
        pesticide supply.

    Actions (Discrete 7):
        0: move +x   1: move -x
        2: move +y   3: move -y
        4: move +z   5: move -z
        6: spray pesticide on current cell

    Observation:
        Flattened float32 vector: [x, y, z, pesticide, *crop_states].

    Reward:
        -0.1 per step, +10 spray unhealthy, -5 spray healthy/treated,
        +50 when all unhealthy crops are treated.

    Termination:
        All unhealthy crops treated **or** max steps reached (truncation).
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": _FPS}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialise the environment.

        Args:
            config: Environment parameters. Uses defaults if None.
            render_mode: 'human' to open a Pygame window, 'ansi' for text,
                         or None for no rendering.
        """
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.config.num_actions)

        # Internal state (properly initialised in reset)
        self._drone_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._pesticide: int = self.config.pesticide_capacity
        self._crop_states: np.ndarray = np.zeros(
            self.config.num_crops, dtype=np.float32
        )
        self._step_count: int = 0
        self._total_unhealthy: int = 0

        # Pygame state (lazy-initialised)
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the farm and return the initial observation.

        Args:
            seed: Optional random seed for reproducibility.
            options: Unused; kept for Gymnasium compatibility.

        Returns:
            Tuple of (observation, info dict).
        """
        super().reset(seed=seed)
        rng = self.np_random

        self._drone_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._pesticide = self.config.pesticide_capacity
        self._step_count = 0

        # Randomly assign unhealthy crops
        self._crop_states = np.zeros(self.config.num_crops, dtype=np.float32)
        num_unhealthy = int(self.config.num_crops * self.config.unhealthy_ratio)
        unhealthy_indices = rng.choice(
            self.config.num_crops, size=num_unhealthy, replace=False
        )
        self._crop_states[unhealthy_indices] = 1.0
        self._total_unhealthy = num_unhealthy

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one action and return the outcome.

        Args:
            action: Discrete action index (0-6).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self._step_count += 1
        reward = self.config.movement_penalty
        terminated = False
        truncated = False

        self._update_state(action)

        if action == 6:
            reward += self._spray()

        # Check completion — all unhealthy crops treated
        if not np.any(self._crop_states == 1.0):
            reward += self.config.completion_bonus
            terminated = True

        # Check truncation — max steps reached
        if self._step_count >= self.config.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[str]:
        """Render the current environment state.

        Returns:
            Grid string when render_mode is 'ansi', None otherwise.
        """
        if self.render_mode == "human":
            self._render_frame()
            return None
        elif self.render_mode == "ansi":
            return self._render_ansi()
        # No rendering when render_mode is None
        return None

    def close(self) -> None:
        """Release rendering resources."""
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._clock = None
            self._font = None

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def _update_state(self, action: int) -> None:
        """Apply movement actions to the drone position.

        Args:
            action: Discrete action index.
        """
        gs = self.config.grid_size
        if action == 0:    # +x
            self._drone_pos[0] = min(self._drone_pos[0] + 1, gs - 1)
        elif action == 1:  # -x
            self._drone_pos[0] = max(self._drone_pos[0] - 1, 0)
        elif action == 2:  # +y
            self._drone_pos[1] = min(self._drone_pos[1] + 1, gs - 1)
        elif action == 3:  # -y
            self._drone_pos[1] = max(self._drone_pos[1] - 1, 0)
        elif action == 4:  # +z
            self._drone_pos[2] = min(self._drone_pos[2] + 1, 3)
        elif action == 5:  # -z
            self._drone_pos[2] = max(self._drone_pos[2] - 1, 0)

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _spray(self) -> float:
        """Attempt to spray pesticide on the crop below the drone.

        Returns:
            Reward delta from the spray action.
        """
        if self._pesticide <= 0:
            return self.config.spray_penalty

        x = int(np.clip(self._drone_pos[0], 0, self.config.grid_size - 1))
        y = int(np.clip(self._drone_pos[1], 0, self.config.grid_size - 1))
        idx = y * self.config.grid_size + x

        if self._crop_states[idx] == 1.0:  # unhealthy
            self._crop_states[idx] = 2.0
            self._pesticide -= 1
            return self.config.spray_reward
        else:
            self._pesticide -= 1
            return self.config.spray_penalty

    def _compute_reward(self, action: int, spray_reward: float) -> float:
        """Compute the total reward for a step.

        Args:
            action: The action taken.
            spray_reward: Additional reward from spraying.

        Returns:
            Total reward for this step.
        """
        reward = self.config.movement_penalty + spray_reward
        if not np.any(self._crop_states == 1.0):
            reward += self.config.completion_bonus
        return reward

    # ------------------------------------------------------------------
    # Observation / info
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the flattened observation vector.

        Returns:
            Float32 array: [x, y, z, pesticide, *crop_states].
        """
        drone_state = np.array(
            [*self._drone_pos, float(self._pesticide)], dtype=np.float32
        )
        return np.concatenate([drone_state, self._crop_states])

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary information about the current state.

        Returns:
            Dictionary with step count, pesticide, and crop statistics.
        """
        remaining = int(np.sum(self._crop_states == 1.0))
        treated = int(np.sum(self._crop_states == 2.0))
        return {
            "step": self._step_count,
            "pesticide_remaining": self._pesticide,
            "unhealthy_remaining": remaining,
            "crops_treated": treated,
            "total_unhealthy": self._total_unhealthy,
        }

    # ------------------------------------------------------------------
    # Pygame rendering
    # ------------------------------------------------------------------

    def _init_pygame(self) -> None:
        """Lazily initialise Pygame display and resources."""
        if not _PYGAME_AVAILABLE:
            raise ImportError(
                "pygame is required for human rendering. "
                "Install it with: pip install pygame"
            )
        pygame.init()
        gs = self.config.grid_size
        win_w = gs * _CELL_SIZE
        win_h = gs * _CELL_SIZE + _INFO_BAR_HEIGHT
        self._screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Crop Treatment Drone — RL Simulation")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 20)

    def _render_frame(self) -> None:
        """Draw the current state to the Pygame window."""
        if self._screen is None:
            self._init_pygame()

        # Process Pygame events (required to keep the window responsive)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        gs = self.config.grid_size
        self._screen.fill(_COLOR_BG)

        # Draw grid cells
        for y in range(gs):
            for x in range(gs):
                idx = y * gs + x
                state = self._crop_states[idx]
                if state == 0.0:
                    color = _COLOR_HEALTHY
                elif state == 1.0:
                    color = _COLOR_UNHEALTHY
                else:
                    color = _COLOR_TREATED

                # Screen y is inverted: grid row 0 at bottom
                screen_y = (gs - 1 - y) * _CELL_SIZE
                screen_x = x * _CELL_SIZE
                rect = pygame.Rect(
                    screen_x + 2, screen_y + 2,
                    _CELL_SIZE - 4, _CELL_SIZE - 4,
                )
                pygame.draw.rect(self._screen, color, rect)

        # Draw grid lines
        for i in range(gs + 1):
            # Vertical
            pygame.draw.line(
                self._screen, _COLOR_GRID_LINE,
                (i * _CELL_SIZE, 0),
                (i * _CELL_SIZE, gs * _CELL_SIZE),
            )
            # Horizontal
            pygame.draw.line(
                self._screen, _COLOR_GRID_LINE,
                (0, i * _CELL_SIZE),
                (gs * _CELL_SIZE, i * _CELL_SIZE),
            )

        # Draw drone
        dx = int(self._drone_pos[0])
        dy = int(self._drone_pos[1])
        drone_screen_x = dx * _CELL_SIZE + _CELL_SIZE // 2
        drone_screen_y = (gs - 1 - dy) * _CELL_SIZE + _CELL_SIZE // 2
        pygame.draw.circle(
            self._screen, _COLOR_DRONE,
            (drone_screen_x, drone_screen_y),
            _DRONE_RADIUS,
        )
        # Drone outline for visibility
        pygame.draw.circle(
            self._screen, (0, 0, 0),
            (drone_screen_x, drone_screen_y),
            _DRONE_RADIUS, 2,
        )

        # Info bar
        info_y = gs * _CELL_SIZE + 5
        remaining = int(np.sum(self._crop_states == 1.0))
        treated = int(np.sum(self._crop_states == 2.0))
        info_text = (
            f"Step: {self._step_count}  |  "
            f"Pos: ({dx},{dy})  |  "
            f"Alt: {int(self._drone_pos[2])}  |  "
            f"Pesticide: {self._pesticide}  |  "
            f"Sick: {remaining}  Treated: {treated}"
        )
        text_surface = self._font.render(info_text, True, _COLOR_TEXT)
        self._screen.blit(text_surface, (10, info_y))

        # Legend
        legend_y = info_y + 25
        legend_items = [
            (_COLOR_HEALTHY, "Healthy"),
            (_COLOR_UNHEALTHY, "Sick"),
            (_COLOR_TREATED, "Treated"),
            (_COLOR_DRONE, "Drone"),
        ]
        lx = 10
        for color, label in legend_items:
            pygame.draw.rect(self._screen, color, (lx, legend_y, 14, 14))
            lbl = self._font.render(label, True, _COLOR_TEXT)
            self._screen.blit(lbl, (lx + 18, legend_y - 3))
            lx += 18 + lbl.get_width() + 15

        pygame.display.flip()
        self._clock.tick(_FPS)

    def _render_ansi(self) -> str:
        """Return a text representation of the farm grid.

        Returns:
            Multi-line string showing the grid, drone, and status.
        """
        gs = self.config.grid_size
        symbols = {0.0: ".", 1.0: "X", 2.0: "T"}
        lines = []
        for y in range(gs - 1, -1, -1):
            row = ""
            for x in range(gs):
                idx = int(y * gs + x)
                if (
                    int(self._drone_pos[0]) == x
                    and int(self._drone_pos[1]) == y
                ):
                    row += "D "
                else:
                    row += symbols.get(self._crop_states[idx], "?") + " "
            lines.append(row)
        grid_str = "\n".join(lines)
        grid_str += (
            f"\nDrone pos: {self._drone_pos}  "
            f"Pesticide: {self._pesticide}  "
            f"Step: {self._step_count}"
        )
        print(grid_str)
        return grid_str


# ---------------------------------------------------------------------------
# Backwards-compatible alias
# ---------------------------------------------------------------------------

DummyCropTreatmentEnv = CropTreatmentEnv
