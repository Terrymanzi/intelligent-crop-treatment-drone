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
        Position (x, y) on the grid and remaining pesticide supply.

    Actions (Discrete 5):
        0: move +x   1: move -x
        2: move +y   3: move -y
        4: spray pesticide on current cell

    Observation (normalised to [0, 1]):
        Flattened float32 vector: [x/gs, y/gs, pesticide/cap, *crop_states/2].

    Reward:
        -0.05 per step, +15 spray unhealthy, -2 spray healthy/treated,
        +100 when all unhealthy crops are treated.
        Proximity shaping: +0.5 per Manhattan-distance unit closer to
        nearest unhealthy crop, -0.3 per unit farther.

    Termination:
        All unhealthy crops treated **or** max steps reached (truncation).
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": _FPS}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        # Gymnasium spaces — observations are normalised to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.config.observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.config.num_actions)

        # Internal state (properly initialised in reset)
        self._drone_pos = np.array([0.0, 0.0], dtype=np.float32)
        self._pesticide: int = self.config.pesticide_capacity
        self._crop_states: np.ndarray = np.zeros(
            self.config.num_crops, dtype=np.float32
        )
        self._step_count: int = 0
        self._total_unhealthy: int = 0
        self._prev_min_dist: float = 0.0

        # Pygame state (lazy-initialised)
        self._screen: Optional["pygame.Surface"] = None
        self._clock: Optional["pygame.time.Clock"] = None
        self._font: Optional["pygame.font.Font"] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random

        self._drone_pos = np.array([0.0, 0.0], dtype=np.float32)
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

        # Initialise proximity tracking
        self._prev_min_dist = self._min_dist_to_unhealthy()

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        reward = self.config.movement_penalty
        terminated = False
        truncated = False

        self._update_state(action)

        # --- Proximity shaping reward ---
        new_min_dist = self._min_dist_to_unhealthy()
        if new_min_dist is not None and self._prev_min_dist is not None:
            dist_delta = self._prev_min_dist - new_min_dist  # positive = got closer
            if dist_delta > 0:
                reward += 0.5 * dist_delta
            elif dist_delta < 0:
                reward += 0.3 * dist_delta  # negative value (penalty)
        self._prev_min_dist = new_min_dist

        # --- Spray action ---
        if action == 4:
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
        if self.render_mode == "human":
            self._render_frame()
            return None
        elif self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def close(self) -> None:
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
        gs = self.config.grid_size
        if action == 0:    # +x
            self._drone_pos[0] = min(self._drone_pos[0] + 1, gs - 1)
        elif action == 1:  # -x
            self._drone_pos[0] = max(self._drone_pos[0] - 1, 0)
        elif action == 2:  # +y
            self._drone_pos[1] = min(self._drone_pos[1] + 1, gs - 1)
        elif action == 3:  # -y
            self._drone_pos[1] = max(self._drone_pos[1] - 1, 0)
        # action == 4 is spray, handled separately

    # ------------------------------------------------------------------
    # Proximity helper
    # ------------------------------------------------------------------

    def _min_dist_to_unhealthy(self) -> Optional[float]:
        """Return Manhattan distance from drone to the nearest unhealthy crop,
        or None if no unhealthy crops remain."""
        unhealthy_indices = np.where(self._crop_states == 1.0)[0]
        if len(unhealthy_indices) == 0:
            return None
        gs = self.config.grid_size
        dx = self._drone_pos[0]
        dy = self._drone_pos[1]
        min_dist = float("inf")
        for idx in unhealthy_indices:
            cx = idx % gs
            cy = idx // gs
            dist = abs(dx - cx) + abs(dy - cy)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _spray(self) -> float:
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

    # ------------------------------------------------------------------
    # Observation / info
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the normalised observation vector.

        Returns:
            Float32 array in [0, 1]: [x_norm, y_norm, pest_norm, *crop_norm].
        """
        gs_max = max(self.config.grid_size - 1, 1)
        drone_state = np.array([
            self._drone_pos[0] / gs_max,
            self._drone_pos[1] / gs_max,
            float(self._pesticide) / max(self.config.pesticide_capacity, 1),
        ], dtype=np.float32)
        crop_norm = self._crop_states / 2.0
        return np.concatenate([drone_state, crop_norm])

    def _get_info(self) -> Dict[str, Any]:
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
        if self._screen is None:
            self._init_pygame()

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

                screen_y = (gs - 1 - y) * _CELL_SIZE
                screen_x = x * _CELL_SIZE
                rect = pygame.Rect(
                    screen_x + 2, screen_y + 2,
                    _CELL_SIZE - 4, _CELL_SIZE - 4,
                )
                pygame.draw.rect(self._screen, color, rect)

        # Draw grid lines
        for i in range(gs + 1):
            pygame.draw.line(
                self._screen, _COLOR_GRID_LINE,
                (i * _CELL_SIZE, 0),
                (i * _CELL_SIZE, gs * _CELL_SIZE),
            )
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
        pygame.draw.circle(
            self._screen, (0, 0, 0),
            (drone_screen_x, drone_screen_y),
            _DRONE_RADIUS, 2,
        )

        # Info bar (no altitude — z-axis removed)
        info_y = gs * _CELL_SIZE + 5
        remaining = int(np.sum(self._crop_states == 1.0))
        treated = int(np.sum(self._crop_states == 2.0))
        info_text = (
            f"Step: {self._step_count}  |  "
            f"Pos: ({dx},{dy})  |  "
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
            f"\nDrone pos: ({int(self._drone_pos[0])},{int(self._drone_pos[1])})  "
            f"Pesticide: {self._pesticide}  "
            f"Step: {self._step_count}"
        )
        print(grid_str)
        return grid_str


# ---------------------------------------------------------------------------
# Backwards-compatible alias
# ---------------------------------------------------------------------------

DummyCropTreatmentEnv = CropTreatmentEnv
