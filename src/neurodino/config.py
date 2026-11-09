"""Configs."""

from dataclasses import dataclass
from pathlib import Path

# UI
IMAGES_DIR = Path("assets")
RENDER_SCALE = 2  # the sprite pack is ~2x Chromium's native sprite sizes

# World, from Chromium's default dimensions and bottom padding
WORLD_WIDTH = 600
WORLD_HEIGHT = 150
BOTTOM_PAD = 10
GROUND_Y = WORLD_HEIGHT - BOTTOM_PAD  # y of the ground line, where feet and cacti stand

FPS = 60
STEP_MS = 1000 / FPS  # milliseconds per frame

# Runner
START_SPEED = 6  # px/frame
MAX_SPEED = 13  # px/frame
ACCELERATION = 0.001  # px/frame added to the speed every frame, until MAX_SPEED
CLEAR_FRAMES = 180  # clearTime 3000 ms: no obstacles before this many frames have run
GAP_COEFFICIENT = 0.6
MAX_GAP_COEFFICIENT = 1.5
MAX_OBSTACLE_SIZE = 3  # cacti come in groups of 1 to 3
MAX_OBSTACLE_DUPLICATION = 2  # at most this many obstacles of one type in a row
SCORE_COEFFICIENT = 0.025  # score = round(distance * coefficient)

# Dino, from Chromium's T-Rex config
DINO_X = 50
DINO_WIDTH = 44
DINO_HEIGHT = 47
GRAVITY = 0.6  # px/frame²
INITIAL_JUMP_VELOCITY = -10  # px/frame, speed / 10 is added on top of it
DROP_VELOCITY = -5  # px/frame, upward speed left after releasing jump
MIN_JUMP_HEIGHT = 30  # px above the ground that must be reached before a release counts
MAX_JUMP_HEIGHT = 30  # absolute y: rising above it releases the jump automatically
SPEED_DROP_COEFFICIENT = 3  # fall speed multiplier while pressing duck in the air


@dataclass(frozen=True, slots=True)
class Config:
    """Shared training hyperparameters."""

    episodes: int = 8  # games per network per generation
    hall_size: int = 20  # per-generation champions re-evaluated on held-out seeds
    holdout_episodes: int = 50  # fixed seeds used only to pick the network that gets saved
    workers: int = 4  # parallel evaluation processes
    patience: int = 15  # stop after this many generations without a 1% mean gain; 0 disables


@dataclass(frozen=True, slots=True)
class TrainingConfig(Config):
    """Hyperparameters."""

    population: int = 200
    elites: int = 20  # copied unchanged into the next generation
    tournament_size: int = 5  # parents are the best of this many random picks
    crossover_prob: float = 0.5  # chance a child mixes two parents instead of cloning one
    mutation_start: float = 0.05  # noise std in the first generation
    mutation_end: float = 0.01  # noise std in the last generation
    mutation_rate: float = 0.5  # fraction of weights that get noise in each child


@dataclass(frozen=True, slots=True)
class CMAConfig(Config):
    """Hyperparameters for CMA-ES training."""

    holdout_episodes: int = 70
    sigma: float = 0.05  # preserves the seeded reflex; 0.5 from random init killed every starter


DEFAULT_CMA = CMAConfig()
DEFAULT_TRAINING = TrainingConfig()
