"""Configs."""

from dataclasses import dataclass
from pathlib import Path

# UI
IMAGES_DIR = Path("assets")
WORLD_WIDTH = 1100
WORLD_HEIGHT = 600
GROUND_Y = 480

FPS = 60
STEP_MS = 1000 / FPS  # milliseconds per frame
SCROLL_SPEED = 8  # px/frame
GRAVITY = 0.8  # px/frame²

# Dino
DINO_HEIGHT = 94
DINO_DUCK_HEIGHT = 60
DINO_WIDTH = 88
DINO_DUCK_WIDTH = 118
JUMP_VELOCITY = -20  # px/frame
JUMP_CUT_VELOCITY = -10  # px/frame, upward speed left after releasing jump early

# Objects
SPAWN_GAP_MIN = 400  # px
SPAWN_GAP_MAX = 900  # px
GAP_SHRINK_PER_SCORE = 10  # px removed from every spawn gap per passed object
MIN_SPAWN_GAP = 250  # px, floor even at high scores
MAX_SCROLL_SPEED = 20  # px/frame, cap for the score-driven speed ramp
SPEED_GAIN_PER_SCORE = 0.2  # px/frame added per passed object
BIRD_ALTITUDES = (
    GROUND_Y - 220,  # above standing head, safe to run under
    GROUND_Y - 148,  # between standing and ducking, duck under it
    GROUND_Y - 100,  # below ducking, jump over it
)


# Training
@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters."""

    population: int = 200
    elites: int = 10  # copied unchanged into the next generation
    tournament_size: int = 5  # parents are the best of this many random picks
    crossover_prob: float = 0.5  # chance a child mixes two parents instead of cloning one
    episodes: int = 8  # games per network per generation
    hall_size: int = 20  # per-generation champions re-evaluated on held-out seeds
    holdout_episodes: int = 50  # fixed seeds used only to pick the network that gets saved
    mutation_start: float = 0.2  # noise std in the first generation
    mutation_end: float = 0.02  # noise std in the last generation
    mutation_rate: float = 0.5  # fraction of weights that get noise in each child
    workers: int = 4  # parallel evaluation processes


@dataclass(frozen=True, slots=True)
class CMAConfig:
    """Hyperparameters for CMA-ES training."""

    sigma: float = 0.5
    episodes: int = 8
    hall_size: int = 20
    holdout_episodes: int = 70
    workers: int = 4


DEFAULT_CMA = CMAConfig()
DEFAULT_TRAINING = TrainingConfig()
