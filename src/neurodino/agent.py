"""Glue."""

from random import Random
from typing import TYPE_CHECKING

import numpy as np

from neurodino.config import DINO_WIDTH, GROUND_Y, MAX_SPEED, SCORE_COEFFICIENT, WORLD_WIDTH
from neurodino.sim import DINO_GROUND_Y, Game
from neurodino.specs import ObjectType

if TYPE_CHECKING:
    from neurodino.nn import NeuralNetwork

MAX_FRAMES = 5000  # the speed only tops out after 7000 frames, so this reaches speed ~11
OBJECT_FEATURES = 6
FEATURE_SIZE = 2 * OBJECT_FEATURES + 5  # two obstacles + on ground, vy, height, speed, ducking

MAX_GROUP_WIDTH = 75  # three large cacti
MAX_OBJECT_HEIGHT = 50
MAX_JUMP_VELOCITY = 15
MAX_DINO_HEIGHT = 70


def _features(game: Game) -> list[float]:
    dino = game.dino
    ahead = [ob for ob in game.objects if ob.x + ob.width > dino.x]
    feats: list[float] = []
    for i in range(2):  # nearest + second-nearest
        if i < len(ahead):
            ob = ahead[i]
            feats += [
                (ob.x - (dino.x + DINO_WIDTH)) / WORLD_WIDTH,  # distance, timing
                ob.width / MAX_GROUP_WIDTH,  # a cactus group is wider than a single cactus
                ob.height / MAX_OBJECT_HEIGHT,
                (GROUND_Y - (ob.y + ob.height)) / MAX_OBJECT_HEIGHT,  # bird altitude, 0 for cacti
                1.0 if ob.spec.kind is ObjectType.BIRD else 0.0,
                ob.speed_offset,  # birds fly faster or slower than the ground moves
            ]
        else:
            feats += [1.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # sentinel: nothing within reach
    feats += [
        0.0 if dino.jumping else 1.0,
        dino.velocity / MAX_JUMP_VELOCITY,  # vertical speed (jump physics), not world scroll
        (DINO_GROUND_Y - dino.y) / MAX_DINO_HEIGHT,  # height above the ground
        game.speed / MAX_SPEED,
        1.0 if dino.ducking else 0.0,
    ]
    return feats


def actions(nn: NeuralNetwork, game: Game) -> tuple[bool, bool]:
    """Return the `(jump, duck)` key states for this frame."""
    match nn.forward(np.array(_features(game))):
        case 1:
            return False, True
        case 2:
            return True, False
        case _:
            return False, False


def run_episode(nn: NeuralNetwork, seed: int, max_frames: int = MAX_FRAMES) -> float:
    """Play one game with the network; fitness is the in-game score, without rounding."""
    game = Game(Random(seed))
    while game.alive and game.frames < max_frames:
        jump, duck = actions(nn, game)
        game.step(jump=jump, duck=duck)
    return game.distance * SCORE_COEFFICIENT
