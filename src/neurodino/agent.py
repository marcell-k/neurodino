"""Glue."""

from random import Random
from typing import TYPE_CHECKING

import numpy as np

from neurodino.config import GROUND_Y, MAX_SCROLL_SPEED, WORLD_HEIGHT
from neurodino.sim import Dino, Game
from neurodino.specs import ObjectType

if TYPE_CHECKING:
    from neurodino.nn import NeuralNetwork

MAX_FRAMES = 2000
SCORE_BONUS = 10
OBJECT_FEATURES = 4
FEATURE_SIZE = 2 * OBJECT_FEATURES + 4  # two obstacles + airborne, vy, height, speed


def _norm_y(y: float) -> float:
    """Map to [-1, 1]."""
    return 2 * y / WORLD_HEIGHT - 1.0


def _features(game: Game) -> list[float]:
    dino = game.dino
    ahead = [ob for ob in game.objects if ob.x + ob.width > dino.x]
    feats: list[float] = []
    for i in range(2):  # nearest + second-nearest
        if i < len(ahead):
            ob = ahead[i]
            feats += [
                (ob.x - (dino.x + dino.width)) / 400,  # dx, timing
                1.0 if ob.spec.kind is ObjectType.BIRD else 0.0,
                ob.height / 100,  # cactus size
                (GROUND_Y - (ob.y + ob.height)) / 300,  # bird altitude, 0 for cacti
            ]
        else:
            feats += [10, 0.0, 0.0, 1.0]  # sentinel: nothing within reach
    feats += [
        0.0 if dino.airborne else 1.0,
        game.velocity / 20,  # vertical speed (jump physics), not world scroll
        _norm_y(dino.y),
        game.speed / MAX_SCROLL_SPEED,
    ]
    return feats


def actions(nn: NeuralNetwork, game: Game, *, holding: bool) -> tuple[bool, bool, bool]:
    """Return `(jump, duck, holding)`; holding keeps jump down for a long jump."""
    if game.dino.airborne:
        holding = holding and game.velocity < 0  # hold until the apex
        return holding, False, holding

    action = nn.forward(np.array(_features(game)))
    match action:
        case 1:
            return False, True, False  # duck
        case 2:
            return True, False, False  # short jump
        case 3:
            return True, False, True  # long jump
        case _:
            return False, False, False  # nothing


def run_episode(nn: NeuralNetwork, seed: int, max_frames: int = MAX_FRAMES) -> float:
    """Play one game with the network; fitness is frames survived plus a score bonus."""
    game = Game(Random(seed), Dino.default())
    holding = False
    while game.alive and game.frames < max_frames:
        jump, duck, holding = actions(nn, game, holding=holding)
        game.step(jump=jump, duck=duck)
    return game.frames + SCORE_BONUS * game.score
