"""Glue."""

from random import Random
from typing import TYPE_CHECKING

import numpy as np

from neurodino.config import GROUND_Y, WORLD_HEIGHT
from neurodino.sim import Dino, Game
from neurodino.specs import ObjectType

if TYPE_CHECKING:
    from neurodino.nn import NeuralNetwork

MAX_FRAMES = 1000
SCORE_BONUS = 100


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
            feats += [1.0, 0.0, 0.0, 1.0]  # sentinel: nothing there
    feats += [
        0.0 if dino.airborne else 1.0,
        game.velocity / 20,
    ]
    return feats


def _actions(nn: NeuralNetwork, game: Game) -> tuple[bool, bool]:
    # TODO: use np.forward()
    vector = np.array([*_features(game)])
    action = nn.forward(vector)
    match action:
        case 0:
            return False, False
        case 1:
            return False, True  # duck
        case 2:
            return True, False  # jump
        case _:
            msg = f"invalid action, got: {action}"
            raise ValueError(msg)


def run_episode(nn: NeuralNetwork, seed: int, max_frames: int = MAX_FRAMES) -> float:
    """Play one game with the network; fitness is frames survived plus a score bonus."""
    game = Game(Random(seed), Dino.default())
    while game.alive and game.frames < max_frames:
        jump, duck = _actions(nn, game)
        game.step(jump=jump, duck=duck)
    return game.frames + SCORE_BONUS * game.score
