"""Application entry point."""

import argparse
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

import pygame
from pygame import Surface

from neurodino.agent import actions
from neurodino.config import (
    DINO_HEIGHT,
    FPS,
    GROUND_Y,
    RENDER_SCALE,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from neurodino.nn import NeuralNetwork
from neurodino.sim import Game
from neurodino.utils import load_images

if TYPE_CHECKING:
    from pygame.font import Font

type Images = dict[str, Surface]

SCALE = RENDER_SCALE


def _dino_key(game: Game) -> str:
    if not game.alive:
        return "dino_dead"
    if game.dino.jumping:
        return "dino_jump"
    if game.dino.ducking:
        return "dino_duck_1" if (game.frames // 6) % 2 == 0 else "dino_duck_2"
    return "dino_run_1" if (game.frames // 6) % 2 == 0 else "dino_run_2"


def _draw(display: Surface, images: Images, game: Game, font: Font) -> None:
    """Draw current state of the game."""
    display.fill((255, 255, 255))

    track = images["track"]
    w = track.get_width()
    offset = -(int(game.distance * SCALE) % w)
    display.blit(track, (offset, GROUND_Y * SCALE))
    display.blit(track, (offset + w, GROUND_Y * SCALE))

    for ob in game.objects:
        img = images[ob.image_key(game.frames)]
        bbox = img.get_bounding_rect()
        x = round((ob.x + ob.width / 2) * SCALE - bbox.width / 2) - bbox.left
        y = (ob.y + ob.height) * SCALE - bbox.bottom
        display.blit(img, (x, y))

    img = images[_dino_key(game)]
    bbox = img.get_bounding_rect()
    x = game.dino.x * SCALE - bbox.left
    y = (game.dino.y + DINO_HEIGHT) * SCALE - bbox.bottom
    display.blit(img, (x, y))

    score = font.render(f"{game.score:05d}", True, (50, 50, 50))  # noqa: FBT003
    display.blit(score, (WORLD_WIDTH * SCALE - score.get_width() - 20, 20))


def main() -> None:
    """Run the application."""
    parser = argparse.ArgumentParser("Dino")
    parser.add_argument("--ai", type=Path, default=None, metavar="FILE")
    args = parser.parse_args()
    pygame.init()
    display = pygame.display.set_mode((WORLD_WIDTH * SCALE, WORLD_HEIGHT * SCALE))
    pygame.display.set_caption("Dino")
    brain = NeuralNetwork.load(args.ai) if args.ai else None
    clock = pygame.time.Clock()
    images = load_images()
    font = pygame.font.Font(None, 25)
    game = Game(Random())

    running = True
    while running:
        clock.tick(FPS)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        if brain is not None:
            jump, duck = actions(brain, game)
            game.step(jump=jump, duck=duck)
        else:
            keys = pygame.key.get_pressed()
            game.step(jump=keys[pygame.K_k], duck=keys[pygame.K_j])

        _draw(display, images, game, font)
        pygame.display.flip()
        if not game.alive:
            print(f"Game over. Score: {game.score}")  # noqa: T201
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
