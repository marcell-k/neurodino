"""Application entry point."""

from random import Random

import pygame
from pygame import Surface

from neurodino.config import (
    BACKGROUND,
    DINO_HEIGHT,
    DINO_WIDTH,
    FPS,
    GROUND_Y,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from neurodino.sim import Dino, Game
from neurodino.utils import load_images

type Images = dict[str, Surface]


def _dino_key(game: Game) -> str:
    if not game.alive:
        return "dino_dead"
    if game.dino.airborne:
        return "dino_jump"
    if game.dino.ducking:
        return "dino_duck_1" if (game.frames // 6) % 2 == 0 else "dino_duck_2"
    return "dino_run_1" if (game.frames // 6) % 2 == 0 else "dino_run_2"


def _draw(display: Surface, images: Images, game: Game) -> None:
    """Draw current state of the game."""
    display.fill(BACKGROUND)

    track = images["track"]
    w = track.get_width()
    offset = -(int(game.distance) % w)
    display.blit(track, (offset, GROUND_Y))
    display.blit(track, (offset + w, GROUND_Y))

    for ob in game.objects:
        img = images[ob.image_key(game.frames)]
        display.blit(img, (int(ob.x), ob.y))

    img = images[_dino_key(game)]
    display.blit(img, (game.dino.x, game.dino.y))


def main() -> None:
    """Run the application."""
    pygame.init()
    display = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
    pygame.display.set_caption("Dino")
    clock = pygame.time.Clock()
    images = load_images()
    game = Game(Random(), Dino(80, GROUND_Y - DINO_HEIGHT, DINO_WIDTH, DINO_HEIGHT))

    running = True
    while running:
        clock.tick(FPS)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        game.step(jump=keys[pygame.K_k], duck=keys[pygame.K_j])
        _draw(display, images, game)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
