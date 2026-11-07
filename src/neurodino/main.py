"""Application entry point."""

from random import Random

import pygame
from pygame import Surface
from pygame.font import Font, SysFont

from neurodino.config import DINO_HEIGHT, DINO_WIDTH, FPS, WORLD_HEIGHT, WORLD_WIDTH
from neurodino.sim import Dino, Game, Object
from neurodino.utils import load_images

type Images = dict[str, Surface]


def _draw_object(surface: Surface, images: Images, ob: Object, game: Game) -> None:
    image = images[ob.file_name(game.frames)]
    surface.blit(image, (ob.x, ob.y))


def _draw(display: Surface, images: dict[str, Surface], font: Font, game: Game) -> None:
    """Draw current state of the game."""
    display.blit(images["dino_start"], (40, 400))
    display.blit(images["track"], (0, 400 + game.dino.width - 18))
    for ob in game.objects:
        _draw_object(display, images, ob, game)


def main() -> None:
    """Run the application."""
    pygame.init()
    display = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
    pygame.display.set_caption("Dino")
    clock = pygame.time.Clock()
    images = load_images()
    font = SysFont(None, 32, bold=True)
    game = Game(Random(), Dino(300, 20, DINO_WIDTH, DINO_HEIGHT))

    running = True
    jump = False
    duck = False
    while running:
        clock.tick(FPS)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_k:
                jump = True
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_j:
                duck = True
        game.step(jump=jump, duck=duck)
        _draw(display, images, font, game)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
