"""Utils and helper functions."""

import pygame
from pygame import Surface

from neurodino.config import IMAGES_DIR


def load_images() -> dict[str, Surface]:
    """Load all images."""

    def _load_image(file_name: str) -> Surface:
        return pygame.image.load(IMAGES_DIR / file_name).convert_alpha()

    return {
        "bird_up": _load_image("Bird/Bird1.png"),
        "bird_down": _load_image("Bird/Bird2.png"),
        "large_cactus_small": _load_image("Cactus/LargeCactus1.png"),
        "large_cactus_mid": _load_image("Cactus/LargeCactus2.png"),
        "large_cactus_big": _load_image("Cactus/LargeCactus3.png"),
        "small_cactus_small": _load_image("Cactus/SmallCactus1.png"),
        "small_cactus_mid": _load_image("Cactus/SmallCactus2.png"),
        "small_cactus_big": _load_image("Cactus/SmallCactus3.png"),
        "dino_duck_1": _load_image("Dino/DinoDuck1.png"),
        "dino_duck_2": _load_image("Dino/DinoDuck2.png"),
        "dino_jump": _load_image("Dino/DinoJump.png"),
        "dino_run_1": _load_image("Dino/DinoRun1.png"),
        "dino_run_2": _load_image("Dino/DinoRun2.png"),
        "dino_start": _load_image("Dino/DinoStart.png"),
        "cloud": _load_image("Other/Cloud.png"),
        "track": _load_image("Other/Track.png"),
    }
