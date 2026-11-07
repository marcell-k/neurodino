"""Configs."""

from pathlib import Path

# UI
IMAGES_DIR = Path("assets")
WORLD_WIDTH = 1100
WORLD_HEIGHT = 600
GROUND_Y = 480
BACKGROUND = (255, 255, 255)

FPS = 60
STEP_MSC = 1000 / FPS
GRAVITY = 0.8  # px/ms

# Dino
DINO_HEIGHT = 94
DINO_DUCK_HEIGHT = 60
DINO_WIDTH = 88
JUMP_VELOCITY = -20  # px/frame
JUMP_CUT_VELOCITY = -10  # upwarrd speed left after releasing jump


# Objects
SPAWN_CACTUS = 3000
BIRD_WIDTH = 68
