"""Configs."""

from pathlib import Path

# UI
IMAGES_DIR = Path("assets")
WORLD_WIDTH = 1100
WORLD_HEIGHT = 600
GROUND_Y = 480
TRACK_GROUND_ROW = 10  # row in Track.png that contains the solid ground line
BACKGROUND = (255, 255, 255)

FPS = 60
STEP_MSC = 1000 / FPS  # milliseconds per frame
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
BIRD_ALTITUDES = (
    GROUND_Y - 220,  # above standing head, safe to run under
    GROUND_Y - 148,  # between standing and ducking, duck under it
    GROUND_Y - 100,  # below ducking, jump over it
)
