"""Descriptions of the objects that can be spawned (Chromium's obstacle definitions)."""

from dataclasses import dataclass
from enum import Enum, auto

from neurodino.config import GROUND_Y


class ObjectType(Enum):
    """Object type."""

    SMALL_CACTUS = auto()
    LARGE_CACTUS = auto()
    BIRD = auto()


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned collision box, relative to the top-left corner of its owner."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    """Immutable description of one obstacle type.

    `width` is the width of a single sprite; a group of `size` cacti is `size * width` wide.
    """

    kind: ObjectType
    sprite: str  # image key prefix
    width: int
    height: int
    y_positions: tuple[int, ...]  # top edge; birds pick one at random
    multiple_speed: float  # groups of more than one only appear from this speed on
    min_gap: int
    min_speed: float  # the type only appears from this speed on
    boxes: tuple[Box, ...]  # three boxes for cacti, widened as one when grouped
    speed_offset: float = 0.0  # birds move this much faster or slower than the ground
    frame_ms: float = 0.0

    def image_key(self, size: int, elapsed_ms: float) -> str:
        """Return the image key for a group of `size` at `elapsed_ms`."""
        if self.kind is ObjectType.BIRD:
            return "bird_up" if int(elapsed_ms // self.frame_ms) % 2 == 0 else "bird_down"
        return f"{self.sprite}_{('small', 'mid', 'big')[size - 1]}"


SMALL_CACTUS = ObjectSpec(
    kind=ObjectType.SMALL_CACTUS,
    sprite="small_cactus",
    width=17,
    height=35,
    y_positions=(GROUND_Y - 35,),
    multiple_speed=4,
    min_gap=120,
    min_speed=0,
    boxes=(Box(0, 7, 5, 27), Box(4, 0, 6, 34), Box(10, 4, 7, 14)),
)

LARGE_CACTUS = ObjectSpec(
    kind=ObjectType.LARGE_CACTUS,
    sprite="large_cactus",
    width=25,
    height=50,
    y_positions=(GROUND_Y - 50,),
    multiple_speed=7,
    min_gap=120,
    min_speed=0,
    boxes=(Box(0, 12, 7, 38), Box(8, 0, 7, 49), Box(13, 10, 10, 38)),
)

BIRD = ObjectSpec(
    kind=ObjectType.BIRD,
    sprite="bird",
    width=46,
    height=40,
    y_positions=(100, 75, 50),  # run into it, duck under it, run under it
    multiple_speed=999,
    min_gap=150,
    min_speed=8.5,
    boxes=(
        Box(15, 15, 16, 5),
        Box(18, 21, 24, 6),
        Box(2, 14, 4, 3),
        Box(6, 10, 4, 7),
        Box(10, 8, 6, 9),
    ),
    speed_offset=0.8,
    frame_ms=1000 / 6,
)

SPAWN_SPECS: tuple[ObjectSpec, ...] = (SMALL_CACTUS, LARGE_CACTUS, BIRD)

# Dino collision boxes, relative to the dino's 1px-inset box.
RUNNING_BOXES: tuple[Box, ...] = (
    Box(22, 0, 17, 16),
    Box(1, 18, 30, 9),
    Box(10, 35, 14, 8),
    Box(1, 24, 29, 5),
    Box(5, 30, 21, 4),
    Box(9, 34, 15, 4),
)
DUCKING_BOXES: tuple[Box, ...] = (Box(1, 18, 55, 25),)
