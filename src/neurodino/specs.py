"""Descriptions of the objects that can be spawned."""

from dataclasses import dataclass
from enum import Enum, auto

from neurodino.config import GROUND_Y


class ObjectType(Enum):
    """Object type."""

    SMALL_CACTUS = auto()
    LARGE_CACTUS = auto()
    BIRD = auto()


class Size(Enum):
    """Sprite variant within an object type."""

    SMALL = auto()
    MID = auto()
    BIG = auto()


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    """Immutable description of one spawnable object variant."""

    kind: ObjectType
    frames: tuple[str, ...]
    width: int
    height: int
    on_ground: bool = True
    frame_ms: float = 0.0

    @property
    def ground_y(self) -> int:
        """Top edge at which the object sits on the ground."""
        return GROUND_Y - self.height

    def frame_name(self, elapsed_ms: float) -> str:
        """Return the image key for the frame at `elapsed_ms`."""
        if not self.frame_ms or len(self.frames) == 1:
            return self.frames[0]
        return self.frames[int(elapsed_ms // self.frame_ms) % len(self.frames)]


CACTUS_SPECS: dict[tuple[ObjectType, Size], ObjectSpec] = {
    (ObjectType.SMALL_CACTUS, Size.SMALL): ObjectSpec(
        kind=ObjectType.SMALL_CACTUS,
        frames=("small_cactus_small",),
        width=40,
        height=71,
    ),
    (ObjectType.SMALL_CACTUS, Size.MID): ObjectSpec(
        kind=ObjectType.SMALL_CACTUS,
        frames=("small_cactus_mid",),
        width=68,
        height=71,
    ),
    (ObjectType.SMALL_CACTUS, Size.BIG): ObjectSpec(
        kind=ObjectType.SMALL_CACTUS,
        frames=("small_cactus_big",),
        width=105,
        height=71,
    ),
    (ObjectType.LARGE_CACTUS, Size.SMALL): ObjectSpec(
        kind=ObjectType.LARGE_CACTUS,
        frames=("large_cactus_small",),
        width=48,
        height=95,
    ),
    (ObjectType.LARGE_CACTUS, Size.MID): ObjectSpec(
        kind=ObjectType.LARGE_CACTUS,
        frames=("large_cactus_mid",),
        width=99,
        height=95,
    ),
    (ObjectType.LARGE_CACTUS, Size.BIG): ObjectSpec(
        kind=ObjectType.LARGE_CACTUS,
        frames=("large_cactus_big",),
        width=102,
        height=95,
    ),
}

BIRD_SPEC = ObjectSpec(
    kind=ObjectType.BIRD,
    frames=("bird_up", "bird_down"),
    width=97,
    height=68,
    on_ground=False,
    frame_ms=250,
)

SPAWN_SPECS: tuple[ObjectSpec, ...] = (*CACTUS_SPECS.values(), BIRD_SPEC)
