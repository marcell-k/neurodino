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


def _cactus_specs(
    kind: ObjectType, prefix: str, height: int, widths: dict[Size, int]
) -> dict[tuple[ObjectType, Size], ObjectSpec]:
    """Build one spec per cactus width variant."""
    return {
        (kind, size): ObjectSpec(
            kind=kind,
            frames=(f"{prefix}_{size.name.lower()}",),
            width=width,
            height=height,
        )
        for size, width in widths.items()
    }


CACTUS_SPECS: dict[tuple[ObjectType, Size], ObjectSpec] = {
    **_cactus_specs(
        ObjectType.SMALL_CACTUS,
        "small_cactus",
        height=71,
        widths={Size.SMALL: 40, Size.MID: 68, Size.BIG: 105},
    ),
    **_cactus_specs(
        ObjectType.LARGE_CACTUS,
        "large_cactus",
        height=95,
        widths={Size.SMALL: 48, Size.MID: 99, Size.BIG: 102},
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
