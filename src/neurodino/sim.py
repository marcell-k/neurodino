"""Simulation."""

from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from random import Random

from neurodino.config import GRAVITY, JUMP_VELOCITY, STEP_MSC, WORLD_HEIGHT, WORLD_WIDTH


class ObjectType(Enum):
    """Object type."""

    SMALL_CACTUS = 0
    LARGE_CACTUS = 1
    BIRD = 2


Size = IntEnum("Size", [("SMALL", 48), ("MID", 99), ("BIG", 102)])


@dataclass(slots=True)
class Object:
    """Cactus."""

    x: float
    y: int
    type: ObjectType
    size: Size
    width: int
    age: int = 0
    scored: bool = False

    def advance(self) -> None:
        """Advance."""
        self.age += 1

    def hits(self, dino: Dino) -> bool:
        """Whether dino whose top edge is at `y` overlaps with cactus."""
        return False

    def file_name(self, frame: int) -> str:
        """Name of the object, used for finding image."""
        if self.type.name == "BIRD":
            bird_up = (frame * STEP_MSC) % 500 >= 250  # noqa: PLR2004
            return "bird_up" if bird_up else "bird_down"

        size = self.size.name.lower()
        cactus_type = "large" if self.type.name == "LARGE_CACTUS" else "small"
        return f"{cactus_type}_cactus_{size}"


@dataclass(slots=True)
class Dino:
    """Dino ."""

    x: int
    y: int
    width: int
    height: int
    ducking: bool = False
    airborne: bool = False


class Game:
    """Game."""

    def __init__(self, rng: Random, dino: Dino) -> None:
        self._rng = rng
        self.y: float = WORLD_HEIGHT
        self.objects: deque[Object] = deque()
        self.velocity = 0.0
        self.score = 0
        self.alive: bool = True
        self.dino = dino
        self.frames = 0
        self.distance = 0.0

    def step(self, *, jump: bool, duck: bool) -> None:
        """Advance one time."""
        if not self.alive:
            return
        if self._should_spawn():
            self._spawn()
        if jump:
            self.velocity = JUMP_VELOCITY
        if duck:
            self.dino.width = 80
        else:
            self.dino.width = 100
        self.velocity = self.velocity + GRAVITY * STEP_MSC
        self.y += self.velocity * STEP_MSC
        self._advance_objects(STEP_MSC, self.dino)
        # if self._collides():
        #     self.alive = False
        self.frames += 1

    def _advance_objects(self, dx: float, dino: Dino) -> None:
        for ob in self.objects:
            ob.x -= dx
            if not ob.scored and ob.x + ob.width < dino.width:
                ob.scored = True
                self.score += 1
        while self.objects and self.objects[0].x <= -self.objects[0].width:
            self.objects.popleft()

    def _should_spawn(self) -> bool:
        if not self.objects:
            return True
        return False

    def _spawn(self) -> None:
        self.objects.append(Object(WORLD_WIDTH - 1, 300, ObjectType.LARGE_CACTUS, Size.MID, 100))
        # object_type = random.choice(list(ObjectType))
        # size = random.choice(list(Size))
        # if object_type.name == "Cactus":
        #     cactus_type = random.choice(list(CactusType))
        #     ob = Object(WORLD_WIDTH - 1, 200, object_type, size, cactus_type.value)
        # else:
        #     ob = Object(WORLD_WIDTH - 1, 200, object_type, size, BIRD_WIDTH)
        # self.objects.append(ob)
