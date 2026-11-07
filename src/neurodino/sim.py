"""Simulation."""

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neurodino.config import (
    DINO_DUCK_HEIGHT,
    DINO_HEIGHT,
    GRAVITY,
    GROUND_Y,
    JUMP_CUT_VELOCITY,
    JUMP_VELOCITY,
    STEP_MSC,
    WORLD_WIDTH,
)
from neurodino.specs import CACTUS_SPECS, ObjectSpec, ObjectType, Size

if TYPE_CHECKING:
    from random import Random


@dataclass(slots=True)
class Object:
    """A moving obstacle."""

    x: float
    y: int
    spec: ObjectSpec
    scored: bool = False

    @property
    def width(self) -> int:
        """Width of the object."""
        return self.spec.width

    @property
    def height(self) -> int:
        """Height of the object."""
        return self.spec.height

    def image_key(self, frame: int) -> str:
        """Image key for the object's current animation frame."""
        return self.spec.frame_name(frame * STEP_MSC)

    def hits(self, dino: Dino) -> bool:
        """Whether the dino overlaps with this object."""
        return False


@dataclass(slots=True)
class Dino:
    """Dino ."""

    x: int
    y: float
    width: int
    height: int
    ducking: bool = False
    airborne: bool = False


class Game:
    """Game."""

    def __init__(self, rng: Random, dino: Dino) -> None:
        self._rng = rng
        self.objects: deque[Object] = deque()
        self.velocity = 0.0
        self.score = 0
        self.alive: bool = True
        self.dino = dino
        self.frames = 0
        self.distance = 0.0
        self.jump_was_down: bool = False

    def step(self, *, jump: bool, duck: bool) -> None:
        """Advance one time."""
        if not self.alive:
            return
        if self._should_spawn():
            self._spawn()
        jump_pressed = jump and not self.jump_was_down
        self.jump_was_down = jump

        if not self.dino.airborne:
            self.dino.ducking = duck
            self.dino.height = DINO_DUCK_HEIGHT if duck else DINO_HEIGHT
            self.dino.y = GROUND_Y - self.dino.height

        if jump_pressed and not self.dino.airborne:
            self.dino.ducking = False
            self.dino.height = DINO_HEIGHT
            self.dino.y = GROUND_Y - DINO_HEIGHT
            self.velocity = JUMP_VELOCITY

        if self.velocity < 0 and not jump:
            self.velocity = max(self.velocity, JUMP_CUT_VELOCITY)

        self.velocity += GRAVITY
        self.dino.y += self.velocity
        if self.dino.y + self.dino.height >= GROUND_Y:
            self.dino.y = GROUND_Y - self.dino.height
            self.velocity = 0
            self.dino.airborne = False
        else:
            self.dino.airborne = True

        self._advance_objects(STEP_MSC, self.dino)
        # if self._collides():
        #     self.alive = False
        self.frames += 1

    def _advance_objects(self, dx: float, dino: Dino) -> None:
        for ob in self.objects:
            ob.x -= dx
            if not ob.scored and ob.x + ob.width < dino.x:
                ob.scored = True
                self.score += 1
        while self.objects and self.objects[0].x <= -self.objects[0].width:
            self.objects.popleft()

    def _should_spawn(self) -> bool:
        if not self.objects:
            return True
        return False

    def _spawn(self) -> None:
        spec = CACTUS_SPECS[(ObjectType.LARGE_CACTUS, Size.MID)]
        self.objects.append(Object(x=WORLD_WIDTH - 1, y=spec.ground_y, spec=spec))
