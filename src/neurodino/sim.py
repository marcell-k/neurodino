"""Simulation."""

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neurodino.config import (
    BIRD_ALTITUDES,
    DINO_DUCK_HEIGHT,
    DINO_DUCK_WIDTH,
    DINO_HEIGHT,
    DINO_WIDTH,
    GRAVITY,
    GROUND_Y,
    JUMP_CUT_VELOCITY,
    JUMP_VELOCITY,
    SCROLL_SPEED,
    SPAWN_GAP_MAX,
    SPAWN_GAP_MIN,
    STEP_MSC,
    WORLD_WIDTH,
)
from neurodino.specs import SPAWN_SPECS, ObjectSpec

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


@dataclass(slots=True)
class Dino:
    """The player-controlled dino."""

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
        self.next_spawn_distance = 0.0
        self.jump_was_down: bool = False

    def step(self, *, jump: bool, duck: bool) -> None:
        """Advance one time."""
        if not self.alive:
            return
        self.distance += SCROLL_SPEED
        if self._should_spawn():
            self._spawn()
        jump_pressed = jump and not self.jump_was_down
        self.jump_was_down = jump

        if not self.dino.airborne:
            self.dino.ducking = duck
            self.dino.width = DINO_DUCK_WIDTH if duck else DINO_WIDTH
            self.dino.height = DINO_DUCK_HEIGHT if duck else DINO_HEIGHT
            self.dino.y = GROUND_Y - self.dino.height

        if jump_pressed and not self.dino.airborne:
            self.dino.ducking = False
            self.dino.width = DINO_WIDTH
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

        self._advance_objects(SCROLL_SPEED, self.dino)
        if self._collides():
            self.alive = False
            self.velocity = 0
            self.dino.airborne = False
            self.dino.ducking = False
            self.dino.width = DINO_WIDTH
            self.dino.height = DINO_HEIGHT
            self.dino.y = GROUND_Y - DINO_HEIGHT
        self.frames += 1

    def _collides(self) -> bool:
        """Whether the dino overlaps with any object."""
        dino = self.dino
        return any(
            dino.x < ob.x + ob.width
            and dino.x + dino.width > ob.x
            and dino.y < ob.y + ob.height
            and dino.y + dino.height > ob.y
            for ob in self.objects
        )

    def _advance_objects(self, dx: float, dino: Dino) -> None:
        for ob in self.objects:
            ob.x -= dx
            if not ob.scored and ob.x + ob.width < dino.x:
                ob.scored = True
                self.score += 1
        while self.objects and self.objects[0].x <= -self.objects[0].width:
            self.objects.popleft()

    def _should_spawn(self) -> bool:
        """Whether another object should be spawned."""
        return self.distance >= self.next_spawn_distance

    def _spawn(self) -> None:
        """Spawn one random object at the right edge."""
        spec = self._rng.choice(SPAWN_SPECS)
        y = spec.ground_y if spec.on_ground else self._rng.choice(BIRD_ALTITUDES)
        self.objects.append(Object(x=WORLD_WIDTH - 1, y=y, spec=spec))
        self.next_spawn_distance = self.distance + self._rng.uniform(
            SPAWN_GAP_MIN,
            SPAWN_GAP_MAX,
        )
