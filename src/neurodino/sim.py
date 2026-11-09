"""Simulation: a frame-by-frame port of Chromium's dino game."""

from collections import deque
from dataclasses import dataclass
from math import floor
from typing import TYPE_CHECKING

from neurodino.config import (
    ACCELERATION,
    CLEAR_FRAMES,
    DINO_HEIGHT,
    DINO_WIDTH,
    DINO_X,
    DROP_VELOCITY,
    GAP_COEFFICIENT,
    GRAVITY,
    GROUND_Y,
    INITIAL_JUMP_VELOCITY,
    MAX_GAP_COEFFICIENT,
    MAX_JUMP_HEIGHT,
    MAX_OBSTACLE_DUPLICATION,
    MAX_OBSTACLE_SIZE,
    MAX_SPEED,
    MIN_JUMP_HEIGHT,
    SCORE_COEFFICIENT,
    SPEED_DROP_COEFFICIENT,
    START_SPEED,
    STEP_MS,
    WORLD_WIDTH,
)
from neurodino.specs import (
    DUCKING_BOXES,
    RUNNING_BOXES,
    SPAWN_SPECS,
    Box,
    ObjectSpec,
    ObjectType,
)

if TYPE_CHECKING:
    from random import Random

DINO_GROUND_Y = GROUND_Y - DINO_HEIGHT  # top edge of the dino while it stands on the ground
MIN_JUMP_Y = DINO_GROUND_Y - MIN_JUMP_HEIGHT  # a release only counts once the dino rose above it


@dataclass(slots=True)
class Object:
    """A moving obstacle: one cactus, a group of up to three cacti, or a bird."""

    spec: ObjectSpec
    size: int
    x: int
    y: int
    gap: int  # free space required behind it before the next obstacle may spawn
    speed_offset: float
    boxes: tuple[Box, ...]
    followed: bool = False  # the next obstacle has already been spawned

    @staticmethod
    def spawn(spec: ObjectSpec, rng: Random, speed: float) -> Object:
        """Create an obstacle just right of the screen, with a random size, height and gap."""
        size = rng.randint(1, MAX_OBSTACLE_SIZE)
        if size > 1 and spec.multiple_speed > speed:
            size = 1  # groups only appear from `multiple_speed` on
        width = spec.width * size

        boxes = spec.boxes
        if size > 1:  # the middle box stretches over the group, the last box moves to its end
            first, middle, last = boxes
            boxes = (
                first,
                Box(middle.x, middle.y, width - first.width - last.width, middle.height),
                Box(width - last.width, last.y, last.width, last.height),
            )

        offset = 0.0
        if spec.speed_offset:
            offset = rng.choice((spec.speed_offset, -spec.speed_offset))

        min_gap = round(width * speed + spec.min_gap * GAP_COEFFICIENT)
        max_gap = round(min_gap * MAX_GAP_COEFFICIENT)
        return Object(
            spec=spec,
            size=size,
            x=WORLD_WIDTH + spec.width,
            y=rng.choice(spec.y_positions),
            gap=rng.randint(min_gap, max_gap),
            speed_offset=offset,
            boxes=boxes,
        )

    @property
    def width(self) -> int:
        """Width of the whole group."""
        return self.spec.width * self.size

    @property
    def height(self) -> int:
        """Height of the object."""
        return self.spec.height

    def image_key(self, frame: int) -> str:
        """Image key for the object's animation frame."""
        return self.spec.image_key(self.size, frame * STEP_MS)


@dataclass(slots=True)
class Dino:
    """The player-controlled dino (`Trex`)."""

    x: int = DINO_X
    y: int = DINO_GROUND_Y
    velocity: float = 0.0  # vertical, px/frame, negative is up
    jumping: bool = False
    ducking: bool = False
    speed_drop: bool = False
    reached_min_height: bool = False

    def start_jump(self, speed: float) -> None:
        """Start a jump; the faster the game, the stronger the jump."""
        if not self.jumping:
            self.velocity = INITIAL_JUMP_VELOCITY - speed / 10
            self.jumping = True
            self.reached_min_height = False
            self.speed_drop = False

    def end_jump(self) -> None:
        """Release the jump key: cut the rise short, once the minimum height was reached."""
        if self.reached_min_height and self.velocity < DROP_VELOCITY:
            self.velocity = DROP_VELOCITY

    def set_speed_drop(self) -> None:
        """Duck in the air: cancel the jump and fall fast."""
        self.speed_drop = True
        self.velocity = 1

    def update_jump(self) -> None:
        """Advance the jump by one frame."""
        if self.speed_drop:
            self.y += round(self.velocity * SPEED_DROP_COEFFICIENT)
        else:
            self.y += round(self.velocity)
        self.velocity += GRAVITY

        if self.y < MIN_JUMP_Y or self.speed_drop:
            self.reached_min_height = True
        if self.y < MAX_JUMP_HEIGHT or self.speed_drop:
            self.end_jump()
        if self.y > DINO_GROUND_Y:
            self.reset()

    def reset(self) -> None:
        """Back to running on the ground."""
        self.y = DINO_GROUND_Y
        self.velocity = 0
        self.jumping = False
        self.ducking = False
        self.speed_drop = False


class Game:
    """One run: the speed ramps up with time, obstacles spawn according to the current speed."""

    def __init__(self, rng: Random) -> None:
        self._rng = rng
        self._history: deque[ObjectType] = deque(maxlen=MAX_OBSTACLE_DUPLICATION)
        self._jump_down = False
        self._duck_down = False
        self.objects: deque[Object] = deque()
        self.dino = Dino()
        self.speed: float = START_SPEED
        self.alive = True
        self.frames = 0
        self.distance = 0.0  # `distanceRan`, in px

    @property
    def score(self) -> int:
        """The score shown in the game."""
        return round(self.distance * SCORE_COEFFICIENT)

    def step(self, *, jump: bool, duck: bool) -> None:
        """Advance one frame with the given key states."""
        if not self.alive:
            return
        self._handle_keys(jump=jump, duck=duck)
        if self.dino.jumping:
            self.dino.update_jump()

        spawning = self.frames >= CLEAR_FRAMES
        self._update_objects(spawning=spawning)
        if spawning and self.objects and self._collides(self.objects[0]):
            self.alive = False
        else:
            self.distance += self.speed
            if self.speed < MAX_SPEED:
                self.speed += ACCELERATION
        self.frames += 1

    def _handle_keys(self, *, jump: bool, duck: bool) -> None:
        """Keyboard handlers: key presses and releases act on the dino."""
        dino = self.dino
        if jump and not self._jump_down and not dino.jumping and not dino.ducking:
            dino.start_jump(self.speed)
        elif self._jump_down and not jump:
            dino.end_jump()

        if duck:
            if dino.jumping:
                if not self._duck_down:
                    dino.set_speed_drop()
            elif not dino.ducking:  # held down, so it ducks again right after landing
                dino.ducking = True
        elif self._duck_down:
            dino.speed_drop = False
            dino.ducking = False
        self._jump_down = jump
        self._duck_down = duck

    def _update_objects(self, *, spawning: bool) -> None:
        """Move obstacles, drop the ones that left the screen, and spawn the next one."""
        if not spawning:
            return
        for ob in self.objects:
            ob.x -= floor(self.speed + ob.speed_offset)
        while self.objects and self.objects[0].x + self.objects[0].width <= 0:
            self.objects.popleft()

        if not self.objects:
            self._add_object()
            return
        last = self.objects[-1]
        if (
            not last.followed
            and last.x + last.width > 0
            and last.x + last.width + last.gap < WORLD_WIDTH
        ):
            self._add_object()
            last.followed = True

    def _add_object(self) -> None:
        """Spawn a random obstacle; types repeat at most twice and need their minimum speed."""
        while True:
            spec = self._rng.choice(SPAWN_SPECS)
            if self.speed >= spec.min_speed and not self._is_duplicate(spec.kind):
                break
        self.objects.append(Object.spawn(spec, self._rng, self.speed))
        self._history.appendleft(spec.kind)

    def _is_duplicate(self, kind: ObjectType) -> bool:
        """Whether the previous obstacles all have this type already."""
        count = 0
        for previous in self._history:
            count = count + 1 if previous is kind else 0
        return count >= MAX_OBSTACLE_DUPLICATION

    def _collides(self, ob: Object) -> bool:
        """Two-phase check against one obstacle: outer boxes, then the detailed boxes."""
        dino = self.dino
        dx, dy = dino.x + 1, dino.y + 1  # both boxes are inset by the 1px sprite border
        ox, oy = ob.x + 1, ob.y + 1
        if not _overlap(
            (dx, dy, DINO_WIDTH - 2, DINO_HEIGHT - 2), (ox, oy, ob.width - 2, ob.height - 2)
        ):
            return False
        for mine in DUCKING_BOXES if dino.ducking else RUNNING_BOXES:
            mine_abs = (dx + mine.x, dy + mine.y, mine.width, mine.height)
            for theirs in ob.boxes:
                theirs_abs = (ox + theirs.x, oy + theirs.y, theirs.width, theirs.height)
                if _overlap(mine_abs, theirs_abs):
                    return True
        return False


type Rect = tuple[float, float, float, float]  # x, y, width, height


def _overlap(a: Rect, b: Rect) -> bool:
    """Axis-aligned rectangles overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by
