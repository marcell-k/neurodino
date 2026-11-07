from enum import Enum, IntEnum


class CactusType(IntEnum):
    """Cactus type can be `(Small, Large)."""

    SMALL = 71
    LARGE = 95


ObjectType = Enum("ObjectType", [("Cactus", CactusType), ("Bird", 1)])


Size = IntEnum("Size", [("SMALL", 48), ("MID", 99), ("BIG", 102)])


import random

for _ in range(20):
    v = random.choice(list(ObjectType))
    print(v)
    if v.name == "Cactus":
        s = random.choice(list(Size))
        print(s, v)
