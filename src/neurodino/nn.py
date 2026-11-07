"""Neural network."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.random import Generator

type Matrix = npt.NDArray[np.float64]
type Vector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class NeuralNetwork:
    """Input -> tanh -> output (0 - nothing, 1 - duck, 2 - short jump, 3 - long jump)."""

    w1: Matrix  # (hidden, inputs)
    b1: Vector  # (hidden, )
    w2: Matrix  # (4, hidden)
    b2: Vector  # (4,)

    @classmethod
    def random(cls, rng: Generator, input_size: int, hidden_size: int = 6) -> NeuralNetwork:
        """Create a network with normally distributed weights scaled by 1/sqrt(fan_in)."""
        w1 = rng.normal(0.0, 1.0 / np.sqrt(input_size), (hidden_size, input_size))
        w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_size), (4, hidden_size))
        return NeuralNetwork(w1, np.zeros(hidden_size), w2, np.zeros(4))

    def forward(self, x: Vector) -> int:
        """Run one input vector through the network and return the chosen action index."""
        hidden = np.tanh(self.w1 @ x + self.b1)
        return int(np.argmax(self.w2 @ hidden + self.b2))

    def mutate(self, rng: Generator, strength: float, rate: float = 1.0) -> NeuralNetwork:
        """Return a new network with Gaussian noise added to a random `rate` fraction of values."""

        def noisy(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            mask = rng.random(values.shape) < rate
            return values + mask * rng.normal(0.0, strength, values.shape)

        return NeuralNetwork(noisy(self.w1), noisy(self.b1), noisy(self.w2), noisy(self.b2))

    def crossover(self, rng: Generator, other: NeuralNetwork) -> NeuralNetwork:
        """Build a child by taking each hidden unit (its inputs, bias and output) from a parent."""
        from_self = rng.random(self.b1.shape) < 0.5  # noqa: PLR2004
        from_self_out = rng.random(self.b2.shape) < 0.5  # noqa: PLR2004
        return NeuralNetwork(
            np.where(from_self[:, None], self.w1, other.w1),
            np.where(from_self, self.b1, other.b1),
            np.where(from_self[None, :], self.w2, other.w2),
            np.where(from_self_out, self.b2, other.b2),
        )

    @classmethod
    def load(cls, path: Path) -> NeuralNetwork:
        """Load a network written by `save`."""
        with np.load(path) as data:
            return cls(data["w1"], data["b1"], data["w2"], data["b2"])

    def save(self, path: Path) -> None:
        """Write the weights and biases to a .npz file."""
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
