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

HIDDEN_SIZE = 6
N_ACTIONS = 3


@dataclass(frozen=True, slots=True, eq=False)
class NeuralNetwork:
    """Input -> tanh -> output (key state: 0 - nothing, 1 - hold duck, 2 - hold jump)."""

    w1: Matrix  # (hidden, inputs)
    b1: Vector  # (hidden, )
    w2: Matrix  # (actions, hidden)
    b2: Vector  # (actions,)

    @classmethod
    def random(
        cls, rng: Generator, input_size: int, hidden_size: int = HIDDEN_SIZE
    ) -> NeuralNetwork:
        """Create a network with normally distributed weights scaled by 1/sqrt(fan_in)."""
        w1 = rng.normal(0.0, 1.0 / np.sqrt(input_size), (hidden_size, input_size))
        w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_size), (N_ACTIONS, hidden_size))
        return NeuralNetwork(w1, np.zeros(hidden_size), w2, np.zeros(N_ACTIONS))

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


def _layer_shapes(input_size: int, hidden_size: int) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Return `(shape, size)` for each parameter array, in vector order."""
    return (
        ((hidden_size, input_size), hidden_size * input_size),
        ((hidden_size,), hidden_size),
        ((N_ACTIONS, hidden_size), N_ACTIONS * hidden_size),
        ((N_ACTIONS,), N_ACTIONS),
    )


def param_count(input_size: int, hidden_size: int = HIDDEN_SIZE) -> int:
    """Return the number of scalar parameters in a network with these layer sizes."""
    return sum(size for _, size in _layer_shapes(input_size, hidden_size))


def to_vector(network: NeuralNetwork) -> Vector:
    """Flatten the weights and biases into one vector, in w1, b1, w2, b2 order."""
    return np.concatenate((network.w1.ravel(), network.b1, network.w2.ravel(), network.b2))


def from_vector(vector: Vector, input_size: int, hidden_size: int = HIDDEN_SIZE) -> NeuralNetwork:
    """Rebuild a network from a vector written by `to_vector`."""
    parts: list[Vector] = []
    offset = 0
    for shape, size in _layer_shapes(input_size, hidden_size):
        parts.append(np.array(vector[offset : offset + size]).reshape(shape))
        offset += size
    w1, b1, w2, b2 = parts
    return NeuralNetwork(w1, b1, w2, b2)
