"""Covariance matrix adaptation evolution strategy (CMA-ES).

- link: https://en.wikipedia.org/wiki/CMA-ES
"""

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from numpy.random import Generator

    from neurodino.nn import Vector


class CMAES:
    """Covariance matrix adaptation evolution strategy."""

    def __init__(self, dim: int, mean: Vector | None = None, sigma: float = 0.5) -> None:
        self.dim = dim
        self.lambda_ = 4 + int(3 * math.log(dim))  # population size
        self.mu = self.lambda_ // 2  # number of parents
        self.weights = self._calculate_weights()
        self.mu_eff = 1 / np.sum(self.weights**2)
        self.mean = np.zeros(dim) if mean is None else mean
        self.C = np.eye(dim)  # covariance matrix
        self.sigma = sigma  # step size

        # step-size adaptation (CSA) constants
        self.c_sigma = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0.0, math.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.c_sigma
        # E||N(0, I)||, standard approximation
        self.chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # covariance adaptation constants
        self.c_c = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.c_1 = 2 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2) ** 2 + self.mu_eff),
        )

        self.p_sigma = np.zeros(dim)  # evolution path for sigma
        self.p_c = np.zeros(dim)  # evolution path for C
        self.generation = 0
        self._c_inv_sqrt = np.eye(dim)  # C^(-1/2), refreshed in ask()

    def _calculate_weights(self) -> npt.NDArray[np.float64]:
        raw = np.array([math.log(self.mu + 0.5) - math.log(i) for i in range(1, self.mu + 1)])
        return raw / raw.sum()

    def ask(self, rng: Generator) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Sample `lambda_` candidates from `N(mean, sigma^2 * C)`."""
        eigvals, B = np.linalg.eigh(self.C)  # noqa: N806
        eigvals = np.maximum(eigvals, 1e-20)  # guard against round-off negatives
        c_sqrt = B @ np.diag(np.sqrt(eigvals)) @ B.T
        self._c_inv_sqrt = B @ np.diag(eigvals**-0.5) @ B.T

        z = rng.standard_normal((self.lambda_, self.dim))  # z_k ~ N(0, I)
        y = z @ c_sqrt  # y_k = C^(1/2) z_k
        x = self.mean + self.sigma * y  # x_k = m + sigma * y_k
        return x, y

    def tell(self, y: npt.NDArray[np.float64], fitness: npt.NDArray[np.float64]) -> None:
        """Rank candidates (lowest fitness first), move the mean, adapt C and sigma."""
        order = np.argsort(fitness)  # ascending: best first when minimizing
        y_best = y[order[: self.mu]]  # (mu, dim)
        y_w = self.weights @ y_best  # (dim, )
        self.mean = self.mean + self.sigma * y_w

        c = self.c_sigma
        self.p_sigma = (1 - c) * self.p_sigma + math.sqrt(c * (2 - c) * self.mu_eff) * (
            self._c_inv_sqrt @ y_w
        )
        self.generation += 1
        # h_sigma stalls the p_c update when sigma is growing too fast
        decay = 1 - (1 - c) ** (2 * self.generation)
        h_sigma = float(
            np.linalg.norm(self.p_sigma) / math.sqrt(decay) / self.chi_n < 1.4 + 2 / (self.dim + 1)
        )

        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * math.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff
        ) * y_w

        delta_h = (1 - h_sigma) * self.c_c * (2 - self.c_c)
        rank_mu = (y_best.T * self.weights) @ y_best  # sum_i w_i * y_i y_i^T
        self.C = (
            (1 + self.c_1 * delta_h - self.c_1 - self.c_mu) * self.C
            + self.c_1 * np.outer(self.p_c, self.p_c)
            + self.c_mu * rank_mu
        )
        self.C = (self.C + self.C.T) / 2  # keep exactly symmetric

        # sigma <- sigma * exp((c / d) * (||p_sigma|| / E||N(0, I)|| - 1))
        self.sigma *= math.exp((c / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.chi_n - 1))
