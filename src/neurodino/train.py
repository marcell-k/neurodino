"""Train `NeuralNetwork`."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from neurodino.agent import run_episode
from neurodino.config import DEFAULT_TRAINING
from neurodino.nn import NeuralNetwork

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

    from numpy.random import Generator

    from neurodino.config import TrainingConfig

type Hall = list[tuple[float, NeuralNetwork]]


def evaluate(nn: NeuralNetwork, seeds: list[int]) -> float:
    """Average fitness of a network."""
    return float(np.median([run_episode(nn, seed) for seed in seeds]))


def _evaluate_all(
    pool: ProcessPoolExecutor, networks: list[NeuralNetwork], seeds: list[int]
) -> list[float]:
    return list(pool.map(evaluate, networks, [seeds] * len(networks)))


def train(generations: int, config: TrainingConfig, rng: Generator) -> NeuralNetwork:
    """Run the evoluation loop and return the champion."""
    population = [NeuralNetwork.random(rng) for _ in range(config.population)]
    holdout_seeds = rng.integers(0, 2**31, config.holdout_episodes).tolist()
    hall: Hall = []  # (training_score, champion) per generation
    for generation in range(generations):
        seeds = rng.integers(0, 2**31, 8).tolist()
        scores = [evaluate(nn, seeds) for nn in population]
        order = np.argsort(scores)[::-1]
        ranked = [population[i] for i in order]
        hall.append((scores[order[0]], ranked[0]))

        if generation % 10 == 0:
            print(  # noqa: T201
                f"gen {generation:4d} best {scores[order[0]]:9.1f} "
                f"mean {float(np.mean(scores)):9.1f}"
            )

        elites = ranked[: config.elites]
        children = [
            elites[int(rng.integers(config.elites))].mutate(rng, config.mutation_rate)
            for _ in range(config.population - config.elites)
        ]
        population = [*elites, *children]
    return _pick_champion(hall, holdout_seeds, config)


def _pick_champion(
    hall: Hall, holdout_seeds: list[int], training_config: TrainingConfig
) -> NeuralNetwork:
    top: Hall = sorted(hall, key=lambda entry: entry[0], reverse=True)[: training_config.hall_size]
    held_out = [evaluate(nn, holdout_seeds) for _, nn in top]
    winner = int(np.argmax(held_out))
    print(  # noqa: T201
        f"hall of fame: {len(top)} canditates, "
        f"held-out best {held_out[winner]:.1f} (training score {top[winner][0]:.1f})"
    )
    return top[winner][1]


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser("Train a neural network")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("best.npz"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    best = train(args.generations, DEFAULT_TRAINING, rng)
    best.save(args.output)
    print(f"Saved network to {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
