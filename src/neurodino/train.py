"""Train `NeuralNetwork`."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from neurodino.agent import FEATURE_SIZE, run_episode
from neurodino.config import DEFAULT_TRAINING
from neurodino.nn import NeuralNetwork

if TYPE_CHECKING:
    from numpy.random import Generator

    from neurodino.config import TrainingConfig

type Hall = list[tuple[float, NeuralNetwork]]

PRINT_EVERY = 10


def evaluate(nn: NeuralNetwork, seeds: list[int]) -> float:
    """Median fitness of a network over `seeds`."""
    return float(np.median([run_episode(nn, seed) for seed in seeds]))


def _evaluate_all(
    pool: ProcessPoolExecutor,
    networks: list[NeuralNetwork],
    seeds: list[int],
) -> list[float]:
    return list(pool.map(evaluate, networks, [seeds] * len(networks)))


def _mutation_strength(generation: int, generations: int, config: TrainingConfig) -> float:
    """Linearly anneal the noise std from `mutation_start` to `mutation_end`."""
    progress = generation / max(generations - 1, 1)
    return config.mutation_start + progress * (config.mutation_end - config.mutation_start)


def _pick_parent(
    ranked: list[NeuralNetwork], rng: Generator, tournament_size: int
) -> NeuralNetwork:
    """Return the best of `tournament_size` random population members."""
    picks = rng.integers(0, len(ranked), tournament_size)
    return ranked[int(np.min(picks))]


def train(generations: int, config: TrainingConfig, rng: Generator) -> NeuralNetwork:
    """Run the evolution loop and return the champion."""
    if generations < 1:
        msg = f"generations must be >= 1, got {generations}"
        raise ValueError(msg)

    population = [NeuralNetwork.random(rng, FEATURE_SIZE) for _ in range(config.population)]
    validation_seeds = rng.integers(0, 2**31, config.episodes).tolist()
    holdout_seeds = rng.integers(0, 2**31, config.holdout_episodes).tolist()
    hall: Hall = []
    seen: set[int] = set()  # elites can stay champion for several generations

    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        for generation in range(generations):
            seeds = rng.integers(0, 2**31, config.episodes).tolist()
            scores = _evaluate_all(pool, population, seeds)
            order = np.argsort(scores)[::-1]
            ranked = [population[i] for i in order]

            champion = ranked[0]
            if id(champion) not in seen:
                hall.append((evaluate(champion, validation_seeds), champion))
                seen.add(id(champion))

            if generation % PRINT_EVERY == 0:
                print(  # noqa: T201
                    f"gen {generation:4d} best {scores[order[0]]:9.1f} "
                    f"mean {float(np.mean(scores)):9.1f}"
                )

            elites = ranked[: config.elites]
            strength = _mutation_strength(generation, generations, config)
            children: list[NeuralNetwork] = []
            while len(children) < config.population - config.elites:
                parent = _pick_parent(ranked, rng, config.tournament_size)
                child = parent
                if rng.random() < config.crossover_prob:
                    other = _pick_parent(ranked, rng, config.tournament_size)
                    child = parent.crossover(rng, other)
                children.append(child.mutate(rng, strength, config.mutation_rate))
            population = [*elites, *children]

    return _pick_champion(hall, holdout_seeds, config)


def _pick_champion(
    hall: Hall, holdout_seeds: list[int], training_config: TrainingConfig
) -> NeuralNetwork:
    """Re-evaluate the best hall candidates on the holdout seeds and return the winner."""
    top: Hall = sorted(hall, key=lambda entry: entry[0], reverse=True)[: training_config.hall_size]
    held_out = [evaluate(nn, holdout_seeds) for _, nn in top]
    winner = int(np.argmax(held_out))
    print(  # noqa: T201
        f"hall of fame: {len(top)} candidates, "
        f"held-out best {held_out[winner]:.1f} (validation score {top[winner][0]:.1f})"
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
