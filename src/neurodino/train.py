"""Train `NeuralNetwork`."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from neurodino.agent import FEATURE_SIZE, run_episode
from neurodino.cma_es import CMAES
from neurodino.config import DEFAULT_CMA, DEFAULT_TRAINING, CMAConfig
from neurodino.nn import (
    HIDDEN_SIZE,
    N_ACTIONS,
    NeuralNetwork,
    Vector,
    from_vector,
    param_count,
    to_vector,
)

if TYPE_CHECKING:
    from numpy.random import Generator

    from neurodino.config import TrainingConfig

type Hall = list[tuple[float, NeuralNetwork]]

_IMPROVEMENT_FACTOR = 1.01  # mean score must gain 1% over the best to reset patience
_REFLEX_BIAS = 0.8  # jump when the nearest obstacle is ~96 px away
_REFLEX_BIAS_RANGE = (0.4, 0.9)  # seed diversity: jump thresholds from ~48 to ~108 px
_SEEDED_FRACTION = 0.75  # share of the first GA generation that inherits the reflex


def _reflex_network(jump_bias: float = _REFLEX_BIAS) -> NeuralNetwork:
    """Jump when the nearest obstacle enters range, otherwise do nothing."""
    w1 = np.zeros((HIDDEN_SIZE, FEATURE_SIZE))
    b1 = np.zeros(HIDDEN_SIZE)
    w2 = np.zeros((N_ACTIONS, HIDDEN_SIZE))
    w1[0, 0] = -5.0  # nearest-obstacle distance feature
    b1[0] = jump_bias
    w2[2, 0] = 1.0  # jump action
    return NeuralNetwork(w1, b1, w2, np.zeros(N_ACTIONS))


def _initial_population(rng: Generator, size: int) -> list[NeuralNetwork]:
    """Seed most of the population with reflexes, keep the rest random for exploration."""
    seeded_count = max(1, int(size * _SEEDED_FRACTION))
    biases = np.linspace(*_REFLEX_BIAS_RANGE, seeded_count)
    biases[0] = _REFLEX_BIAS
    seeded = [_reflex_network(float(bias)) for bias in biases]
    random = [NeuralNetwork.random(rng, FEATURE_SIZE) for _ in range(size - seeded_count)]
    return [*seeded, *random]


def evaluate(nn: NeuralNetwork, seeds: list[int]) -> float:
    """Median fitness of a network over `seeds`."""
    return float(np.median([run_episode(nn, seed) for seed in seeds]))


def evaluate_vector(vector: Vector, seeds: list[int]) -> float:
    """Median fitness of a flat parameter vector over `seeds`."""
    return evaluate(from_vector(vector, FEATURE_SIZE), seeds)


def train_cma(generations: int, config: CMAConfig, rng: Generator) -> NeuralNetwork:
    """Run the CMA-ES loop and return the champion."""
    start = _reflex_network()
    cma = CMAES(param_count(FEATURE_SIZE), to_vector(start), config.sigma)
    validation_seeds = rng.integers(0, 2**31, config.episodes).tolist()
    holdout_seeds = rng.integers(0, 2**31, config.holdout_episodes).tolist()
    hall: Hall = []
    best = -np.inf
    best_mean = -np.inf
    stale = 0
    print_every = max(1, int(generations / 10))

    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        for generation in range(generations):
            x, y = cma.ask(rng)
            seeds = rng.integers(0, 2**31, config.episodes).tolist()
            scores = np.array(list(pool.map(evaluate_vector, x, [seeds] * len(x))))
            cma.tell(y, -scores)  # CMA-ES minimises, fitness is maximised

            i = int(np.argmax(scores))
            if scores[i] > best:
                best = scores[i]
                champion = from_vector(x[i], FEATURE_SIZE)
                hall.append((evaluate(champion, validation_seeds), champion))

            mean_score = float(np.mean(scores))
            if generation % print_every == 0:
                print(  # noqa: T201
                    f"gen {generation:4d} best {scores[i]:9.1f} mean {mean_score:9.1f}"
                )

            if mean_score > best_mean * _IMPROVEMENT_FACTOR:
                best_mean = mean_score
                stale = 0
            else:
                stale += 1

            if config.patience > 0 and stale >= config.patience:
                print(  # noqa: T201
                    f"early stop at gen {generation:4d}: mean {mean_score:9.1f} did not beat "
                    f"{best_mean * _IMPROVEMENT_FACTOR:9.1f} for {stale} generations"
                )
                break
    return _pick_champion(hall, holdout_seeds, config.hall_size)


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

    population = _initial_population(rng, config.population)
    validation_seeds = rng.integers(0, 2**31, config.episodes).tolist()
    holdout_seeds = rng.integers(0, 2**31, config.holdout_episodes).tolist()
    hall: Hall = []
    seen: set[int] = set()  # elites can stay champion for several generations
    best_mean = -np.inf
    stale = 0

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

            mean_score = float(np.mean(scores))
            if generation % 10 == 0:
                print(  # noqa: T201
                    f"gen {generation:4d} best {scores[order[0]]:9.1f} mean {mean_score:9.1f}"
                )

            if mean_score > best_mean * _IMPROVEMENT_FACTOR:
                best_mean = mean_score
                stale = 0
            else:
                stale += 1

            if config.patience > 0 and stale >= config.patience:
                print(  # noqa: T201
                    f"early stop at gen {generation:4d}: mean {mean_score:9.1f} did not beat "
                    f"{best_mean * _IMPROVEMENT_FACTOR:9.1f} for {stale} generations"
                )
                break

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

    return _pick_champion(hall, holdout_seeds, config.hall_size)


def _pick_champion(hall: Hall, holdout_seeds: list[int], hall_size: int) -> NeuralNetwork:
    """Re-evaluate the best hall candidates on the holdout seeds and return the winner."""
    top: Hall = sorted(hall, key=lambda entry: entry[0], reverse=True)[:hall_size]
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
    parser.add_argument("--algorithm", choices=("ga", "cma"), default="ga")
    parser.add_argument("--sigma", type=float, default=DEFAULT_CMA.sigma)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("best.npz"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.algorithm == "cma":
        best = train_cma(args.generations, replace(DEFAULT_CMA, sigma=args.sigma), rng)
    else:
        best = train(args.generations, DEFAULT_TRAINING, rng)
    best.save(args.output)
    print(f"Saved network to {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
