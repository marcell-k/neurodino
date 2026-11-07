# neurodino

Chrome-dino clone in pygame: play it yourself or watch a neural network
evolved by a genetic algorithm.

<img src="assets/Dino/DinoRun1.png" width="88" alt="Dino">

## Setup

Requires Python 3.15+ and [uv](https://docs.astral.sh/uv/).

    uv sync

## Play

    uv run neurodino                 # K = jump, J = duck
    uv run neurodino --ai best.npz   # watch a trained network

Run from the repository root so `assets/` is found.

## Train

    uv run neurodino-train --generations 50 --output best.npz

Each generation scores a population in parallel, keeps the elites, and breeds
children via tournament selection, crossover and Gaussian mutation. A hall of
fame re-evaluates champions on fixed holdout seeds before saving the best one.

## How it works

12 inputs (two nearest obstacles, airborne, vertical speed, height, scroll
speed) feed a 6-unit tanh layer that picks one of four actions: nothing, duck,
short jump or long jump.

## Checks

    uv run ruff check . && uv run ty check
