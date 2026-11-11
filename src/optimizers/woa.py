"""Whale Optimization Algorithm."""
import numpy as np


def whale_optimization(objective, dimensions, pop_size=40, iterations=300,
                       seed=0, lower_bound=-3.0, upper_bound=3.0):
    rng = np.random.default_rng(seed)
    population = rng.uniform(lower_bound, upper_bound, (pop_size, dimensions))
    values = np.array([objective(ind)[0] for ind in population])
    idx = int(np.argmin(values))
    best, best_value = population[idx].copy(), float(values[idx])

    for iteration in range(iterations):
        a = 2.0 - 2.0 * iteration / max(iterations, 1)
        for i in range(pop_size):
            r1, r2 = rng.random(dimensions), rng.random(dimensions)
            A, C = 2.0 * a * r1 - a, 2.0 * r2
            probability, spiral = rng.random(), rng.uniform(-1.0, 1.0)
            if probability < 0.5:
                if np.linalg.norm(A) < 1.0:
                    distance = np.abs(C * best - population[i])
                    candidate = best - A * distance
                else:
                    random_whale = population[rng.integers(pop_size)]
                    distance = np.abs(C * random_whale - population[i])
                    candidate = random_whale - A * distance
            else:
                distance = np.abs(best - population[i])
                candidate = distance * np.exp(spiral) * np.cos(2.0 * np.pi * spiral) + best
            candidate = np.clip(candidate, lower_bound, upper_bound)
            value = objective(candidate)[0]
            if value < values[i]:
                population[i], values[i] = candidate, value
                if value < best_value:
                    best, best_value = candidate.copy(), float(value)
    return best, best_value, objective(best)[1:]
