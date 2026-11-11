"""White Eagle / Bald Eagle Search implementation used in the experiments."""
import numpy as np


def bes(objective, dimensions, pop_size=40, iterations=300, seed=0,
        alpha=2.0, a=10.0, radius=1.5, c1=2.0, c2=2.0):
    rng = np.random.default_rng(seed)
    population = rng.random((pop_size, dimensions))
    fitness = np.array([objective(ind)[0] for ind in population])
    best_idx = int(np.argmin(fitness))
    best, best_fit = population[best_idx].copy(), float(fitness[best_idx])

    for _ in range(iterations):
        for i in range(pop_size):
            candidate = best + alpha * rng.random() * (
                population.mean(axis=0) - population[i]
            )
            value = objective(candidate)[0]
            if value < fitness[i]:
                population[i], fitness[i] = candidate, value
                if value < best_fit:
                    best, best_fit = candidate.copy(), float(value)

            angle = a * np.pi * rng.random()
            r = angle + radius * rng.random()
            x = (r * np.sin(angle)) / (abs(r * np.sin(angle)) + 1e-8)
            y = (r * np.cos(angle)) / (abs(r * np.cos(angle)) + 1e-8)
            candidate = (population[i] + y * (population[i] - population[(i + 1) % pop_size])
                         + x * (population[i] - population.mean(axis=0)))
            value = objective(candidate)[0]
            if value < fitness[i]:
                population[i], fitness[i] = candidate, value
                if value < best_fit:
                    best, best_fit = candidate.copy(), float(value)

            angle = a * np.pi * rng.random()
            x_raw, y_raw = angle * np.sinh(angle), angle * np.cosh(angle)
            x = x_raw / (abs(x_raw) + 1e-8)
            y = y_raw / (abs(y_raw) + 1e-8)
            candidate = (rng.random() * best
                         + x * (population[i] - c1 * population.mean(axis=0))
                         + y * (population[i] - c2 * best))
            value = objective(candidate)[0]
            if value < fitness[i]:
                population[i], fitness[i] = candidate, value
                if value < best_fit:
                    best, best_fit = candidate.copy(), float(value)
    return best, best_fit, objective(best)[1:]
