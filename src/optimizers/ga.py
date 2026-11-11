"""Genetic Algorithm with tournament selection and elitism."""
import numpy as np


def genetic_algorithm(objective, dimensions, pop_size=40, generations=300,
                      seed=0, crossover_rate=0.8, mutation_rate=0.1,
                      mutation_scale=0.1, bounds=(-2.0, 2.0), tournament_size=3):
    rng = np.random.default_rng(seed)
    low, high = bounds
    population = rng.uniform(low, high, (pop_size, dimensions))
    values = np.array([objective(ind)[0] for ind in population])
    best_idx = int(np.argmin(values))
    best, best_value = population[best_idx].copy(), float(values[best_idx])

    def select_parent():
        candidates = rng.choice(pop_size, size=min(tournament_size, pop_size), replace=False)
        return population[candidates[np.argmin(values[candidates])]].copy()

    for _ in range(generations):
        next_population = [best.copy()]  # elitism
        while len(next_population) < pop_size:
            p1, p2 = select_parent(), select_parent()
            if rng.random() < crossover_rate:
                alpha = rng.random(dimensions)
                c1 = alpha * p1 + (1.0 - alpha) * p2
                c2 = alpha * p2 + (1.0 - alpha) * p1
            else:
                c1, c2 = p1.copy(), p2.copy()
            for child in (c1, c2):
                mask = rng.random(dimensions) < mutation_rate
                if np.any(mask):
                    child[mask] += rng.normal(0.0, mutation_scale, mask.sum())
                next_population.append(np.clip(child, low, high))
                if len(next_population) == pop_size:
                    break
        population = np.asarray(next_population)
        values = np.array([objective(ind)[0] for ind in population])
        idx = int(np.argmin(values))
        if values[idx] < best_value:
            best, best_value = population[idx].copy(), float(values[idx])
    return best, best_value, objective(best)[1:]
