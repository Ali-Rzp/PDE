"""Particle Swarm Optimization."""
import numpy as np


def pso(objective, dimensions, pop_size=40, iterations=300, seed=0,
        inertia=0.7, cognitive=1.5, social=1.5):
    rng = np.random.default_rng(seed)
    positions = rng.random((pop_size, dimensions))
    velocities = np.zeros_like(positions)
    personal = positions.copy()
    values = np.array([objective(x)[0] for x in positions])
    personal_values = values.copy()
    best_idx = int(np.argmin(values))
    global_best, global_value = positions[best_idx].copy(), float(values[best_idx])

    for _ in range(iterations):
        for i in range(pop_size):
            r1, r2 = rng.random(dimensions), rng.random(dimensions)
            velocities[i] = (inertia * velocities[i]
                             + cognitive * r1 * (personal[i] - positions[i])
                             + social * r2 * (global_best - positions[i]))
            positions[i] += velocities[i]
            value = objective(positions[i])[0]
            if value < personal_values[i]:
                personal[i], personal_values[i] = positions[i].copy(), value
                if value < global_value:
                    global_best, global_value = positions[i].copy(), float(value)
    return global_best, global_value, objective(global_best)[1:]
