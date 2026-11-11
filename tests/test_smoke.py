import numpy as np

from src.problems import PROBLEMS
from src.objective import fitness_function
from src.optimizers import bes, pso, genetic_algorithm, whale_optimization


def test_problem_definitions_are_finite():
    for problem in PROBLEMS.values():
        assert np.isfinite(problem.exact(0.25, 0.1))


def test_objective_returns_finite_components():
    result = fitness_function(PROBLEMS["sine"], np.array([1.0, 0.0]))
    assert len(result) == 5
    assert np.all(np.isfinite(result))


def test_optimizer_smoke_runs():
    objective = lambda theta: fitness_function(PROBLEMS["sine"], theta)
    optimizers = [
        lambda: bes(objective, dimensions=2, pop_size=8, iterations=3),
        lambda: pso(objective, dimensions=2, pop_size=8, iterations=3),
        lambda: genetic_algorithm(objective, dimensions=2, pop_size=8, generations=3),
        lambda: whale_optimization(objective, dimensions=2, pop_size=8, iterations=3),
    ]
    for run in optimizers:
        solution, loss, components = run()
        assert solution.shape == (2,)
        assert np.isfinite(loss)
        assert len(components) == 4
