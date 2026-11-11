"""Runtime smoke tests for the extracted PDE experiment modules.

These tests import the modules and evaluate their objective functions once with
a deterministic parameter vector. They do not run BES, PSO, GA, or WOA searches.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    (problem, algorithm)
    for problem in range(1, 5)
    for algorithm in ("bes", "pso", "ga", "woa")
]


@pytest.mark.parametrize(("problem", "algorithm"), MODULES)
def test_module_import_and_objective_smoke(problem, algorithm):
    path = (
        ROOT
        / "src"
        / "experiments"
        / f"problem_{problem}"
        / f"{algorithm}.py"
    )
    module_name = f"pde_problem_{problem}_{algorithm}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.U_approx)
    assert callable(module.U_exact)

    objective_name = "fitness_function_fd" if problem == 1 else "fitness_function"
    objective = getattr(module, objective_name)
    assert callable(objective)

    # Eight coefficients support all four approximation parameterizations;
    # Problem 1 uses the first six, while Problem 4 uses coefficient pairs.
    theta = np.zeros(8, dtype=float)
    result = objective(theta)

    values = np.asarray(result, dtype=float)
    assert values.ndim == 1
    assert values.size >= 1
    assert np.isfinite(values).all()
