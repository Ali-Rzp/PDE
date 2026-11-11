"""Benchmark heat-equation problems and their approximation bases."""
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class HeatEquationProblem:
    name: str
    description: str
    exact: Callable
    initial: Callable
    left_boundary: Callable
    right_boundary: Callable
    approximation: Callable


def polynomial_approximation(x, t, theta):
    """Six-parameter polynomial ansatz used for the first benchmark."""
    return (theta[0] + theta[1] * x + theta[2] * t
            + theta[3] * x**2 + theta[4] * x * t + theta[5] * t**2)


def sine_series(x, t, theta):
    return sum(theta[k] * np.sin((k + 1) * np.pi * x)
               * np.exp(-((k + 1) * np.pi) ** 2 * t)
               for k in range(len(theta)))


def cosine_series(x, t, theta):
    return sum(theta[k] * np.cos((k + 1) * np.pi * x)
               * np.exp(-((k + 1) * np.pi) ** 2 * t)
               for k in range(len(theta)))


def mixed_series(x, t, theta):
    modes = len(theta) // 2
    value = 0.0
    for k in range(modes):
        eigenvalue = ((k + 1) * np.pi) ** 2
        decay = np.exp(-eigenvalue * t)
        value += theta[2 * k] * np.cos((k + 1) * np.pi * x) * decay
        value += theta[2 * k + 1] * np.sin((k + 1) * np.pi * x) * decay
    return value


PROBLEMS = {
    "polynomial": HeatEquationProblem(
        name="Polynomial heat-equation benchmark",
        description="u_t = u_xx, u(x,0)=x^2, u(0,t)=2t, u(1,t)=1+2t",
        exact=lambda x, t: x**2 + 2 * t,
        initial=lambda x: x**2,
        left_boundary=lambda t: 2 * t,
        right_boundary=lambda t: 1 + 2 * t,
        approximation=polynomial_approximation,
    ),
    "sine": HeatEquationProblem(
        name="Sine heat-equation benchmark",
        description="u_t = u_xx, u(x,0)=sin(pi*x), with zero boundary values",
        exact=lambda x, t: np.exp(-np.pi**2 * t) * np.sin(np.pi * x),
        initial=lambda x: np.sin(np.pi * x),
        left_boundary=lambda t: np.zeros_like(t, dtype=float),
        right_boundary=lambda t: np.zeros_like(t, dtype=float),
        approximation=sine_series,
    ),
    "cosine": HeatEquationProblem(
        name="Cosine heat-equation benchmark",
        description="u_t = u_xx, cosine initial profile and time-dependent boundary values",
        exact=lambda x, t: np.exp(-np.pi**2 * t) * np.cos(np.pi * x),
        initial=lambda x: np.cos(np.pi * x),
        left_boundary=lambda t: np.exp(-np.pi**2 * t),
        right_boundary=lambda t: -np.exp(-np.pi**2 * t),
        approximation=cosine_series,
    ),
    "mixed": HeatEquationProblem(
        name="Mixed sine-cosine heat-equation benchmark",
        description="u_t = u_xx with initial profile sin(pi*x)+cos(pi*x)",
        exact=lambda x, t: np.exp(-np.pi**2 * t) * (
            np.sin(np.pi * x) + np.cos(np.pi * x)
        ),
        initial=lambda x: np.sin(np.pi * x) + np.cos(np.pi * x),
        left_boundary=lambda t: np.exp(-np.pi**2 * t),
        right_boundary=lambda t: -np.exp(-np.pi**2 * t),
        approximation=mixed_series,
    ),
}
