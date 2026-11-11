"""Objective components for one-dimensional heat-equation benchmarks."""
import numpy as np


def compute_components(problem, theta, num_x=20, num_t=20):
    """Return PDE, initial-condition and boundary RMSE values.

    The PDE residual follows the finite-difference/Crank-Nicolson-style
    approximation used in the original notebook. Its value therefore depends
    on grid resolution and should not be interpreted as an exact derivative.
    """
    x = np.linspace(0.0, 1.0, num_x)
    t = np.linspace(0.0, 1.0, num_t)
    dx, dt = x[1] - x[0], t[1] - t[0]
    u = np.array([
        [problem.approximation(xi, ti, theta) for xi in x]
        for ti in t
    ])

    u_xx_n = (u[:-1, 2:] - 2 * u[:-1, 1:-1] + u[:-1, :-2]) / dx**2
    u_xx_np1 = (u[1:, 2:] - 2 * u[1:, 1:-1] + u[1:, :-2]) / dx**2
    u_t = (u[1:, 1:-1] - u[:-1, 1:-1]) / dt
    residual = u_t - 0.5 * (u_xx_n + u_xx_np1)
    pde_rmse = float(np.sqrt(np.mean(residual**2)))

    init_rmse = float(np.sqrt(np.mean(
        (u[0] - problem.initial(x))**2
    )))
    left_rmse = float(np.sqrt(np.mean(
        (u[:, 0] - problem.left_boundary(t))**2
    )))
    right_rmse = float(np.sqrt(np.mean(
        (u[:, -1] - problem.right_boundary(t))**2
    )))
    return pde_rmse, init_rmse, left_rmse, right_rmse


def fitness_function(problem, theta, lambda_initial=1.0, lambda_boundary=1.0,
                     num_x=20, num_t=20):
    components = compute_components(problem, theta, num_x, num_t)
    pde, initial, left, right = components
    loss = pde + lambda_initial * initial + lambda_boundary * (left + right)
    return (float(loss), *components)
