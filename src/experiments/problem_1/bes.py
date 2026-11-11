import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)

def U_approx(x, t, theta):
    return (theta[0] + theta[1]*x + theta[2]*t +
            theta[3]*x**2 + theta[4]*x*t + theta[5]*t**2)

def U_exact(x, t):
    return x**2 + 2*t

def fitness_function_fd(theta):
    num_x, num_t = 20, 20
    x_vals = np.linspace(0, 1, num_x)
    t_vals = np.linspace(0, 1, num_t)
    dx = x_vals[1] - x_vals[0]
    dt = t_vals[1] - t_vals[0]

    U = np.zeros((num_t, num_x))
    for i, t in enumerate(t_vals):
        for j, x in enumerate(x_vals):
            U[i, j] = U_approx(x, t, theta)

    pde_err = []
    for n in range(num_t - 1):
        u_xx_n   = (U[n, 2:] - 2*U[n, 1:-1] + U[n, :-2]) / dx**2
        u_xx_np1 = (U[n+1, 2:] - 2*U[n+1, 1:-1] + U[n+1, :-2]) / dx**2
        u_t      = (U[n+1, 1:-1] - U[n, 1:-1]) / dt
        rhs      = 0.5 * (u_xx_n + u_xx_np1)
        pde_err.extend((u_t - rhs)**2)
    pde_rmse = np.sqrt(np.mean(pde_err))

    init_err = [(U[0, j] - x_vals[j]**2)**2 for j in range(num_x)]
    init_rmse = np.sqrt(np.mean(init_err))

    bc_left_err  = [(U[i, 0] - 2*t_vals[i])**2 for i in range(num_t)]
    bc_right_err = [(U[i, -1] - (1 + 2*t_vals[i]))**2 for i in range(num_t)]
    bc_left_rmse  = np.sqrt(np.mean(bc_left_err))
    bc_right_rmse = np.sqrt(np.mean(bc_right_err))

    λ_init, λ_bc = 1, 1
    total_loss = pde_rmse + λ_init*init_rmse + λ_bc*(bc_left_rmse + bc_right_rmse)

    return total_loss, pde_rmse, init_rmse, bc_left_rmse, bc_right_rmse

def bes_algorithm(fitness_function, pop_size, dimensions, max_iterations,
                  alpha=2, a=10, R=1.5, c1=2, c2=2):
    np.random.seed(0)
    population = np.random.rand(pop_size, dimensions)
    fitness = np.array([fitness_function(ind)[0] for ind in population])

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    for _ in range(max_iterations):
        for i in range(pop_size):
            # Exploitation
            r = np.random.rand()
            new_pos = best_solution + alpha * r * (population.mean(axis=0) - population[i])
            new_fit = fitness_function(new_pos)[0]
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_fitness:
                    best_solution, best_fitness = new_pos.copy(), new_fit

            # Exploration
            theta = a * np.pi * np.random.rand()
            r_val = theta + R * np.random.rand()
            xr = r_val * np.sin(theta)
            yr = r_val * np.cos(theta)
            x = xr / (np.max(np.abs(xr)) + 1e-8)
            y = yr / (np.max(np.abs(yr)) + 1e-8)
            new_pos = (population[i]
                       + y * (population[i] - population[(i + 1) % pop_size])
                       + x * (population[i] - population.mean(axis=0)))
            new_fit = fitness_function(new_pos)[0]
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_fitness:
                    best_solution, best_fitness = new_pos.copy(), new_fit

            # Swoop
            theta = a * np.pi * np.random.rand()
            r_val = theta
            xr = r_val * np.sinh(theta)
            yr = r_val * np.cosh(theta)
            x = xr / (np.max(np.abs(xr)) + 1e-8)
            y = yr / (np.max(np.abs(yr)) + 1e-8)
            new_pos = (np.random.rand() * best_solution
                       + x * (population[i] - c1 * population.mean(axis=0))
                       + y * (population[i] - c2 * best_solution))
            new_fit = fitness_function(new_pos)[0]
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                if new_fit < best_fitness:
                    best_solution, best_fitness = new_pos.copy(), new_fit

    return best_solution, best_fitness
