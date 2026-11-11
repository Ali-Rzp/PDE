import numpy as np

import matplotlib.pyplot as plt

np.random.seed(0)

def U_approx(x, t, theta):
    N = len(theta)
    return sum(theta[k] * np.sin((k+1) * np.pi * x) * np.exp(-((k+1) * np.pi)**2 * t) for k in range(N))

def U_exact(x, t):
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

def compute_components(theta):
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
        u_xx_n   = (U[n, 2:] - 2 * U[n, 1:-1] + U[n, :-2]) / dx**2
        u_xx_np1 = (U[n+1, 2:] - 2 * U[n+1, 1:-1] + U[n+1, :-2]) / dx**2
        u_t      = (U[n+1, 1:-1] - U[n, 1:-1]) / dt
        rhs      = 0.5 * (u_xx_n + u_xx_np1)
        pde_err.extend((u_t - rhs)**2)
    pde_rmse = np.sqrt(np.mean(pde_err))

    init_err = (U[0, :] - np.sin(np.pi * x_vals))**2
    init_rmse = np.sqrt(np.mean(init_err))

    bc_left_rmse  = np.sqrt(np.mean(U[:, 0]**2))
    bc_right_rmse = np.sqrt(np.mean(U[:, -1]**2))

    return pde_rmse, init_rmse, bc_left_rmse, bc_right_rmse

def fitness_function(theta):
    λ_init, λ_bc = 1.0, 1.0
    pde_rmse, init_rmse, bc_left_rmse, bc_right_rmse = compute_components(theta)
    total_loss = pde_rmse + λ_init * init_rmse + λ_bc * (bc_left_rmse + bc_right_rmse)
    return total_loss, pde_rmse, init_rmse, bc_left_rmse, bc_right_rmse

def bes_algorithm(fitness_function, pop_size, dimensions, max_iterations,
                   alpha=2, a=10, R=1.5, c1=2, c2=2):
    np.random.seed(0)
    population = np.random.rand(pop_size, dimensions)
    fitness_results = [fitness_function(individual) for individual in population]
    fitness = np.array([f[0] for f in fitness_results])

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    best_errors = fitness_results[best_idx][1:]

    for _ in range(max_iterations):
        for i in range(pop_size):
            r = np.random.rand()
            new_pos = best_solution + alpha * r * (population.mean(axis=0) - population[i])
            new_fit_all = fitness_function(new_pos)
            if new_fit_all[0] < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit_all[0]
                if new_fit_all[0] < best_fitness:
                    best_solution, best_fitness = new_pos.copy(), new_fit_all[0]
                    best_errors = new_fit_all[1:]

            theta = a * np.pi * np.random.rand()
            r_val = theta + R * np.random.rand()
            xr = r_val * np.sin(theta)
            yr = r_val * np.cos(theta)
            x = xr / (np.max(np.abs(xr)) + 1e-8)
            y = yr / (np.max(np.abs(yr)) + 1e-8)
            new_pos = population[i] + y * (population[i] - population[(i + 1) % pop_size]) + x * (population[i] - population.mean(axis=0))
            new_fit_all = fitness_function(new_pos)
            if new_fit_all[0] < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit_all[0]
                if new_fit_all[0] < best_fitness:
                    best_solution, best_fitness = new_pos.copy(), new_fit_all[0]
                    best_errors = new_fit_all[1:]

            theta_s = a * np.pi * np.random.rand()
            r_val = theta_s
            xr = r_val * np.sinh(theta_s)
            yr = r_val * np.cosh(theta_s)
            x = xr / (np.max(np.abs(xr)) + 1e-8)
            y = yr / (np.max(np.abs(yr)) + 1e-8)
            new_pos = np.random.rand() * best_solution + x * (population[i] - c1 * population.mean(axis=0)) + y * (population[i] - c2 * best_solution)
            new_fit_all = fitness_function(new_pos)
            if new_fit_all[0] < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit_all[0]
                if new_fit_all[0] < best_fitness:
                    best_solution, best_fitness = new_pos.copy(), new_fit_all[0]
                    best_errors = new_fit_all[1:]

    return best_solution, best_fitness, best_errors
