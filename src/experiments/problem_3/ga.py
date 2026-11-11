import numpy as np

import matplotlib.pyplot as plt

np.random.seed(0)

def U_approx(x, t, theta):
    K = len(theta) // 2
    u = 0.0
    for k in range(K):
        A = theta[2 * k]
        B = theta[2 * k + 1]
        λ = ((k + 1) * np.pi) ** 2
        u += A * np.cos((k + 1) * np.pi * x) * np.exp(-λ * t)
        u += B * np.sin((k + 1) * np.pi * x) * np.exp(-λ * t)
    return u

def U_exact(x, t):
    return (np.sin(np.pi * x) + np.cos(np.pi * x)) * np.exp(-np.pi**2 * t)

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

    # PDE RMSE
    pde_err = []
    for n in range(num_t - 1):
        u_xx_n = (U[n, 2:] - 2 * U[n, 1:-1] + U[n, :-2]) / dx**2
        u_xx_np1 = (U[n+1, 2:] - 2 * U[n+1, 1:-1] + U[n+1, :-2]) / dx**2
        u_t = (U[n+1, 1:-1] - U[n, 1:-1]) / dt
        rhs = 0.5 * (u_xx_n + u_xx_np1)
        pde_err.extend((u_t - rhs)**2)
    pde_rmse = np.sqrt(np.mean(pde_err))

    # Initial Condition
    init_target = np.sin(np.pi * x_vals) + np.cos(np.pi * x_vals)
    init_rmse = np.sqrt(np.mean((U[0, :] - init_target)**2))

    # Boundary Conditions
    bc_left_target = np.exp(-np.pi**2 * t_vals)
    bc_right_target = -np.exp(-np.pi**2 * t_vals)
    bc_left_rmse = np.sqrt(np.mean((U[:, 0] - bc_left_target)**2))
    bc_right_rmse = np.sqrt(np.mean((U[:, -1] - bc_right_target)**2))

    return pde_rmse, init_rmse, bc_left_rmse, bc_right_rmse

def fitness_function(theta):
    λ_init, λ_bc = 1, 1
    pde_rmse, init_rmse, bc_left_rmse, bc_right_rmse = compute_components(theta)
    total_loss = pde_rmse + λ_init * init_rmse + λ_bc * (bc_left_rmse + bc_right_rmse)
    return total_loss, pde_rmse, init_rmse, bc_left_rmse, bc_right_rmse

def ga_algorithm(fitness_func, pop_size, dimensions, max_generations,
                 crossover_rate=0.8, mutation_rate=0.1):
    population = np.random.uniform(-1, 1, (pop_size, dimensions))
    fitness = np.array([fitness_func(ind)[0] for ind in population])
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]
    best_errors = fitness_func(best_solution)[1:]

    for gen in range(max_generations):
        new_population = []
        for _ in range(pop_size // 2):
            # انتخاب والد با تورنومنت
            idxs = np.random.choice(pop_size, 4, replace=False)
            p1 = population[idxs[:2]][np.argmin(fitness[idxs[:2]])]
            p2 = population[idxs[2:]][np.argmin(fitness[idxs[2:]])]

            # کراس اور
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand(dimensions)
                child1 = alpha * p1 + (1 - alpha) * p2
                child2 = alpha * p2 + (1 - alpha) * p1
            else:
                child1, child2 = p1.copy(), p2.copy()

            # جهش
            for child in [child1, child2]:
                mutation_mask = np.random.rand(dimensions) < mutation_rate
                child[mutation_mask] += np.random.normal(0, 0.1, size=dimensions)[mutation_mask]
                new_population.append(child)

        population = np.clip(np.array(new_population), -2, 2)
        fitness = np.array([fitness_func(ind)[0] for ind in population])
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_solution = population[current_best_idx]
            best_fitness = fitness[current_best_idx]
            best_errors = fitness_func(best_solution)[1:]

    return best_solution, best_fitness, best_errors
