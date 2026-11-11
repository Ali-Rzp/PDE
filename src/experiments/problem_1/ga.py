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

def ga_algorithm(fitness_func, pop_size, dim, generations, crossover_rate=0.8, mutation_rate=0.1):
    def crossover(p1, p2):
        point = np.random.randint(1, dim)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2

    def mutate(ind):
        idx = np.random.randint(dim)
        ind[idx] += np.random.normal(0, 0.1)
        return ind

    population = np.random.rand(pop_size, dim)
    fitness_vals = np.array([fitness_func(ind)[0] for ind in population])
    best_idx = np.argmin(fitness_vals)
    best_solution = population[best_idx].copy()
    best_fit = fitness_vals[best_idx]
    best_errors = fitness_func(best_solution)[1:]

    for _ in range(generations):
        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = population[np.random.choice(pop_size, 2, replace=False)]
            if np.random.rand() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            if np.random.rand() < mutation_rate:
                c1 = mutate(c1)
            if np.random.rand() < mutation_rate:
                c2 = mutate(c2)
            new_pop.extend([c1, c2])
        population = np.array(new_pop[:pop_size])
        fitness_vals = np.array([fitness_func(ind)[0] for ind in population])
        best_idx = np.argmin(fitness_vals)
        if fitness_vals[best_idx] < best_fit:
            best_solution = population[best_idx].copy()
            best_fit = fitness_vals[best_idx]
            best_errors = fitness_func(best_solution)[1:]

    return best_solution, best_fit, best_errors
