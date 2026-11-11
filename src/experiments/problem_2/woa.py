import numpy as np

import matplotlib.pyplot as plt

def U_approx(x, t, theta):
    K = len(theta)
    return sum(theta[k] * np.sin((k + 1) * np.pi * x) * np.exp(-((k + 1) * np.pi)**2 * t) for k in range(K))

def U_exact(x, t):
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

def compute_errors(theta):
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

    init_rmse = np.sqrt(np.mean((U[0, :] - np.sin(np.pi * x_vals))**2))
    bc_left_rmse  = np.sqrt(np.mean(U[:, 0]**2))
    bc_right_rmse = np.sqrt(np.mean(U[:, -1]**2))

    return pde_rmse, init_rmse, bc_left_rmse, bc_right_rmse

def fitness_function(theta, λ_init=1.0, λ_bc=1.0):
    pde_rmse, init_rmse, bc_left, bc_right = compute_errors(theta)
    loss = pde_rmse + λ_init * init_rmse + λ_bc * (bc_left + bc_right)
    return loss, pde_rmse, init_rmse, bc_left, bc_right

def whale_optimization(fitness_fn, pop_size, dim, max_iter, lb=-5, ub=5):
    population = lb + (ub - lb) * np.random.rand(pop_size, dim)
    fitness_values = np.array([fitness_fn(ind)[0] for ind in population])

    best_idx = np.argmin(fitness_values)
    best_sol = population[best_idx].copy()
    best_fit = fitness_values[best_idx]
    best_errs = fitness_fn(best_sol)[1:]

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)
        for i in range(pop_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.rand()
            l = np.random.uniform(-1, 1)
            b = 1

            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    D = np.abs(C * best_sol - population[i])
                    new_pos = best_sol - A * D
                else:
                    rand_idx = np.random.randint(pop_size)
                    X_rand = population[rand_idx]
                    D = np.abs(C * X_rand - population[i])
                    new_pos = X_rand - A * D
            else:
                D = np.abs(best_sol - population[i])
                new_pos = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_sol

            new_pos = np.clip(new_pos, lb, ub)
            new_fit = fitness_fn(new_pos)[0]

            if new_fit < fitness_values[i]:
                population[i] = new_pos
                fitness_values[i] = new_fit
                if new_fit < best_fit:
                    best_sol = new_pos.copy()
                    best_fit = new_fit
                    best_errs = fitness_fn(best_sol)[1:]

    return best_sol, best_fit, best_errs
