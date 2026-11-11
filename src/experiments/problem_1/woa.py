import numpy as np

import matplotlib.pyplot as plt

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
        u_xx_n = (U[n, 2:] - 2*U[n, 1:-1] + U[n, :-2]) / dx**2
        u_xx_np1 = (U[n+1, 2:] - 2*U[n+1, 1:-1] + U[n+1, :-2]) / dx**2
        u_t = (U[n+1, 1:-1] - U[n, 1:-1]) / dt
        rhs = 0.5 * (u_xx_n + u_xx_np1)
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

def woa_algorithm(fitness_func, pop_size, dim, max_iter):
    np.random.seed(0)
    lb = -1.0
    ub = 1.0
    population = np.random.uniform(lb, ub, (pop_size, dim))
    fitness_vals = [fitness_func(ind)[0] for ind in population]

    best_idx = np.argmin(fitness_vals)
    best_pos = population[best_idx].copy()
    best_score = fitness_vals[best_idx]

    for iter in range(max_iter):
        a = 2 - iter * (2 / max_iter)  # عامل کاهش خطی a ∈ [2, 0]
        for i in range(pop_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.rand()

            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * best_pos - population[i])
                    new_pos = best_pos - A * D
                else:
                    rand_idx = np.random.randint(pop_size)
                    rand_pos = population[rand_idx]
                    D = abs(C * rand_pos - population[i])
                    new_pos = rand_pos - A * D
            else:
                b = 1
                l = np.random.uniform(-1, 1)
                D = abs(best_pos - population[i])
                new_pos = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos

            # محدودسازی به بازه تعریف‌شده
            new_pos = np.clip(new_pos, lb, ub)

            new_score = fitness_func(new_pos)[0]
            if new_score < fitness_vals[i]:
                population[i] = new_pos
                fitness_vals[i] = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_pos = new_pos.copy()

    return best_pos, best_score
