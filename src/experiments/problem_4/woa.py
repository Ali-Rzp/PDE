import numpy as np

import matplotlib.pyplot as plt

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

    pde_err = []
    for n in range(num_t - 1):
        u_xx_n   = (U[n, 2:] - 2 * U[n, 1:-1] + U[n, :-2]) / dx**2
        u_xx_np1 = (U[n+1, 2:] - 2 * U[n+1, 1:-1] + U[n+1, :-2]) / dx**2
        u_t      = (U[n+1, 1:-1] - U[n, 1:-1]) / dt
        rhs      = 0.5 * (u_xx_n + u_xx_np1)
        pde_err.extend((u_t - rhs)**2)
    pde_rmse = np.sqrt(np.mean(pde_err))

    init_target = np.sin(np.pi * x_vals) + np.cos(np.pi * x_vals)
    init_rmse = np.sqrt(np.mean((U[0, :] - init_target)**2))

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

def woa(fitness_function, pop_size, dim, max_iter, lb=-3, ub=3):
    np.random.seed(0)

    # مقداردهی اولیه جمعیت
    population = lb + (ub - lb) * np.random.rand(pop_size, dim)
    fitnesses = np.array([fitness_function(ind)[0] for ind in population])

    # بهترین راه‌حل فعلی
    best_idx = np.argmin(fitnesses)
    best_solution = population[best_idx].copy()
    best_fitness = fitnesses[best_idx]
    best_errors = fitness_function(best_solution)[1:]

    for t in range(max_iter):
        a = 2.0 - t * (2.0 / max_iter)  # کاهش خطی پارامتر a

        for i in range(pop_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.rand()
            l = np.random.uniform(-1, 1)

            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    # احاطه طعمه (بهترین راه‌حل)
                    D = np.abs(C * best_solution - population[i])
                    new_pos = best_solution - A * D
                else:
                    # جستجوی تصادفی طعمه
                    rand_idx = np.random.randint(pop_size)
                    X_rand = population[rand_idx]
                    D = np.abs(C * X_rand - population[i])
                    new_pos = X_rand - A * D
            else:
                # حرکت مارپیچی به سمت طعمه (حمله حبابی)
                D_prime = np.abs(best_solution - population[i])
                b = 1  # پارامتر شکل مارپیچ
                new_pos = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + best_solution

            # محدود کردن به بازه مشخص شده
            new_pos = np.clip(new_pos, lb, ub)

            # ارزیابی و به‌روزرسانی
            fit_new = fitness_function(new_pos)[0]
            if fit_new < fitnesses[i]:
                population[i] = new_pos
                fitnesses[i] = fit_new

                if fit_new < best_fitness:
                    best_solution = new_pos.copy()
                    best_fitness = fit_new
                    best_errors = fitness_function(best_solution)[1:]

    return best_solution, best_fitness, best_errors
