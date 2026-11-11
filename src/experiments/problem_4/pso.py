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

def pso_algorithm(fitness_function, pop_size, dimensions, max_iterations,
                  w=0.7, c1=1.5, c2=1.5):
    np.random.seed(0)
    X = np.random.rand(pop_size, dimensions)
    V = np.zeros((pop_size, dimensions))
    P = X.copy()

    fitness_all = [fitness_function(ind) for ind in X]
    fitness_vals = np.array([f[0] for f in fitness_all])
    P_fit = fitness_vals.copy()

    g_idx = np.argmin(fitness_vals)
    G = X[g_idx].copy()
    G_fit = fitness_vals[g_idx]
    G_errors = fitness_all[g_idx][1:]

    for _ in range(max_iterations):
        for i in range(pop_size):
            r1 = np.random.rand(dimensions)
            r2 = np.random.rand(dimensions)
            V[i] = w * V[i] + c1 * r1 * (P[i] - X[i]) + c2 * r2 * (G - X[i])
            X[i] += V[i]

            fit_all = fitness_function(X[i])
            fit = fit_all[0]
            if fit < P_fit[i]:
                P[i] = X[i].copy()
                P_fit[i] = fit
                if fit < G_fit:
                    G = X[i].copy()
                    G_fit = fit
                    G_errors = fit_all[1:]
    return G, G_fit, G_errors
