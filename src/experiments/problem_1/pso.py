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

def pso_algorithm(fitness_function, pop_size, dimensions, max_iter,
                  w=0.7, c1=1.5, c2=1.5, bounds=(-2, 2)):
    np.random.seed(0)
    pos = np.random.uniform(bounds[0], bounds[1], (pop_size, dimensions))
    vel = np.zeros_like(pos)
    personal_best = pos.copy()
    personal_best_val = np.array([fitness_function(ind)[0] for ind in pos])
    global_best = personal_best[np.argmin(personal_best_val)]
    global_best_val = np.min(personal_best_val)

    for _ in range(max_iter):
        for i in range(pop_size):
            r1, r2 = np.random.rand(dimensions), np.random.rand(dimensions)
            vel[i] = (w * vel[i]
                      + c1 * r1 * (personal_best[i] - pos[i])
                      + c2 * r2 * (global_best - pos[i]))
            pos[i] += vel[i]
            pos[i] = np.clip(pos[i], bounds[0], bounds[1])
            fit = fitness_function(pos[i])[0]
            if fit < personal_best_val[i]:
                personal_best[i] = pos[i].copy()
                personal_best_val[i] = fit
                if fit < global_best_val:
                    global_best = pos[i].copy()
                    global_best_val = fit
    return global_best, global_best_val
