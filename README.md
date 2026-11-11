# Metaheuristic Optimization for Heat-Equation PDEs

This repository contains the computational experiments from my research on solving one-dimensional heat-equation problems using metaheuristic optimization algorithms.

The original research notebook is preserved as the primary implementation and source of the reported experimental outputs:

- [Open the original notebook](./Finall_ALL_AL_PDE.ipynb)

## Research overview

The notebook studies four heat-equation benchmark cases and compares four population-based optimization algorithms:

- Bald Eagle Search (BES)
- Particle Swarm Optimization (PSO)
- Genetic Algorithm (GA)
- Whale Optimization Algorithm (WOA)

The four approximation settings in the notebook are:

1. A polynomial approximation for the solution \(u(x,t)=x^2+2t\).
2. A Fourier sine-series approximation for \(u(x,t)=e^{-\pi^2t}\sin(\pi x)\).
3. A Fourier cosine-series approximation for \(u(x,t)=e^{-\pi^2t}\cos(\pi x)\).
4. A combined Fourier sine-cosine approximation for \(u(x,t)=e^{-\pi^2t}(\sin(\pi x)+\cos(\pi x))\).

For each case, the notebook defines an approximate solution, an analytical solution, error components, an objective function, and optimizer implementations. The PDE residual is evaluated numerically using finite differences in a Crank–Nicolson-style expression; initial and boundary-condition errors are also included in the objective.

## Repository structure

```text
PDE/
├── README.md
├── requirements.txt
└── Finall_ALL_AL_PDE.ipynb
```

The notebook is intentionally retained as the authoritative research artifact. No separate optimizer package is presented as an equivalent replacement.

## Run the notebook

Python 3.10 or newer is recommended.

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
jupyter notebook Finall_ALL_AL_PDE.ipynb
```

You can also open the notebook in Google Colab by uploading it or using GitHub's notebook preview.

## Reproducibility notes

- The notebook contains saved outputs from the original experiments.
- Re-running the notebook may produce different optimization results depending on random-number state, execution order, and environment.
- The saved outputs should be treated as the recorded results of the original notebook, not as results from a later refactor.
- The numerical PDE residual depends on the finite-difference grid. When interpreting very small solution errors, distinguish approximation error from discretization error in the residual.

## Dependencies

The notebook imports NumPy and Matplotlib. Jupyter is included to run it interactively. The `mpl_toolkits` package used for 3D plotting is provided by Matplotlib.

## Author

**Ali Rezapour**

- GitHub: https://github.com/Ali-Rzp
- Portfolio: https://ali-rezapour.vercel.app/
