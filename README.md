# Solving Heat-Equation PDEs with Metaheuristic Optimization

This repository studies population-based metaheuristics for estimating parameters of approximate solutions to one-dimensional heat-equation problems.

## Research workflow

1. Represent the unknown solution with a problem-specific approximation basis.
2. Build an objective from the PDE residual, initial condition, and boundary conditions.
3. Search for approximation parameters using metaheuristic optimizers.
4. Compare the approximation with the analytical solution where available.

Four benchmark problems and four optimizers are included: Bald Eagle Search (BES), Particle Swarm Optimization (PSO), Genetic Algorithm (GA), and Whale Optimization Algorithm (WOA).

## Repository structure

```text
PDE/
├── README.md
├── requirements.txt
├── Finall_ALL_AL_PDE.ipynb       # Original thesis notebook, preserved
├── src/
│   ├── problems.py               # Benchmark problems and approximation bases
│   ├── objective.py              # Shared finite-difference objective
│   └── optimizers/
│       ├── bes.py
│       ├── pso.py
│       ├── ga.py
│       └── woa.py
└── tests/
    └── test_smoke.py
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Quick start

Run from the repository root:

```python
from src.problems import PROBLEMS
from src.objective import fitness_function
from src.optimizers import bes, pso, genetic_algorithm, whale_optimization

problem = PROBLEMS["sine"]
dimensions = 2

def objective(theta):
    return fitness_function(problem, theta)

for optimizer in (bes, pso, genetic_algorithm, whale_optimization):
    if optimizer is genetic_algorithm:
        best, loss, components = optimizer(objective, dimensions, generations=300)
    else:
        best, loss, components = optimizer(objective, dimensions, iterations=300)
    print(optimizer.__name__, best, loss, components)
```

Use six parameters for the polynomial benchmark, one parameter per mode for sine/cosine series, and an even number of parameters for the mixed sine-cosine basis.

## Benchmark problems

| Key | Approximation | Analytical solution |
|---|---|---|
| `polynomial` | Six-term polynomial | `x² + 2t` |
| `sine` | Decaying sine series | `exp(-π²t) sin(πx)` |
| `cosine` | Decaying cosine series | `exp(-π²t) cos(πx)` |
| `mixed` | Combined sine-cosine series | `exp(-π²t)(sin(πx)+cos(πx))` |

## Objective and numerical note

The objective returns four components: PDE residual RMSE, initial-condition RMSE, left-boundary RMSE, and right-boundary RMSE. The PDE residual retains the finite-difference / Crank-Nicolson-style calculation used in the original notebook. Its value depends on spatial and temporal grid resolution and can remain nonzero even when the approximation is close to the analytical solution. A convergence study or an analytical-derivative residual is needed for stronger numerical claims.

## Genetic Algorithm note

The new GA module uses tournament selection and elitism. This differs from the original notebook's GA variant, which selected parents uniformly at random. Therefore, its results must not be mixed with the notebook's saved GA outputs without rerunning the experiments.

## Reproducibility

All optimizer functions accept a `seed` argument. The original notebook is preserved as a record of the thesis experiments. Its saved outputs are historical and are not results from the refactored package. No new benchmark scores are claimed here.

## Original notebook

Open [Finall_ALL_AL_PDE.ipynb](./Finall_ALL_AL_PDE.ipynb) in Jupyter or Google Colab.

## Author

**Ali Rezapour** · [GitHub](https://github.com/Ali-Rzp) · [Portfolio](https://ali-rezapour.vercel.app/)
