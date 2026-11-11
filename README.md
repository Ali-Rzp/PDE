# Metaheuristic Optimization for Heat-Equation PDEs

This repository contains the original computational experiments from my research on solving one-dimensional heat-equation problems using metaheuristic optimization algorithms.

## Original research notebook

The original notebook is the authoritative research artifact. It retains the four benchmark problems, optimizer experiment cells, saved outputs, and plots:

- [Finall_ALL_AL_PDE.ipynb](./Finall_ALL_AL_PDE.ipynb)

The notebook compares Bald Eagle Search (BES), Particle Swarm Optimization (PSO), Genetic Algorithm (GA), and Whale Optimization Algorithm (WOA) across four problem settings.

## Repository structure

```text
PDE/
├── Finall_ALL_AL_PDE.ipynb
├── README.md
├── requirements.txt
├── scripts/
│   └── extract_experiments.py
├── src/
│   └── experiments/
│       ├── problem_1/
│       ├── problem_2/
│       ├── problem_3/
│       └── problem_4/
└── tests/
    ├── test_extraction.py
    └── test_runtime_smoke.py
```

Each problem folder contains the original BES, PSO, GA, and WOA implementations extracted from its corresponding notebook cells.

## Generate the modular source files

The extraction script copies function definitions from the original notebook cells into separate Python modules. It preserves the original function bodies, imports, and explicit top-level NumPy seed calls. It does not redesign, merge, or tune any optimizer.

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/extract_experiments.py
```

Problem-specific helper functions are intentionally not unified because some definitions differ between cells. Notebook-only plotting and experiment-driver statements remain in the notebook.

## Validation

Run the tests:

```bash
python -m pytest -q
```

The tests cover three different levels:

1. **Extraction structure:** all 16 mapped notebook cells are code cells and contain valid function definitions.
2. **Source preservation:** generated function ASTs match the original notebook function ASTs.
3. **Runtime smoke checks:** all 16 modules are imported, and each problem's objective function is evaluated once with a simple finite parameter vector.

Runtime smoke checks are not full optimizer runs. They do not rerun BES, PSO, GA, or WOA searches, reproduce the saved research results, or prove numerical equivalence across all parameter settings.

GitHub Actions runs these checks on pushes and pull requests to `main`.

## Reproducibility and scope

- Saved notebook outputs remain the recorded outputs of the original experiments.
- Optimizers are stochastic; reruns can depend on random state and execution order.
- The extraction process is for code organization and basic runtime validation only.
- The original notebook remains the source of truth for reported research results.

## Dependencies

NumPy and Matplotlib are used by the notebook and extracted modules. Jupyter is used to run the notebook interactively. Pytest is used for automated tests.

## Author

**Ali Rezapour**

- GitHub: https://github.com/Ali-Rzp
- Portfolio: https://ali-rezapour.vercel.app/
