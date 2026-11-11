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
└── tests/
    └── test_extraction.py
```

## Generate the modular source files

The extraction script copies function definitions from the original notebook cells into separate Python modules. It preserves the original function bodies, imports, and explicit top-level NumPy seed calls. It does not redesign, merge, or tune any optimizer.

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/extract_experiments.py
```

Generated structure:

```text
src/
└── experiments/
    ├── __init__.py
    ├── problem_1/
    │   ├── __init__.py
    │   ├── bes.py
    │   ├── pso.py
    │   ├── ga.py
    │   └── woa.py
    ├── problem_2/
    ├── problem_3/
    └── problem_4/
```

Each of the 16 optimizer modules comes from its corresponding code cell in the notebook. Problem-specific helper functions are intentionally not unified because some definitions differ between cells. Notebook-only plotting and experiment-driver statements remain in the notebook.

## Validation

Run the lightweight extraction tests:

```bash
python -m pytest -q
```

The tests check that all mapped notebook cells are valid, all 16 modules can be generated and parsed, and the extracted function ASTs match the original notebook function ASTs. They do not rerun the 16 optimization experiments or claim new numerical results.

GitHub Actions runs these checks on pushes and pull requests to `main`.

## Reproducibility

- Saved notebook outputs remain the recorded outputs of the original experiments.
- Optimizers are stochastic; reruns can depend on random state and execution order.
- The extraction process is for code organization only. The notebook remains the source of truth for reported research results.

## Dependencies

NumPy and Matplotlib are used by the notebook. Jupyter is used to run it interactively. Pytest is used only for the extraction tests.

## Author

**Ali Rezapour**

- GitHub: https://github.com/Ali-Rzp
- Portfolio: https://ali-rezapour.vercel.app/
