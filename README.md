# Metaheuristic Optimization for Heat-Equation PDEs

This repository contains the original computational experiments from my research on solving one-dimensional heat-equation problems using metaheuristic optimization algorithms.

## Original research notebook

The original notebook is preserved as the authoritative research artifact, including its saved outputs, plots, and experiment cells:

- [Finall_ALL_AL_PDE.ipynb](./Finall_ALL_AL_PDE.ipynb)

It covers four benchmark problems and compares four population-based optimizers: Bald Eagle Search (BES), Particle Swarm Optimization (PSO), Genetic Algorithm (GA), and Whale Optimization Algorithm (WOA).

## Project organization

```text
PDE/
├── Finall_ALL_AL_PDE.ipynb
├── README.md
├── requirements.txt
└── scripts/
    └── extract_experiments.py
```

The extraction script copies the original function definitions from the notebook's corresponding code cells into a problem/algorithm folder structure under `src/experiments/`. It does not replace the algorithm logic with newly written implementations.

## Generate the modular source files

Run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/extract_experiments.py
```

The script creates this structure:

```text
src/
└── experiments/
    ├── problem_1/
    │   ├── bes.py
    │   ├── pso.py
    │   ├── ga.py
    │   └── woa.py
    ├── problem_2/
    ├── problem_3/
    └── problem_4/
```

Each optimizer module is extracted from its own original notebook cell. This is intentional: some definitions differ between cells, so they are not silently unified. The generated modules contain the function definitions from those cells, while notebook-only plotting and experiment-driver code remain in the original notebook.

## Benchmark overview

1. Polynomial approximation for the first heat-equation case.
2. Fourier sine-series approximation.
3. Fourier cosine-series approximation.
4. Combined Fourier sine-cosine approximation.

The exact approximation definitions, objective terms, finite-difference calculation, optimizer parameters, and experimental settings are documented in the notebook and retained in the extracted function source.

## Reproducibility

- Saved notebook outputs are the recorded outputs of the original experiments.
- Re-running stochastic optimizers may produce different results depending on random-number state and execution order.
- The extraction script is a source-organization utility; it does not rerun experiments or claim new benchmark results.
- Do not compare results from altered implementations with the notebook's saved outputs unless the experimental setup is deliberately matched.

## Dependencies

```text
numpy
matplotlib
jupyter
```

## Author

**Ali Rezapour**

- GitHub: https://github.com/Ali-Rzp
- Portfolio: https://ali-rezapour.vercel.app/
