# Metaheuristic Optimization for Partial Differential Equations

This repository contains the computational work associated with my research on solving partial differential equation (PDE) problems using metaheuristic optimization.

The experiments compare four algorithms:

- **Bald Eagle Search (BES)**
- **Particle Swarm Optimization (PSO)**
- **Genetic Algorithm (GA)**
- **Whale Optimization Algorithm (WOA)**

The original research notebook is preserved as the primary record of the experiments and their saved outputs.

> [!IMPORTANT]
> **The Python package layout in this repository is preliminary and has not yet completed a full manual review.** The folders and modules under `src/experiments/` are an initial organization of code extracted from selected notebook cells. Their current layout should not be treated as a finalized software architecture or as a fully validated, independently reproducible implementation. The structure, dependencies, function interfaces, and experiment flow will be reviewed and tested before being considered final.

## Research source of truth

The original notebook is:

- [`Finall_ALL_AL_PDE.ipynb`](./Finall_ALL_AL_PDE.ipynb)

It contains the original experiment code, the four problem settings, optimizer runs, saved outputs, and plots. When there is any uncertainty or discrepancy, consult the notebook first. The extracted Python files do not replace it.

## Current repository layout

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
├── tests/
│   ├── test_extraction.py
│   └── test_runtime_smoke.py
└── .github/
    └── workflows/
        └── extraction-tests.yml
```

The current folders group selected optimizer functions by problem and algorithm. **This grouping is provisional.** We still need to review whether the problem boundaries, shared functions, module responsibilities, imports, constants, and naming are the best representation of the actual research workflow.

## Extracted modules

The script `scripts/extract_experiments.py` reads selected code cells from the original notebook and writes separate Python files under `src/experiments/`.

At present, it copies:

- Top-level imports from each selected cell.
- Top-level function definitions.
- Explicit top-level calls to `np.random.seed(...)`, when present.

It does not extract every notebook variable, constant, experiment driver, plotting statement, or execution dependency. It also does not refactor or redesign the algorithms.

To regenerate the current modules, run from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/extract_experiments.py
```

Because extraction is based on selected notebook cells, the generated files should be reviewed against the complete notebook before being used as a standalone package.

## Automated checks

Run the current checks with:

```bash
python -m pytest -q
```

The tests currently provide limited structural and runtime checks:

1. **Notebook-cell checks:** verify that the 16 mapped cells are code cells and contain top-level function definitions.
2. **Extraction checks:** generate the modules in a temporary directory and verify that the resulting Python files compile.
3. **Function-source comparison:** compare the extracted function ASTs with the corresponding notebook function ASTs.
4. **Runtime smoke checks:** import each of the 16 generated modules and call its objective function once with a simple zero-valued parameter vector. The returned values are checked for basic shape and finiteness.

These checks are useful safeguards, but they do **not** establish that:

- The current folder/module boundaries are the best design.
- Every dependency and required global variable has been captured.
- The extracted modules reproduce the complete notebook execution flow.
- The optimizers produce the same numerical results as the original experiments.
- The reported research results have been independently reproduced.

GitHub Actions runs the automated checks when relevant changes are pushed or submitted through a pull request to `main`.

## Planned manual review

Before treating the Python structure as final, the following work remains:

- Review the four problem definitions and confirm the mapping to the notebook.
- Trace every optimizer's inputs, outputs, helper functions, constants, and dependencies.
- Check whether shared code should be separated or whether problem-specific implementations should remain independent.
- Verify that each extracted module can be used in a clear and documented way.
- Add tests for mathematical behavior and representative parameter values, based on the original problem definitions.
- Decide on a reproducible experiment runner and compare selected runs with the original notebook without overwriting its saved research outputs.

Until this review is complete, the original notebook remains the authoritative research artifact.

## Dependencies

The current `requirements.txt` lists:

- NumPy
- Matplotlib
- Jupyter
- pytest

The dependency list and supported Python environment will also be reviewed as the project is made reproducible.

## Author

**Ali Rezapour**

- GitHub: https://github.com/Ali-Rzp
- Portfolio: https://ali-rezapour.vercel.app/
