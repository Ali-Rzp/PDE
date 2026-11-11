# Solving Partial Differential Equations with Metaheuristic Algorithms

This repository contains implementations and experiments for solving partial differential equation (PDE) problems by formulating them as optimization tasks.

## Approach

1. Express the differential-equation problem as an objective (fitness) function.
2. Choose a suitable approximation/basis function to represent the unknown solution.
3. Use a metaheuristic optimizer to search for the approximation parameters.
4. Evaluate the resulting approximation against the problem formulation.

The repository currently includes a notebook with implementations of four metaheuristic approaches.

## Repository Contents

| File | Description |
|---|---|
| [Finall_ALL_AL_PDE.ipynb](./Finall_ALL_AL_PDE.ipynb) | Main notebook containing the algorithm implementations and PDE experiments |

## Getting Started

1. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab.
2. Review the problem definition and objective-function sections.
3. Set the PDE and its boundary/initial conditions as required by the notebook.
4. Run the notebook cells in order.

> **Note:** The exact approximation function and objective function depend on the PDE being solved. Review the notebook before adapting it to a different equation.

## Research Context

This work explores the use of the Eagle metaheuristic for differential-equation optimization and compares its performance with other optimization methods, including Whale Optimization, Particle Swarm Optimization (PSO), and Genetic Algorithm (GA).

## Author

**Ali Rezapour** — [GitHub](https://github.com/Ali-Rzp) · [Portfolio](https://ali-rezapour.vercel.app/)
