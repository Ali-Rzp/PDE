"""Extract the original experiment functions from the thesis notebook.

This script copies function source verbatim from the notebook's code cells.
It does not rewrite algorithm logic, objective functions, or parameters.
Run from the repository root:
    python scripts/extract_experiments.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOK = Path("Finall_ALL_AL_PDE.ipynb")
OUTPUT = Path("src/experiments")

# Zero-based notebook cell index for each original problem/algorithm section.
CELL_MAP = {
    "problem_1": {"bes": 3, "pso": 5, "ga": 7, "woa": 9},
    "problem_2": {"bes": 13, "pso": 15, "ga": 17, "woa": 19},
    "problem_3": {"bes": 23, "pso": 25, "ga": 27, "woa": 29},
    "problem_4": {"bes": 33, "pso": 35, "ga": 37, "woa": 39},
}


def extract_functions(source: str) -> str:
    """Return top-level function definitions exactly as written in source."""
    tree = ast.parse(source)
    lines = source.splitlines()
    blocks = [
        "\n".join(lines[node.lineno - 1:node.end_lineno])
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not blocks:
        raise ValueError("No top-level function definitions found in source cell.")
    return "\n\n\n".join(blocks) + "\n"


def main() -> None:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK}")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "__init__.py").write_text(
        '"""Original notebook experiment implementations."""\n',
        encoding="utf-8",
    )

    for problem, algorithms in CELL_MAP.items():
        problem_dir = OUTPUT / problem
        problem_dir.mkdir(parents=True, exist_ok=True)
        (problem_dir / "__init__.py").write_text(
            f'"""Original experiment cells for {problem}."""\n',
            encoding="utf-8",
        )

        for algorithm, cell_index in algorithms.items():
            cell = notebook["cells"][cell_index]
            if cell.get("cell_type") != "code":
                raise TypeError(f"Notebook cell {cell_index} is not a code cell.")
            source = "".join(cell.get("source", []))
            extracted = extract_functions(source)
            header = (
                '"""\n'
                f"Function definitions copied from notebook code cell {cell_index} "
                f"({problem}, {algorithm.upper()}).\n"
                "Algorithm and objective logic are intentionally unchanged.\n"
                '"""\n'
                "import numpy as np\n\n"
                "np.random.seed(0)\n\n"
            )
            target = problem_dir / f"{algorithm}.py"
            target.write_text(header + extracted, encoding="utf-8")
            print(f"Extracted {problem}/{algorithm}.py from notebook cell {cell_index}")


if __name__ == "__main__":
    main()
