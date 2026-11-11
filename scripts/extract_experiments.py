"""Extract original optimizer functions from the research notebook without rewriting them.

Run from the repository root:
    python scripts/extract_experiments.py

The generated modules retain each cell's imports, explicit NumPy seed statement
(if present), and top-level function definitions verbatim. Plotting and
experiment-driver statements remain in the original notebook.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOK = Path("Finall_ALL_AL_PDE.ipynb")
OUTPUT = Path("src/experiments")

# Zero-based code-cell indices in the original notebook.
CELL_MAP = {
    "problem_1": {"bes": 3, "pso": 5, "ga": 7, "woa": 9},
    "problem_2": {"bes": 13, "pso": 15, "ga": 17, "woa": 19},
    "problem_3": {"bes": 23, "pso": 25, "ga": 27, "woa": 29},
    "problem_4": {"bes": 33, "pso": 35, "ga": 37, "woa": 39},
}


def extract_original_definitions(source: str) -> str:
    """Copy imports, explicit NumPy seed calls, and function definitions verbatim."""
    tree = ast.parse(source)
    lines = source.splitlines()
    selected = []

    for node in tree.body:
        is_import = isinstance(node, (ast.Import, ast.ImportFrom))
        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        is_seed = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "np.random.seed"
        )
        if is_import or is_function or is_seed:
            selected.append("\n".join(lines[node.lineno - 1:node.end_lineno]))

    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in tree.body):
        raise ValueError("No top-level function definitions found in source cell.")
    return "\n\n".join(selected).rstrip() + "\n"


def main() -> None:
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK}")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "__init__.py").write_text(
        '"""Original optimizer implementations extracted from the notebook."""\n',
        encoding="utf-8",
    )

    for problem, algorithms in CELL_MAP.items():
        problem_dir = OUTPUT / problem
        problem_dir.mkdir(parents=True, exist_ok=True)
        (problem_dir / "__init__.py").write_text(
            f'"""Original optimizer implementations for {problem}."""\n',
            encoding="utf-8",
        )

        for algorithm, cell_index in algorithms.items():
            cell = notebook["cells"][cell_index]
            if cell.get("cell_type") != "code":
                raise TypeError(f"Notebook cell {cell_index} is not a code cell.")

            source = "".join(cell.get("source", []))
            extracted = extract_original_definitions(source)
            target = problem_dir / f"{algorithm}.py"
            target.write_text(extracted, encoding="utf-8")
            print(f"Extracted {target} from notebook code cell {cell_index}")


if __name__ == "__main__":
    main()
