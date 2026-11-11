import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Finall_ALL_AL_PDE.ipynb"
SCRIPT = ROOT / "scripts" / "extract_experiments.py"
CELL_MAP = {
    "problem_1": {"bes": 3, "pso": 5, "ga": 7, "woa": 9},
    "problem_2": {"bes": 13, "pso": 15, "ga": 17, "woa": 19},
    "problem_3": {"bes": 23, "pso": 25, "ga": 27, "woa": 29},
    "problem_4": {"bes": 33, "pso": 35, "ga": 37, "woa": 39},
}


def test_all_sixteen_original_cells_are_valid():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for algorithms in CELL_MAP.values():
        for index in algorithms.values():
            cell = notebook["cells"][index]
            assert cell["cell_type"] == "code"
            tree = ast.parse("".join(cell.get("source", [])))
            assert any(isinstance(n, ast.FunctionDef) for n in tree.body)


def test_extraction_generates_sixteen_valid_modules(tmp_path):
    shutil.copy2(NOTEBOOK, tmp_path / NOTEBOOK.name)
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    subprocess.run([sys.executable, "scripts/extract_experiments.py"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    for problem, algorithms in CELL_MAP.items():
        for algorithm in algorithms:
            path = tmp_path / "src/experiments" / problem / f"{algorithm}.py"
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")


def test_extracted_function_bodies_match_original_notebook(tmp_path):
    shutil.copy2(NOTEBOOK, tmp_path / NOTEBOOK.name)
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    subprocess.run([sys.executable, "scripts/extract_experiments.py"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for problem, algorithms in CELL_MAP.items():
        for algorithm, index in algorithms.items():
            original = ast.parse("".join(notebook["cells"][index]["source"]))
            generated_path = tmp_path / "src/experiments" / problem / f"{algorithm}.py"
            generated = ast.parse(generated_path.read_text(encoding="utf-8"))
            original_defs = [ast.dump(n, include_attributes=False) for n in original.body
                             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            generated_defs = [ast.dump(n, include_attributes=False) for n in generated.body
                              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            assert generated_defs == original_defs
