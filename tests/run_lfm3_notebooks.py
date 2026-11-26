"""Execute bounded copies of all notebooks with actual pretrained weights.

Usage: python tests/run_lfm3_notebooks.py --weights /path/to/lfm3
       --output /tmp/lfm3-notebooks [--notebook ch02/...ipynb]

Artifacts include the transformed source, cell outputs, and a manifest.
Caps: 8 generated tokens, 2 samples/rollouts, 1 training step/epoch/example.
Small local dataset fixtures exercise plumbing, not benchmark accuracy.
Live Ollama judge cells are explicitly skipped; compilation is opt-in in the
source notebooks. Historical log plots still use the original remote logs.
"""

import argparse
import ast
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
CAPS = {"max_new_tokens": 8, "num_samples": 2, "num_rollouts": 2,
        "steps": 1, "epochs": 1, "iterations": 1, "max_examples": 1,
        "num_examples": 1, "max_items": 1, "eval_max_items": 1}


class BoundWork(ast.NodeTransformer):
    def bounded(self, name, value):
        if name in CAPS and isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            return ast.copy_location(ast.Constant(min(value.value, CAPS[name])), value)
        return value

    def visit_keyword(self, node):
        self.generic_visit(node)
        node.value = self.bounded(node.arg, node.value)
        if node.arg in ("train_examples", "val_examples"):
            node.value = ast.Subscript(node.value, ast.Slice(upper=ast.Constant(1)), ast.Load())
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Name):
                node.value = self.bounded(target.id, node.value)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        args = node.args.args[-len(node.args.defaults):] if node.args.defaults else []
        node.args.defaults = [self.bounded(arg.arg, default)
                              for arg, default in zip(args, node.args.defaults)]
        return node


def fixture_data(directory):
    row = {"problem": "What is 1 + 1?", "answer": "2", "solution": r"\boxed{2}",
           "prediction": r"\boxed{2}", "generated_text": r"\boxed{2}",
           "message_content": r"The answer is \boxed{2}.", "message_thinking": "1 + 1 = 2."}
    for name in ("math_train.json", "math500_test.json", "deepseek-r1-math-train.json",
                 "math500_gpt_answers.json", "math500_qwen3_answers.json"):
        (directory / name).write_text(json.dumps([row] * 30))
    # Chapter 3's exercise reads the JSONL emitted by the evaluation notebook.
    (directory / "math500-mps.jsonl").write_text(json.dumps(row) + "\n")


def setup_source():
    return f'''
import os, sys, inspect, functools
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
get_ipython().run_line_magic("matplotlib", "inline")
import torch
from reasoning_from_scratch import ch02, ch03, ch04, ch05, ch06, ch08
_caps = {CAPS!r}
def _bound_call(fn):
    signature = inspect.signature(fn)
    @functools.wraps(fn)
    def run(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        for name, limit in _caps.items():
            if name in signature.parameters:
                value = bound.arguments.get(name, signature.parameters[name].default)
                if isinstance(value, int): bound.arguments[name] = min(value, limit)
        return fn(*bound.args, **bound.kwargs)
    return run
for _module in (ch02, ch03, ch04, ch05, ch06, ch08):
    for _name, _fn in list(vars(_module).items()):
        if inspect.isfunction(_fn) and _fn.__module__ == _module.__name__:
            if any(_key in inspect.signature(_fn).parameters for _key in _caps):
                setattr(_module, _name, _bound_call(_fn))
print("BOUNDED PRETRAINED VALIDATION: local synthetic data; caps:", _caps)
'''


def run_notebook(path, weights, output):
    relative = path.relative_to(ROOT)
    work = output / str(relative).replace("/", "__").removesuffix(".ipynb")
    work.mkdir(parents=True, exist_ok=True)
    local_weights = work / "lfm3"
    if not local_weights.exists():
        local_weights.symlink_to(weights, target_is_directory=True)
    fixture_data(work)
    nb = nbformat.read(path, as_version=4)
    changes, skipped = [], []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        original = cell.source
        cell.outputs = []
        cell.execution_count = None
        if path.name == "chF_main.ipynb" and index in (41, 44, 49):
            cell.source = f"print('SKIPPED live Ollama dependency: source cell {index}')"
            skipped.append(index)
        elif original.lstrip().startswith("!"):
            cell.source = original.replace("!uv run", "!" + sys.executable)
        else:
            try:
                cell.source = ast.unparse(ast.fix_missing_locations(BoundWork().visit(ast.parse(original))))
            except SyntaxError:
                pass
        if cell.source != original:
            changes.append(index)
    nb.cells.insert(0, nbformat.v4.new_code_cell(setup_source()))
    entry = {"notebook": str(relative), "changed_cells": changes, "skipped_cells": skipped,
             "caps": CAPS, "dataset": "30 synthetic arithmetic examples",
             "weights": str(weights), "artifact": str(work / "executed.ipynb")}
    try:
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(work)}}).execute()
        entry["status"] = "passed"
    except Exception as error:
        entry["status"] = "failed"
        entry["error"] = str(error)[-5000:]
    nbformat.write(nb, work / "executed.ipynb")
    # Training was real, but keep artifacts small after recording the outputs.
    for checkpoint in work.rglob("*.pth"):
        if not checkpoint.is_symlink():
            checkpoint.unlink()
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--notebook", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / p for p in args.notebook] if args.notebook else sorted(ROOT.glob("ch*/**/*.ipynb"))
    results = []
    for path in paths:
        print(f"RUNNING {path.relative_to(ROOT)}", flush=True)
        result = run_notebook(path, args.weights.resolve(), args.output.resolve())
        results.append(result)
        (args.output / "manifest.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
    return int(any(entry["status"] == "failed" for entry in results))


if __name__ == "__main__":
    sys.exit(main())
