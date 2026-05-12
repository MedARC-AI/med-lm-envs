import ast
import importlib.util
from pathlib import Path

from verifiers.types import flatten_task_input

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_longhealth_module():
    module_path = REPO_ROOT / "environments" / "longhealth" / "longhealth.py"
    spec = importlib.util.spec_from_file_location("longhealth_local", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_code_does_not_emit_reserved_task_key() -> None:
    offenders = []
    for path in (REPO_ROOT / "environments").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and key.value == "task":
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{key.lineno}")
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                if isinstance(node.slice, ast.Constant) and node.slice.value == "task":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_copied_info_payloads_drop_reserved_task_key() -> None:
    offenders = []
    for path in (REPO_ROOT / "environments").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for fn in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
            copies_raw_payload = False
            drops_task = False
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if not (isinstance(target, ast.Name) and target.id == "info"):
                            continue
                        value = node.value
                        if (
                            isinstance(value, ast.Call)
                            and isinstance(value.func, ast.Name)
                            and value.func.id == "dict"
                            and value.args
                        ):
                            copies_raw_payload = True
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "info"):
                        continue
                    if node.func.attr == "pop" and node.args:
                        arg = node.args[0]
                        if isinstance(arg, ast.Constant) and arg.value == "task":
                            drops_task = True
            if copies_raw_payload and not drops_task:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{fn.lineno}:{fn.name}")

    assert offenders == []


def test_longhealth_task1_metadata_does_not_use_verifiers_task_key() -> None:
    module = _load_longhealth_module()

    env = module.load_environment(task="task1", max_examples=3, shuffle_docs=False)

    seen_tasks = set()
    for row in env.eval_dataset:
        info = row["info"]
        assert "task" not in info
        seen_tasks.add(info["longhealth_task"])
        assert flatten_task_input(row)["info"]["longhealth_task"] == info["longhealth_task"]

    assert seen_tasks == {"task1"}


def test_longhealth_task2_metadata_does_not_use_verifiers_task_key() -> None:
    module = _load_longhealth_module()

    env = module.load_environment(task="task2", max_examples=2, shuffle_docs=False)

    seen_tasks = set()
    for row in env.eval_dataset:
        info = row["info"]
        assert "task" not in info
        seen_tasks.add(info["longhealth_task"])
        assert flatten_task_input(row)["info"]["longhealth_task"] == info["longhealth_task"]

    assert seen_tasks == {"task2_negation", "task2_identification"}
