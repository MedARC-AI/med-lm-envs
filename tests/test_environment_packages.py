"""Validate structural consistency of all environment packages.

These tests auto-discover every environment under ``environments/`` and verify
that each package has the required files, valid pyproject.toml configuration,
and a discoverable ``load_environment`` loader function.  All checks are
offline — no network calls, API keys, or dataset downloads are needed.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENTS_DIR = REPO_ROOT / "environments"


def _discover_envs() -> list[str]:
    """Return sorted list of environment directory names."""
    if not ENVIRONMENTS_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in ENVIRONMENTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_"))
    )


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file, using tomllib (3.11+) or tomli as fallback."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(path, "rb") as f:
        return tomllib.load(f)


def _find_load_environment(env_dir: Path) -> bool:
    """Search for a ``load_environment`` function in any .py file.

    Handles both single-module envs (``env.py``) and sub-package envs
    (``env/env/__init__.py`` or ``env/env/module.py``).
    """
    py_files: list[Path] = list(env_dir.rglob("*.py"))
    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "load_environment":
                    return True
    return False


ENV_NAMES = _discover_envs()

# Environments that are known to be missing [tool.prime.environment] metadata.
# These are pre-existing upstream issues and are marked as expected failures.
_KNOWN_MISSING_LOADER_METADATA: set[str] = {
    "healthbench",
    "med_dialog",
    "medagentbench",
    "medcasereasoning",
    "mtsamples_procedures",
    "mtsamples_replicate",
    "pubmedqa",
}


# ---------------------------------------------------------------------------
# 1. pyproject.toml exists and is parseable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_pyproject_exists_and_parses(env_name: str) -> None:
    """Every environment must have a parseable pyproject.toml."""
    toml_path = ENVIRONMENTS_DIR / env_name / "pyproject.toml"
    assert toml_path.exists(), f"{env_name}: missing pyproject.toml"
    data = _load_toml(toml_path)
    assert "project" in data, f"{env_name}: pyproject.toml missing [project] table"


# ---------------------------------------------------------------------------
# 2. [project] has required fields
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_project_has_name_and_version(env_name: str) -> None:
    """[project] must declare name and version."""
    data = _load_toml(ENVIRONMENTS_DIR / env_name / "pyproject.toml")
    project = data.get("project", {})
    assert "name" in project, f"{env_name}: [project] missing 'name'"
    assert "version" in project, f"{env_name}: [project] missing 'version'"


# ---------------------------------------------------------------------------
# 3. Build system is configured
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_build_system_configured(env_name: str) -> None:
    """pyproject.toml must have [build-system]."""
    data = _load_toml(ENVIRONMENTS_DIR / env_name / "pyproject.toml")
    assert "build-system" in data, f"{env_name}: missing [build-system]"


# ---------------------------------------------------------------------------
# 4. Loader is discoverable via [tool.prime.environment] or entry-points
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_loader_discoverable(env_name: str) -> None:
    """Environment loader must be declared in pyproject.toml.

    Accepted mechanisms:
      - [tool.prime.environment] with a ``loader`` key
      - [project.entry-points."verifiers.environments"]
    """
    if env_name in _KNOWN_MISSING_LOADER_METADATA:
        pytest.xfail(f"{env_name}: known to be missing loader metadata (upstream issue)")
    data = _load_toml(ENVIRONMENTS_DIR / env_name / "pyproject.toml")

    has_prime = (
        "tool" in data
        and "prime" in data.get("tool", {})
        and "environment" in data["tool"]["prime"]
        and "loader" in data["tool"]["prime"]["environment"]
    )

    has_entry_points = (
        "project" in data
        and "entry-points" in data.get("project", {})
        and "verifiers.environments" in data["project"]["entry-points"]
    )

    assert has_prime or has_entry_points, (
        f"{env_name}: no loader discoverable. Add [tool.prime.environment] "
        f"with 'loader' key, or [project.entry-points.\"verifiers.environments\"]"
    )


# ---------------------------------------------------------------------------
# 5. A load_environment function exists somewhere in the package
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_load_environment_exists(env_name: str) -> None:
    """The environment package must contain a load_environment function."""
    env_dir = ENVIRONMENTS_DIR / env_name
    assert _find_load_environment(env_dir), (
        f"{env_name}: no load_environment function found in any .py file"
    )


# ---------------------------------------------------------------------------
# 6. Dependencies include verifiers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_dependencies_include_verifiers(env_name: str) -> None:
    """All environments must depend on verifiers."""
    data = _load_toml(ENVIRONMENTS_DIR / env_name / "pyproject.toml")
    deps = data.get("project", {}).get("dependencies", [])
    dep_names = [
        d.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()
        for d in deps
    ]
    assert "verifiers" in dep_names, f"{env_name}: 'verifiers' not in dependencies"
