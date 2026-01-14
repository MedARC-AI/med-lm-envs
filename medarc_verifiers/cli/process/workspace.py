"""Workspace helpers for processing outputs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from medarc_verifiers.cli.hf import HFSyncConfig, download_hf_repo


@dataclass(slots=True)
class BaselineResult:
    policy: str
    files_copied: list[Path] = field(default_factory=list)
    files_overwritten: list[Path] = field(default_factory=list)
    files_skipped: list[Path] = field(default_factory=list)
    snapshot_dir: Path | None = None


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def is_nonempty_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for entry in path.iterdir():
        return True
    return False


def prepare_hf_baseline(
    *,
    output_dir: Path,
    hf_config: HFSyncConfig,
    pull_policy: str | None,
    is_tty: bool,
    prompt_func: Callable[[str], str] | None = None,
) -> BaselineResult:
    """Ensure output_dir is populated with HF snapshot per pull policy."""
    ensure_output_dir(output_dir)
    if not hf_config.repo_id:
        return BaselineResult(policy="local")

    policy = _resolve_pull_policy(pull_policy, is_tty=is_tty)
    result = BaselineResult(policy=policy)
    if not is_nonempty_dir(output_dir):
        snapshot_dir = download_hf_repo(
            repo_id=hf_config.repo_id,
            branch=hf_config.branch,
            token=hf_config.token,
            allow_patterns=["**/*.parquet", "env_index.json", "dataset_infos.json"],
            local_dir=None,
            local_only=False,
        )
        result.snapshot_dir = snapshot_dir
        _copy_snapshot(snapshot_dir, output_dir, result, overwrite=True)
        return result

    prompt_conflicts = False
    if policy == "prompt":
        choice = _prompt_baseline_choice(prompt_func, is_tty=is_tty)
        policy = choice
        result.policy = policy
        prompt_conflicts = policy == "pull"

    if policy == "clean":
        snapshot_dir = download_hf_repo(
            repo_id=hf_config.repo_id,
            branch=hf_config.branch,
            token=hf_config.token,
            allow_patterns=["**/*.parquet", "env_index.json", "dataset_infos.json"],
            local_dir=None,
            local_only=False,
        )
        result.snapshot_dir = snapshot_dir
        _clear_directory(output_dir)
        _copy_snapshot(snapshot_dir, output_dir, result, overwrite=True)
        return result

    if policy == "pull":
        if _has_complete_hf_baseline(output_dir):
            return BaselineResult(policy=policy)
        snapshot_dir = download_hf_repo(
            repo_id=hf_config.repo_id,
            branch=hf_config.branch,
            token=hf_config.token,
            allow_patterns=["**/*.parquet", "env_index.json", "dataset_infos.json"],
            local_dir=None,
            local_only=False,
        )
        result.snapshot_dir = snapshot_dir
        _copy_snapshot(
            snapshot_dir,
            output_dir,
            result,
            overwrite=False,
            prompt_func=prompt_func if prompt_conflicts else None,
            is_tty=is_tty if prompt_conflicts else False,
        )
        return result

    raise ValueError(f"Unsupported HF pull policy: {policy}")


def _resolve_pull_policy(pull_policy: str | None, *, is_tty: bool) -> str:
    if pull_policy:
        return pull_policy
    return "prompt" if is_tty else "pull"


def _prompt_baseline_choice(prompt_func: Callable[[str], str] | None, *, is_tty: bool) -> str:
    if not is_tty or prompt_func is None:
        return "pull"
    if prompt_func is not input:
        prompt = (
            "HF baseline exists locally.\n"
            "  pull  -> download missing data without deleting local files\n"
            "  clean -> redownload everything after deleting local files\n"
            "Choose [pull/clean]: "
        )
        return _read_choice(prompt_func, prompt, {"pull", "clean"})
    from rich.console import Console
    from rich.prompt import Prompt

    console = Console()
    console.print("[bold yellow]HF baseline exists locally.[/bold yellow]")
    console.print("  [cyan]pull[/cyan]  -> download missing data without deleting local files")
    console.print("  [cyan]clean[/cyan] -> redownload everything after deleting local files")
    return Prompt.ask("Choose", choices=["pull", "clean"], default="pull")


def _prompt_overwrite_file(prompt_func: Callable[[str], str] | None, *, path: Path, is_tty: bool) -> bool:
    if not is_tty or prompt_func is None:
        return False
    prompt = f"File exists: {path}. Overwrite? [y/N]: "
    response = prompt_func(prompt).strip().lower()
    return response in {"y", "yes"}


def _read_choice(prompt_func: Callable[[str], str], prompt: str, choices: Sequence[str]) -> str:
    choices_set = {c.lower() for c in choices}
    while True:
        try:
            response = prompt_func(prompt).strip().lower()
        except EOFError:  # noqa: PERF203
            raise RuntimeError("Aborted HF baseline selection.") from None
        if response in choices_set:
            return response


def _iter_snapshot_files(snapshot_dir: Path) -> Iterable[Path]:
    for path in snapshot_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in {"env_index.json", "dataset_infos.json"}:
            yield path
            continue
        if path.suffix == ".parquet":
            yield path


def _has_complete_hf_baseline(output_dir: Path) -> bool:
    index_path = output_dir / "env_index.json"
    if not index_path.is_file():
        return False
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        return False
    for rel_path in files.keys():
        try:
            candidate = output_dir / str(rel_path)
        except Exception:
            return False
        if not candidate.is_file():
            return False
    return True


def _copy_snapshot(
    snapshot_dir: Path,
    output_dir: Path,
    result: BaselineResult,
    *,
    overwrite: bool,
    prompt_func: Callable[[str], str] | None = None,
    is_tty: bool = False,
) -> None:
    for src in _iter_snapshot_files(snapshot_dir):
        rel_path = src.relative_to(snapshot_dir)
        dest = output_dir / rel_path
        do_overwrite = overwrite
        if dest.exists():
            if not do_overwrite:
                if _prompt_overwrite_file(prompt_func, path=dest, is_tty=is_tty):
                    do_overwrite = True
                else:
                    result.files_skipped.append(dest)
                    continue
            if do_overwrite:
                result.files_overwritten.append(dest)
        else:
            result.files_copied.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _clear_directory(path: Path) -> None:
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)


def clear_output_dir(output_dir: Path) -> None:
    """Remove processed outputs."""
    if output_dir.exists():
        _clear_directory(output_dir)


__all__ = [
    "BaselineResult",
    "clear_output_dir",
    "ensure_output_dir",
    "is_nonempty_dir",
    "prepare_hf_baseline",
]
