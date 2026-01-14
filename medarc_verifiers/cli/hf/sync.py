"""Hugging Face dataset sync helpers for exporter process pipeline."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from medarc_verifiers.cli.process.writer import EnvWriteSummary

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class HFSyncConfig:
    repo_id: str | None
    branch: str | None = None
    private: bool = False
    dry_run: bool = False
    token: str | None = None
    merge_strategy: str = "file"

    @classmethod
    def from_cli(
        cls,
        *,
        repo: str | None,
        branch: str | None = None,
        token: str | None = None,
        private: bool | None = None,
        dry_run: bool | None = None,
    ) -> "HFSyncConfig" | None:
        """Build an HFSyncConfig from CLI args while tolerating absence of a repo."""
        if not repo:
            return None
        return cls(
            repo_id=repo,
            branch=branch,
            token=token,
            private=bool(private) if private is not None else False,
            dry_run=bool(dry_run) if dry_run is not None else False,
        )


@dataclass(slots=True)
class HFSyncSummary:
    repo_id: str
    strategy: str
    total_rows: int
    total_files: int
    files: Sequence[str]


def sync_files_to_hub(
    *,
    repo_id: str,
    output_dir: Path,
    files: Sequence[str | Path],
    token: str | None,
    private: bool,
    message: str,
    branch: str | None = None,
    dry_run: bool = False,
) -> None:
    """Upload explicit file paths from output_dir to a HF dataset repo."""
    if not repo_id:
        logger.debug("HF sync skipped: no repo_id provided.")
        return
    file_list = []
    for path in files:
        rel_path = Path(path).as_posix() if not isinstance(path, str) else Path(path).as_posix()
        if rel_path:
            file_list.append(rel_path)
    if not file_list:
        logger.debug("HF sync skipped: no files provided.")
        return
    if dry_run:
        logger.debug("HF sync dry-run; skipping push.")
        return

    try:
        from huggingface_hub import CommitOperationAdd, HfApi  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise ImportError("huggingface_hub is required for HF uploads.") from exc

    api = HfApi(token=token)
    if private:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
    operations = []
    output_dir = Path(output_dir)
    for rel_path in file_list:
        operations.append(
            CommitOperationAdd(
                path_in_repo=rel_path,
                path_or_fileobj=str(output_dir / rel_path),
            )
        )
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=message,
        revision=branch,
    )


def sync_to_hub(
    env_summaries: Sequence[EnvWriteSummary],
    config: HFSyncConfig,
    *,
    output_dir: Path,
    metadata_paths: Sequence[Path] | None = None,
) -> HFSyncSummary | None:
    """Upload changed artifacts to a HF dataset repo."""
    if not config.repo_id:
        logger.debug("HF sync skipped: no repo_id provided.")
        return None
    if not env_summaries:
        logger.debug("HF sync skipped: no environment summaries available.")
        return None
    if all(summary.dry_run for summary in env_summaries):
        logger.debug("HF sync skipped: only dry-run summaries available.")
        return None

    changed = [summary for summary in env_summaries if summary.changed]
    if not changed:
        logger.debug("HF sync skipped: no changed outputs.")
        return None

    output_dir = Path(output_dir)
    changed_paths = {summary.output_path for summary in changed}
    if metadata_paths:
        for path in metadata_paths:
            candidate = Path(path)
            if not candidate.is_absolute():
                output_parts = output_dir.parts
                if output_parts and candidate.parts[: len(output_parts)] != output_parts:
                    candidate = output_dir / candidate
            changed_paths.add(candidate)

    files = []
    for path in changed_paths:
        try:
            rel_path = path.relative_to(output_dir)
        except ValueError:
            continue
        files.append(rel_path.as_posix())
    files = sorted(set(files))
    summary = HFSyncSummary(
        repo_id=config.repo_id,
        strategy="file",
        total_rows=sum(summary.row_count for summary in changed),
        total_files=len(files),
        files=files,
    )

    message = f"Update {summary.total_files} file(s) from medarc-eval process"
    sync_files_to_hub(
        repo_id=config.repo_id,
        output_dir=output_dir,
        files=files,
        token=config.token,
        private=config.private,
        message=message,
        branch=config.branch,
        dry_run=config.dry_run,
    )
    return summary


def download_hf_repo(
    *,
    repo_id: str,
    branch: str | None,
    token: str | None,
    allow_patterns: str | Sequence[str] = "*.parquet",
    local_dir: Path | None = None,
    local_only: bool = False,
) -> Path:
    """Download a HF dataset repo snapshot to a temp dir and return the path."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise ImportError("huggingface_hub is required for HF-backed downloads.") from exc

    if local_only and local_dir is not None:
        temp_root = Path(local_dir)
        if temp_root.is_dir() and any(temp_root.iterdir()):
            return temp_root
        raise FileNotFoundError(f"Local HF repo not found at {temp_root}")

    temp_root = Path(tempfile.mkdtemp(prefix="hf-sync-")) if local_dir is None else Path(local_dir)

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=branch,
            token=token,
            allow_patterns=allow_patterns,
            local_dir=temp_root,
        )
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        status = getattr(exc, "response", None)
        status_code = getattr(status, "status_code", None)
        if status_code == 404 or "Repository Not Found" in message:
            logger.warning("HF repo %s not found; continuing without baseline.", repo_id)
            return temp_root
        raise
    return temp_root


__all__ = [
    "HFSyncSummary",
    "HFSyncConfig",
    "sync_files_to_hub",
    "sync_to_hub",
]
