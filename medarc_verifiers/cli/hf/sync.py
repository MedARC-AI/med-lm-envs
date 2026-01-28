"""Hugging Face dataset sync helpers for exporter process pipeline."""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from medarc_verifiers.cli.process.writer import EnvWriteSummary

logger = logging.getLogger(__name__)


def _configure_hf_http_timeout(request_timeout_s: float) -> None:
    """Configure huggingface_hub's shared httpx client with a longer request timeout."""
    if request_timeout_s <= 0:
        return
    try:
        import httpx  # type: ignore[import-not-found]

        from huggingface_hub.utils._http import (  # type: ignore[import-not-found]
            hf_request_event_hook,
            set_client_factory,
        )
    except Exception:
        return

    timeout_value = float(request_timeout_s)
    write_timeout = max(60.0, timeout_value)

    def _factory() -> httpx.Client:
        return httpx.Client(
            event_hooks={"request": [hf_request_event_hook]},
            follow_redirects=True,
            timeout=httpx.Timeout(timeout_value, write=write_timeout),
        )

    set_client_factory(_factory)


def _sleep_backoff_seconds(attempt: int) -> float:
    # 1, 2, 4, 8, ... capped
    return min(60.0, 2.0**attempt)


@dataclass(slots=True)
class HFSyncConfig:
    repo_id: str | None
    branch: str | None = None
    private: bool = False
    dry_run: bool = False
    token: str | None = None
    merge_strategy: str = "file"
    request_timeout_s: float = 300.0
    retries: int = 3
    max_files_per_commit: int | None = None

    @classmethod
    def from_cli(
        cls,
        *,
        repo: str | None,
        branch: str | None = None,
        token: str | None = None,
        private: bool | None = None,
        dry_run: bool | None = None,
        request_timeout: float | None = None,
        retries: int | None = None,
        max_files_per_commit: int | None = None,
    ) -> "HFSyncConfig" | None:
        """Build an HFSyncConfig from CLI args while tolerating absence of a repo."""
        if not repo:
            return None
        payload: dict[str, object] = dict(
            repo_id=repo,
            branch=branch,
            token=token,
            private=bool(private) if private is not None else False,
            dry_run=bool(dry_run) if dry_run is not None else False,
        )
        if request_timeout is not None:
            try:
                payload["request_timeout_s"] = max(1.0, float(request_timeout))
            except Exception:
                pass
        if retries is not None:
            try:
                payload["retries"] = max(0, int(retries))
            except Exception:
                pass
        if max_files_per_commit is not None:
            try:
                value = int(max_files_per_commit)
                payload["max_files_per_commit"] = value if value > 0 else None
            except Exception:
                pass
        return cls(
            **payload,  # type: ignore[arg-type]
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
    request_timeout_s: float | None = None,
    retries: int = 3,
    max_files_per_commit: int | None = None,
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

    if request_timeout_s is not None:
        _configure_hf_http_timeout(float(request_timeout_s))

    api = HfApi(token=token)
    if private:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )

    if max_files_per_commit is None or max_files_per_commit <= 0:
        batches = [file_list]
    else:
        batches = [
            file_list[index : index + max_files_per_commit] for index in range(0, len(file_list), max_files_per_commit)
        ]

    output_dir = Path(output_dir)

    for batch_index, batch_files in enumerate(batches, start=1):
        operations = [
            CommitOperationAdd(path_in_repo=rel_path, path_or_fileobj=str(output_dir / rel_path))
            for rel_path in batch_files
        ]
        commit_message = message
        if len(batches) > 1:
            commit_message = f"{message} ({batch_index}/{len(batches)})"

        for attempt in range(max(0, int(retries)) + 1):
            try:
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=commit_message,
                    revision=branch,
                )
                break
            except Exception as exc:  # noqa: BLE001
                try:
                    import httpx  # type: ignore[import-not-found]

                    is_retryable = isinstance(exc, (httpx.TimeoutException, httpx.TransportError))
                except Exception:
                    is_retryable = False
                if not is_retryable or attempt >= int(retries):
                    raise
                delay = _sleep_backoff_seconds(attempt)
                logger.warning(
                    "HF create_commit failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1,
                    int(retries) + 1,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)


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
        request_timeout_s=config.request_timeout_s,
        retries=config.retries,
        max_files_per_commit=config.max_files_per_commit,
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
