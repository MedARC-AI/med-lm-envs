"""Hugging Face dataset sync helpers for exporter process pipeline."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from medarc_verifiers.utils.pathing import resolve_under

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
    # Fixed pause between retry attempts to avoid hammering the Hub.
    return 5.0


def _is_repo_not_found_error(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code == 404:
        return True
    message = str(exc)
    if "Repository Not Found" in message:
        return True
    if "404" in message and "Not Found" in message:
        return True
    return False


def _status_code_from_exc(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "status_code", None)
    try:
        return int(status_code) if status_code is not None else None
    except Exception:
        return None


def _is_transient_hf_error(exc: BaseException) -> bool:
    status_code = _status_code_from_exc(exc)
    if status_code == 429 or (status_code is not None and 500 <= status_code < 600):
        return True
    try:
        import httpx  # type: ignore[import-not-found]

        return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))
    except Exception:
        return False


def _confirm_create_repo(
    *,
    repo_id: str,
    private: bool,
    is_tty: bool,
    assume_yes: bool,
    prompt_func: Callable[[str], str] | None,
) -> bool:
    if assume_yes:
        return True
    if not is_tty:
        return False
    try:
        from rich.console import Console
        from rich.prompt import Confirm

        console = Console()
        console.print(f"[bold yellow]HF dataset repo not found:[/bold yellow] {repo_id}")
        visibility = "private" if private else "public"
        console.print(f"[dim]Will create as a {visibility} dataset repo.[/dim]")
        return bool(Confirm.ask("Create it now?", default=False))
    except Exception:
        if prompt_func is None:
            return False
        response = prompt_func(f"HF dataset repo '{repo_id}' not found. Create it? [y/N]: ").strip().lower()
        return response in {"y", "yes"}


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


def _local_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_tree_entry_path(entry: Any) -> str | None:
    for attr in ("path", "rfilename"):
        value = getattr(entry, attr, None)
        if isinstance(value, str) and value.strip():
            return Path(value).as_posix()
    if isinstance(entry, dict):
        value = entry.get("path") or entry.get("rfilename")
        if isinstance(value, str) and value.strip():
            return Path(value).as_posix()
    return None


def _repo_tree_entry_lfs_sha256(entry: Any) -> str | None:
    lfs = getattr(entry, "lfs", None)
    if lfs is None and isinstance(entry, dict):
        lfs = entry.get("lfs")
    if isinstance(lfs, dict):
        sha256 = lfs.get("sha256")
        return str(sha256) if sha256 else None
    sha256 = getattr(lfs, "sha256", None)
    return str(sha256) if sha256 else None


def _normalize_output_files(output_dir: Path, files: Iterable[str | Path]) -> list[str]:
    normalized: list[str] = []
    for path in files:
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                rel_path = candidate.relative_to(output_dir)
            except ValueError:
                continue
        else:
            # Accept caller inputs like "runs/processed/foo.parquet" when output_dir is also relative.
            output_parts = output_dir.parts
            if output_parts and candidate.parts[: len(output_parts)] == output_parts:
                try:
                    rel_path = candidate.relative_to(output_dir)
                except ValueError:
                    continue
            else:
                rel_path = candidate
        rel_text = rel_path.as_posix()
        if rel_text:
            normalized.append(rel_text)
    return sorted(set(normalized))


def _prepare_upload_file_entries(output_dir: Path, files: Sequence[str | Path]) -> list[tuple[str, Path]]:
    output_dir = output_dir.resolve()
    prepared: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for path in files:
        candidate = Path(path)
        raw_text = candidate.as_posix()
        if not raw_text:
            continue
        if candidate.is_absolute():
            try:
                rel_path = candidate.resolve().relative_to(output_dir).as_posix()
            except ValueError as exc:
                raise ValueError(f"Upload file path must be under output_dir: {candidate}") from exc
        else:
            resolved = resolve_under(output_dir, raw_text)
            if resolved is None:
                raise ValueError(f"Upload file path must be relative to output_dir without traversal: {raw_text!r}")
            try:
                rel_path = resolved.resolve().relative_to(output_dir).as_posix()
            except ValueError as exc:
                raise ValueError(f"Upload file path resolves outside output_dir: {raw_text!r}") from exc
        local_path = (output_dir / rel_path).resolve()
        try:
            local_path.relative_to(output_dir)
        except ValueError as exc:
            raise ValueError(f"Upload file path resolves outside output_dir: {raw_text!r}") from exc
        if rel_path in seen:
            continue
        prepared.append((rel_path, local_path))
        seen.add(rel_path)
    return prepared


def collect_changed_output_files(
    env_summaries: Sequence[EnvWriteSummary],
    *,
    output_dir: Path,
    metadata_paths: Sequence[Path] | None = None,
) -> list[str]:
    changed_paths = {summary.output_path for summary in env_summaries if summary.changed}
    if metadata_paths:
        for path in metadata_paths:
            candidate = Path(path)
            if not candidate.is_absolute():
                output_parts = output_dir.parts
                if output_parts and candidate.parts[: len(output_parts)] != output_parts:
                    candidate = output_dir / candidate
            changed_paths.add(candidate)
    return _normalize_output_files(output_dir, changed_paths)


def _collect_changed_output_files(
    env_summaries: Sequence[EnvWriteSummary],
    *,
    output_dir: Path,
    metadata_paths: Sequence[Path] | None = None,
) -> list[str]:
    return collect_changed_output_files(env_summaries, output_dir=output_dir, metadata_paths=metadata_paths)


def compute_pending_parquet_uploads(
    output_dir: Path,
    repo_id: str,
    branch: str | None,
    token: str | None,
) -> set[str]:
    """Return local parquet paths that are missing remotely or differ from remote lfs.sha256."""
    output_dir = Path(output_dir)
    local_parquets = sorted(path for path in output_dir.rglob("*.parquet") if path.is_file())
    if not local_parquets:
        return set()

    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise ImportError("huggingface_hub is required for HF upload recovery.") from exc

    api = HfApi(token=token)
    list_kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": branch,
        "recursive": True,
        "expand": True,
    }
    try:
        try:
            tree_entries = list(api.list_repo_tree(**list_kwargs))
        except TypeError as exc:
            if "expand" not in str(exc):
                raise
            list_kwargs.pop("expand", None)
            tree_entries = list(api.list_repo_tree(**list_kwargs))
    except Exception as exc:  # noqa: BLE001
        if _is_repo_not_found_error(exc):
            tree_entries = []
        else:
            raise

    remote_parquets: dict[str, str | None] = {}
    for entry in tree_entries:
        rel_path = _repo_tree_entry_path(entry)
        if not rel_path or not rel_path.endswith(".parquet"):
            continue
        remote_parquets[rel_path] = _repo_tree_entry_lfs_sha256(entry)

    pending: set[str] = set()
    for parquet_path in local_parquets:
        rel_path = parquet_path.relative_to(output_dir).as_posix()
        if rel_path not in remote_parquets:
            pending.add(rel_path)
            continue
        remote_sha256 = remote_parquets[rel_path]
        if remote_sha256 is None or remote_sha256 != _local_sha256(parquet_path):
            pending.add(rel_path)
    return pending


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
    path_in_repo_prefix: str | None = None,
    is_tty: bool = False,
    assume_yes: bool = False,
    prompt_func: Callable[[str], str] | None = None,
) -> bool:
    """Upload explicit file paths from output_dir to a HF dataset repo.

    Returns False only when upload is skipped because repo creation was declined.
    """
    if not repo_id:
        logger.debug("HF sync skipped: no repo_id provided.")
        return True
    output_dir = Path(output_dir)
    prepared_files = _prepare_upload_file_entries(output_dir, files)
    file_list = [rel_path for rel_path, _ in prepared_files]
    if not file_list:
        logger.debug("HF sync skipped: no files provided.")
        return True
    if dry_run:
        logger.debug("HF sync dry-run; skipping push.")
        return True

    try:
        from huggingface_hub import CommitOperationAdd, HfApi  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise ImportError("huggingface_hub is required for HF uploads.") from exc

    if request_timeout_s is not None:
        _configure_hf_http_timeout(float(request_timeout_s))

    api = HfApi(token=token)
    repo_prefix = _normalize_repo_path_prefix(path_in_repo_prefix)

    file_map = dict(prepared_files)

    if max_files_per_commit is None or max_files_per_commit <= 0:
        batches = [file_list]
    else:
        batches = [
            file_list[index : index + max_files_per_commit] for index in range(0, len(file_list), max_files_per_commit)
        ]

    for batch_index, batch_files in enumerate(batches, start=1):
        operations = [
            CommitOperationAdd(
                path_in_repo=_join_repo_path(repo_prefix, rel_path),
                path_or_fileobj=str(file_map[rel_path]),
            )
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
                if _is_repo_not_found_error(exc):
                    should_create = _confirm_create_repo(
                        repo_id=repo_id,
                        private=private,
                        is_tty=is_tty,
                        assume_yes=assume_yes,
                        prompt_func=prompt_func,
                    )
                    if not should_create:
                        logger.warning(
                            "HF dataset repo '%s' not found; skipping upload because repo creation was declined.",
                            repo_id,
                        )
                        return False
                    api.create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        private=private,
                        exist_ok=True,
                    )
                    # Retry the commit immediately after repo creation.
                    continue
                if not _is_transient_hf_error(exc) or attempt >= int(retries):
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
    return True


def _normalize_repo_path_prefix(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().replace("\\", "/").strip("/")
    if not raw:
        return None
    candidate = resolve_under(Path("."), raw)
    if candidate is None:
        raise ValueError(f"Invalid path_in_repo_prefix: {value!r}")
    normalized = candidate.as_posix().lstrip("./")
    return normalized or None


def _join_repo_path(prefix: str | None, rel_path: str) -> str:
    rel = rel_path.strip().replace("\\", "/").lstrip("/")
    if not prefix:
        return rel
    return f"{prefix}/{rel}" if rel else prefix


def sync_to_hub(
    env_summaries: Sequence[EnvWriteSummary],
    config: HFSyncConfig,
    *,
    output_dir: Path,
    metadata_paths: Sequence[Path] | None = None,
    files: Sequence[str | Path] | None = None,
    is_tty: bool = False,
    assume_yes: bool = False,
    prompt_func: Callable[[str], str] | None = None,
) -> HFSyncSummary | None:
    """Upload changed artifacts to a HF dataset repo."""
    if not config.repo_id:
        logger.debug("HF sync skipped: no repo_id provided.")
        return None
    if config.dry_run:
        logger.debug("HF sync dry-run; skipping summary generation and upload.")
        return None

    output_dir = Path(output_dir)
    changed = [summary for summary in env_summaries if summary.changed]
    if files is None:
        if not env_summaries:
            logger.debug("HF sync skipped: no environment summaries available.")
            return None
        if all(summary.dry_run for summary in env_summaries):
            logger.debug("HF sync skipped: only dry-run summaries available.")
            return None
        files = collect_changed_output_files(env_summaries, output_dir=output_dir, metadata_paths=metadata_paths)
    else:
        files = _normalize_output_files(output_dir, files)

    if not files:
        logger.debug("HF sync skipped: no files selected for upload.")
        return None

    summary = HFSyncSummary(
        repo_id=config.repo_id,
        strategy="file",
        total_rows=sum(summary.row_count for summary in changed),
        total_files=len(files),
        files=files,
    )

    message = f"Update {summary.total_files} file(s) from medarc-eval process"
    uploaded = sync_files_to_hub(
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
        is_tty=is_tty,
        assume_yes=assume_yes,
        prompt_func=prompt_func,
    )
    if not uploaded:
        return None
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
    "collect_changed_output_files",
    "compute_pending_parquet_uploads",
    "sync_files_to_hub",
    "sync_to_hub",
]
