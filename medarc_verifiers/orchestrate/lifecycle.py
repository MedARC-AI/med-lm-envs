"""Construct and teardown lifecycle helpers for Slurm/Pyxis orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from medarc_verifiers.orchestrate.bundle import ExecutionAllocation, load_execution_allocation, load_task_spec
from medarc_verifiers.orchestrate.config import ConstructConfig
from medarc_verifiers.orchestrate.env import apply_env, load_runtime_env

HF_CONTAINER_HOME = "/root/.cache/huggingface"
HF_CONTAINER_HUB = "/root/.cache/huggingface/hub"


@dataclass(frozen=True)
class ConstructCache:
    hf_home: str | None = None
    hub_cache: str | None = None
    image_dir: str | None = None
    latest_link: bool = True
    isolated: bool = False
    container_hf_home: str | None = None
    container_hub_cache: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ConstructCache":
        payload = dict(payload or {})
        return cls(
            hf_home=str(payload["hf_home"]) if payload.get("hf_home") is not None else None,
            hub_cache=str(payload["hub_cache"]) if payload.get("hub_cache") is not None else None,
            image_dir=str(payload["image_dir"]) if payload.get("image_dir") is not None else None,
            latest_link=bool(payload.get("latest_link", True)),
            isolated=bool(payload.get("isolated", False)),
            container_hf_home=(
                str(payload["container_hf_home"]) if payload.get("container_hf_home") is not None else None
            ),
            container_hub_cache=(
                str(payload["container_hub_cache"]) if payload.get("container_hub_cache") is not None else None
            ),
        )


def resolve_construct_cache(
    *,
    config: ConstructConfig,
    volume_mounts: list[str],
    env: Mapping[str, str] | None = None,
    require_image_dir: bool | None = None,
) -> ConstructCache:
    env = env or os.environ
    hf_home = str(config.cache.hf_home) if config.cache.hf_home is not None else None
    container_hf_home: str | None = None
    if hf_home is None:
        mount = _find_hf_mount(volume_mounts)
        if mount is not None:
            hf_home, container_hf_home = mount
    if hf_home is None and env.get("HF_HOME"):
        hf_home = str(env["HF_HOME"])
    if config.prefetch_enabled and hf_home is None:
        raise ValueError("construct prefetch_model_weights requires [construct.cache].hf_home or a volume mounted at /root/.cache/huggingface.")
    hub_cache = str(config.cache.hub_cache) if config.cache.hub_cache is not None else None
    if hub_cache is None and hf_home:
        hub_cache = str(Path(hf_home) / "hub")
    if config.cache.hub_cache is None and env.get("HUGGINGFACE_HUB_CACHE") and hf_home:
        env_hub = str(env["HUGGINGFACE_HUB_CACHE"])
        if _is_relative_to(Path(env_hub), Path(hf_home)):
            hub_cache = env_hub
    image_dir = str(config.cache.image_dir) if config.cache.image_dir is not None else None
    if require_image_dir is None:
        require_image_dir = config.image_materialization_enabled
    if require_image_dir and image_dir is None:
        raise ValueError("construct materialize_images requires [construct.cache].image_dir.")
    if container_hf_home is None and hf_home is not None:
        for mount in volume_mounts:
            parts = str(mount).split(":")
            if len(parts) >= 2 and Path(parts[0]).expanduser() == Path(hf_home).expanduser():
                container_hf_home = parts[1]
                break
    container_hf_home = container_hf_home or (HF_CONTAINER_HOME if hf_home else None)
    container_hub_cache = HF_CONTAINER_HUB if container_hf_home == HF_CONTAINER_HOME else None
    return ConstructCache(
        hf_home=hf_home,
        hub_cache=hub_cache,
        image_dir=image_dir,
        latest_link=config.cache.latest_link,
        isolated=_looks_isolated_cache(hf_home),
        container_hf_home=container_hf_home,
        container_hub_cache=container_hub_cache,
    )


def materialized_image_path(source: str, image_dir: str | Path) -> Path:
    source = str(source).strip()
    if is_absolute_sqsh_image(source):
        return Path(source)
    _reject_latest_image(source)
    normalized = _normalize_image_ref(source)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return Path(image_dir).expanduser() / f"{normalized}--{digest}.sqsh"


def is_absolute_sqsh_image(source: str) -> bool:
    path = Path(str(source).strip()).expanduser()
    return path.is_absolute() and path.suffix == ".sqsh"


def materialize_image(*, source: str, final_path: Path, latest_link: bool = True) -> dict[str, Any]:
    if final_path.exists() and os.access(final_path, os.R_OK):
        return {"source": source, "image_path": str(final_path), "skipped": True}
    if str(source).startswith("/") and str(source).endswith(".sqsh"):
        raise RuntimeError(f"Configured Pyxis image does not exist or is unreadable: {source}")
    if shutil.which("enroot") is None:
        raise RuntimeError("enroot not found; construct image materialization requires the Enroot CLI on CPU nodes.")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = final_path.parent / ".locks" / f"{final_path.stem}.lock"
    _with_lock_dir(lock_dir, lambda: _import_image_locked(source=source, final_path=final_path, latest_link=latest_link))
    return {"source": source, "image_path": str(final_path), "skipped": False}


def prefetch_model(*, model_id: str, hub_cache: str) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(repo_id=model_id, cache_dir=hub_cache, local_files_only=False, resume_download=True)
    commit = Path(snapshot_path).name if Path(snapshot_path).parent.name == "snapshots" else None
    return {"repo_id": model_id, "hub_cache": hub_cache, "snapshot_path": str(snapshot_path), "commit_hash": commit}


def run_construct(
    *,
    task_path: Path,
    allocation_path: Path,
    env_file: Path | None,
    prefetch_model_flag: bool,
    materialize_image_flag: bool,
) -> int:
    spec = load_task_spec(task_path)
    allocation = load_execution_allocation(allocation_path) or ExecutionAllocation(task_id=spec.task_id)
    env = load_runtime_env(spec, allocation=allocation, env_file=env_file)
    apply_env(env)
    cache = ConstructCache.from_dict(spec.construct_cache)
    if cache.hf_home:
        os.environ["HF_HOME"] = cache.hf_home
    if cache.hub_cache:
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache.hub_cache
    result: dict[str, Any] = {"task_id": spec.task_id, "state": "completed", "cache": cache.to_dict()}
    try:
        if prefetch_model_flag:
            if not cache.hub_cache:
                raise RuntimeError("construct prefetch requested but no Hugging Face hub cache is configured.")
            result["model"] = prefetch_model(model_id=spec.model_id, hub_cache=cache.hub_cache)
        if materialize_image_flag:
            if not spec.container_image_source:
                raise RuntimeError("construct image materialization requested but container_image_source is missing.")
            image_path = Path(spec.container_image)
            if cache.image_dir and not _is_relative_to(image_path, Path(cache.image_dir)):
                raise RuntimeError(f"Materialized image path {image_path} is not under image cache {cache.image_dir}.")
            result["image"] = materialize_image(
                source=spec.container_image_source,
                final_path=image_path,
                latest_link=cache.latest_link,
            )
    except Exception as exc:  # noqa: BLE001
        result.update({"state": "failed", "error": str(exc)})
        _write_json(Path(spec.output_paths.construct_result_path), result)
        return 1
    _write_json(Path(spec.output_paths.construct_result_path), result)
    return 0


def run_teardown(*, task_path: Path, allocation_path: Path, env_file: Path | None) -> int:
    spec = load_task_spec(task_path)
    allocation = load_execution_allocation(allocation_path) or ExecutionAllocation(task_id=spec.task_id)
    env = load_runtime_env(spec, allocation=allocation, env_file=env_file)
    apply_env(env)
    result_path = Path(spec.output_paths.teardown_result_path)
    construct_result_path = Path(spec.output_paths.construct_result_path)
    payload: dict[str, Any] = {"task_id": spec.task_id, "state": "completed", "removed": []}
    try:
        construct_result = json.loads(construct_result_path.read_text(encoding="utf-8")) if construct_result_path.exists() else {}
        teardown = dict(spec.teardown or {})
        cache = ConstructCache.from_dict(spec.construct_cache)
        if teardown.get("remove_model_weights"):
            if not cache.isolated:
                raise RuntimeError("teardown remove_model_weights is only supported for isolated per-run cache roots.")
            model_info = dict(construct_result.get("model") or {})
            repo_id = str(model_info.get("repo_id") or spec.model_id)
            repo_dir = Path(str(cache.hub_cache)) / ("models--" + repo_id.replace("/", "--"))
            _safe_delete(repo_dir, root=Path(str(cache.hub_cache)))
            payload["removed"].append(str(repo_dir))
        if teardown.get("remove_images"):
            image_path = Path(spec.container_image)
            if cache.image_dir is None or not _is_relative_to(image_path, Path(cache.image_dir)):
                raise RuntimeError("Refusing to remove image outside configured image cache.")
            _safe_delete(image_path, root=Path(cache.image_dir))
            payload["removed"].append(str(image_path))
    except Exception as exc:  # noqa: BLE001
        payload.update({"state": "failed", "error": str(exc)})
        _write_json(result_path, payload)
        return 1
    _write_json(result_path, payload)
    return 0


def build_construct_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medarc-orchestrate construct")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--prefetch-model", action="store_true")
    parser.add_argument("--materialize-image", action="store_true")
    return parser


def build_teardown_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medarc-orchestrate teardown")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    return parser


def _find_hf_mount(volumes: list[str]) -> tuple[str, str] | None:
    for mount in volumes:
        parts = str(mount).split(":")
        if len(parts) >= 2 and parts[1] == HF_CONTAINER_HOME:
            return parts[0], parts[1]
    return None


def _reject_latest_image(source: str) -> None:
    tail = source.rsplit("/", maxsplit=1)[-1]
    if "@" in tail:
        return
    if ":" not in tail or tail.rsplit(":", maxsplit=1)[-1] == "latest":
        raise ValueError(f"Construct image materialization requires a non-latest image tag or digest: {source}")


def _normalize_image_ref(source: str) -> str:
    value = source
    if "/" in value and "." not in value.split("/", maxsplit=1)[0] and ":" not in value.split("/", maxsplit=1)[0]:
        value = "docker.io/" + value
    return re.sub(r"[^A-Za-z0-9_.-]+", lambda m: "__" if "/" in m.group(0) else "--", value.replace(":", "--"))


def _with_lock_dir(lock_dir: Path, func) -> None:
    while True:
        try:
            lock_dir.mkdir(parents=True)
            break
        except FileExistsError:
            time.sleep(1)
    try:
        func()
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


def _import_image_locked(*, source: str, final_path: Path, latest_link: bool) -> None:
    if final_path.exists() and os.access(final_path, os.R_OK):
        return
    tmp_prefix = final_path.parent / f".{final_path.stem}.{uuid.uuid4().hex}"
    command = ["enroot", "import", "--output", str(tmp_prefix), f"docker://{source}"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "enroot import failed")
    tmp_sqsh = tmp_prefix.with_suffix(".sqsh")
    if not tmp_sqsh.exists():
        raise RuntimeError(f"enroot import did not produce expected image: {tmp_sqsh}")
    tmp_sqsh.replace(final_path)
    if latest_link:
        tmp_link = final_path.parent / f".latest_tmp_{uuid.uuid4().hex}"
        tmp_link.symlink_to(final_path.name)
        tmp_link.replace(final_path.parent / "latest")


def _safe_delete(path: Path, *, root: Path) -> None:
    if not _is_relative_to(path, root):
        raise RuntimeError(f"Refusing to delete {path}; it is outside {root}.")
    if str(root.resolve()) in {"/", str(Path.home().resolve())}:
        raise RuntimeError(f"Refusing unsafe cache root deletion scope: {root}")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _looks_isolated_cache(hf_home: str | None) -> bool:
    if not hf_home:
        return False
    text = str(hf_home)
    return "/tasks/" in text or "/orchestrate/" in text or "/runs/" in text


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.expanduser().resolve())
        return True
    except ValueError:
        return False


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "ConstructCache",
    "build_construct_parser",
    "build_teardown_parser",
    "is_absolute_sqsh_image",
    "materialize_image",
    "materialized_image_path",
    "prefetch_model",
    "resolve_construct_cache",
    "run_construct",
    "run_teardown",
]
