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

import httpx

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
    if _is_latest_image(source):
        return Path(image_dir).expanduser() / "latest.sqsh"
    normalized = _normalize_image_ref(source)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return Path(image_dir).expanduser() / f"{normalized}--{digest}.sqsh"


def is_absolute_sqsh_image(source: str) -> bool:
    path = Path(str(source).strip()).expanduser()
    return path.is_absolute() and path.suffix == ".sqsh"


def materialize_image(*, source: str, final_path: Path, latest_link: bool = True) -> dict[str, Any]:
    source = str(source).strip()
    latest_source = _is_latest_image(source)
    if not latest_source and final_path.exists() and os.access(final_path, os.R_OK):
        return {"source": source, "image_path": str(final_path), "skipped": True}
    if is_absolute_sqsh_image(source):
        raise RuntimeError(f"Configured Pyxis image does not exist or is unreadable: {source}")
    if shutil.which("enroot") is None:
        raise RuntimeError("enroot not found; construct image materialization requires the Enroot CLI on CPU nodes.")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = final_path.parent / ".locks" / f"{final_path.stem}.lock"
    result: dict[str, Any] = {}
    _with_lock_dir(
        lock_dir,
        lambda: result.update(
            _import_image_locked(source=source, final_path=final_path, latest_link=latest_link)
        ),
    )
    return {"source": source, **result}


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


def _is_latest_image(source: str) -> bool:
    tail = source.rsplit("/", maxsplit=1)[-1]
    if "@" in tail:
        return False
    return ":" not in tail or tail.rsplit(":", maxsplit=1)[-1] == "latest"


def resolve_image_digest(source: str) -> str:
    registry, repository, tag = _parse_image_tag(source)
    manifest_url = f"https://{registry}/v2/{repository}/manifests/{tag}"
    headers = {
        "Accept": ", ".join(
            (
                "application/vnd.oci.image.index.v1+json",
                "application/vnd.oci.image.manifest.v1+json",
                "application/vnd.docker.distribution.manifest.list.v2+json",
                "application/vnd.docker.distribution.manifest.v2+json",
            )
        )
    }
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = client.get(manifest_url, headers=headers)
        if response.status_code == 401:
            auth_header = response.headers.get("www-authenticate")
            if not auth_header:
                response.raise_for_status()
            token = _registry_bearer_token(client, auth_header)
            response = client.get(manifest_url, headers={**headers, "Authorization": f"Bearer {token}"})
        response.raise_for_status()
    digest = response.headers.get("docker-content-digest")
    if not digest:
        raise RuntimeError(f"Registry did not return Docker-Content-Digest for image tag: {source}")
    import_registry = "docker.io" if registry == "registry-1.docker.io" else registry
    return f"{import_registry}/{repository}@{digest}"


def _parse_image_tag(source: str) -> tuple[str, str, str]:
    source = source.removeprefix("docker://").strip()
    if "@" in source:
        raise ValueError(f"Image is already digest-pinned: {source}")
    name, tag = source.rsplit(":", maxsplit=1) if ":" in source.rsplit("/", maxsplit=1)[-1] else (source, "latest")
    parts = name.split("/")
    if len(parts) == 1:
        return "registry-1.docker.io", f"library/{parts[0]}", tag
    first = parts[0]
    if "." in first or ":" in first or first == "localhost":
        registry = "registry-1.docker.io" if first == "docker.io" else first
        repository = "/".join(parts[1:])
    else:
        registry = "registry-1.docker.io"
        repository = "/".join(parts)
    if not repository:
        raise ValueError(f"Invalid image reference: {source}")
    return registry, repository, tag


def _registry_bearer_token(client: httpx.Client, auth_header: str) -> str:
    scheme, _, params = auth_header.partition(" ")
    if scheme.lower() != "bearer":
        raise RuntimeError(f"Unsupported registry authentication challenge: {auth_header}")
    parsed = _parse_auth_params(params)
    realm = parsed.get("realm")
    if not realm:
        raise RuntimeError(f"Registry authentication challenge is missing realm: {auth_header}")
    response = client.get(
        realm,
        params={key: value for key, value in parsed.items() if key != "realm"},
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("token") or payload.get("access_token")
    if not token:
        raise RuntimeError("Registry token response did not include a token.")
    return str(token)


def _parse_auth_params(value: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for match in re.finditer(r'(\w+)="([^"]*)"', value):
        params[match.group(1)] = match.group(2)
    return params


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


def _import_image_locked(*, source: str, final_path: Path, latest_link: bool) -> dict[str, Any]:
    source = str(source).strip()
    resolved_source = resolve_image_digest(source) if _is_latest_image(source) else source
    import_path = final_path
    if resolved_source != source:
        import_path = materialized_image_path(resolved_source, final_path.parent)
    if import_path.exists() and os.access(import_path, os.R_OK):
        if resolved_source != source:
            _update_symlink(final_path, target=import_path)
            if latest_link and final_path.name != "latest":
                _update_symlink(final_path.parent / "latest", target=import_path)
        return {
            "resolved_source": resolved_source,
            "image_path": str(final_path),
            "resolved_image_path": str(import_path),
            "skipped": True,
        }
    tmp_prefix = import_path.parent / f".{import_path.stem}.{uuid.uuid4().hex}"
    command = ["enroot", "import", "--output", str(tmp_prefix), f"docker://{resolved_source}"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "enroot import failed")
    tmp_sqsh = tmp_prefix.with_suffix(".sqsh")
    if not tmp_sqsh.exists():
        raise RuntimeError(f"enroot import did not produce expected image: {tmp_sqsh}")
    tmp_sqsh.replace(import_path)
    if resolved_source != source:
        _update_symlink(final_path, target=import_path)
        if latest_link and final_path.name != "latest":
            _update_symlink(final_path.parent / "latest", target=import_path)
    elif latest_link:
        _update_symlink(final_path.parent / "latest", target=import_path)
    return {
        "resolved_source": resolved_source,
        "image_path": str(final_path),
        "resolved_image_path": str(import_path),
        "skipped": False,
    }


def _update_symlink(path: Path, *, target: Path) -> None:
    tmp_link = path.parent / f".{path.name}_tmp_{uuid.uuid4().hex}"
    tmp_link.symlink_to(target.name)
    tmp_link.replace(path)


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
    "resolve_image_digest",
    "resolve_construct_cache",
    "run_construct",
    "run_teardown",
]
