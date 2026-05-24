"""Construct and teardown lifecycle helpers for Slurm/Pyxis orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import httpx

from medarc_verifiers.orchestrate.config import ConstructConfig
from medarc_verifiers.orchestrate.env import apply_env, load_explicit_runtime_env

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


def configure_enroot_paths(image_dir: Path) -> dict[str, str]:
    uid = os.getuid()
    root = image_dir.expanduser() / ".enroot"
    paths = {
        "ENROOT_CACHE_PATH": root / f"cache-{uid}",
        "ENROOT_DATA_PATH": root / f"data-{uid}",
        "ENROOT_RUNTIME_PATH": root / f"run-{uid}-{os.environ.get('SLURM_JOB_ID', 'local')}",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    return {key: str(path) for key, path in paths.items()}


def run_prepare(
    *,
    model: str | None,
    result_path: Path | None,
    env_file: Path | None,
    hf_home: Path | None = None,
    hub_cache: Path | None = None,
    image: str | None = None,
    image_dir: Path | None = None,
    image_output: Path | None = None,
    latest_link: bool = True,
    prefetch_model_flag: bool | None = None,
    materialize_image_flag: bool | None = None,
) -> int:
    if model is None and image is None:
        raise ValueError("prepare requires at least one of --model or --image.")
    prefetch_model_flag = model is not None if prefetch_model_flag is None else bool(prefetch_model_flag)
    materialize_image_flag = image is not None if materialize_image_flag is None else bool(materialize_image_flag)
    env = load_explicit_runtime_env(env_file=env_file)
    apply_env(env)
    resolved_hf_home = str(hf_home) if hf_home is not None else None
    resolved_hub_cache = (
        str(hub_cache or (hf_home / "hub" if hf_home is not None else None)) if (hub_cache or hf_home) else None
    )
    cache = ConstructCache(
        hf_home=resolved_hf_home,
        hub_cache=resolved_hub_cache,
        image_dir=str(image_dir) if image_dir is not None else None,
        latest_link=latest_link,
        isolated=_looks_isolated_cache(resolved_hf_home),
    )
    if prefetch_model_flag and not model:
        raise ValueError("prepare model prefetch requires --model.")
    if materialize_image_flag and not image:
        raise ValueError("prepare image materialization requires --image.")
    if prefetch_model_flag and not cache.hub_cache:
        raise ValueError("prepare --model requires --hub-cache or --hf-home.")
    if image and not is_absolute_sqsh_image(image) and image_output is None and image_dir is None:
        raise ValueError("prepare --image requires --image-dir or --image-output unless the image is an absolute .sqsh.")
    if cache.hf_home:
        os.environ["HF_HOME"] = cache.hf_home
    if cache.hub_cache:
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache.hub_cache
    result: dict[str, Any] = {"state": "completed", "cache": cache.to_dict()}
    try:
        if prefetch_model_flag:
            result["model"] = prefetch_model(model_id=str(model), hub_cache=str(cache.hub_cache))
        if materialize_image_flag:
            assert image is not None
            if is_absolute_sqsh_image(image):
                image_path = Path(image).expanduser()
                if not image_path.is_file() or not os.access(image_path, os.R_OK):
                    raise RuntimeError(f"Configured Pyxis image does not exist or is unreadable: {image}")
                result["image"] = {"source": image, "image_path": str(image_path), "skipped": True}
            elif image_output is not None:
                image_path = image_output.expanduser()
                result["image"] = materialize_image(source=image, final_path=image_path, latest_link=latest_link)
            else:
                assert image_dir is not None
                image_path = materialized_image_path(image, image_dir)
                result["enroot"] = configure_enroot_paths(image_dir)
                if not _is_relative_to(image_path, image_dir):
                    raise RuntimeError(f"Materialized image path {image_path} is not under image cache {image_dir}.")
                result["image"] = materialize_image(source=image, final_path=image_path, latest_link=latest_link)
    except Exception as exc:  # noqa: BLE001
        result.update({"state": "failed", "error": str(exc)})
        _emit_prepare_result(result_path, result)
        return 1
    _emit_prepare_result(result_path, result)
    return 0


def run_teardown(
    *,
    result_path: Path,
    model: str,
    env_file: Path | None,
    hub_cache: Path | None = None,
    remove_model_weights: bool = False,
    remove_image: Path | None = None,
    image_root: Path | None = None,
    prepare_result: Path | None = None,
) -> int:
    env = load_explicit_runtime_env(env_file=env_file)
    apply_env(env)
    payload: dict[str, Any] = {"model": model, "state": "completed", "removed": []}
    try:
        prepare_payload = json.loads(prepare_result.read_text(encoding="utf-8")) if prepare_result and prepare_result.exists() else {}
        cache = ConstructCache.from_dict(prepare_payload.get("cache") if isinstance(prepare_payload, Mapping) else {})
        effective_hub_cache = str(hub_cache or cache.hub_cache or "")
        if remove_model_weights:
            if not _looks_isolated_cache(str(effective_hub_cache)):
                raise RuntimeError("teardown remove_model_weights is only supported for isolated per-run cache roots.")
            if not effective_hub_cache:
                raise RuntimeError("teardown remove_model_weights requires --hub-cache or prepare cache metadata.")
            repo_dir = Path(effective_hub_cache) / ("models--" + model.replace("/", "--"))
            _safe_delete(repo_dir, root=Path(effective_hub_cache))
            payload["removed"].append(str(repo_dir))
        if remove_image is not None:
            if image_root is None or not _is_relative_to(remove_image, image_root):
                raise RuntimeError("Refusing to remove image outside configured image cache.")
            _safe_delete(remove_image, root=image_root)
            payload["removed"].append(str(remove_image))
    except Exception as exc:  # noqa: BLE001
        payload.update({"state": "failed", "error": str(exc)})
        _write_json(result_path, payload)
        return 1
    _write_json(result_path, payload)
    return 0


def build_prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medarc-orchestrate prepare")
    parser.add_argument("--model")
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--hub-cache", type=Path)
    parser.add_argument("--image")
    image_output = parser.add_mutually_exclusive_group()
    image_output.add_argument("--image-dir", type=Path)
    image_output.add_argument("--image-output", type=Path)
    latest = parser.add_mutually_exclusive_group()
    latest.add_argument("--latest-link", action="store_true", dest="latest_link", default=True)
    latest.add_argument("--no-latest-link", action="store_false", dest="latest_link")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--prefetch-model", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--materialize-image", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path)
    return parser


def build_teardown_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medarc-orchestrate teardown")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hub-cache", type=Path)
    parser.add_argument("--remove-model-weights", action="store_true")
    parser.add_argument("--remove-image", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--prepare-result", type=Path)
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
            _write_json(
                lock_dir / "owner.json",
                {
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "created_at": time.time(),
                },
            )
            break
        except FileExistsError:
            if _lock_dir_is_stale(lock_dir):
                try:
                    shutil.rmtree(lock_dir)
                    continue
                except OSError:
                    pass
            time.sleep(1)
    try:
        func()
    finally:
        try:
            shutil.rmtree(lock_dir)
        except OSError:
            pass


def _lock_dir_is_stale(lock_dir: Path) -> bool:
    try:
        owner = json.loads((lock_dir / "owner.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return time.time() - lock_dir.stat().st_mtime > 3600
    slurm_job_id = str(owner.get("slurm_job_id") or "")
    if slurm_job_id and shutil.which("squeue") is not None:
        completed = subprocess.run(
            ["squeue", "-h", "-j", slurm_job_id, "-t", "PENDING,RUNNING,CONFIGURING,COMPLETING"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and not completed.stdout.strip():
            return True
    if owner.get("host") == socket.gethostname():
        try:
            os.kill(int(owner["pid"]), 0)
        except (KeyError, TypeError, ValueError, ProcessLookupError):
            return True
        except PermissionError:
            return False
    return False


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
    import_source = source if resolved_source != source else resolved_source
    command = ["enroot", "import", "--output", str(tmp_prefix), f"docker://{import_source}"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "enroot import failed")
    tmp_sqsh = tmp_prefix.with_suffix(".sqsh") if tmp_prefix.with_suffix(".sqsh").exists() else tmp_prefix
    if not tmp_sqsh.exists():
        raise RuntimeError(f"enroot import did not produce expected image: {tmp_prefix} or {tmp_prefix.with_suffix('.sqsh')}")
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


def _emit_prepare_result(result_path: Path | None, payload: Mapping[str, Any]) -> None:
    if result_path is not None:
        _write_json(result_path, payload)
        return
    if payload.get("state") == "failed":
        print(f"prepare failed: {payload.get('error')}", file=sys.stderr)
        return
    parts = ["prepare completed"]
    model = payload.get("model")
    if isinstance(model, Mapping) and model.get("repo_id"):
        parts.append(f"model={model['repo_id']}")
    image = payload.get("image")
    if isinstance(image, Mapping) and image.get("image_path"):
        parts.append(f"image={image['image_path']}")
    print(" ".join(parts))


__all__ = [
    "ConstructCache",
    "build_prepare_parser",
    "build_teardown_parser",
    "configure_enroot_paths",
    "is_absolute_sqsh_image",
    "materialize_image",
    "materialized_image_path",
    "prefetch_model",
    "resolve_image_digest",
    "resolve_construct_cache",
    "run_prepare",
    "run_teardown",
]
