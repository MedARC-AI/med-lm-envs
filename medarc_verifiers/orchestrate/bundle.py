"""Shared task bundle planning and path helpers for medarc-orchestrate."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from medarc_verifiers.orchestrate.config import TaskSpec, render_toml_mapping
from medarc_verifiers.orchestrate.task_naming import sanitize_task_dirname

_AUXILIARY_IMAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_AUXILIARY_IMAGE_SRUN_FLAGS = {
    "--overlap",
    "--nodes",
    "--ntasks",
    "--container-image",
    "--cpus-per-task",
    "--cpus-per-gpu",
    "--tres-per-task",
    "--gpus",
    "--gpus-per-task",
    "--gres",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp-{uuid.uuid4().hex}")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


@dataclass(frozen=True)
class TaskBundlePaths:
    root: Path

    @property
    def eval_config_path(self) -> Path:
        return self.root / "eval-config.toml"

    @property
    def runtime_dir(self) -> Path:
        return self.root / "runtime"

    @property
    def prepare_result_path(self) -> Path:
        return self.runtime_dir / "prepare_result.json"

    @property
    def teardown_result_path(self) -> Path:
        return self.runtime_dir / "teardown_result.json"

    @property
    def state_path(self) -> Path:
        return self.runtime_dir / "state.json"

    @property
    def serve_dir(self) -> Path:
        return self.root / "serve"

    @property
    def bench_dir(self) -> Path:
        return self.root / "bench"

    @property
    def auxiliary_image_dir(self) -> Path:
        return self.root / "auxiliary_images"

    @property
    def prepare_dir(self) -> Path:
        return self.root / "prepare"

    @property
    def teardown_dir(self) -> Path:
        return self.root / "teardown"

    @property
    def submit_script_path(self) -> Path:
        return self.root / "submit.sh"

    @property
    def prepare_script_path(self) -> Path:
        return self.root / "prepare.sh"

    @property
    def teardown_script_path(self) -> Path:
        return self.root / "teardown.sh"

    @property
    def orchestrate_snapshot_path(self) -> Path:
        return self.root / "orchestrate-snapshot.toml"

    @property
    def eval_images_snapshot_path(self) -> Path:
        return self.root / "eval_images-snapshot.toml"


@dataclass(frozen=True)
class PlannedRuntimePaths:
    root: str
    eval_config_path: str
    prepare_result_path: str
    teardown_result_path: str
    state_path: str
    serve_dir: str
    bench_dir: str
    auxiliary_image_dir: str
    prepare_dir: str
    teardown_dir: str
    submit_script_path: str
    prepare_script_path: str
    teardown_script_path: str


@dataclass(frozen=True)
class AuxiliaryImageReadinessSpec:
    enabled: bool = True
    url: str | None = None
    timeout_s: int = 240
    interval_s: int = 2


@dataclass(frozen=True)
class AuxiliaryImageSpec:
    name: str
    evals: list[str]
    envs: list[str]
    runtime: str
    image: str
    srun_args: list[str]
    env: Mapping[str, str]
    command: list[str]
    readiness: AuxiliaryImageReadinessSpec
    inject_env_args: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedTaskRuntime:
    task_id: str
    task_slug: str
    model_key: str
    model_id: str
    mode: str
    runtime: str
    suite_path: str
    suite_checksum: str
    target_endpoint_id: str
    bundled_eval_config_path: str
    bundled_eval_config_checksum: str
    output_dir: str
    gpus: int
    tensor_parallel_size: int
    data_parallel_size: int | None
    container_image: str
    container_port: int
    container_ipc_mode: str | None
    container_env_file: str | None
    volume_mounts: list[str]
    pyxis_srun_extra_args: list[str]
    serve_args: Mapping[str, Any]
    auxiliary_images: list[AuxiliaryImageSpec]
    output_paths: PlannedRuntimePaths
    container_image_source: str | None = None
    endpoints_path: str | None = None
    orchestrate_registry_path: str | None = None
    orchestrate_registry_checksum: str | None = None
    matched_model: Mapping[str, Any] = field(default_factory=dict)
    eval_images_registry_path: str | None = None
    eval_images_registry_checksum: str | None = None
    selected_eval_images: list[Mapping[str, Any]] = field(default_factory=list)
    construct_cache: Mapping[str, Any] = field(default_factory=dict)
    teardown: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeState:
    task_id: str
    state: str = "pending"
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunBundleEntry:
    task_id: str
    task_slug: str
    model_key: str
    model_id: str
    mode: str
    runtime: str
    suite_path: str
    suite_checksum: str
    target_endpoint_id: str
    bundled_eval_config_path: str
    bundled_eval_config_checksum: str
    state_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunBundleEntry":
        return cls(**dict(payload))


@dataclass
class RunBundleManifest:
    run_id: str
    output_root: str
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    tasks: list[RunBundleEntry] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = _now()

    def entry_map(self) -> dict[str, RunBundleEntry]:
        return {entry.task_id: entry for entry in self.tasks}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_root": self.output_root,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tasks": [entry.to_dict() for entry in self.tasks],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunBundleManifest":
        return cls(
            run_id=str(payload["run_id"]),
            output_root=str(payload["output_root"]),
            created_at=str(payload.get("created_at") or _now()),
            updated_at=str(payload.get("updated_at") or _now()),
            tasks=[RunBundleEntry.from_dict(entry) for entry in payload.get("tasks", [])],
        )


@dataclass(frozen=True)
class PlannedTaskBundle:
    task: TaskSpec
    runtime: PlannedTaskRuntime
    paths: TaskBundlePaths
    state: RuntimeState


@dataclass(frozen=True)
class BundlePlan:
    manifest: RunBundleManifest
    tasks: Mapping[str, PlannedTaskBundle]


def run_manifest_path(output_root: Path) -> Path:
    return output_root / "run_manifest.json"


def task_bundle_paths(output_root: Path, task_id: str) -> TaskBundlePaths:
    return TaskBundlePaths((output_root / "tasks" / sanitize_task_dirname(task_id)).resolve())


def load_run_bundle_manifest(path: Path) -> RunBundleManifest:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Run bundle manifest must be a mapping: {path}")
    return RunBundleManifest.from_dict(payload)


def load_runtime_state(path: Path) -> RuntimeState | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Runtime state must be a mapping: {path}")
    return RuntimeState(
        task_id=str(payload["task_id"]),
        state=str(payload.get("state") or "pending"),
        updated_at=str(payload.get("updated_at") or _now()),
    )


def write_runtime_state(path: Path, state: RuntimeState) -> None:
    _write_json_atomic(path, state.to_dict())


def write_run_bundle_manifest(path: Path, manifest: RunBundleManifest) -> None:
    manifest.touch()
    _write_json_atomic(path, manifest.to_dict())


def ensure_run_bundle(
    *,
    tasks: list[TaskSpec],
    run_id: str = "bundle",
    output_root: Path,
    mode: str,
    runtime: str,
    existing_manifest: RunBundleManifest | None = None,
    construct_cache_by_task: Mapping[str, Mapping[str, Any]] | None = None,
    teardown_by_task: Mapping[str, Mapping[str, Any]] | None = None,
    container_image_by_task: Mapping[str, str] | None = None,
) -> BundlePlan:
    if mode != "slurm":
        raise ValueError("Task bundles are only supported in slurm mode.")
    if runtime != "pyxis":
        raise ValueError("Slurm task bundles require pyxis runtime.")
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_manifest_path(output_root)
    existing = _load_existing_run_bundle(
        output_root=output_root,
        manifest_path=manifest_path,
        run_id=run_id,
        existing_manifest=existing_manifest,
    )
    bundles: dict[str, PlannedTaskBundle] = {}
    manifest_entries: list[RunBundleEntry] = []

    for task in tasks:
        bundle = _ensure_task_bundle(
            task=task,
            output_root=output_root,
            mode=mode,
            runtime=runtime,
            construct_cache=dict((construct_cache_by_task or {}).get(task.task_id) or {}),
            teardown=dict((teardown_by_task or {}).get(task.task_id) or {}),
            container_image_override=(container_image_by_task or {}).get(task.task_id),
        )
        bundles[task.task_id] = bundle
        manifest_entries.append(
            RunBundleEntry(
                task_id=task.task_id,
                task_slug=bundle.runtime.task_slug,
                model_key=task.model_key,
                model_id=task.model_id,
                mode=mode,
                runtime=runtime,
                suite_path=str(task.suite_path),
                suite_checksum=bundle.runtime.suite_checksum,
                target_endpoint_id=task.target_endpoint_id,
                bundled_eval_config_path=bundle.runtime.bundled_eval_config_path,
                bundled_eval_config_checksum=bundle.runtime.bundled_eval_config_checksum,
                state_path=bundle.runtime.output_paths.state_path,
            )
        )

    manifest = RunBundleManifest(run_id=run_id, output_root=str(output_root), tasks=manifest_entries)
    if existing is not None:
        manifest.created_at = existing.created_at
    write_run_bundle_manifest(manifest_path, manifest)
    return BundlePlan(manifest=manifest, tasks=bundles)


def _load_existing_run_bundle(
    *,
    output_root: Path,
    manifest_path: Path,
    run_id: str,
    existing_manifest: RunBundleManifest | None,
) -> RunBundleManifest | None:
    existing = existing_manifest or (load_run_bundle_manifest(manifest_path) if manifest_path.exists() else None)
    if existing is not None:
        if existing.run_id != run_id:
            raise ValueError(
                f"Existing run bundle at {manifest_path} belongs to run_id={existing.run_id}, not {run_id}."
            )
        return existing
    if _has_orphaned_bundle_artifacts(output_root):
        raise ValueError(
            f"Output root {output_root} already contains orchestrate task bundle artifacts without a run manifest; "
            "remove the stale artifacts or choose a different output directory."
        )
    return None


def _has_orphaned_bundle_artifacts(output_root: Path) -> bool:
    tasks_root = output_root / "tasks"
    if not tasks_root.exists():
        return False
    return any(tasks_root.iterdir())


def _ensure_task_bundle(
    *,
    task: TaskSpec,
    output_root: Path,
    mode: str,
    runtime: str,
    construct_cache: Mapping[str, Any],
    teardown: Mapping[str, Any],
    container_image_override: str | None,
) -> PlannedTaskBundle:
    paths = task_bundle_paths(output_root, task.task_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.serve_dir.mkdir(parents=True, exist_ok=True)
    paths.bench_dir.mkdir(parents=True, exist_ok=True)
    paths.auxiliary_image_dir.mkdir(parents=True, exist_ok=True)
    paths.prepare_dir.mkdir(parents=True, exist_ok=True)
    paths.teardown_dir.mkdir(parents=True, exist_ok=True)
    expected_payload, expected_runtime = _build_task_runtime(
        task=task,
        paths=paths,
        mode=mode,
        runtime=runtime,
        construct_cache=construct_cache,
        teardown=teardown,
        container_image_override=container_image_override,
    )

    _write_task_bundle(paths=paths, eval_payload=expected_payload, runtime=expected_runtime)
    runtime_plan = expected_runtime

    state = load_runtime_state(paths.state_path)
    if state is None:
        state = RuntimeState(task_id=task.task_id)
        write_runtime_state(paths.state_path, state)
    return PlannedTaskBundle(task=task, runtime=runtime_plan, paths=paths, state=state)


def _build_task_runtime(
    *,
    task: TaskSpec,
    paths: TaskBundlePaths,
    mode: str,
    runtime: str,
    construct_cache: Mapping[str, Any],
    teardown: Mapping[str, Any],
    container_image_override: str | None,
) -> tuple[bytes, PlannedTaskRuntime]:
    source_checksum = _sha256_file(task.suite_path)
    eval_payload = deepcopy(dict(task.generated_eval_config))
    eval_payload["endpoint_id"] = task.target_endpoint_id
    if task.endpoints_path is not None:
        eval_payload["endpoints_path"] = str(task.endpoints_path)
    output_dir = str(eval_payload.get("output_dir") or paths.bench_dir)
    eval_payload["output_dir"] = output_dir
    bundled_bytes = render_toml_mapping(eval_payload).encode("utf-8")
    bundled_checksum = _sha256_bytes(bundled_bytes)

    model_cfg = dict(_mapping_section(task.orchestrate, "vllm", task_id=task.task_id))
    container_cfg = dict(_mapping_section(task.orchestrate, "container", task_id=task.task_id))
    pyxis_cfg = dict(task.orchestrate.get("pyxis") or {})
    auxiliary_images = _parse_eval_image_auxiliary_images(task.eval_images, task_id=task.task_id, mode=mode, runtime=runtime)
    pyxis_srun_extra_args = [str(item) for item in pyxis_cfg.get("srun_extra_args", []) or []]
    if auxiliary_images and not _srun_args_include_flag(pyxis_srun_extra_args, "--overlap"):
        pyxis_srun_extra_args.append("--overlap")
    container_image_source = _required_string(
        container_cfg.get("image"), f"Task {task.task_id} orchestrate.container.image"
    )
    runtime_plan = PlannedTaskRuntime(
        task_id=task.task_id,
        task_slug=paths.root.name,
        model_key=task.model_key,
        model_id=task.model_id,
        mode=mode,
        runtime=runtime,
        suite_path=str(task.suite_path.resolve()),
        suite_checksum=source_checksum,
        target_endpoint_id=task.target_endpoint_id,
        bundled_eval_config_path=str(paths.eval_config_path),
        bundled_eval_config_checksum=bundled_checksum,
        output_dir=output_dir,
        gpus=_required_int(model_cfg.get("gpus"), f"Task {task.task_id} orchestrate.vllm.gpus"),
        tensor_parallel_size=_required_int(
            model_cfg.get("tensor_parallel_size"), f"Task {task.task_id} orchestrate.vllm.tensor_parallel_size"
        ),
        data_parallel_size=(
            int(model_cfg["data_parallel_size"]) if model_cfg.get("data_parallel_size") is not None else None
        ),
        container_image=container_image_override or container_image_source,
        container_port=_required_int(
            container_cfg.get("container_port"), f"Task {task.task_id} orchestrate.container.container_port"
        ),
        container_ipc_mode=str(container_cfg.get("ipc_mode")) if container_cfg.get("ipc_mode") is not None else None,
        container_env_file=(
            str(container_cfg["env_file"]).strip() if container_cfg.get("env_file") is not None else None
        ),
        volume_mounts=[str(item) for item in container_cfg.get("volumes", []) or []],
        pyxis_srun_extra_args=pyxis_srun_extra_args,
        serve_args=dict(model_cfg.get("serve") or {}),
        auxiliary_images=auxiliary_images,
        output_paths=PlannedRuntimePaths(
            root=str(paths.root),
            eval_config_path=str(paths.eval_config_path),
            prepare_result_path=str(paths.prepare_result_path),
            teardown_result_path=str(paths.teardown_result_path),
            state_path=str(paths.state_path),
            serve_dir=str(paths.serve_dir),
            bench_dir=str(paths.bench_dir),
            auxiliary_image_dir=str(paths.auxiliary_image_dir),
            prepare_dir=str(paths.prepare_dir),
            teardown_dir=str(paths.teardown_dir),
            submit_script_path=str(paths.submit_script_path),
            prepare_script_path=str(paths.prepare_script_path),
            teardown_script_path=str(paths.teardown_script_path),
        ),
        container_image_source=container_image_source if container_image_override else None,
        endpoints_path=str(task.endpoints_path) if task.endpoints_path is not None else None,
        orchestrate_registry_path=task.orchestrate_registry.path,
        orchestrate_registry_checksum=task.orchestrate_registry.checksum,
        matched_model=dict(task.matched_model),
        eval_images_registry_path=task.eval_images_registry.path,
        eval_images_registry_checksum=task.eval_images_registry.checksum,
        selected_eval_images=[dict(item) for item in task.eval_images],
        construct_cache=dict(construct_cache),
        teardown=dict(teardown),
    )
    return bundled_bytes, runtime_plan


def _write_snapshot_toml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_toml_mapping(payload), encoding="utf-8")


def _render_toml_mapping(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    arrays: list[tuple[str, list[Mapping[str, Any]]]] = []
    tables: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
            arrays.append((str(key), [dict(item) for item in value]))
        elif isinstance(value, Mapping):
            tables.append((str(key), value))
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    for name, table in tables:
        lines.append("")
        lines.append(f"[{name}]")
        _append_toml_table(lines, table)
    for name, items in arrays:
        for item in items:
            lines.append("")
            lines.append(f"[[{name}]]")
            _append_toml_table(lines, item)
    return "\n".join(lines).strip() + "\n"


def _append_toml_table(lines: list[str], table: Mapping[str, Any], *, prefix: str = "") -> None:
    nested: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in table.items():
        if value is None:
            continue
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            nested.append((full_key, value))
        elif isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
            continue
        else:
            lines.append(f"{full_key} = {_toml_value(value)}")
    for full_key, value in nested:
        _append_toml_table(lines, value, prefix=full_key)


def _toml_value(value: Any) -> str:
    import json as _json

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return _json.dumps(str(value))


def _parse_eval_image_auxiliary_images(
    eval_images: list[Mapping[str, Any]],
    *,
    task_id: str,
    mode: str,
    runtime: str,
) -> list[AuxiliaryImageSpec]:
    if not eval_images:
        return []
    if mode != "slurm":
        raise ValueError(f"Task {task_id} configures eval images, but eval images are only supported in slurm mode.")
    specs: list[AuxiliaryImageSpec] = []
    suffixes: dict[str, str] = {}
    for entry in eval_images:
        name = str(entry["id"])
        if _AUXILIARY_IMAGE_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"Task {task_id} eval image id {name!r} must match [A-Za-z0-9_.-]+.")
        suffix = _auxiliary_image_shell_suffix(name)
        previous = suffixes.get(suffix)
        if previous is not None:
            raise ValueError(
                f"Task {task_id} eval image ids {previous!r} and {name!r} produce the same shell variable suffix."
            )
        suffixes[suffix] = name
        auxiliary_image_runtime = str(entry.get("runtime", "")).strip().lower()
        if auxiliary_image_runtime != "pyxis":
            raise ValueError(f"Task {task_id} eval image {name} runtime must be 'pyxis' in v1.")
        if runtime != "pyxis":
            raise ValueError(f"Task {task_id} eval image {name} requires orchestrate runtime 'pyxis'.")
        image = _required_string(entry.get("image"), f"Task {task_id} eval image {name} image")
        command = _required_string_list(entry.get("command"), f"Task {task_id} eval image {name} command")
        srun_args = _optional_string_list(entry.get("srun_args", []), f"Task {task_id} eval image {name} srun_args")
        _validate_auxiliary_image_srun_args(srun_args, task_id=task_id, auxiliary_image_name=name)
        env = _auxiliary_image_env(entry.get("env", {}), task_id=task_id, auxiliary_image_name=name)
        readiness = _auxiliary_image_readiness(entry.get("readiness", {}), task_id=task_id, auxiliary_image_name=name)
        inject_env_args = _eval_image_inject_env_args(entry.get("inject", {}), task_id=task_id, image_name=name)
        if inject_env_args and any(":" in str(item) for item in entry.get("evals", []) or []):
            raise ValueError(
                f"Task {task_id} eval image {name} inject.env_args does not support variant eval selectors."
            )
        specs.append(
            AuxiliaryImageSpec(
                name=name,
                evals=[str(item) for item in entry.get("evals", []) or []],
                envs=[str(item) for item in (entry.get("envs", []) or []) + (entry.get("env_ids", []) or [])],
                runtime=auxiliary_image_runtime,
                image=image,
                srun_args=srun_args,
                env=env,
                command=command,
                readiness=readiness,
                inject_env_args=inject_env_args,
            )
        )
    return specs


def _auxiliary_image_shell_suffix(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required and must be a non-empty string.")
    return value


def _required_int(value: object, label: str) -> int:
    if value is None:
        raise ValueError(f"{label} is required.")
    return int(value)


def _required_string_list(value: object, label: str) -> list[str]:
    items = _optional_string_list(value, label)
    if not items:
        raise ValueError(f"{label} is required and must be a non-empty list of strings.")
    return items


def _optional_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of strings.")
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{label} must be a list of strings.")
    return list(value)


def _auxiliary_image_env(value: object, *, task_id: str, auxiliary_image_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Task {task_id} auxiliary image {auxiliary_image_name} env must be a mapping.")
    env: dict[str, str] = {}
    for key, item in value.items():
        env_key = str(key)
        if _ENV_NAME_RE.fullmatch(env_key) is None:
            raise ValueError(f"Task {task_id} auxiliary image {auxiliary_image_name} env key {env_key!r} is not shell-safe.")
        if not isinstance(item, str):
            raise ValueError(f"Task {task_id} auxiliary image {auxiliary_image_name} env values must be strings.")
        env[env_key] = item
    return env


def _auxiliary_image_readiness(value: object, *, task_id: str, auxiliary_image_name: str) -> AuxiliaryImageReadinessSpec:
    if not isinstance(value, Mapping):
        raise ValueError(f"Task {task_id} auxiliary image {auxiliary_image_name} readiness must be a mapping.")
    enabled = bool(value.get("enabled", True))
    url = value.get("url")
    if enabled and (not isinstance(url, str) or not url.strip()):
        raise ValueError(
            f"Task {task_id} auxiliary image {auxiliary_image_name} readiness.url is required unless readiness.enabled=false."
        )
    timeout_s = int(value.get("timeout_s", 240) or 240)
    interval_s = int(value.get("interval_s", 2) or 2)
    if timeout_s <= 0:
        raise ValueError(f"Task {task_id} auxiliary image {auxiliary_image_name} readiness.timeout_s must be positive.")
    if interval_s <= 0:
        raise ValueError(f"Task {task_id} auxiliary image {auxiliary_image_name} readiness.interval_s must be positive.")
    return AuxiliaryImageReadinessSpec(
        enabled=enabled, url=str(url) if url is not None else None, timeout_s=timeout_s, interval_s=interval_s
    )


def _eval_image_inject_env_args(value: object, *, task_id: str, image_name: str) -> dict[str, str]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Task {task_id} eval image {image_name} inject must be a mapping.")
    env_args = value.get("env_args", {})
    if not isinstance(env_args, Mapping):
        raise ValueError(f"Task {task_id} eval image {image_name} inject.env_args must be a mapping.")
    rendered: dict[str, str] = {}
    for key, item in env_args.items():
        if not isinstance(item, str):
            raise ValueError(f"Task {task_id} eval image {image_name} inject.env_args values must be strings.")
        rendered[str(key)] = item
    return rendered


def _validate_auxiliary_image_srun_args(args: list[str], *, task_id: str, auxiliary_image_name: str) -> None:
    for arg in args:
        if not arg.startswith("--"):
            continue
        key = _srun_arg_key(arg)
        if key in _RESERVED_AUXILIARY_IMAGE_SRUN_FLAGS:
            raise ValueError(
                f"Task {task_id} auxiliary image {auxiliary_image_name} srun_args cannot set renderer-owned flag {key}."
            )


def _srun_args_include_flag(args: list[str], flag: str) -> bool:
    return any(_srun_arg_key(arg) == flag for arg in args if arg.startswith("--"))


def _srun_arg_key(arg: str) -> str:
    return arg.split("=", maxsplit=1)[0]


def _write_task_bundle(*, paths: TaskBundlePaths, eval_payload: bytes, runtime: PlannedTaskRuntime) -> None:
    paths.eval_config_path.write_bytes(eval_payload)
    _write_snapshot_toml(
        paths.orchestrate_snapshot_path,
        {
            "registry_path": runtime.orchestrate_registry_path,
            "registry_checksum": runtime.orchestrate_registry_checksum,
            "model": runtime.matched_model,
        },
    )
    _write_snapshot_toml(
        paths.eval_images_snapshot_path,
        {
            "registry_path": runtime.eval_images_registry_path,
            "registry_checksum": runtime.eval_images_registry_checksum,
            "eval_image": runtime.selected_eval_images,
        },
    )


def _mapping_section(payload: Mapping[str, Any], key: str, *, task_id: str) -> dict[str, Any]:
    section = payload.get(key)
    if not isinstance(section, Mapping):
        raise ValueError(f"Task {task_id} orchestrate.{key} must be a mapping.")
    return dict(section)


__all__ = [
    "BundlePlan",
    "PlannedTaskBundle",
    "PlannedTaskRuntime",
    "RunBundleEntry",
    "RunBundleManifest",
    "RuntimeState",
    "AuxiliaryImageReadinessSpec",
    "AuxiliaryImageSpec",
    "TaskBundlePaths",
    "ensure_run_bundle",
    "load_run_bundle_manifest",
    "load_runtime_state",
    "run_manifest_path",
    "task_bundle_paths",
    "write_run_bundle_manifest",
    "write_runtime_state",
]
