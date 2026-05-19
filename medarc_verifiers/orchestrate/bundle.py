"""Shared task bundle models and persistence for medarc-orchestrate."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from medarc_verifiers.orchestrate.config import TaskSpec
from medarc_verifiers.orchestrate.internal_io import load_internal_mapping
from medarc_verifiers.orchestrate.task_naming import sanitize_task_dirname

SPEC_VERSION = 2
_SIDECAR_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_SIDECAR_SRUN_FLAGS = {
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


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp-{uuid.uuid4().hex}")
    OmegaConf.save(config=OmegaConf.create(payload), f=str(tmp_path))
    tmp_path.replace(path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


@dataclass(frozen=True)
class TaskBundlePaths:
    root: Path

    @property
    def task_spec_path(self) -> Path:
        return self.root / "task.yaml"

    @property
    def eval_config_path(self) -> Path:
        return self.root / "eval-config.toml"

    @property
    def runtime_dir(self) -> Path:
        return self.root / "runtime"

    @property
    def allocation_path(self) -> Path:
        return self.runtime_dir / "allocation.json"

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
    def sidecar_dir(self) -> Path:
        return self.root / "sidecars"

    @property
    def submit_script_path(self) -> Path:
        return self.root / "submit.sh"

    @property
    def orchestrate_snapshot_path(self) -> Path:
        return self.root / "orchestrate-snapshot.toml"

    @property
    def eval_images_snapshot_path(self) -> Path:
        return self.root / "eval_images-snapshot.toml"


@dataclass(frozen=True)
class TaskOutputPaths:
    root: str
    task_spec_path: str
    eval_config_path: str
    allocation_path: str
    state_path: str
    serve_dir: str
    bench_dir: str
    sidecar_dir: str
    submit_script_path: str


@dataclass(frozen=True)
class SidecarReadinessSpec:
    enabled: bool = True
    url: str | None = None
    timeout_s: int = 240
    interval_s: int = 2


@dataclass(frozen=True)
class SidecarSpec:
    name: str
    runtime: str
    image: str
    srun_args: list[str]
    env: Mapping[str, str]
    command: list[str]
    readiness: SidecarReadinessSpec


@dataclass(frozen=True)
class ResolvedTaskSpec:
    spec_version: int
    task_id: str
    task_slug: str
    model_key: str
    model_id: str
    mode: str
    runtime: str
    original_job_config_path: str
    original_job_config_checksum: str
    bundled_eval_config_path: str
    bundled_eval_config_checksum: str
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
    sidecars: list[SidecarSpec]
    restart_source: str | None
    restart_source_strategy: str
    output_paths: TaskOutputPaths
    endpoints_path: str | None = None
    orchestrate_registry_path: str | None = None
    orchestrate_registry_checksum: str | None = None
    orchestrate_registry_schema_version: int | None = None
    matched_model: Mapping[str, Any] = field(default_factory=dict)
    eval_images_registry_path: str | None = None
    eval_images_registry_checksum: str | None = None
    eval_images_registry_schema_version: int | None = None
    selected_eval_images: list[Mapping[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, allow_missing_v2_fields: bool = False) -> "ResolvedTaskSpec":
        output_paths = payload.get("output_paths")
        if not isinstance(output_paths, Mapping):
            raise ValueError("Resolved task spec output_paths must be a mapping.")
        if "sidecars" not in payload and not allow_missing_v2_fields:
            raise ValueError("Resolved task spec is missing required sidecars field.")
        if "sidecar_dir" not in output_paths and not allow_missing_v2_fields:
            raise ValueError("Resolved task spec output_paths is missing required sidecar_dir field.")
        sidecars = [_sidecar_from_dict(item) for item in payload.get("sidecars", [])]
        root = str(output_paths["root"])
        return cls(
            spec_version=int(payload["spec_version"]),
            task_id=str(payload["task_id"]),
            task_slug=str(payload["task_slug"]),
            model_key=str(payload["model_key"]),
            model_id=str(payload["model_id"]),
            mode=str(payload["mode"]),
            runtime=str(payload["runtime"]),
            original_job_config_path=str(payload["original_job_config_path"]),
            original_job_config_checksum=str(payload["original_job_config_checksum"]),
            bundled_eval_config_path=str(payload["bundled_eval_config_path"]),
            bundled_eval_config_checksum=str(payload["bundled_eval_config_checksum"]),
            gpus=int(payload["gpus"]),
            tensor_parallel_size=int(payload["tensor_parallel_size"]),
            data_parallel_size=(
                int(payload["data_parallel_size"]) if payload.get("data_parallel_size") is not None else None
            ),
            container_image=str(payload["container_image"]),
            container_port=int(payload.get("container_port") or 8000),
            container_ipc_mode=(
                str(payload["container_ipc_mode"]) if payload.get("container_ipc_mode") is not None else None
            ),
            container_env_file=(
                str(payload["container_env_file"]) if payload.get("container_env_file") is not None else None
            ),
            volume_mounts=[str(item) for item in payload.get("volume_mounts", [])],
            pyxis_srun_extra_args=[str(item) for item in payload.get("pyxis_srun_extra_args", [])],
            serve_args=dict(payload.get("serve_args") or {}),
            sidecars=sidecars,
            restart_source=(str(payload["restart_source"]) if payload.get("restart_source") is not None else None),
            restart_source_strategy=str(payload.get("restart_source_strategy") or "none"),
            output_paths=TaskOutputPaths(
                root=root,
                task_spec_path=str(output_paths["task_spec_path"]),
                eval_config_path=str(output_paths["eval_config_path"]),
                allocation_path=str(output_paths["allocation_path"]),
                state_path=str(output_paths["state_path"]),
                serve_dir=str(output_paths["serve_dir"]),
                bench_dir=str(output_paths["bench_dir"]),
                sidecar_dir=str(output_paths.get("sidecar_dir") or (Path(root) / "sidecars")),
                submit_script_path=str(output_paths["submit_script_path"]),
            ),
            endpoints_path=(str(payload["endpoints_path"]) if payload.get("endpoints_path") is not None else None),
            orchestrate_registry_path=(
                str(payload["orchestrate_registry_path"])
                if payload.get("orchestrate_registry_path") is not None
                else None
            ),
            orchestrate_registry_checksum=(
                str(payload["orchestrate_registry_checksum"])
                if payload.get("orchestrate_registry_checksum") is not None
                else None
            ),
            orchestrate_registry_schema_version=(
                int(payload["orchestrate_registry_schema_version"])
                if payload.get("orchestrate_registry_schema_version") is not None
                else None
            ),
            matched_model=dict(payload.get("matched_model") or {}),
            eval_images_registry_path=(
                str(payload["eval_images_registry_path"])
                if payload.get("eval_images_registry_path") is not None
                else None
            ),
            eval_images_registry_checksum=(
                str(payload["eval_images_registry_checksum"])
                if payload.get("eval_images_registry_checksum") is not None
                else None
            ),
            eval_images_registry_schema_version=(
                int(payload["eval_images_registry_schema_version"])
                if payload.get("eval_images_registry_schema_version") is not None
                else None
            ),
            selected_eval_images=[dict(item) for item in payload.get("selected_eval_images", [])],
        )


class UnsupportedTaskSpecVersionError(ValueError):
    """Raised when a persisted task bundle uses a spec version this code cannot execute."""


@dataclass(frozen=True)
class ExecutionAllocation:
    task_id: str
    allocated_gpus: int | None = None
    gpu_ids: list[int] = field(default_factory=list)
    server_port: int | None = None
    require_contiguous_gpus: bool | None = None
    slurm_job_id: str | None = None
    constraints: Mapping[str, Any] = field(default_factory=dict)
    runtime_env: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeState:
    task_id: str
    state: str = "pending"
    restart_source: str | None = None
    restart_source_strategy: str = "none"
    bench_run_id: str | None = None
    bench_run_dir: str | None = None
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
    original_job_config_path: str
    original_job_config_checksum: str
    bundled_eval_config_path: str
    bundled_eval_config_checksum: str
    task_spec_path: str
    task_spec_checksum: str
    allocation_path: str
    state_path: str
    restart_source: str | None = None
    restart_source_strategy: str = "none"

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
    spec: ResolvedTaskSpec
    paths: TaskBundlePaths
    allocation: ExecutionAllocation
    state: RuntimeState


@dataclass(frozen=True)
class BundlePlan:
    manifest: RunBundleManifest
    tasks: Mapping[str, PlannedTaskBundle]


def default_output_root(run_id: str) -> Path:
    return Path("outputs") / "orchestrate" / run_id


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
        restart_source=(str(payload["restart_source"]) if payload.get("restart_source") is not None else None),
        restart_source_strategy=str(payload.get("restart_source_strategy") or "none"),
        bench_run_id=(str(payload["bench_run_id"]) if payload.get("bench_run_id") is not None else None),
        bench_run_dir=(str(payload["bench_run_dir"]) if payload.get("bench_run_dir") is not None else None),
        updated_at=str(payload.get("updated_at") or _now()),
    )


def write_runtime_state(path: Path, state: RuntimeState) -> None:
    _write_json_atomic(path, state.to_dict())


def load_execution_allocation(path: Path) -> ExecutionAllocation | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Execution allocation must be a mapping: {path}")
    return ExecutionAllocation(
        task_id=str(payload["task_id"]),
        allocated_gpus=(int(payload["allocated_gpus"]) if payload.get("allocated_gpus") is not None else None),
        gpu_ids=[int(item) for item in payload.get("gpu_ids", [])],
        server_port=(int(payload["server_port"]) if payload.get("server_port") is not None else None),
        require_contiguous_gpus=payload.get("require_contiguous_gpus"),
        slurm_job_id=(str(payload["slurm_job_id"]) if payload.get("slurm_job_id") is not None else None),
        constraints=dict(payload.get("constraints") or {}),
        runtime_env={str(key): str(value) for key, value in dict(payload.get("runtime_env") or {}).items()},
    )


def write_execution_allocation(path: Path, allocation: ExecutionAllocation) -> None:
    _write_json_atomic(path, allocation.to_dict())


def write_run_bundle_manifest(path: Path, manifest: RunBundleManifest) -> None:
    manifest.touch()
    _write_json_atomic(path, manifest.to_dict())


def load_task_spec(path: Path) -> ResolvedTaskSpec:
    payload = dict(load_internal_mapping(path, label="task spec"))
    raw_version = payload.get("spec_version")
    try:
        spec_version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise UnsupportedTaskSpecVersionError(f"Unsupported task spec_version={raw_version!r}; expected {SPEC_VERSION}.") from exc
    if spec_version != SPEC_VERSION:
        raise UnsupportedTaskSpecVersionError(
            f"Unsupported task spec_version={spec_version}; expected {SPEC_VERSION}."
        )
    spec = ResolvedTaskSpec.from_dict(payload)
    bundled_eval_path = Path(spec.bundled_eval_config_path)
    if not bundled_eval_path.exists():
        raise FileNotFoundError(f"Bundled eval config not found: {bundled_eval_path}")
    checksum = _sha256_file(bundled_eval_path)
    if checksum != spec.bundled_eval_config_checksum:
        raise ValueError(
            f"Bundled eval config checksum mismatch for task {spec.task_id}: "
            f"expected {spec.bundled_eval_config_checksum}, got {checksum}."
        )
    return spec


def ensure_run_bundle(
    *,
    tasks: list[TaskSpec],
    run_id: str,
    output_root: Path,
    mode: str,
    runtime: str,
    eval_config_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    allocation_defaults: Mapping[str, ExecutionAllocation] | None = None,
    existing_manifest: RunBundleManifest | None = None,
) -> BundlePlan:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_manifest_path(output_root)
    existing = _load_existing_run_bundle(
        output_root=output_root,
        manifest_path=manifest_path,
        run_id=run_id,
        existing_manifest=existing_manifest,
    )
    existing_entries = existing.entry_map() if existing else {}
    bundles: dict[str, PlannedTaskBundle] = {}
    manifest_entries: list[RunBundleEntry] = []

    for task in tasks:
        bundle = _ensure_task_bundle(
            task=task,
            output_root=output_root,
            mode=mode,
            runtime=runtime,
            eval_config_override=dict(eval_config_overrides.get(task.task_id) or {}) if eval_config_overrides else {},
            allocation_default=allocation_defaults.get(task.task_id) if allocation_defaults else None,
            existing_entry=existing_entries.get(task.task_id),
        )
        bundles[task.task_id] = bundle
        manifest_entries.append(
            RunBundleEntry(
                task_id=task.task_id,
                task_slug=bundle.spec.task_slug,
                model_key=task.model_key,
                model_id=task.model_id,
                mode=mode,
                runtime=runtime,
                original_job_config_path=str(task.job_config_path),
                original_job_config_checksum=bundle.spec.original_job_config_checksum,
                bundled_eval_config_path=bundle.spec.bundled_eval_config_path,
                bundled_eval_config_checksum=bundle.spec.bundled_eval_config_checksum,
                task_spec_path=bundle.spec.output_paths.task_spec_path,
                task_spec_checksum=_sha256_file(Path(bundle.spec.output_paths.task_spec_path)),
                allocation_path=bundle.spec.output_paths.allocation_path,
                state_path=bundle.spec.output_paths.state_path,
                restart_source=bundle.state.restart_source,
                restart_source_strategy=bundle.state.restart_source_strategy,
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
    eval_config_override: Mapping[str, Any],
    allocation_default: ExecutionAllocation | None,
    existing_entry: RunBundleEntry | None,
) -> PlannedTaskBundle:
    paths = task_bundle_paths(output_root, task.task_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.serve_dir.mkdir(parents=True, exist_ok=True)
    paths.bench_dir.mkdir(parents=True, exist_ok=True)
    paths.sidecar_dir.mkdir(parents=True, exist_ok=True)
    expected_payload, expected_spec = _build_task_spec(
        task=task,
        paths=paths,
        mode=mode,
        runtime=runtime,
        eval_config_override=eval_config_override,
    )

    if paths.task_spec_path.exists() and paths.eval_config_path.exists():
        try:
            loaded_spec = load_task_spec(paths.task_spec_path)
        except UnsupportedTaskSpecVersionError:
            _write_task_bundle(paths=paths, eval_payload=expected_payload, spec=expected_spec)
            spec = expected_spec
        else:
            if loaded_spec.to_dict() == expected_spec.to_dict():
                spec = loaded_spec
            else:
                _write_task_bundle(paths=paths, eval_payload=expected_payload, spec=expected_spec)
                spec = expected_spec
    else:
        _write_task_bundle(paths=paths, eval_payload=expected_payload, spec=expected_spec)
        spec = expected_spec

    state = load_runtime_state(paths.state_path)
    resolved_restart_source, resolved_restart_strategy = _resolve_restart_source(
        task=task,
        spec=spec,
        state=state,
        existing_entry=existing_entry,
    )
    if state is None:
        state = RuntimeState(
            task_id=task.task_id,
            restart_source=resolved_restart_source,
            restart_source_strategy=resolved_restart_strategy,
        )
        write_runtime_state(paths.state_path, state)
    allocation = load_execution_allocation(paths.allocation_path)
    if allocation is None:
        allocation = allocation_default or ExecutionAllocation(task_id=task.task_id)
        write_execution_allocation(paths.allocation_path, allocation)
    return PlannedTaskBundle(task=task, spec=spec, paths=paths, allocation=allocation, state=state)


def _resolve_restart_source(
    *,
    task: TaskSpec,
    spec: ResolvedTaskSpec,
    state: RuntimeState | None,
    existing_entry: RunBundleEntry | None,
) -> tuple[str | None, str]:
    del task, spec, existing_entry
    if state is not None and state.restart_source:
        return state.restart_source, state.restart_source_strategy or "runtime_state"
    return None, "none"


def _create_task_spec(
    *,
    task: TaskSpec,
    paths: TaskBundlePaths,
    mode: str,
    runtime: str,
    eval_config_override: Mapping[str, Any],
) -> ResolvedTaskSpec:
    payload, spec = _build_task_spec(
        task=task,
        paths=paths,
        mode=mode,
        runtime=runtime,
        eval_config_override=eval_config_override,
    )
    _write_task_bundle(paths=paths, eval_payload=payload, spec=spec)
    return spec


def _build_task_spec(
    *,
    task: TaskSpec,
    paths: TaskBundlePaths,
    mode: str,
    runtime: str,
    eval_config_override: Mapping[str, Any],
) -> tuple[bytes, ResolvedTaskSpec]:
    if eval_config_override:
        raise ValueError("Task-local eval config overrides are not supported for TOML bench bundles.")
    source_bytes = task.job_config_path.read_bytes()
    source_checksum = _sha256_file(task.job_config_path)
    bundled_bytes = _rewrite_eval_toml_paths(source_bytes, base_dir=task.job_config_path.parent)
    bundled_checksum = _sha256_bytes(bundled_bytes)

    model_cfg = dict(_mapping_section(task.orchestrate, "vllm", task_id=task.task_id))
    container_cfg = dict(_mapping_section(task.orchestrate, "container", task_id=task.task_id))
    pyxis_cfg = dict(task.orchestrate.get("pyxis") or {})
    sidecars = _parse_eval_image_sidecars(task.eval_images, task_id=task.task_id, mode=mode, runtime=runtime)
    pyxis_srun_extra_args = [str(item) for item in pyxis_cfg.get("srun_extra_args", []) or []]
    if sidecars and not _srun_args_include_flag(pyxis_srun_extra_args, "--overlap"):
        pyxis_srun_extra_args.append("--overlap")
    spec = ResolvedTaskSpec(
        spec_version=SPEC_VERSION,
        task_id=task.task_id,
        task_slug=paths.root.name,
        model_key=task.model_key,
        model_id=task.model_id,
        mode=mode,
        runtime=runtime,
        original_job_config_path=str(task.job_config_path.resolve()),
        original_job_config_checksum=source_checksum,
        bundled_eval_config_path=str(paths.eval_config_path),
        bundled_eval_config_checksum=bundled_checksum,
        gpus=int(model_cfg.get("gpus", 1) or 1),
        tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1) or 1),
        data_parallel_size=(
            int(model_cfg["data_parallel_size"]) if model_cfg.get("data_parallel_size") is not None else None
        ),
        container_image=str(container_cfg.get("image", "")),
        container_port=int(container_cfg.get("container_port", 8000) or 8000),
        container_ipc_mode=str(container_cfg.get("ipc_mode")) if container_cfg.get("ipc_mode") is not None else None,
        container_env_file=(
            str(container_cfg["env_file"]).strip() if container_cfg.get("env_file") is not None else None
        ),
        volume_mounts=[str(item) for item in container_cfg.get("volumes", []) or []],
        pyxis_srun_extra_args=pyxis_srun_extra_args,
        serve_args=dict(model_cfg.get("serve") or {}),
        sidecars=sidecars,
        restart_source=None,
        restart_source_strategy="none",
        output_paths=TaskOutputPaths(
            root=str(paths.root),
            task_spec_path=str(paths.task_spec_path),
            eval_config_path=str(paths.eval_config_path),
            allocation_path=str(paths.allocation_path),
            state_path=str(paths.state_path),
            serve_dir=str(paths.serve_dir),
            bench_dir=str(paths.bench_dir),
            sidecar_dir=str(paths.sidecar_dir),
            submit_script_path=str(paths.submit_script_path),
        ),
        endpoints_path=str(task.endpoints_path) if task.endpoints_path is not None else None,
        orchestrate_registry_path=task.orchestrate_registry.path,
        orchestrate_registry_checksum=task.orchestrate_registry.checksum,
        orchestrate_registry_schema_version=task.orchestrate_registry.schema_version,
        matched_model=dict(task.matched_model),
        eval_images_registry_path=task.eval_images_registry.path,
        eval_images_registry_checksum=task.eval_images_registry.checksum,
        eval_images_registry_schema_version=task.eval_images_registry.schema_version,
        selected_eval_images=[dict(item) for item in task.eval_images],
    )
    return bundled_bytes, spec


def _sidecar_from_dict(payload: Mapping[str, Any]) -> SidecarSpec:
    readiness_payload = payload.get("readiness") or {}
    if not isinstance(readiness_payload, Mapping):
        raise ValueError("Sidecar readiness must be a mapping.")
    return SidecarSpec(
        name=str(payload["name"]),
        runtime=str(payload["runtime"]),
        image=str(payload["image"]),
        srun_args=[str(item) for item in payload.get("srun_args", [])],
        env={str(key): str(value) for key, value in dict(payload.get("env") or {}).items()},
        command=[str(item) for item in payload.get("command", [])],
        readiness=SidecarReadinessSpec(
            enabled=bool(readiness_payload.get("enabled", True)),
            url=str(readiness_payload["url"]) if readiness_payload.get("url") is not None else None,
            timeout_s=int(readiness_payload.get("timeout_s", 240) or 240),
            interval_s=int(readiness_payload.get("interval_s", 2) or 2),
        ),
    )


def _parse_sidecars(
    orchestrate: object,
    *,
    task_id: str,
    mode: str,
    runtime: str,
) -> list[SidecarSpec]:
    if not isinstance(orchestrate, Mapping):
        return []
    raw_sidecars = orchestrate.get("sidecars")
    if raw_sidecars is None:
        return []
    if mode != "slurm":
        raise ValueError(f"Task {task_id} configures sidecars, but sidecars are only supported in slurm mode.")
    if not isinstance(raw_sidecars, Mapping):
        raise ValueError(f"Task {task_id} orchestrate.sidecars must be a mapping.")
    specs: list[SidecarSpec] = []
    suffixes: dict[str, str] = {}
    for raw_name, raw_cfg in raw_sidecars.items():
        name = str(raw_name)
        if _SIDECAR_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"Task {task_id} sidecar name {name!r} must match [A-Za-z0-9_.-]+.")
        suffix = _sidecar_shell_suffix(name)
        previous = suffixes.get(suffix)
        if previous is not None:
            raise ValueError(
                f"Task {task_id} sidecar names {previous!r} and {name!r} produce the same shell variable suffix."
            )
        suffixes[suffix] = name
        if not isinstance(raw_cfg, Mapping):
            raise ValueError(f"Task {task_id} sidecar {name} must be a mapping.")
        cfg = dict(raw_cfg)
        sidecar_runtime = str(cfg.get("runtime", "")).strip().lower()
        if sidecar_runtime != "pyxis":
            raise ValueError(f"Task {task_id} sidecar {name} runtime must be 'pyxis' in v1.")
        if runtime != "pyxis":
            raise ValueError(f"Task {task_id} sidecar {name} requires orchestrate runtime 'pyxis'.")
        image = _required_string(cfg.get("image"), f"Task {task_id} sidecar {name} image")
        command = _required_string_list(cfg.get("command"), f"Task {task_id} sidecar {name} command")
        srun_args = _optional_string_list(cfg.get("srun_args", []), f"Task {task_id} sidecar {name} srun_args")
        _validate_sidecar_srun_args(srun_args, task_id=task_id, sidecar_name=name)
        env = _sidecar_env(cfg.get("env", {}), task_id=task_id, sidecar_name=name)
        readiness = _sidecar_readiness(cfg.get("readiness", {}), task_id=task_id, sidecar_name=name)
        specs.append(
            SidecarSpec(
                name=name,
                runtime=sidecar_runtime,
                image=image,
                srun_args=srun_args,
                env=env,
                command=command,
                readiness=readiness,
            )
        )
    return specs


def _rewrite_eval_toml_paths(source_bytes: bytes, *, base_dir: Path) -> bytes:
    text = source_bytes.decode("utf-8")
    lines = text.splitlines(keepends=True)
    changed = False
    rewritten: list[str] = []
    in_array_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[[") and stripped.endswith("]]"):
            in_array_table = True
        if not in_array_table:
            updated = _rewrite_top_level_path_line(line, key="endpoints_path", base_dir=base_dir)
            if updated is not None:
                rewritten.append(updated)
                changed = True
                continue
            updated = _rewrite_top_level_path_line(line, key="env_dir_path", base_dir=base_dir)
            if updated is not None:
                rewritten.append(updated)
                changed = True
                continue
        rewritten.append(line)
    if not changed:
        return source_bytes
    return "".join(rewritten).encode("utf-8")


def _rewrite_top_level_path_line(line: str, *, key: str, base_dir: Path) -> str | None:
    import json as _json
    import re as _re

    match = _re.match(rf"^(?P<prefix>\s*{_re.escape(key)}\s*=\s*)(?P<quote>['\"])(?P<value>.*?)(?P=quote)(?P<suffix>\s*(?:#.*)?\n?)$", line)
    if match is None:
        return None
    value = match.group("value").strip()
    path = Path(value).expanduser()
    if path.is_absolute():
        return line
    resolved = (base_dir / path).resolve()
    return f"{match.group('prefix')}{_json.dumps(str(resolved))}{match.group('suffix')}"


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


def _parse_eval_image_sidecars(
    eval_images: list[Mapping[str, Any]],
    *,
    task_id: str,
    mode: str,
    runtime: str,
) -> list[SidecarSpec]:
    if not eval_images:
        return []
    if mode != "slurm":
        raise ValueError(f"Task {task_id} configures eval images, but eval images are only supported in slurm mode.")
    specs: list[SidecarSpec] = []
    for entry in eval_images:
        name = str(entry["id"])
        if _SIDECAR_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"Task {task_id} eval image id {name!r} must match [A-Za-z0-9_.-]+.")
        sidecar_runtime = str(entry.get("runtime", "")).strip().lower()
        if sidecar_runtime != "pyxis":
            raise ValueError(f"Task {task_id} eval image {name} runtime must be 'pyxis' in v1.")
        if runtime != "pyxis":
            raise ValueError(f"Task {task_id} eval image {name} requires orchestrate runtime 'pyxis'.")
        image = _required_string(entry.get("image"), f"Task {task_id} eval image {name} image")
        command = _required_string_list(entry.get("command"), f"Task {task_id} eval image {name} command")
        srun_args = _optional_string_list(entry.get("srun_args", []), f"Task {task_id} eval image {name} srun_args")
        _validate_sidecar_srun_args(srun_args, task_id=task_id, sidecar_name=name)
        env = _sidecar_env(entry.get("env", {}), task_id=task_id, sidecar_name=name)
        readiness = _sidecar_readiness(entry.get("readiness", {}), task_id=task_id, sidecar_name=name)
        specs.append(
            SidecarSpec(
                name=name,
                runtime=sidecar_runtime,
                image=image,
                srun_args=srun_args,
                env=env,
                command=command,
                readiness=readiness,
            )
        )
    return specs


def _sidecar_shell_suffix(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required and must be a non-empty string.")
    return value


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


def _sidecar_env(value: object, *, task_id: str, sidecar_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Task {task_id} sidecar {sidecar_name} env must be a mapping.")
    env: dict[str, str] = {}
    for key, item in value.items():
        env_key = str(key)
        if _ENV_NAME_RE.fullmatch(env_key) is None:
            raise ValueError(f"Task {task_id} sidecar {sidecar_name} env key {env_key!r} is not shell-safe.")
        if not isinstance(item, str):
            raise ValueError(f"Task {task_id} sidecar {sidecar_name} env values must be strings.")
        env[env_key] = item
    return env


def _sidecar_readiness(value: object, *, task_id: str, sidecar_name: str) -> SidecarReadinessSpec:
    if not isinstance(value, Mapping):
        raise ValueError(f"Task {task_id} sidecar {sidecar_name} readiness must be a mapping.")
    enabled = bool(value.get("enabled", True))
    url = value.get("url")
    if enabled and (not isinstance(url, str) or not url.strip()):
        raise ValueError(f"Task {task_id} sidecar {sidecar_name} readiness.url is required unless readiness.enabled=false.")
    timeout_s = int(value.get("timeout_s", 240) or 240)
    interval_s = int(value.get("interval_s", 2) or 2)
    if timeout_s <= 0:
        raise ValueError(f"Task {task_id} sidecar {sidecar_name} readiness.timeout_s must be positive.")
    if interval_s <= 0:
        raise ValueError(f"Task {task_id} sidecar {sidecar_name} readiness.interval_s must be positive.")
    return SidecarReadinessSpec(enabled=enabled, url=str(url) if url is not None else None, timeout_s=timeout_s, interval_s=interval_s)


def _validate_sidecar_srun_args(args: list[str], *, task_id: str, sidecar_name: str) -> None:
    for arg in args:
        if not arg.startswith("--"):
            continue
        key = _srun_arg_key(arg)
        if key in _RESERVED_SIDECAR_SRUN_FLAGS:
            raise ValueError(f"Task {task_id} sidecar {sidecar_name} srun_args cannot set renderer-owned flag {key}.")


def _srun_args_include_flag(args: list[str], flag: str) -> bool:
    return any(_srun_arg_key(arg) == flag for arg in args if arg.startswith("--"))


def _srun_arg_key(arg: str) -> str:
    return arg.split("=", maxsplit=1)[0]


def _write_task_bundle(*, paths: TaskBundlePaths, eval_payload: bytes, spec: ResolvedTaskSpec) -> None:
    paths.eval_config_path.write_bytes(eval_payload)
    _write_task_spec(paths.task_spec_path, spec)
    _write_snapshot_toml(
        paths.orchestrate_snapshot_path,
        {
            "schema_version": spec.orchestrate_registry_schema_version,
            "registry_path": spec.orchestrate_registry_path,
            "registry_checksum": spec.orchestrate_registry_checksum,
            "model": spec.matched_model,
        },
    )
    _write_snapshot_toml(
        paths.eval_images_snapshot_path,
        {
            "schema_version": spec.eval_images_registry_schema_version,
            "registry_path": spec.eval_images_registry_path,
            "registry_checksum": spec.eval_images_registry_checksum,
            "eval_image": spec.selected_eval_images,
        },
    )


def _write_task_spec(path: Path, spec: ResolvedTaskSpec) -> None:
    _write_yaml_atomic(path, spec.to_dict())


def _mapping_section(payload: Mapping[str, Any], key: str, *, task_id: str) -> dict[str, Any]:
    section = payload.get(key)
    if not isinstance(section, Mapping):
        raise ValueError(f"Task {task_id} orchestrate.{key} must be a mapping.")
    return dict(section)


__all__ = [
    "BundlePlan",
    "ExecutionAllocation",
    "PlannedTaskBundle",
    "ResolvedTaskSpec",
    "RunBundleEntry",
    "RunBundleManifest",
    "RuntimeState",
    "SidecarReadinessSpec",
    "SidecarSpec",
    "SPEC_VERSION",
    "TaskBundlePaths",
    "default_output_root",
    "ensure_run_bundle",
    "load_execution_allocation",
    "load_run_bundle_manifest",
    "load_runtime_state",
    "load_task_spec",
    "run_manifest_path",
    "task_bundle_paths",
    "write_execution_allocation",
    "write_run_bundle_manifest",
    "write_runtime_state",
]
