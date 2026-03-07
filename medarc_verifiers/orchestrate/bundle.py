"""Shared task bundle models and persistence for medarc-orchestrate."""

from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from medarc_verifiers.orchestrate.config import TaskSpec, load_job_config
from medarc_verifiers.orchestrate.task_naming import sanitize_task_dirname

SPEC_VERSION = 1


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


def _yaml_bytes(payload: Mapping[str, Any]) -> bytes:
    return OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True).encode("utf-8")


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _extract_restart_source(payload: Mapping[str, Any]) -> str | None:
    orchestrate = payload.get("orchestrate")
    if not isinstance(orchestrate, Mapping):
        return None
    value = orchestrate.get("restart")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _apply_restart_source(payload: dict[str, Any], restart_source: str | None) -> dict[str, Any]:
    orchestrate = dict(payload.get("orchestrate") or {})
    if restart_source:
        orchestrate["restart"] = restart_source
    else:
        orchestrate.pop("restart", None)
    payload["orchestrate"] = orchestrate
    return payload


@dataclass(frozen=True)
class TaskBundlePaths:
    root: Path

    @property
    def task_spec_path(self) -> Path:
        return self.root / "task.yaml"

    @property
    def eval_config_path(self) -> Path:
        return self.root / "eval-config.yaml"

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
    def submit_script_path(self) -> Path:
        return self.root / "submit.sh"


@dataclass(frozen=True)
class TaskOutputPaths:
    root: str
    task_spec_path: str
    eval_config_path: str
    allocation_path: str
    state_path: str
    serve_dir: str
    bench_dir: str
    submit_script_path: str


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
    data_parallel_size: int
    container_image: str
    container_port: int
    container_ipc_mode: str | None
    container_env_file: str | None
    volume_mounts: list[str]
    pyxis_srun_extra_args: list[str]
    serve_args: Mapping[str, Any]
    restart_source: str | None
    restart_source_strategy: str
    output_paths: TaskOutputPaths

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResolvedTaskSpec":
        output_paths = payload.get("output_paths")
        if not isinstance(output_paths, Mapping):
            raise ValueError("Resolved task spec output_paths must be a mapping.")
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
            data_parallel_size=int(payload.get("data_parallel_size") or 1),
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
            restart_source=(str(payload["restart_source"]) if payload.get("restart_source") is not None else None),
            restart_source_strategy=str(payload.get("restart_source_strategy") or "none"),
            output_paths=TaskOutputPaths(
                root=str(output_paths["root"]),
                task_spec_path=str(output_paths["task_spec_path"]),
                eval_config_path=str(output_paths["eval_config_path"]),
                allocation_path=str(output_paths["allocation_path"]),
                state_path=str(output_paths["state_path"]),
                serve_dir=str(output_paths["serve_dir"]),
                bench_dir=str(output_paths["bench_dir"]),
                submit_script_path=str(output_paths["submit_script_path"]),
            ),
        )


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
    payload = dict(load_job_config(path))
    spec = ResolvedTaskSpec.from_dict(payload)
    if spec.spec_version != SPEC_VERSION:
        raise ValueError(f"Unsupported task spec_version={spec.spec_version}; expected {SPEC_VERSION}.")
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
    existing = existing_manifest or (load_run_bundle_manifest(manifest_path) if manifest_path.exists() else None)
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
    expected_payload, expected_spec = _build_task_spec(
        task=task,
        paths=paths,
        mode=mode,
        runtime=runtime,
        eval_config_override=eval_config_override,
    )

    if paths.task_spec_path.exists() and paths.eval_config_path.exists():
        loaded_spec = load_task_spec(paths.task_spec_path)
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
    if state is not None and state.restart_source:
        return state.restart_source, state.restart_source_strategy or "runtime_state"
    if existing_entry is not None and existing_entry.restart_source:
        return existing_entry.restart_source, existing_entry.restart_source_strategy or "persisted"
    if spec.restart_source:
        return spec.restart_source, spec.restart_source_strategy
    source_payload = dict(load_job_config(task.job_config_path))
    source_restart = _extract_restart_source(source_payload)
    if source_restart:
        return source_restart, "source_config"
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
) -> tuple[dict[str, Any], ResolvedTaskSpec]:
    source_payload = deepcopy(dict(load_job_config(task.job_config_path)))
    source_checksum = _sha256_file(task.job_config_path)
    restart_source = _extract_restart_source(source_payload)
    restart_strategy = "source_config" if restart_source else "none"
    if eval_config_override:
        source_payload = _deep_merge(source_payload, eval_config_override)
    _apply_restart_source(source_payload, restart_source)
    bundled_checksum = _sha256_bytes(_yaml_bytes(source_payload))

    model_cfg = _resolve_orchestrate_section(source_payload, key=task.model_key, task_id=task.task_id)
    container_cfg = _resolve_orchestrate_section(
        source_payload,
        key="vllm-container",
        task_id=task.task_id,
        fallback_key="vllm-docker",
    )
    pyxis_cfg = _resolve_orchestrate_section(source_payload, key="pyxis", task_id=task.task_id)
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
        data_parallel_size=int(model_cfg.get("data_parallel_size", 1) or 1),
        container_image=str(container_cfg.get("image", "")),
        container_port=int(container_cfg.get("container_port", 8000) or 8000),
        container_ipc_mode=str(container_cfg.get("ipc_mode")) if container_cfg.get("ipc_mode") is not None else None,
        container_env_file=(
            str(container_cfg["env_file"]).strip() if container_cfg.get("env_file") is not None else None
        ),
        volume_mounts=[str(item) for item in container_cfg.get("volumes", []) or []],
        pyxis_srun_extra_args=[str(item) for item in pyxis_cfg.get("srun_extra_args", []) or []],
        serve_args=dict(model_cfg.get("serve") or {}),
        restart_source=restart_source,
        restart_source_strategy=restart_strategy,
        output_paths=TaskOutputPaths(
            root=str(paths.root),
            task_spec_path=str(paths.task_spec_path),
            eval_config_path=str(paths.eval_config_path),
            allocation_path=str(paths.allocation_path),
            state_path=str(paths.state_path),
            serve_dir=str(paths.serve_dir),
            bench_dir=str(paths.bench_dir),
            submit_script_path=str(paths.submit_script_path),
        ),
    )
    return source_payload, spec


def _write_task_bundle(*, paths: TaskBundlePaths, eval_payload: Mapping[str, Any], spec: ResolvedTaskSpec) -> None:
    _write_yaml_atomic(paths.eval_config_path, eval_payload)
    _write_task_spec(paths.task_spec_path, spec)


def _write_task_spec(path: Path, spec: ResolvedTaskSpec) -> None:
    _write_yaml_atomic(path, spec.to_dict())


def _resolve_orchestrate_section(
    payload: Mapping[str, Any],
    *,
    key: str,
    task_id: str,
    fallback_key: str | None = None,
) -> dict[str, Any]:
    orchestrate = payload.get("orchestrate")
    if not isinstance(orchestrate, Mapping):
        raise ValueError(f"Task {task_id} bundled payload is missing a valid orchestrate mapping.")
    section = orchestrate.get(key)
    if section is None and fallback_key is not None:
        section = orchestrate.get(fallback_key)
    if section is None:
        return {}
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
