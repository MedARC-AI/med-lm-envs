import json
import tomllib
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.bundle import (
    ensure_run_bundle,
    load_execution_allocation,
    load_run_bundle_manifest,
    load_task_spec,
)
from medarc_verifiers.orchestrate.cli import main as orchestrate_main
from medarc_verifiers.orchestrate.config import TaskSpec, expand_tasks, load_job_config, load_plan
from medarc_verifiers.orchestrate.internal_io import load_internal_mapping
from medarc_verifiers.orchestrate.slurm.manifest import SlurmBundleManifest
from medarc_verifiers.orchestrate.slurm.cli import build_parser as build_slurm_parser
from medarc_verifiers.orchestrate.slurm.plan import DEFAULT_SLURM_ACCOUNT, SlurmCliOverrides, build_submission_plan
from medarc_verifiers.orchestrate.slurm.render import render_bundle
from medarc_verifiers.orchestrate.slurm.submit import submit_bundle


_JOB_META: dict[Path, dict[str, object]] = {}
_SIDECAR_META: dict[Path, dict[str, object]] = {}


def _write_job_config(
    path: Path,
    *,
    model_id: str | None = None,
    gpus: int = 1,
    tensor_parallel_size: int | None = None,
    data_parallel_size: int | None = None,
    slurm: dict[str, object] | None = None,
) -> None:
    toml_path = path.with_suffix(".toml")
    resolved_model = model_id or f"Foo/{toml_path.stem}"
    endpoint_id = toml_path.stem
    toml_path.write_text(
        f"""
endpoint_id = "{endpoint_id}"
endpoints_path = "endpoints.toml"

[[eval]]
env_id = "{toml_path.stem}"
num_examples = 1
rollouts_per_example = 1
""".lstrip(),
        encoding="utf-8",
    )
    _JOB_META[toml_path.resolve()] = {
        "endpoint_id": endpoint_id,
        "model_id": resolved_model,
        "gpus": gpus,
        "tensor_parallel_size": tensor_parallel_size or gpus,
        "data_parallel_size": data_parallel_size,
        "slurm": dict(slurm or {}),
        "serve": {},
    }


def _write_sidecar_job_config(
    path: Path, *, name: str = "medagentbench-fhir", image: str = "/tmp/image with space.sqsh"
) -> None:
    _write_job_config(path, model_id="Foo/Bar", gpus=1)
    toml_path = path.with_suffix(".toml").resolve()
    _SIDECAR_META[toml_path] = [
        {
            "id": name,
            "env": path.with_suffix(".toml").stem,
            "runtime": "pyxis",
            "image": image,
            "srun_args": ["--mem=16G", "--container-env=JAVA_TOOL_OPTIONS"],
            "env_vars": {"JAVA_TOOL_OPTIONS": "-XX:+UseSerialGC -Xms256m -Xmx1024m"},
            "command": ["/usr/bin/java", "--class-path", "/app/main war"],
            "readiness": {
                "url": "http://127.0.0.1:8080/fhir/metadata?x=a b",
                "timeout_s": 12,
                "interval_s": 3,
            },
        }
    ]


def _task(
    tmp_path: Path, name: str, *, gpus: int, tp: int | None = None, slurm: dict[str, object] | None = None
) -> TaskSpec:
    job_cfg = tmp_path / f"{name}.toml"
    _write_job_config(job_cfg, model_id="foo", gpus=gpus, tensor_parallel_size=tp, slurm=slurm)
    return TaskSpec(
        task_id=f"{name}:foo",
        job_config_path=job_cfg.resolve(),
        model_key="foo",
        model_id="foo",
        orchestrate={
            "vllm": {"gpus": gpus, "tensor_parallel_size": tp or gpus, "serve": {}},
            "container": {"image": "fake"},
            "pyxis": {},
        },
        slurm=dict(slurm or {}),
    )


def _write_plan(tmp_path: Path, job_configs: list[Path]) -> Path:
    toml_paths = [path.with_suffix(".toml").resolve() for path in job_configs]
    plan_path = tmp_path / "plan.yaml"
    eval_images_path = tmp_path / "eval_images.toml"
    config_lines = "\n".join(f"  - {path}" for path in toml_paths)
    plan_lines = ["job_configs:", config_lines]
    sidecar_entries = [
        (entry, path)
        for path in toml_paths
        for entry in (
            _SIDECAR_META.get(path, []) if isinstance(_SIDECAR_META.get(path, []), list) else [_SIDECAR_META[path]]
        )
    ]
    if sidecar_entries:
        plan_lines.append(f"eval_images_config: {eval_images_path}")
    plan_path.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    endpoint_lines = []
    seen_endpoints: set[str] = set()
    for path in toml_paths:
        meta = _JOB_META[path]
        endpoint_id = str(meta["endpoint_id"])
        if endpoint_id in seen_endpoints:
            continue
        seen_endpoints.add(endpoint_id)
        endpoint_lines.extend(
            [
                "[[endpoint]]",
                f'endpoint_id = "{endpoint_id}"',
                f'model = "{meta["model_id"]}"',
                "",
                "[endpoint.orchestrate.vllm]",
                f"gpus = {meta['gpus']}",
                f"tensor_parallel_size = {meta['tensor_parallel_size']}",
            ]
        )
        if meta.get("data_parallel_size") is not None:
            endpoint_lines.append(f"data_parallel_size = {meta['data_parallel_size']}")
        endpoint_lines.extend(["", "[endpoint.orchestrate.vllm.serve]"])
        for key, value in dict(meta.get("serve") or {}).items():
            rendered = "true" if value is True else ("false" if value is False else str(value))
            if isinstance(value, str):
                rendered = f'"{value}"'
            endpoint_lines.append(f"{key} = {rendered}")
        endpoint_lines.extend(
            [
                "",
                "[endpoint.orchestrate.container]",
                'image = "fake"',
                "",
                "[endpoint.orchestrate.pyxis]",
                "srun_extra_args = []",
                "",
                "[endpoint.orchestrate.slurm]",
            ]
        )
        for key, value in dict(meta.get("slurm") or {}).items():
            rendered = "true" if value is True else ("false" if value is False else f'"{value}"')
            endpoint_lines.append(f"{key} = {rendered}")
        endpoint_lines.append("")
    (tmp_path / "endpoints.toml").write_text("\n".join(endpoint_lines), encoding="utf-8")

    if sidecar_entries:
        lines = []
        for entry, _path in sidecar_entries:
            lines.extend(
                [
                    "[[eval_image]]",
                    f'id = "{entry["id"]}"',
                    f'envs = ["{entry["env"]}"]',
                    'runtime = "pyxis"',
                    f'image = "{entry["image"]}"',
                    "srun_args = [" + ", ".join(f'"{arg}"' for arg in entry["srun_args"]) + "]",
                    "command = [" + ", ".join(f'"{arg}"' for arg in entry["command"]) + "]",
                    "",
                    "[eval_image.env]",
                ]
            )
            for key, value in entry["env_vars"].items():
                lines.append(f'{key} = "{value}"')
            lines.extend(["", "[eval_image.readiness]"])
            for key, value in entry["readiness"].items():
                if isinstance(value, bool):
                    rendered = "true" if value else "false"
                elif isinstance(value, int):
                    rendered = str(value)
                else:
                    rendered = f'"{value}"'
                lines.append(f"{key} = {rendered}")
            lines.append("")
        eval_images_path.write_text("\n".join(lines), encoding="utf-8")
    return plan_path


def test_slurm_parser_accepts_direct_job_configs() -> None:
    parser = build_slurm_parser()

    args = parser.parse_args(["--job-config", "a.yaml", "--job-config", "b.yaml", "--node-gpus", "8"])

    assert args.job_configs == [Path("a.yaml"), Path("b.yaml")]
    assert args.plan is None
    assert args.node_gpus == 8


def test_build_submission_plan_derives_dp_and_sorts(tmp_path: Path) -> None:
    tasks = [
        _task(tmp_path, "small", gpus=1),
        _task(tmp_path, "large", gpus=8, tp=8),
        _task(tmp_path, "medium", gpus=2, tp=2),
        _task(tmp_path, "medium-low-tp", gpus=2, tp=1),
    ]

    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=8,
        max_simultaneous_nodes=1,
        run_simultaneously=False,
        base_dependency=None,
        cli_overrides=SlurmCliOverrides(),
    )

    assert [task.task.task_id for task in planned] == ["large:foo", "medium:foo", "medium-low-tp:foo", "small:foo"]
    assert [
        (task.gpus, task.tensor_parallel_size, task.data_parallel_size, task.allocated_gpus, task.vllm_world_size)
        for task in planned
    ] == [
        (8, 8, 1, 8, 8),
        (2, 2, 4, 8, 8),
        (2, 1, 8, 8, 8),
        (1, 1, 8, 8, 8),
    ]
    assert planned[1].predecessor_task_id == "large:foo"
    assert all(task.options.account == DEFAULT_SLURM_ACCOUNT for task in planned)


def test_build_submission_plan_round_robins_two_chains(tmp_path: Path) -> None:
    tasks = [
        _task(tmp_path, "a", gpus=4, tp=4),
        _task(tmp_path, "b", gpus=2, tp=2),
        _task(tmp_path, "c", gpus=2, tp=2),
        _task(tmp_path, "d", gpus=1),
    ]

    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=8,
        max_simultaneous_nodes=2,
        run_simultaneously=False,
        base_dependency="afterok:555",
        cli_overrides=SlurmCliOverrides(),
    )

    assert [
        (task.task.task_id, task.chain_index, task.predecessor_task_id, task.base_dependency) for task in planned
    ] == [
        ("a:foo", 0, None, "afterok:555"),
        ("b:foo", 1, None, "afterok:555"),
        ("c:foo", 0, "a:foo", None),
        ("d:foo", 1, "b:foo", None),
    ]


def test_build_submission_plan_run_simultaneously_uses_no_generated_dependencies(tmp_path: Path) -> None:
    tasks = [
        _task(tmp_path, "a", gpus=4, tp=4),
        _task(tmp_path, "b", gpus=1),
    ]

    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=8,
        max_simultaneous_nodes=1,
        run_simultaneously=True,
        base_dependency="afterok:555",
        cli_overrides=SlurmCliOverrides(),
    )

    assert [
        (task.task.task_id, task.chain_index, task.predecessor_task_id, task.base_dependency) for task in planned
    ] == [
        ("a:foo", 0, None, "afterok:555"),
        ("b:foo", 1, None, "afterok:555"),
    ]


def test_build_submission_plan_rejects_tp_larger_than_node_gpus(tmp_path: Path) -> None:
    tasks = [_task(tmp_path, "too-wide", gpus=16, tp=16)]

    with pytest.raises(ValueError, match=r"requires gpus=16 minimum outer allocation, but allocated_gpus=8"):
        build_submission_plan(
            tasks,
            run_id="bundle",
            node_gpus=8,
            max_simultaneous_nodes=1,
            run_simultaneously=False,
            base_dependency=None,
            cli_overrides=SlurmCliOverrides(),
        )


def test_build_submission_plan_rejects_mismatched_explicit_data_parallel_size(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(job_cfg, gpus=2, tensor_parallel_size=2, data_parallel_size=1)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))

    with pytest.raises(ValueError, match=r"explicit data_parallel_size=1 does not match derived value 4"):
        build_submission_plan(
            tasks,
            run_id="bundle",
            node_gpus=8,
            max_simultaneous_nodes=1,
            run_simultaneously=False,
            base_dependency=None,
            cli_overrides=SlurmCliOverrides(),
        )


def test_build_submission_plan_rejects_node_allocation_incompatible_with_tp(tmp_path: Path) -> None:
    tasks = [_task(tmp_path, "bad-shape", gpus=3, tp=3)]

    with pytest.raises(ValueError, match=r"allocated_gpus=4 must be divisible by tensor_parallel_size=3"):
        build_submission_plan(
            tasks,
            run_id="bundle",
            node_gpus=4,
            max_simultaneous_nodes=1,
            run_simultaneously=False,
            base_dependency=None,
            cli_overrides=SlurmCliOverrides(),
        )


def test_render_bundle_writes_script_and_bundled_config(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(
        job_cfg,
        gpus=2,
        tensor_parallel_size=2,
        slurm={"partition": "gpu", "qos": "low", "nice": 500, "slurm_resume": True},
    )
    toml_cfg = job_cfg.with_suffix(".toml")
    toml_cfg.write_text(
        toml_cfg.read_text(encoding="utf-8").replace(
            "[[eval]]",
            'env_dir_path = "envs"\n\n[[eval]]',
            1,
        ),
        encoding="utf-8",
    )
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))
    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=8,
        max_simultaneous_nodes=1,
        run_simultaneously=False,
        base_dependency=None,
        cli_overrides=SlurmCliOverrides(time="04:00:00", cpus_per_gpu=12),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        run_id="bundle",
        node_gpus=8,
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=tmp_path / ".env",
        readiness_timeout_s=300,
        prune_logs_on_success=True,
    )

    entry = manifest.entries[0]
    script = Path(entry.script_path).read_text(encoding="utf-8")

    assert entry.gpus == 2
    assert entry.allocated_gpus == 8
    assert entry.tensor_parallel_size == 2
    assert entry.data_parallel_size == 4
    assert entry.vllm_world_size == 8
    assert entry.original_job_config_checksum
    assert entry.bundled_eval_config_checksum
    assert entry.task_spec_checksum
    assert "#SBATCH --gpus-per-task=8" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-gpu=12" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --qos=low" in script
    assert "#SBATCH --nice=500" in script
    assert "#SBATCH --requeue" in script
    assert "--runtime pyxis" in script
    assert "medarc-orchestrate worker" in script
    assert "--task" in script
    assert "--allocation" in script
    assert "--resume" not in script
    assert f'ACTIVATE_SCRIPT="${{ACTIVATE_SCRIPT:-{tmp_path / ".venv" / "bin" / "activate"}}}"' in script
    assert 'source "$ACTIVATE_SCRIPT"' in script
    assert "Missing activation script:" in script
    assert "MEDARC_ALLOCATED_GPU_COUNT=8" in script
    assert "--mem" not in script
    assert str(job_cfg) not in script

    bundled_payload = load_job_config(Path(entry.effective_job_config_path))
    assert bundled_payload["endpoint_id"] == "job"
    assert bundled_payload["endpoints_path"] == str((tmp_path / "endpoints.toml").resolve())
    assert bundled_payload["env_dir_path"] == str((tmp_path / "envs").resolve())
    assert "orchestrate" not in bundled_payload
    task_spec = load_task_spec(Path(entry.task_spec_path))
    assert task_spec.gpus == 2
    assert task_spec.tensor_parallel_size == 2
    assert task_spec.data_parallel_size is None
    orchestrate_snapshot = tomllib.loads(
        (Path(task_spec.output_paths.root) / "orchestrate-snapshot.toml").read_text(encoding="utf-8")
    )
    assert orchestrate_snapshot["registry_path"] == str((tmp_path / "endpoints.toml").resolve())
    assert orchestrate_snapshot["registry_checksum"]
    assert orchestrate_snapshot["model"]["model"].startswith("Foo/")
    allocation = load_execution_allocation(Path(entry.allocation_path))
    assert allocation is not None
    assert allocation.allocated_gpus == 8
    assert Path(entry.task_spec_path).name == "task.yaml"
    assert Path(entry.script_path).name == "submit.sh"

    run_manifest = load_run_bundle_manifest(tmp_path / "outputs" / "run_manifest.json")
    assert run_manifest.tasks[0].bundled_eval_config_path == entry.effective_job_config_path


def test_bundle_parses_sidecars(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_sidecar_job_config(job_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))

    plan = ensure_run_bundle(
        tasks=tasks,
        run_id="bundle",
        output_root=tmp_path / "outputs",
        mode="slurm",
        runtime="pyxis",
    )

    spec = plan.tasks[tasks[0].task_id].spec
    assert spec.output_paths.sidecar_dir.endswith("/sidecars")
    assert len(spec.sidecars) == 1
    sidecar = spec.sidecars[0]
    assert sidecar.name == "medagentbench-fhir"
    assert sidecar.runtime == "pyxis"
    assert sidecar.image == "/tmp/image with space.sqsh"
    assert sidecar.srun_args == ["--mem=16G", "--container-env=JAVA_TOOL_OPTIONS"]
    assert sidecar.env["JAVA_TOOL_OPTIONS"] == "-XX:+UseSerialGC -Xms256m -Xmx1024m"
    assert sidecar.command == ["/usr/bin/java", "--class-path", "/app/main war"]
    assert sidecar.readiness.url == "http://127.0.0.1:8080/fhir/metadata?x=a b"
    assert sidecar.readiness.timeout_s == 12
    assert sidecar.readiness.interval_s == 3
    assert "--overlap" in spec.pyxis_srun_extra_args
    eval_images_snapshot = tomllib.loads(
        (Path(spec.output_paths.root) / "eval_images-snapshot.toml").read_text(encoding="utf-8")
    )
    assert eval_images_snapshot["registry_path"] == str(tmp_path / "eval_images.toml")
    assert eval_images_snapshot["registry_checksum"]
    assert eval_images_snapshot["eval_image"][0]["id"] == "medagentbench-fhir"


def test_slurm_render_starts_sidecar_before_worker(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_sidecar_job_config(job_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))
    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=1,
        max_simultaneous_nodes=1,
        run_simultaneously=False,
        base_dependency=None,
        cli_overrides=SlurmCliOverrides(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        run_id="bundle",
        node_gpus=1,
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")

    assert script.index("srun --overlap") < script.index("medarc-orchestrate worker")
    assert "'--container-image=/tmp/image with space.sqsh'" in script
    assert "/usr/bin/java --class-path '/app/main war'" in script
    assert "--overlap" in load_task_spec(Path(manifest.entries[0].task_spec_path)).pyxis_srun_extra_args


def test_slurm_render_sidecar_has_exit_trap(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_sidecar_job_config(job_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))
    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=1,
        max_simultaneous_nodes=1,
        run_simultaneously=False,
        base_dependency=None,
        cli_overrides=SlurmCliOverrides(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        run_id="bundle",
        node_gpus=1,
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")

    assert "cleanup_sidecars() {" in script
    assert 'kill "$pid" 2>/dev/null || true' in script
    assert "trap cleanup_sidecars EXIT" in script


def test_slurm_render_sidecar_readiness_failure_tails_log(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_sidecar_job_config(job_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))
    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=1,
        max_simultaneous_nodes=1,
        run_simultaneously=False,
        base_dependency=None,
        cli_overrides=SlurmCliOverrides(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        run_id="bundle",
        node_gpus=1,
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")

    assert script.count('tail -100 "$SIDECAR_LOG_MEDAGENTBENCH_FHIR" >&2 || true') == 2
    assert "record_sidecar_failure sidecar_exited_before_readiness" in script
    assert "record_sidecar_failure sidecar_readiness_timeout" in script


def test_sidecar_validation_rejects_missing_image_or_command(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_sidecar_job_config(job_cfg, image="")

    with pytest.raises(ValueError, match="non-empty image"):
        expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))


def test_sidecar_validation_rejects_non_slurm_mode(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_sidecar_job_config(job_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))

    with pytest.raises(ValueError, match="only supported in slurm mode"):
        ensure_run_bundle(tasks=tasks, run_id="bundle", output_root=tmp_path / "outputs", mode="local", runtime="pyxis")


def test_sidecar_validation_rejects_reserved_srun_args(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_sidecar_job_config(job_cfg)
    _SIDECAR_META[job_cfg.with_suffix(".toml").resolve()][0]["srun_args"].extend(["--nodes", "1"])
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))

    with pytest.raises(ValueError, match="renderer-owned flag --nodes"):
        ensure_run_bundle(tasks=tasks, run_id="bundle", output_root=tmp_path / "outputs", mode="slurm", runtime="pyxis")


def test_sidecar_validation_rejects_shell_suffix_collision(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_sidecar_job_config(job_cfg, name="fhir-v2")
    first = dict(_SIDECAR_META[job_cfg.with_suffix(".toml").resolve()][0])
    second = dict(first)
    second.update(
        {
            "id": "fhir.v2",
            "image": "/tmp/other.sqsh",
            "command": ["/bin/sh", "-lc", "sleep 60"],
            "readiness": {"enabled": False},
        }
    )
    _SIDECAR_META[job_cfg.with_suffix(".toml").resolve()].append(second)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))

    with pytest.raises(ValueError, match="same shell variable suffix"):
        ensure_run_bundle(tasks=tasks, run_id="bundle", output_root=tmp_path / "outputs", mode="slurm", runtime="pyxis")


def test_slurm_render_uses_single_cleanup_trap_for_multiple_sidecars(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_sidecar_job_config(job_cfg)
    first = dict(_SIDECAR_META[job_cfg.with_suffix(".toml").resolve()][0])
    audit = dict(first)
    audit.update(
        {
            "id": "audit",
            "image": "/tmp/audit.sqsh",
            "command": ["/bin/sh", "-lc", "sleep 60"],
            "readiness": {"enabled": False},
        }
    )
    _SIDECAR_META[job_cfg.with_suffix(".toml").resolve()].append(audit)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))
    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=1,
        max_simultaneous_nodes=1,
        run_simultaneously=False,
        base_dependency=None,
        cli_overrides=SlurmCliOverrides(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        run_id="bundle",
        node_gpus=1,
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")

    assert script.count("cleanup_sidecars() {") == 1
    assert script.count("trap cleanup_sidecars EXIT") == 1
    assert script.count("srun --overlap") == 2
    assert script.count("until python3") == 1
    first_readiness_done = script.index("done", script.index("until python3"))
    assert first_readiness_done < script.index("--container-image=/tmp/audit.sqsh")


def test_slurm_render_shell_quotes_sidecar_values(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_sidecar_job_config(job_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))
    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=1,
        max_simultaneous_nodes=1,
        run_simultaneously=False,
        base_dependency=None,
        cli_overrides=SlurmCliOverrides(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs with space",
        run_id="bundle",
        node_gpus=1,
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")

    assert "JAVA_TOOL_OPTIONS='-XX:+UseSerialGC -Xms256m -Xmx1024m' srun --overlap" in script
    assert "SIDECAR_LOG_MEDAGENTBENCH_FHIR='" in script
    assert "'http://127.0.0.1:8080/fhir/metadata?x=a b'" in script
    assert "'/app/main war'" in script
    assert "export JAVA_TOOL_OPTIONS" not in script


def test_record_failure_writes_pre_worker_failure_artifacts(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_sidecar_job_config(job_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))
    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=1,
        max_simultaneous_nodes=1,
        run_simultaneously=False,
        base_dependency=None,
        cli_overrides=SlurmCliOverrides(),
    )
    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        run_id="bundle",
        node_gpus=1,
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    entry = manifest.entries[0]

    rc = orchestrate_main(
        [
            "record-failure",
            "--task-spec",
            entry.task_spec_path,
            "--allocation",
            entry.allocation_path,
            "--reason",
            "sidecar_readiness_timeout",
            "--message",
            "Timed out waiting for sidecar medagentbench-fhir",
        ]
    )

    task_root = Path(entry.task_spec_path).parent
    state = json.loads((task_root / "runtime" / "state.json").read_text(encoding="utf-8"))
    result = json.loads((task_root / "runtime" / "result.json").read_text(encoding="utf-8"))
    task_manifest = json.loads((task_root / "runtime" / "task_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "outputs" / "summary.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert state["state"] == "failed"
    assert result["failure_reason"] == "sidecar_readiness_timeout"
    assert task_manifest["state"] == "failed"
    assert task_manifest["error"] == "Timed out waiting for sidecar medagentbench-fhir"
    assert summary["tasks"][0]["failure_reason"] == "sidecar_readiness_timeout"


def test_old_task_spec_version_fails_before_missing_field_parsing(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(job_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))
    plan = ensure_run_bundle(
        tasks=tasks,
        run_id="bundle",
        output_root=tmp_path / "outputs",
        mode="slurm",
        runtime="pyxis",
    )
    task_spec_path = plan.tasks[tasks[0].task_id].paths.task_spec_path
    payload = dict(load_internal_mapping(task_spec_path, label="task spec"))
    payload["spec_version"] = 1
    payload.pop("sidecars", None)
    payload["output_paths"].pop("sidecar_dir", None)
    from omegaconf import OmegaConf

    OmegaConf.save(config=OmegaConf.create(payload), f=str(task_spec_path))

    with pytest.raises(ValueError, match="Unsupported task spec_version=1"):
        load_task_spec(task_spec_path)


def test_v2_task_spec_requires_sidecar_fields(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(job_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))
    plan = ensure_run_bundle(
        tasks=tasks,
        run_id="bundle",
        output_root=tmp_path / "outputs",
        mode="slurm",
        runtime="pyxis",
    )
    task_spec_path = plan.tasks[tasks[0].task_id].paths.task_spec_path
    payload = dict(load_internal_mapping(task_spec_path, label="task spec"))
    payload.pop("sidecars", None)
    from omegaconf import OmegaConf

    OmegaConf.save(config=OmegaConf.create(payload), f=str(task_spec_path))

    with pytest.raises(ValueError, match="missing required sidecars"):
        load_task_spec(task_spec_path)


def test_render_bundle_assigns_unique_ports_per_task(tmp_path: Path) -> None:
    job_a = tmp_path / "job-a.yaml"
    job_b = tmp_path / "job-b.yaml"
    _write_job_config(job_a, gpus=1)
    _write_job_config(job_b, gpus=1)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_a, job_b])))
    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=1,
        max_simultaneous_nodes=2,
        run_simultaneously=True,
        base_dependency=None,
        cli_overrides=SlurmCliOverrides(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        run_id="bundle",
        node_gpus=1,
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )

    allocations = [load_execution_allocation(Path(entry.allocation_path)) for entry in manifest.entries]

    ports = [allocation.server_port for allocation in allocations if allocation is not None]

    assert len(ports) == 2
    assert len(set(ports)) == 2
    assert all(port is not None and 8000 <= port <= 65000 for port in ports)


def test_render_bundle_refreshes_stale_task_bundle_when_source_changes(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(job_cfg, gpus=1)
    _write_plan(tmp_path, [job_cfg])
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))
    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=8,
        max_simultaneous_nodes=1,
        run_simultaneously=False,
        base_dependency=None,
        cli_overrides=SlurmCliOverrides(),
    )

    first_manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        run_id="bundle",
        node_gpus=8,
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    first_entry = first_manifest.entries[0]
    first_config_path = Path(first_entry.effective_job_config_path)
    first_payload = load_job_config(first_config_path)

    assert "orchestrate" not in first_payload

    job_cfg.with_suffix(".toml").write_text(
        job_cfg.with_suffix(".toml").read_text(encoding="utf-8") + '\nvariant_id = "changed"\n', encoding="utf-8"
    )

    second_manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        run_id="bundle",
        node_gpus=8,
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
        existing_manifest=first_manifest,
    )
    second_entry = second_manifest.entries[0]
    second_payload = load_job_config(Path(second_entry.effective_job_config_path))

    assert second_entry.effective_job_config_path == first_entry.effective_job_config_path
    assert "orchestrate" not in second_payload
    assert first_payload != second_payload



def test_rendered_task_local_config_is_loadable_by_orchestrator(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(job_cfg, gpus=2, tensor_parallel_size=2)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))
    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=8,
        max_simultaneous_nodes=1,
        run_simultaneously=False,
        base_dependency=None,
        cli_overrides=SlurmCliOverrides(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        run_id="bundle",
        node_gpus=8,
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )

    patched_path = Path(manifest.entries[0].effective_job_config_path)
    patched_plan = tmp_path / "patched-plan.yaml"
    patched_plan.write_text(f"job_configs:\n  - {patched_path}\n", encoding="utf-8")
    patched_tasks = expand_tasks(load_plan(patched_plan))

    assert len(patched_tasks) == 1
    assert patched_tasks[0].orchestrate["vllm"]["gpus"] == 2


def test_slurm_dry_run_writes_manifest_and_prints_commands(tmp_path: Path, capsys) -> None:
    job_a = tmp_path / "job-a.yaml"
    job_b = tmp_path / "job-b.yaml"
    _write_job_config(job_a, gpus=4, tensor_parallel_size=4)
    _write_job_config(job_b, gpus=1)
    _write_plan(tmp_path, [job_a, job_b])

    rc = orchestrate_main(
        [
            "slurm",
            "--job-config",
            str(job_a.with_suffix(".toml")),
            "--job-config",
            str(job_b.with_suffix(".toml")),
            "--run-id",
            "bundle",
            "--output-dir",
            str(tmp_path / "bundle"),
            "--dry-run",
        ]
    )

    assert rc == 0
    stdout_lines = capsys.readouterr().out.strip().splitlines()
    assert len(stdout_lines) == 2
    assert stdout_lines[0].startswith(f"sbatch --account {DEFAULT_SLURM_ACCOUNT} ")
    assert f"sbatch --account {DEFAULT_SLURM_ACCOUNT} " in stdout_lines[1]
    assert "--dependency=afterany:$JOBID_1" in stdout_lines[1]

    manifest = json.loads((tmp_path / "bundle" / "submission_manifest.json").read_text())
    assert [entry["state"] for entry in manifest["entries"]] == ["dry-run", "dry-run"]
    assert all(entry["slurm_job_id"] is None for entry in manifest["entries"])
    assert all(entry["account"] == DEFAULT_SLURM_ACCOUNT for entry in manifest["entries"])

    run_manifest = json.loads((tmp_path / "bundle" / "run_manifest.json").read_text())
    assert len(run_manifest["tasks"]) == 2


def test_slurm_rerender_loads_pre_phase3_submission_manifest(tmp_path: Path, capsys) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(job_cfg, gpus=2, tensor_parallel_size=2)
    _write_plan(tmp_path, [job_cfg])

    output_root = tmp_path / "bundle"
    rc = orchestrate_main(
        [
            "slurm",
            "--job-config",
            str(job_cfg.with_suffix(".toml")),
            "--run-id",
            "bundle",
            "--output-dir",
            str(output_root),
            "--dry-run",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    manifest_path = output_root / "submission_manifest.json"
    current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_entry = dict(current_manifest["entries"][0])
    old_entry = dict(current_entry)
    old_entry["tp_size"] = old_entry.pop("tensor_parallel_size")
    old_entry["dp_size"] = old_entry.pop("data_parallel_size")
    old_entry["effective_gpus"] = old_entry.pop("allocated_gpus")
    old_entry.pop("gpus")
    old_entry.pop("vllm_world_size")
    old_entry["state"] = "submitted"
    old_entry["slurm_job_id"] = "12345"
    old_manifest = {
        "run_id": current_manifest["run_id"],
        "bundle_root": current_manifest["bundle_root"],
        "node_gpus": current_manifest["node_gpus"],
        "created_at": current_manifest["created_at"],
        "updated_at": current_manifest["updated_at"],
        "entries": [old_entry],
    }
    manifest_path.write_text(json.dumps(old_manifest, indent=2), encoding="utf-8")

    rc = orchestrate_main(
        [
            "slurm",
            "--job-config",
            str(job_cfg.with_suffix(".toml")),
            "--run-id",
            "bundle",
            "--output-dir",
            str(output_root),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert capsys.readouterr().out.strip() == ""
    rerendered = json.loads(manifest_path.read_text(encoding="utf-8"))
    rerendered_entry = rerendered["entries"][0]
    assert rerendered_entry["slurm_job_id"] == "12345"
    assert rerendered_entry["state"] == "submitted"
    assert rerendered_entry["gpus"] == 2
    assert rerendered_entry["allocated_gpus"] == 8
    assert rerendered_entry["tensor_parallel_size"] == 2
    assert rerendered_entry["data_parallel_size"] == 4
    assert rerendered_entry["vllm_world_size"] == 8
    assert "tp_size" not in rerendered_entry
    assert "dp_size" not in rerendered_entry
    assert "effective_gpus" not in rerendered_entry


def test_slurm_cli_default_run_id_uses_shared_generator(tmp_path: Path, monkeypatch) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(job_cfg, gpus=1)
    _write_plan(tmp_path, [job_cfg])

    captured: dict[str, object] = {}

    def fake_render_bundle(**kwargs):
        captured["run_id"] = kwargs["run_id"]
        captured["bundle_root"] = kwargs["bundle_root"]
        return SlurmBundleManifest(
            run_id=kwargs["run_id"], bundle_root=str(kwargs["bundle_root"]), node_gpus=8, entries=[]
        )

    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.generate_run_id", lambda name: "shared-run-id")
    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.cli.render_bundle", fake_render_bundle)
    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.cli.write_bundle_manifest", lambda path, manifest: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.cli.mark_dry_run", lambda path, manifest: [])

    rc = orchestrate_main(
        [
            "slurm",
            "--job-config",
            str(job_cfg.with_suffix(".toml")),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert captured["run_id"] == "shared-run-id"
    assert captured["bundle_root"] == (Path("outputs") / "orchestrate" / "shared-run-id").resolve()


def test_run_backend_slurm_matches_slurm_alias_and_skips_local_probes(tmp_path: Path, monkeypatch) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(job_cfg, gpus=1)
    _write_plan(tmp_path, [job_cfg])
    output_root = tmp_path / "bundle"
    captured: list[dict[str, object]] = []

    def fake_render_bundle(**kwargs):
        captured.append(
            {
                "run_id": kwargs["run_id"],
                "bundle_root": kwargs["bundle_root"],
                "tasks": [task.task.task_id for task in kwargs["planned_tasks"]],
                "readiness_timeout_s": kwargs["readiness_timeout_s"],
            }
        )
        return SlurmBundleManifest(
            run_id=kwargs["run_id"], bundle_root=str(kwargs["bundle_root"]), node_gpus=8, entries=[]
        )

    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.runtime_probe.docker_available",
        lambda: (_ for _ in ()).throw(AssertionError("docker probe should not run for slurm")),
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.runtime_probe.podman_available",
        lambda: (_ for _ in ()).throw(AssertionError("podman probe should not run for slurm")),
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.launch.discover_gpus",
        lambda: (_ for _ in ()).throw(AssertionError("GPU discovery should not run for slurm")),
    )
    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.cli.render_bundle", fake_render_bundle)
    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.cli.write_bundle_manifest", lambda path, manifest: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.cli.mark_dry_run", lambda path, manifest: [])

    common = [
        "--job-config",
        str(job_cfg.with_suffix(".toml")),
        "--run-id",
        "bundle",
        "--output-dir",
        str(output_root),
        "--readiness-timeout-s",
        "111",
        "--dry-run",
    ]

    assert orchestrate_main(["slurm", *common]) == 0
    assert orchestrate_main(["run", "--backend", "slurm", *common]) == 0
    assert captured[0] == captured[1]


def test_submit_bundle_resumes_from_existing_job_ids(tmp_path: Path, monkeypatch) -> None:
    job_a = tmp_path / "job-a.yaml"
    job_b = tmp_path / "job-b.yaml"
    job_c = tmp_path / "job-c.yaml"
    _write_job_config(job_a, gpus=4, tensor_parallel_size=4)
    _write_job_config(job_b, gpus=2, tensor_parallel_size=2)
    _write_job_config(job_c, gpus=1)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_a, job_b, job_c])))
    planned = build_submission_plan(
        tasks,
        run_id="bundle",
        node_gpus=8,
        max_simultaneous_nodes=1,
        run_simultaneously=False,
        base_dependency=None,
        cli_overrides=SlurmCliOverrides(),
    )
    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "bundle",
        run_id="bundle",
        node_gpus=8,
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    manifest.entries[0].state = "submitted"
    manifest.entries[0].slurm_job_id = "101"

    calls: list[list[str]] = []

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout
            self.stderr = ""
            self.returncode = 0

    outputs = iter(["202\n", "303\n"])

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return Result(next(outputs))

    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.submit.subprocess.run", fake_run)

    submit_bundle(tmp_path / "bundle" / "submission_manifest.json", manifest)

    assert calls[0][0:4] == ["sbatch", "--account", DEFAULT_SLURM_ACCOUNT, "--parsable"]
    assert "--dependency=afterany:101" in calls[0]
    assert "--dependency=afterany:202" in calls[1]
    assert [entry.slurm_job_id for entry in manifest.entries] == ["101", "202", "303"]
