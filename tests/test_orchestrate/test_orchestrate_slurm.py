import json
import re
import shlex
import tomllib
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.bundle import (
    ensure_run_bundle,
    load_run_bundle_manifest,
)
from medarc_verifiers.orchestrate.cli import main as orchestrate_main
from medarc_verifiers.orchestrate.config import TaskSpec, expand_tasks, load_plan, load_suite_config
from medarc_verifiers.orchestrate.slurm.manifest import SlurmBundleManifest
from medarc_verifiers.orchestrate.slurm.plan import build_submission_plan
from medarc_verifiers.orchestrate.slurm.render import render_bundle
from medarc_verifiers.orchestrate.slurm.submit import SlurmSubmissionOptions, mark_dry_run, submit_bundle, submit_lifecycle_bundle
from medarc_verifiers.orchestrate.worker import build_parser as build_launch_parser


_JOB_META: dict[Path, dict[str, object]] = {}
_AUX_IMAGE_META: dict[Path, dict[str, object]] = {}


def _write_suite_config(
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


def _write_lifecycle_plan(tmp_path: Path, suite_cfg: Path) -> Path:
    _write_plan(tmp_path, [suite_cfg])
    container_env = tmp_path / "container.env"
    container_env.write_text("HF_HOME=/root/.cache/huggingface\n", encoding="utf-8")
    endpoints = tmp_path / "endpoints.toml"
    endpoints.write_text(
        endpoints.read_text(encoding="utf-8").replace('image = "fake"', 'image = "vllm/vllm-openai:v0.12.0"'),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(
        plan_path.read_text(encoding="utf-8")
        + f"""

[container]
volumes = ["{tmp_path / 'hf'}:/root/.cache/huggingface"]
env_file = "container.env"

[prepare]
enabled = true
cpus = 4
time = "01:00:00"
partition = "cpu"

[prepare.cache]
image_dir = "{tmp_path / 'images'}"

[teardown]
enabled = true
cpus = 2
time = "00:15:00"
partition = "cpu"
""",
        encoding="utf-8",
    )
    return plan_path


def _write_lifecycle_sqsh_plan(tmp_path: Path, suite_cfg: Path, image_path: Path) -> Path:
    _write_plan(tmp_path, [suite_cfg])
    endpoints = tmp_path / "endpoints.toml"
    endpoints.write_text(
        endpoints.read_text(encoding="utf-8").replace('image = "fake"', f'image = "{image_path}"'),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(
        plan_path.read_text(encoding="utf-8")
        + f"""

[container]
volumes = ["{tmp_path / 'hf'}:/root/.cache/huggingface"]

[prepare]
enabled = true
materialize_images = true
prefetch_model_weights = true
cpus = 4
time = "01:00:00"
partition = "cpu"
""",
        encoding="utf-8",
    )
    return plan_path


def _write_prepare_only_plan(
    tmp_path: Path,
    suite_cfg: Path,
    *,
    prefetch: bool,
    materialize: bool,
) -> Path:
    _write_plan(tmp_path, [suite_cfg])
    endpoints = tmp_path / "endpoints.toml"
    endpoints.write_text(
        endpoints.read_text(encoding="utf-8").replace('image = "fake"', 'image = "vllm/vllm-openai:v0.12.0"'),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(
        plan_path.read_text(encoding="utf-8")
        + f"""

[container]
volumes = ["{tmp_path / 'hf'}:/root/.cache/huggingface"]

[prepare]
enabled = true
materialize_images = {"true" if materialize else "false"}
prefetch_model_weights = {"true" if prefetch else "false"}
cpus = 4
time = "01:00:00"
partition = "cpu"

[prepare.cache]
image_dir = "{tmp_path / 'images'}"
""",
        encoding="utf-8",
    )
    return plan_path


def _write_auxiliary_image_suite_config(
    path: Path, *, name: str = "medagentbench-fhir", image: str = "/tmp/image with space.sqsh"
) -> None:
    _write_suite_config(path, model_id="Foo/Bar", gpus=1)
    toml_path = path.with_suffix(".toml").resolve()
    _AUX_IMAGE_META[toml_path] = [
        {
            "id": name,
            "env": path.with_suffix(".toml").stem,
            "runtime": "pyxis",
            "image": image,
            "srun_args": ["--mem=16G", "--container-env=JAVA_TOOL_OPTIONS"],
            "env_vars": {"JAVA_TOOL_OPTIONS": "-XX:+UseSerialGC -Xms256m -Xmx1024m"},
            "command": ["/usr/bin/java", "--class-path", "/app/main war"],
            "inject_env_args": {},
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
    suite_cfg = tmp_path / f"{name}.toml"
    _write_suite_config(suite_cfg, model_id="foo", gpus=gpus, tensor_parallel_size=tp, slurm=slurm)
    return TaskSpec(
        task_id=f"{name}:foo",
        model_key="foo",
        model_id="foo",
        orchestrate={
            "vllm": {"gpus": gpus, "tensor_parallel_size": tp or gpus, "serve": {}},
            "container": {"image": "fake"},
            "pyxis": {},
        },
        suite_path=suite_cfg.resolve(),
        target_endpoint_id=suite_cfg.stem,
        generated_eval_config={
            "endpoint_id": suite_cfg.stem,
            "endpoints_path": str(tmp_path / "endpoints.toml"),
            "output_dir": ".",
            "eval": [{"env_id": suite_cfg.stem, "num_examples": 1, "rollouts_per_example": 1}],
        },
        slurm=dict(slurm or {}),
    )


def _write_plan(tmp_path: Path, suites: list[Path]) -> Path:
    toml_paths = [path.with_suffix(".toml").resolve() for path in suites]
    plan_path = tmp_path / "plan.toml"
    eval_images_path = tmp_path / "eval_images.toml"
    plan_lines = [f'suite = "{toml_paths[0]}"', 'endpoints_path = "endpoints.toml"']
    auxiliary_image_entries = [
        (entry, path)
        for path in toml_paths
        for entry in (
            _AUX_IMAGE_META.get(path, [])
            if isinstance(_AUX_IMAGE_META.get(path, []), list)
            else [_AUX_IMAGE_META[path]]
        )
    ]
    if auxiliary_image_entries:
        plan_lines.append(f'eval_images_config = "{eval_images_path}"')
    for path in toml_paths:
        plan_lines.extend(["", "[[target]]", f'endpoint_id = "{path.stem}"', f'suite = "{path}"'])
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

    if auxiliary_image_entries:
        lines = []
        for entry, _path in auxiliary_image_entries:
            lines.extend(
                [
                    "[[eval_image]]",
                    f'id = "{entry["id"]}"',
                    f'envs = ["{entry["env"]}"]',
                    "evals = [" + ", ".join(f'"{arg}"' for arg in entry.get("evals", [])) + "]",
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
            if entry.get("inject_env_args"):
                lines.extend(["", "[eval_image.inject.env_args]"])
                for key, value in entry["inject_env_args"].items():
                    lines.append(f'{key} = "{value}"')
            lines.append("")
        eval_images_path.write_text("\n".join(lines), encoding="utf-8")
    return plan_path


def test_build_submission_plan_derives_dp_and_sorts(tmp_path: Path) -> None:
    tasks = [
        _task(tmp_path, "small", gpus=1),
        _task(tmp_path, "large", gpus=8, tp=8),
        _task(tmp_path, "medium", gpus=2, tp=2),
        _task(tmp_path, "medium-low-tp", gpus=2, tp=1),
    ]

    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )

    assert [task.task.task_id for task in planned] == ["large:foo", "medium:foo", "medium-low-tp:foo", "small:foo"]
    assert [
        (task.gpus, task.tensor_parallel_size, task.data_parallel_size, task.allocated_gpus, task.vllm_world_size)
        for task in planned
    ] == [
        (8, 8, 1, 8, 8),
        (2, 2, 1, 2, 2),
        (2, 1, 2, 2, 2),
        (1, 1, 1, 1, 1),
    ]
    assert all(task.base_dependency is None for task in planned)
    assert all(task.options.account is None for task in planned)


def test_build_submission_plan_applies_base_dependency_to_each_task(tmp_path: Path) -> None:
    tasks = [
        _task(tmp_path, "a", gpus=4, tp=4),
        _task(tmp_path, "b", gpus=2, tp=2),
        _task(tmp_path, "c", gpus=2, tp=2),
        _task(tmp_path, "d", gpus=1),
    ]

    planned = build_submission_plan(
        tasks,
        base_dependency="afterok:555",
        submission_options=SlurmSubmissionOptions(),
    )

    assert [(task.task.task_id, task.base_dependency) for task in planned] == [
        ("a:foo", "afterok:555"),
        ("b:foo", "afterok:555"),
        ("c:foo", "afterok:555"),
        ("d:foo", "afterok:555"),
    ]


def test_build_submission_plan_does_not_generate_dependencies(tmp_path: Path) -> None:
    tasks = [
        _task(tmp_path, "a", gpus=4, tp=4),
        _task(tmp_path, "b", gpus=1),
    ]

    planned = build_submission_plan(
        tasks,
        base_dependency="afterok:555",
        submission_options=SlurmSubmissionOptions(),
    )

    assert [(task.task.task_id, task.base_dependency) for task in planned] == [
        ("a:foo", "afterok:555"),
        ("b:foo", "afterok:555"),
    ]


def test_build_submission_plan_rejects_invalid_endpoint_gpu_shape(tmp_path: Path) -> None:
    tasks = [_task(tmp_path, "too-wide", gpus=16, tp=16)]

    with pytest.raises(ValueError, match=r"allocated_gpus=16 is invalid"):
        build_submission_plan(
            tasks,
            base_dependency=None,
            submission_options=SlurmSubmissionOptions(),
        )


def test_build_submission_plan_rejects_mismatched_explicit_data_parallel_size(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg, gpus=2, tensor_parallel_size=2, data_parallel_size=2)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))

    with pytest.raises(ValueError, match=r"explicit data_parallel_size=2 does not match derived value 1"):
        build_submission_plan(
            tasks,
            base_dependency=None,
            submission_options=SlurmSubmissionOptions(),
        )


def test_build_submission_plan_rejects_node_allocation_incompatible_with_tp(tmp_path: Path) -> None:
    tasks = [_task(tmp_path, "bad-shape", gpus=3, tp=3)]

    with pytest.raises(ValueError, match=r"allocated_gpus=3 is invalid"):
        build_submission_plan(
            tasks,
            base_dependency=None,
            submission_options=SlurmSubmissionOptions(),
        )


def test_render_bundle_writes_script_and_bundled_config(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(
        suite_cfg,
        gpus=2,
        tensor_parallel_size=2,
        slurm={"partition": "gpu", "qos": "low", "nice": 500, "slurm_resume": True},
    )
    toml_cfg = suite_cfg.with_suffix(".toml")
    toml_cfg.write_text(
        toml_cfg.read_text(encoding="utf-8").replace(
            "[[eval]]",
            'env_dir_path = "envs"\n\n[[eval]]',
            1,
        ),
        encoding="utf-8",
    )
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))
    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(time="04:00:00", cpus_per_gpu=12),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=tmp_path / ".env",
        readiness_timeout_s=300,
        prune_logs_on_success=True,
    )

    entry = manifest.entries[0]
    script = Path(entry.script_path).read_text(encoding="utf-8")

    assert entry.gpus == 2
    assert entry.allocated_gpus == 2
    assert entry.tensor_parallel_size == 2
    assert entry.data_parallel_size == 1
    assert entry.vllm_world_size == 2
    assert entry.suite_checksum
    assert entry.bundled_eval_config_checksum
    assert "#SBATCH --gpus-per-task=2" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-gpu=12" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --qos=low" in script
    assert "#SBATCH --nice=500" in script
    assert "#SBATCH --signal=B:TERM@120" in script
    assert "#SBATCH --requeue" in script
    assert "trap forward_launch_signal TERM INT USR1" in script
    assert 'kill -TERM "$launch_pid"' in script
    assert "--runtime pyxis" in script
    assert "medarc-orchestrate launch" in script
    assert "--task-id" in script
    assert "-- medarc-eval bench" in script
    assert "--api-base-url http://127.0.0.1:" in script
    assert "--provider local" in script
    assert "worker" not in script
    assert "--task " not in script
    assert "--allocation" not in script
    assert "task.yaml" not in script
    assert "allocation.json" not in script
    assert "--no-uv-run" not in script
    assert "--resume" not in script
    assert f'ACTIVATE_SCRIPT="${{ACTIVATE_SCRIPT:-{tmp_path / ".venv" / "bin" / "activate"}}}"' in script
    assert 'source "$ACTIVATE_SCRIPT"' in script
    assert "Missing activation script:" in script
    assert "MEDARC_ALLOCATED_GPU_COUNT=2" in script
    assert "--mem" not in script
    assert str(suite_cfg) not in script

    bundled_payload = load_suite_config(Path(entry.generated_eval_config_path))
    assert bundled_payload["endpoint_id"] == "job"
    assert bundled_payload["endpoints_path"] == str((tmp_path / "endpoints.toml").resolve())
    assert bundled_payload["env_dir_path"] == str((tmp_path / "envs").resolve())
    assert "orchestrate" not in bundled_payload
    orchestrate_snapshot = tomllib.loads(
        (Path(entry.script_path).parent / "orchestrate-snapshot.toml").read_text(encoding="utf-8")
    )
    assert orchestrate_snapshot["registry_path"] == str((tmp_path / "endpoints.toml").resolve())
    assert orchestrate_snapshot["registry_checksum"]
    assert orchestrate_snapshot["model"]["model"].startswith("Foo/")
    assert not (Path(entry.script_path).parent / "task.yaml").exists()
    assert not (Path(entry.script_path).parent / "runtime" / "allocation.json").exists()
    assert Path(entry.script_path).name == "submit.sh"

    run_manifest = load_run_bundle_manifest(tmp_path / "outputs" / "run_manifest.json")
    assert run_manifest.tasks[0].bundled_eval_config_path == entry.generated_eval_config_path


def test_build_submission_plan_rejects_unsupported_slurm_signal(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg, slurm={"signal": "B:USR2@120"})
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))

    with pytest.raises(ValueError, match=r"slurm.signal must use B:TERM@<seconds>"):
        build_submission_plan(
            tasks,
            base_dependency=None,
            submission_options=SlurmSubmissionOptions(),
        )


def test_lifecycle_render_writes_cpu_scripts_manifest_and_symbolic_dependencies(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg, gpus=2, tensor_parallel_size=2)
    plan = load_plan(_write_lifecycle_plan(tmp_path, suite_cfg))
    tasks = expand_tasks(plan)
    planned = build_submission_plan(tasks, base_dependency="afterok:99", submission_options=SlurmSubmissionOptions())

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=tmp_path / ".env",
        readiness_timeout_s=None,
        prune_logs_on_success=False,
        construct=plan.prepare,
        teardown=plan.teardown,
    )

    assert len(manifest.entries) == 1
    assert len(manifest.lifecycle_entries) == 2
    entry = manifest.entries[0]
    prepare = manifest.lifecycle_entry_map()[(entry.task_id, "prepare")]
    teardown = manifest.lifecycle_entry_map()[(entry.task_id, "teardown")]
    assert prepare.cpus == 4
    assert prepare.base_dependency == "afterok:99"
    assert entry.base_dependency is None
    assert entry.generated_dependency == f"afterok:${{{entry.task_id}:prepare}}"
    assert teardown.generated_dependency == f"afterany:${{{entry.task_id}:eval}}"

    prepare_script = Path(prepare.script_path).read_text(encoding="utf-8")
    teardown_script = Path(teardown.script_path).read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=4" in prepare_script
    assert "#SBATCH --gpus-per-task" not in prepare_script
    assert "MEDARC_ALLOCATED_GPU_COUNT" not in prepare_script
    assert "medarc-orchestrate prepare" in prepare_script
    assert "--task" not in prepare_script
    assert "--allocation" not in prepare_script
    assert "--prefetch-model" in prepare_script
    assert "--materialize-image" in prepare_script
    assert "#SBATCH --cpus-per-task=2" in teardown_script
    assert "medarc-orchestrate teardown" in teardown_script
    assert "--task" not in teardown_script
    assert "--allocation" not in teardown_script
    assert "''" not in prepare_script
    assert "''" not in teardown_script

    assert "vllm/vllm-openai:v0.12.0" in prepare_script
    assert str(tmp_path / "images") in prepare_script
    assert str(tmp_path / "hf") in prepare_script
    submit_script = Path(entry.script_path).read_text(encoding="utf-8")
    assert "--container-image-source vllm/vllm-openai:v0.12.0" in submit_script
    assert f"--container-env-file {tmp_path / 'container.env'}" in submit_script
    assert f"--image-dir {tmp_path / 'images'}" in submit_script
    assert "--image-root" not in teardown_script
    assert "--prepare-result" not in teardown_script
    assert "--remove-image" not in teardown_script


def test_lifecycle_render_treats_absolute_sqsh_as_already_materialized(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    image_path = tmp_path / "prebuilt.sqsh"
    _write_suite_config(suite_cfg, gpus=1)
    plan = load_plan(_write_lifecycle_sqsh_plan(tmp_path, suite_cfg, image_path))
    tasks = expand_tasks(plan)
    planned = build_submission_plan(tasks, base_dependency=None, submission_options=SlurmSubmissionOptions())

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
        construct=plan.prepare,
        teardown=plan.teardown,
    )

    entry = manifest.entries[0]
    prepare = manifest.lifecycle_entry_map()[(entry.task_id, "prepare")]
    prepare_script = Path(prepare.script_path).read_text(encoding="utf-8")

    assert "--prefetch-model" in prepare_script
    assert "--materialize-image" not in prepare_script
    assert "--image " not in prepare_script
    assert "--image-output" not in prepare_script
    assert str(image_path) not in prepare_script
    submit_script = Path(entry.script_path).read_text(encoding="utf-8")
    assert "--container-image-source" not in submit_script
    assert "--image-dir" not in submit_script


def test_prepare_render_omits_image_flags_for_model_only_prepare(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg, gpus=1)
    plan = load_plan(_write_prepare_only_plan(tmp_path, suite_cfg, prefetch=True, materialize=False))
    planned = build_submission_plan(expand_tasks(plan), base_dependency=None, submission_options=SlurmSubmissionOptions())

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
        construct=plan.prepare,
        teardown=plan.teardown,
    )

    entry = manifest.entries[0]
    prepare = manifest.lifecycle_entry_map()[(entry.task_id, "prepare")]
    prepare_script = Path(prepare.script_path).read_text(encoding="utf-8")
    submit_script = Path(entry.script_path).read_text(encoding="utf-8")

    assert "--prefetch-model" in prepare_script
    assert "--model Foo/job" in prepare_script
    assert f"--hub-cache {tmp_path / 'hf' / 'hub'}" in prepare_script
    assert "--materialize-image" not in prepare_script
    assert "--image " not in prepare_script
    assert "--image-output" not in prepare_script
    assert "''" not in prepare_script
    assert "--container-image-source" not in submit_script
    assert "--image-dir" not in submit_script


def test_prepare_render_omits_model_flags_for_image_only_prepare(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg, gpus=1)
    plan = load_plan(_write_prepare_only_plan(tmp_path, suite_cfg, prefetch=False, materialize=True))
    planned = build_submission_plan(expand_tasks(plan), base_dependency=None, submission_options=SlurmSubmissionOptions())

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
        construct=plan.prepare,
        teardown=plan.teardown,
    )

    entry = manifest.entries[0]
    prepare = manifest.lifecycle_entry_map()[(entry.task_id, "prepare")]
    prepare_script = Path(prepare.script_path).read_text(encoding="utf-8")
    submit_script = Path(entry.script_path).read_text(encoding="utf-8")

    assert "--materialize-image" in prepare_script
    assert "--image vllm/vllm-openai:v0.12.0" in prepare_script
    assert f"--image-output {tmp_path / 'images'}" in prepare_script
    assert "--prefetch-model" not in prepare_script
    assert "--model " not in prepare_script
    assert "--hub-cache" not in prepare_script
    assert "''" not in prepare_script
    assert "--container-image-source vllm/vllm-openai:v0.12.0" in submit_script
    assert f"--image-dir {tmp_path / 'images'}" in submit_script


def test_lifecycle_dry_run_prints_phase_commands_in_dependency_order(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg)
    plan = load_plan(_write_lifecycle_plan(tmp_path, suite_cfg))
    planned = build_submission_plan(expand_tasks(plan), base_dependency=None, submission_options=SlurmSubmissionOptions())
    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
        construct=plan.prepare,
        teardown=plan.teardown,
    )

    commands = mark_dry_run(tmp_path / "outputs" / "slurm_manifest.json", manifest)

    assert [Path(command.split()[-1]).name for command in commands] == ["prepare.sh", "submit.sh", "teardown.sh"]
    assert "--dependency=afterok:" in commands[1]
    assert "--dependency=afterany:" in commands[2]


def test_lifecycle_fake_sbatch_threads_job_ids_into_dependencies(tmp_path: Path, monkeypatch) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg)
    plan = load_plan(_write_lifecycle_plan(tmp_path, suite_cfg))
    planned = build_submission_plan(expand_tasks(plan), base_dependency="afterok:base", submission_options=SlurmSubmissionOptions())
    manifest_path = tmp_path / "outputs" / "slurm_manifest.json"
    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
        construct=plan.prepare,
        teardown=plan.teardown,
    )
    calls: list[list[str]] = []
    job_ids = iter(["101", "102", "103"])

    class Completed:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(command, check, capture_output, text):
        del check, capture_output, text
        calls.append(list(command))
        return Completed(f"{next(job_ids)}\n")

    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.submit.subprocess.run", fake_run)

    submit_lifecycle_bundle(manifest_path, manifest)

    assert calls[0][-1].endswith("prepare.sh")
    assert "--dependency=afterok:base" in calls[0]
    assert calls[1][-1].endswith("submit.sh")
    assert "--dependency=afterok:101" in calls[1]
    assert calls[2][-1].endswith("teardown.sh")
    assert "--dependency=afterany:102" in calls[2]
    persisted = SlurmBundleManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    entry = persisted.entries[0]
    assert persisted.lifecycle_entry_map()[(entry.task_id, "prepare")].slurm_job_id == "101"
    assert entry.slurm_job_id == "102"
    assert persisted.lifecycle_entry_map()[(entry.task_id, "teardown")].slurm_job_id == "103"


def test_lifecycle_manifest_records_prepare_phase_fields(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg)
    plan = load_plan(_write_lifecycle_plan(tmp_path, suite_cfg))
    planned = build_submission_plan(expand_tasks(plan), base_dependency=None, submission_options=SlurmSubmissionOptions())
    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
        construct=plan.prepare,
        teardown=plan.teardown,
    )
    entry = manifest.entries[0]
    lifecycle = manifest.lifecycle_entry_map()
    lifecycle[(entry.task_id, "prepare")].state = "submitted"
    lifecycle[(entry.task_id, "prepare")].slurm_job_id = "201"
    entry.state = "submitted"
    entry.slurm_job_id = "202"
    lifecycle[(entry.task_id, "teardown")].state = "submitted"
    lifecycle[(entry.task_id, "teardown")].slurm_job_id = "203"

    assert lifecycle[(entry.task_id, "prepare")].state == "submitted"
    assert lifecycle[(entry.task_id, "prepare")].slurm_job_id == "201"
    assert entry.state == "submitted"
    assert entry.slurm_job_id == "202"
    assert lifecycle[(entry.task_id, "teardown")].state == "submitted"
    assert lifecycle[(entry.task_id, "teardown")].slurm_job_id == "203"


def test_bundle_parses_auxiliary_images(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))

    plan = ensure_run_bundle(
        tasks=tasks,
        output_root=tmp_path / "outputs",
        mode="slurm",
        runtime="pyxis",
    )

    spec = plan.tasks[tasks[0].task_id].runtime
    assert spec.output_paths.auxiliary_image_dir.endswith("/auxiliary_images")
    assert len(spec.auxiliary_images) == 1
    auxiliary_image = spec.auxiliary_images[0]
    assert auxiliary_image.name == "medagentbench-fhir"
    assert auxiliary_image.evals == []
    assert auxiliary_image.envs == ["job"]
    assert auxiliary_image.runtime == "pyxis"
    assert auxiliary_image.image == "/tmp/image with space.sqsh"
    assert auxiliary_image.srun_args == ["--mem=16G", "--container-env=JAVA_TOOL_OPTIONS"]
    assert auxiliary_image.env["JAVA_TOOL_OPTIONS"] == "-XX:+UseSerialGC -Xms256m -Xmx1024m"
    assert auxiliary_image.command == ["/usr/bin/java", "--class-path", "/app/main war"]
    assert auxiliary_image.readiness.url == "http://127.0.0.1:8080/fhir/metadata?x=a b"
    assert auxiliary_image.readiness.timeout_s == 12
    assert auxiliary_image.readiness.interval_s == 3
    assert "--overlap" in spec.pyxis_srun_extra_args
    eval_images_snapshot = tomllib.loads(
        (Path(spec.output_paths.root) / "eval_images-snapshot.toml").read_text(encoding="utf-8")
    )
    assert eval_images_snapshot["registry_path"] == str(tmp_path / "eval_images.toml")
    assert eval_images_snapshot["registry_checksum"]
    assert eval_images_snapshot["eval_image"][0]["id"] == "medagentbench-fhir"


def test_slurm_render_starts_auxiliary_image_before_launch(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))
    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")

    assert script.index("srun --overlap") < script.index("medarc-orchestrate launch")
    assert "'--container-image=/tmp/image with space.sqsh'" in script
    assert "/usr/bin/java --class-path '/app/main war'" in script
    assert "--pyxis-srun-arg=--overlap" in script
    launch_line = next(line for line in script.splitlines() if line.startswith("medarc-orchestrate launch "))
    parsed = build_launch_parser().parse_args(shlex.split(launch_line)[2:])
    assert parsed.pyxis_srun_arg == ["--overlap"]


def test_slurm_render_auxiliary_image_has_exit_trap(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))
    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")

    assert "cleanup_auxiliary_images() {" in script
    assert 'kill "$pid" 2>/dev/null || true' in script
    assert "trap cleanup_auxiliary_images EXIT" in script


def test_slurm_render_auxiliary_image_readiness_failure_tails_log(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))
    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")

    assert script.count('tail -100 "$AUX_IMAGE_LOG_MEDAGENTBENCH_FHIR" >&2 || true') == 2
    assert "record_auxiliary_image_failure auxiliary_image_exited_before_readiness" in script
    assert "record_auxiliary_image_failure auxiliary_image_readiness_timeout" in script


def test_auxiliary_image_validation_rejects_missing_image_or_command(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg, image="")

    with pytest.raises(ValueError, match="non-empty image"):
        expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))


def test_auxiliary_image_validation_rejects_non_slurm_mode(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))

    with pytest.raises(ValueError, match="only supported in slurm mode"):
        ensure_run_bundle(tasks=tasks, run_id="bundle", output_root=tmp_path / "outputs", mode="local", runtime="pyxis")


def test_auxiliary_image_validation_rejects_reserved_srun_args(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg)
    _AUX_IMAGE_META[suite_cfg.with_suffix(".toml").resolve()][0]["srun_args"].extend(["--nodes", "1"])
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))

    with pytest.raises(ValueError, match="renderer-owned flag --nodes"):
        ensure_run_bundle(tasks=tasks, run_id="bundle", output_root=tmp_path / "outputs", mode="slurm", runtime="pyxis")


def test_auxiliary_image_injection_rejects_variant_eval_selectors(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg)
    entry = _AUX_IMAGE_META[suite_cfg.with_suffix(".toml").resolve()][0]
    entry["evals"] = [f"{suite_cfg.with_suffix('.toml').stem}:variant-a"]
    entry["inject_env_args"] = {"fhir_api_base": "http://127.0.0.1:{port}/fhir/"}
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))

    with pytest.raises(ValueError, match="variant eval selectors"):
        ensure_run_bundle(tasks=tasks, run_id="bundle", output_root=tmp_path / "outputs", mode="slurm", runtime="pyxis")


def test_auxiliary_image_validation_rejects_shell_suffix_collision(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg, name="fhir-v2")
    first = dict(_AUX_IMAGE_META[suite_cfg.with_suffix(".toml").resolve()][0])
    second = dict(first)
    second.update(
        {
            "id": "fhir.v2",
            "image": "/tmp/other.sqsh",
            "command": ["/bin/sh", "-lc", "sleep 60"],
            "readiness": {"enabled": False},
        }
    )
    _AUX_IMAGE_META[suite_cfg.with_suffix(".toml").resolve()].append(second)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))

    with pytest.raises(ValueError, match="same shell variable suffix"):
        ensure_run_bundle(tasks=tasks, run_id="bundle", output_root=tmp_path / "outputs", mode="slurm", runtime="pyxis")


def test_slurm_render_uses_single_cleanup_trap_for_multiple_auxiliary_images(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg)
    first = dict(_AUX_IMAGE_META[suite_cfg.with_suffix(".toml").resolve()][0])
    audit = dict(first)
    audit.update(
        {
            "id": "audit",
            "image": "/tmp/audit.sqsh",
            "command": ["/bin/sh", "-lc", "sleep 60"],
            "readiness": {"enabled": False},
        }
    )
    _AUX_IMAGE_META[suite_cfg.with_suffix(".toml").resolve()].append(audit)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))
    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")

    assert script.count("cleanup_auxiliary_images() {") == 1
    assert script.count("trap cleanup_auxiliary_images EXIT") == 1
    assert script.count("srun --overlap") == 2
    assert script.count("until python3") == 1
    first_readiness_done = script.index("done", script.index("until python3"))
    assert first_readiness_done < script.index("--container-image=/tmp/audit.sqsh")


def test_slurm_render_shell_quotes_auxiliary_image_values(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))
    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs with space",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")

    assert "JAVA_TOOL_OPTIONS='-XX:+UseSerialGC -Xms256m -Xmx1024m' srun --overlap" in script
    assert "AUX_IMAGE_LOG_MEDAGENTBENCH_FHIR='" in script
    assert "'http://127.0.0.1:8080/fhir/metadata?x=a b'" in script
    assert "'/app/main war'" in script
    assert "export JAVA_TOOL_OPTIONS" not in script


def test_slurm_render_dynamic_aux_image_port_and_env_arg_injection(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg)
    entry = _AUX_IMAGE_META[suite_cfg.with_suffix(".toml").resolve()][0]
    entry["env_vars"]["SERVER_PORT"] = "{port}"
    entry["srun_args"] = ["--mem=16G", "--container-env=JAVA_TOOL_OPTIONS,SERVER_PORT"]
    entry["readiness"]["url"] = "http://127.0.0.1:{port}/fhir/metadata"
    entry["inject_env_args"] = {"fhir_api_base": "http://127.0.0.1:{port}/fhir/"}
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))
    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )

    plan = ensure_run_bundle(
        tasks=tasks,
        output_root=tmp_path / "outputs-bundle",
        mode="slurm",
        runtime="pyxis",
    )
    spec = plan.tasks[tasks[0].task_id].runtime
    assert spec.auxiliary_images[0].inject_env_args == {"fhir_api_base": "http://127.0.0.1:{port}/fhir/"}

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")

    assert "#SBATCH --exclusive" not in script
    assert 'export AUX_IMAGE_PORT_MEDAGENTBENCH_FHIR="$(allocate_auxiliary_image_port)"' in script
    assert "--gpus=0" in script
    assert "SERVER_PORT=\"${AUX_IMAGE_PORT_MEDAGENTBENCH_FHIR}\"" in script
    assert 'http://127.0.0.1:"${AUX_IMAGE_PORT_MEDAGENTBENCH_FHIR}"/fhir/metadata' in script
    assert "env_args[key] = str(template).replace('{port}', port)" in script
    assert "fhir_api_base" in script
    assert "env_match = bool(envs and env_id in envs)" in script


def test_auxiliary_image_failures_are_recorded_directly_in_shell(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_auxiliary_image_suite_config(suite_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))
    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )
    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")

    assert "record_auxiliary_image_failure() {" in script
    assert "medarc-orchestrate record-failure" not in script
    assert "runtime/state.json" in script
    assert "runtime/result.json" in script
    assert "failure_reason" in script
    assert "--task-spec" not in script
    assert "--allocation" not in script


def test_run_bundle_no_longer_writes_task_spec_execution_input(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))
    plan = ensure_run_bundle(
        tasks=tasks,
        output_root=tmp_path / "outputs",
        mode="slurm",
        runtime="pyxis",
    )
    task_bundle = plan.tasks[tasks[0].task_id]

    assert not (task_bundle.paths.root / "task.yaml").exists()
    assert not (task_bundle.paths.runtime_dir / "allocation.json").exists()
    assert task_bundle.paths.eval_config_path.exists()
    assert task_bundle.paths.state_path.exists()


def test_render_bundle_assigns_unique_ports_per_task(tmp_path: Path) -> None:
    job_a = tmp_path / "job-a.yaml"
    job_b = tmp_path / "job-b.yaml"
    _write_suite_config(job_a, gpus=1)
    _write_suite_config(job_b, gpus=1)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_a, job_b])))
    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )

    scripts = [Path(entry.script_path).read_text(encoding="utf-8") for entry in manifest.entries]
    ports = [
        int(match.group(1))
        for script in scripts
        for match in [re.search(r"--host-port (\d+)", script)]
        if match is not None
    ]

    assert len(ports) == 2
    assert len(set(ports)) == 2
    assert all(port is not None and 8000 <= port <= 65000 for port in ports)


def test_render_bundle_refreshes_stale_task_bundle_when_source_changes(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg, gpus=1)
    _write_plan(tmp_path, [suite_cfg])
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))
    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )

    first_manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )
    first_entry = first_manifest.entries[0]
    first_config_path = Path(first_entry.generated_eval_config_path)
    first_payload = load_suite_config(first_config_path)

    assert "orchestrate" not in first_payload

    suite_cfg.with_suffix(".toml").write_text(
        suite_cfg.with_suffix(".toml").read_text(encoding="utf-8") + '\nvariant_id = "changed"\n', encoding="utf-8"
    )

    updated_tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))
    updated_planned = build_submission_plan(
        updated_tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )
    second_manifest = render_bundle(
        planned_tasks=updated_planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
        existing_manifest=first_manifest,
    )
    second_entry = second_manifest.entries[0]
    second_payload = load_suite_config(Path(second_entry.generated_eval_config_path))

    assert second_entry.generated_eval_config_path == first_entry.generated_eval_config_path
    assert "orchestrate" not in second_payload
    assert first_payload != second_payload


def test_rendered_task_local_config_is_loadable_by_orchestrator(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg, gpus=2, tensor_parallel_size=2)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [suite_cfg])))
    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )

    patched_payload = load_suite_config(Path(manifest.entries[0].generated_eval_config_path))

    assert patched_payload["endpoint_id"] == "job"
    assert patched_payload["output_dir"].endswith("/bench")


def test_render_bundle_uses_plan_output_dir_for_bench_results(tmp_path: Path) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg)
    plan_path = _write_plan(tmp_path, [suite_cfg])
    result_root = tmp_path / "durable-results"
    plan_path.write_text(
        plan_path.read_text(encoding="utf-8").replace(
            'endpoints_path = "endpoints.toml"',
            f'endpoints_path = "endpoints.toml"\noutput_dir = "{result_root}"',
            1,
        ),
        encoding="utf-8",
    )
    planned = build_submission_plan(
        expand_tasks(load_plan(plan_path)),
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )

    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "outputs",
        source_dir=tmp_path,
        activate_script=tmp_path / ".venv" / "bin" / "activate",
        env_file=None,
        readiness_timeout_s=None,
        prune_logs_on_success=False,
    )

    payload = load_suite_config(Path(manifest.entries[0].generated_eval_config_path))
    script = Path(manifest.entries[0].script_path).read_text(encoding="utf-8")
    assert payload["output_dir"] == str(result_root)
    assert f"--output-dir {result_root}" in script


def test_slurm_dry_run_writes_manifest_and_prints_commands(tmp_path: Path, capsys) -> None:
    job_a = tmp_path / "job-a.yaml"
    job_b = tmp_path / "job-b.yaml"
    _write_suite_config(job_a, gpus=4, tensor_parallel_size=4)
    _write_suite_config(job_b, gpus=1)
    plan_path = _write_plan(tmp_path, [job_a, job_b])

    rc = orchestrate_main(
        [
            "run",
            "--plan",
            str(plan_path),
            "--run-id",
            "bundle",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--dry-run",
        ]
    )

    assert rc == 0
    stdout_lines = capsys.readouterr().out.strip().splitlines()
    assert len(stdout_lines) == 2
    assert stdout_lines[0].startswith("sbatch --parsable ")
    assert stdout_lines[1].startswith("sbatch --parsable ")
    assert all("--dependency=afterany:$JOBID_" not in line for line in stdout_lines)

    manifest = json.loads((tmp_path / "bundle" / "slurm_manifest.json").read_text())
    assert [entry["state"] for entry in manifest["entries"]] == ["dry-run", "dry-run"]
    assert all(entry["slurm_job_id"] is None for entry in manifest["entries"])
    assert all(entry["account"] is None for entry in manifest["entries"])
    for entry in manifest["entries"]:
        script = Path(entry["script_path"]).read_text(encoding="utf-8")
        assert "#SBATCH --nice" not in script

    run_manifest = json.loads((tmp_path / "bundle" / "run_manifest.json").read_text())
    assert len(run_manifest["tasks"]) == 2


def test_slurm_cli_default_run_id_uses_shared_generator(tmp_path: Path, monkeypatch) -> None:
    suite_cfg = tmp_path / "job.yaml"
    _write_suite_config(suite_cfg, gpus=1)
    plan_path = _write_plan(tmp_path, [suite_cfg])

    captured: dict[str, object] = {}

    def fake_render_bundle(**kwargs):
        captured["run_id"] = kwargs["run_id"]
        captured["bundle_root"] = kwargs["bundle_root"]
        return SlurmBundleManifest(run_id=kwargs["run_id"], bundle_root=str(kwargs["bundle_root"]), entries=[])

    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.generate_run_id", lambda name: "shared-run-id")
    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.submit.render_bundle", fake_render_bundle)
    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.submit.write_bundle_manifest", lambda path, manifest: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.submit.mark_dry_run", lambda path, manifest: [])

    rc = orchestrate_main(
        [
            "run",
            "--plan",
            str(plan_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert captured["run_id"] == "shared-run-id"
    assert captured["bundle_root"] == (Path("outputs") / "orchestrate" / "shared-run-id").resolve()


def test_submit_bundle_resumes_from_existing_job_ids(tmp_path: Path, monkeypatch) -> None:
    job_a = tmp_path / "job-a.yaml"
    job_b = tmp_path / "job-b.yaml"
    job_c = tmp_path / "job-c.yaml"
    _write_suite_config(job_a, gpus=4, tensor_parallel_size=4)
    _write_suite_config(job_b, gpus=2, tensor_parallel_size=2)
    _write_suite_config(job_c, gpus=1)
    tasks = expand_tasks(load_plan(_write_plan(tmp_path, [job_a, job_b, job_c])))
    planned = build_submission_plan(
        tasks,
        base_dependency=None,
        submission_options=SlurmSubmissionOptions(),
    )
    manifest = render_bundle(
        planned_tasks=planned,
        bundle_root=tmp_path / "bundle",
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

    submit_bundle(tmp_path / "bundle" / "slurm_manifest.json", manifest)

    assert calls[0][0:2] == ["sbatch", "--parsable"]
    assert all(not any(arg.startswith("--dependency=afterany:") for arg in call) for call in calls)
    assert [entry.slurm_job_id for entry in manifest.entries] == ["101", "202", "303"]
