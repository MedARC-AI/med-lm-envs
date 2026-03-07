import json
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.bundle import load_run_bundle_manifest, load_runtime_state
from medarc_verifiers.orchestrate.cli import main as orchestrate_main
from medarc_verifiers.orchestrate.config import TaskSpec, expand_tasks, load_job_config, load_plan
from medarc_verifiers.orchestrate.slurm.manifest import SlurmBundleManifest
from medarc_verifiers.orchestrate.slurm.cli import build_parser as build_slurm_parser
from medarc_verifiers.orchestrate.slurm.plan import DEFAULT_SLURM_ACCOUNT, SlurmCliOverrides, build_submission_plan
from medarc_verifiers.orchestrate.slurm.render import render_bundle
from medarc_verifiers.orchestrate.slurm.submit import submit_bundle


def _write_job_config(
    path: Path,
    *,
    model_id: str = "Foo/Bar",
    gpus: int = 1,
    tensor_parallel_size: int | None = None,
    restart: str | None = None,
    slurm: dict[str, object] | None = None,
) -> None:
    tp_block = ""
    if tensor_parallel_size is not None:
        tp_block = f"    tensor_parallel_size: {tensor_parallel_size}\n"
    restart_block = ""
    if restart is not None:
        restart_block = f"  restart: {restart}\n"
    slurm_block = ""
    if slurm:
        body = "\n".join(f"  {key}: {value}" for key, value in slurm.items())
        slurm_block = f"slurm:\n{body}\n"
    path.write_text(
        (
            "models:\n"
            "  foo:\n"
            f"    model: {model_id}\n"
            "orchestrate:\n"
            f"{restart_block}"
            "  vllm-container:\n"
            "    image: fake\n"
            "  foo:\n"
            f"    gpus: {gpus}\n"
            f"{tp_block}"
            "    serve: {}\n"
            f"{slurm_block}"
        ),
        encoding="utf-8",
    )


def _task(tmp_path: Path, name: str, *, gpus: int, tp: int | None = None, slurm: dict[str, object] | None = None) -> TaskSpec:
    job_cfg = tmp_path / f"{name}.yaml"
    _write_job_config(job_cfg, gpus=gpus, tensor_parallel_size=tp, slurm=slurm)
    return expand_tasks(load_plan(_write_plan(tmp_path, [job_cfg])))[0]


def _write_plan(tmp_path: Path, job_configs: list[Path]) -> Path:
    plan_path = tmp_path / "plan.yaml"
    config_lines = "\n".join(f"  - {path.name}" for path in job_configs)
    plan_path.write_text(f"job_configs:\n{config_lines}\n", encoding="utf-8")
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
        _task(tmp_path, "odd", gpus=3, tp=3),
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

    assert [task.task.task_id for task in planned] == ["large:foo", "odd:foo", "medium:foo", "small:foo"]
    assert [(task.tp_size, task.dp_size, task.effective_gpus) for task in planned] == [
        (8, 1, 8),
        (3, 2, 6),
        (2, 4, 8),
        (1, 8, 8),
    ]
    assert planned[1].predecessor_task_id == "large:foo"
    assert all(task.options.account == DEFAULT_SLURM_ACCOUNT for task in planned)


def test_build_submission_plan_round_robins_two_chains(tmp_path: Path) -> None:
    tasks = [
        _task(tmp_path, "a", gpus=4, tp=4),
        _task(tmp_path, "b", gpus=3, tp=3),
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

    assert [(task.task.task_id, task.chain_index, task.predecessor_task_id, task.base_dependency) for task in planned] == [
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

    assert [(task.task.task_id, task.chain_index, task.predecessor_task_id, task.base_dependency) for task in planned] == [
        ("a:foo", 0, None, "afterok:555"),
        ("b:foo", 1, None, "afterok:555"),
    ]


def test_build_submission_plan_rejects_tp_larger_than_node_gpus(tmp_path: Path) -> None:
    tasks = [_task(tmp_path, "too-wide", gpus=16, tp=16)]

    with pytest.raises(ValueError, match=r"tensor_parallel_size=16.*exceeds node_gpus=8"):
        build_submission_plan(
            tasks,
            run_id="bundle",
            node_gpus=8,
            max_simultaneous_nodes=1,
            run_simultaneously=False,
            base_dependency=None,
            cli_overrides=SlurmCliOverrides(),
        )


def test_render_bundle_writes_script_and_bundled_config(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(job_cfg, gpus=2, tensor_parallel_size=2, slurm={"partition": "gpu", "slurm_resume": True})
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

    assert entry.dp_size == 4
    assert entry.effective_gpus == 8
    assert entry.original_job_config_checksum
    assert entry.bundled_eval_config_checksum
    assert entry.task_spec_checksum
    assert "#SBATCH --gpus-per-task=8" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-gpu=12" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --requeue" in script
    assert "--runtime pyxis" in script
    assert "--resume" in script
    assert f'ACTIVATE_SCRIPT="${{ACTIVATE_SCRIPT:-{tmp_path / ".venv" / "bin" / "activate"}}}"' in script
    assert 'source "$ACTIVATE_SCRIPT"' in script
    assert "Missing activation script:" in script
    assert "MEDARC_ALLOCATED_GPU_COUNT=8" in script
    assert "--mem" not in script
    assert str(job_cfg) not in script

    bundled_payload = load_job_config(Path(entry.effective_job_config_path))
    assert bundled_payload["orchestrate"]["foo"]["gpus"] == 8
    assert bundled_payload["orchestrate"]["foo"]["data_parallel_size"] == 4
    assert "restart" not in bundled_payload["orchestrate"]
    assert Path(entry.task_spec_path).name == "task.yaml"
    assert Path(entry.script_path).name == "submit.sh"

    run_manifest = load_run_bundle_manifest(tmp_path / "outputs" / "run_manifest.json")
    assert run_manifest.tasks[0].bundled_eval_config_path == entry.effective_job_config_path


def test_render_bundle_refreshes_stale_task_bundle_when_source_changes(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(job_cfg, gpus=1)
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

    assert "restart" not in first_payload["orchestrate"]

    job_cfg.write_text(
        """
models:
  foo:
    model: Foo/Bar
orchestrate:
  vllm-container:
    image: fake
  foo:
    gpus: 1
    serve:
      dtype: float16
""".lstrip(),
        encoding="utf-8",
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
    assert "restart" not in second_payload["orchestrate"]
    assert first_payload["orchestrate"]["foo"].get("serve", {}) == {}
    assert second_payload["orchestrate"]["foo"].get("serve", {}) == {"dtype": "float16"}


def test_render_bundle_prefers_runtime_state_restart_on_rerender(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(job_cfg, gpus=1, restart="runs/raw/source-run")
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
    state_path = Path(first_entry.state_path)
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    state_payload["restart_source"] = "runs/raw/runtime-run"
    state_payload["restart_source_strategy"] = "runtime_state"
    state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")

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
    runtime_state = load_runtime_state(Path(second_entry.state_path))

    assert second_payload["orchestrate"]["restart"] == "runs/raw/source-run"
    assert runtime_state is not None
    assert runtime_state.restart_source == "runs/raw/runtime-run"
    assert second_entry.restart_source == "runs/raw/runtime-run"


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
    assert patched_tasks[0].orchestrate["foo"]["gpus"] == 8
    assert patched_tasks[0].orchestrate["foo"]["data_parallel_size"] == 4


def test_slurm_dry_run_writes_manifest_and_prints_commands(tmp_path: Path, capsys) -> None:
    job_a = tmp_path / "job-a.yaml"
    job_b = tmp_path / "job-b.yaml"
    _write_job_config(job_a, gpus=4, tensor_parallel_size=4)
    _write_job_config(job_b, gpus=1)

    rc = orchestrate_main(
        [
            "slurm",
            "--job-config",
            str(job_a),
            "--job-config",
            str(job_b),
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


def test_slurm_cli_default_run_id_uses_shared_generator(tmp_path: Path, monkeypatch) -> None:
    job_cfg = tmp_path / "job.yaml"
    _write_job_config(job_cfg, gpus=1)

    captured: dict[str, object] = {}

    def fake_render_bundle(**kwargs):
        captured["run_id"] = kwargs["run_id"]
        captured["bundle_root"] = kwargs["bundle_root"]
        return SlurmBundleManifest(run_id=kwargs["run_id"], bundle_root=str(kwargs["bundle_root"]), node_gpus=8, entries=[])

    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.cli.generate_run_id", lambda name: "shared-run-id")
    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.cli.render_bundle", fake_render_bundle)
    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.cli.write_bundle_manifest", lambda path, manifest: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.slurm.cli.mark_dry_run", lambda path, manifest: [])

    rc = orchestrate_main(["slurm", "--job-config", str(job_cfg), "--dry-run"])

    assert rc == 0
    assert captured["run_id"] == "shared-run-id"
    assert captured["bundle_root"] == (Path("outputs") / "orchestrate" / "shared-run-id").resolve()


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
