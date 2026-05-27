"""CLI entrypoint for the vLLM orchestrator."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

from medarc_verifiers.orchestrate.docker_vllm import cleanup_orphan_containers as cleanup_docker_orphans
from medarc_verifiers.orchestrate.launch import (
    LaunchRequest,
    resolve_launch_plan,
    resolve_status_target,
)
from medarc_verifiers.orchestrate.podman_vllm import cleanup_orphan_containers as cleanup_podman_orphans
from medarc_verifiers.orchestrate.slurm.submit import SlurmSubmissionOptions, submit_slurm_launch_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medarc-orchestrate",
        description="Run Slurm/Pyxis vLLM orchestration.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{run,prepare,launch,teardown,status,cleanup}")
    subparsers.required = True

    run_parser = subparsers.add_parser("run", description="Submit Slurm orchestration jobs.")
    _add_source_arguments(run_parser)
    run_parser.add_argument("--dry-run", action="store_true", help="Render bundle and print sbatch commands.")
    run_parser.add_argument("--run-id", help="Run identifier.")
    run_parser.add_argument("--bundle-dir", type=Path, help="Override orchestrator bundle directory.")
    run_parser.add_argument("--output-dir", type=Path, help="Override medarc-eval bench result directory.")
    run_parser.add_argument("--readiness-timeout-s", type=int, default=None, help="Readiness timeout in seconds.")
    run_parser.add_argument("--prune-logs-on-success", action="store_true")
    run_parser.add_argument("--eval-images-config", type=Path, help="Path to eval auxiliary image registry TOML.")
    run_parser.add_argument("--endpoints-path", type=Path, help="Path to endpoint registry TOML.")
    _add_slurm_arguments(run_parser)
    run_parser.set_defaults(handler=_run_launch)

    prepare_parser = subparsers.add_parser(
        "prepare", description="Prepare model weights and/or materialized images."
    )
    prepare_parser.set_defaults(handler=lambda _args: main(["prepare", "--help"]))

    launch_parser = subparsers.add_parser("launch", description="Start vLLM and run the benchmark command after --.")
    launch_parser.set_defaults(handler=lambda _args: main(["launch", "--help"]))

    teardown_parser = subparsers.add_parser("teardown", description="Remove explicit prepared cache artifacts.")
    teardown_parser.set_defaults(handler=lambda _args: main(["teardown", "--help"]))

    status_parser = subparsers.add_parser("status", description="Print orchestrator status artifacts.")
    status_parser.add_argument("--run-id", help="Run identifier under outputs/orchestrate.")
    status_parser.add_argument("--bundle-dir", type=Path, help="Run bundle directory.")
    status_parser.add_argument("--json", action="store_true", help="Print combined status JSON.")
    status_parser.set_defaults(handler=_run_status)

    cleanup_parser = subparsers.add_parser("cleanup", description="Clean local runtime leftovers from tests/dev.")
    cleanup_parser.add_argument("--runtime", choices=("docker", "podman"), required=True)
    cleanup_parser.add_argument("--run-id", help="Only clean containers for this run id.")
    cleanup_parser.set_defaults(handler=_run_cleanup)
    return parser


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--plan", type=Path, help="Path to orchestrator plan file.")
    source.add_argument("--suite", type=Path, help="Eval suite TOML accepted by medarc-eval bench.")
    parser.add_argument("--endpoint", action="append", default=[], help="Endpoint id target for --suite shorthand.")
    parser.add_argument("--name", default=None, help="Optional bundle name when using --suite directly.")
    parser.add_argument("--env-file", type=Path, default=None, help="Dotenv file shared by runtime launches.")


def _add_slurm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cpus-per-gpu", type=int, default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--nice", type=int, default=None)
    parser.add_argument("--dependency", default=None, help="Base sbatch dependency applied to each chain head.")
    parser.add_argument("--mail-type", default=None)
    parser.add_argument("--mail-user", default=None)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--slurm-resume", action="store_true", default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--activate-script", type=Path, default=None)


def _run_launch(args: argparse.Namespace) -> int:
    _require_source(args)
    request = LaunchRequest(
        plan=args.plan,
        suite=args.suite,
        endpoints=tuple(args.endpoint or ()),
        name=args.name,
        env_file=args.env_file,
        run_id=args.run_id,
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        readiness_timeout_s=args.readiness_timeout_s,
        prune_logs_on_success=bool(args.prune_logs_on_success),
        eval_images_config=args.eval_images_config,
        endpoints_path=args.endpoints_path,
    )
    launch = resolve_launch_plan(request, cwd=Path.cwd())
    source_dir = (args.source_dir or Path.cwd()).expanduser().resolve()
    activate_script = (
        args.activate_script.expanduser() if args.activate_script is not None else source_dir / ".venv/bin/activate"
    )
    if not activate_script.is_absolute():
        activate_script = source_dir / activate_script
    options = SlurmSubmissionOptions(
        base_dependency=args.dependency,
        test_only=bool(args.test_only),
        dry_run=bool(args.dry_run),
        source_dir=source_dir,
        activate_script=activate_script.resolve(),
        cpus_per_gpu=args.cpus_per_gpu,
        time=args.time,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        nice=args.nice,
        mail_type=args.mail_type,
        mail_user=args.mail_user,
        slurm_resume=args.slurm_resume,
    )
    return submit_slurm_launch_plan(launch, options)


def _run_status(args: argparse.Namespace) -> int:
    target = resolve_status_target(run_id=args.run_id, bundle_dir=args.bundle_dir, cwd=Path.cwd())
    payload = _load_combined_status(target.output_root)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for row in payload["tasks"]:
        print(
            "\t".join(
                str(row.get(field) or "")
                for field in (
                    "task_id",
                    "submit_state",
                    "runtime_state",
                    "slurm_job_id",
                    "endpoint_id",
                    "model_id",
                    "suite",
                    "failure_reason",
                    "error",
                    "prepare_state",
                    "prepare_slurm_job_id",
                    "eval_state",
                    "eval_slurm_job_id",
                    "teardown_state",
                    "teardown_slurm_job_id",
                    "eval_slurm_live_state",
                    "eval_slurm_reason",
                    "eval_slurm_restarts",
                    "eval_slurm_preemptions",
                    "eval_slurm_elapsed",
                )
            )
        )
    return 0


def _run_cleanup(args: argparse.Namespace) -> int:
    removed = _cleanup_orphans(runtime=args.runtime, run_id=args.run_id)
    if removed:
        print("\n".join(removed))
    return 0


def _load_combined_status(output_root: Path) -> dict[str, object]:
    manifest_path = output_root / "slurm_manifest.json"
    summary_path = output_root / "summary.json"
    manifest = _load_json_artifact(manifest_path) if manifest_path.exists() else None
    summary = _load_json_artifact(summary_path) if summary_path.exists() else None
    if manifest is None:
        raise SystemExit(f"No orchestrator status found at {output_root}: missing slurm_manifest.json.")
    rows: dict[str, dict[str, object]] = {}
    if isinstance(manifest, dict):
        for entry in manifest.get("entries", []) or []:
            if not isinstance(entry, dict):
                continue
            task_id = str(entry.get("task_id") or "")
            row = rows.setdefault(task_id, {"task_id": task_id})
            row.update(
                {
                    "submit_state": entry.get("state"),
                    "eval_state": entry.get("state"),
                    "slurm_job_id": entry.get("slurm_job_id"),
                    "eval_slurm_job_id": entry.get("slurm_job_id"),
                    "dependency": entry.get("generated_dependency") or entry.get("base_dependency"),
                    "suite": entry.get("suite_path"),
                    "endpoint_id": entry.get("target_endpoint_id"),
                }
            )
        for entry in manifest.get("lifecycle_entries", []) or []:
            if not isinstance(entry, dict):
                continue
            task_id = str(entry.get("task_id") or "")
            phase = str(entry.get("phase") or "")
            if phase not in {"prepare", "teardown"}:
                continue
            row = rows.setdefault(task_id, {"task_id": task_id})
            row[f"{phase}_state"] = entry.get("state")
            row[f"{phase}_slurm_job_id"] = entry.get("slurm_job_id")
            row[f"{phase}_dependency"] = entry.get("generated_dependency") or entry.get("base_dependency")
    if isinstance(summary, dict):
        for entry in summary.get("tasks", []) or []:
            if not isinstance(entry, dict):
                continue
            task_id = str(entry.get("task_id") or "")
            row = rows.setdefault(task_id, {"task_id": task_id})
            row.update(
                {
                    "runtime_state": entry.get("state"),
                    "model_id": entry.get("model_id"),
                    "failure_reason": entry.get("failure_reason"),
                    "error": entry.get("error"),
                }
            )
    slurm_status = _collect_slurm_status(rows.values())
    _merge_slurm_status(rows.values(), slurm_status)
    return {
        "output_root": str(output_root),
        "slurm_manifest": manifest,
        "summary": summary,
        "slurm": slurm_status,
        "tasks": [rows[key] for key in sorted(rows)],
    }


def _collect_slurm_status(rows: object) -> dict[str, object]:
    job_ids = _collect_status_job_ids(rows)
    status: dict[str, object] = {
        "queried_at": dt.datetime.now(dt.UTC).isoformat(),
        "available": True,
        "commands": [],
        "jobs": {},
    }
    if not job_ids:
        return status

    jobs: dict[str, dict[str, object]] = {}
    command = [
        "squeue",
        "-h",
        "-j",
        ",".join(job_ids),
        "-o",
        "%i|%T|%R|%M|%l|%D|%C|%b|%q|%a|%y|%N",
    ]
    status["commands"].append(" ".join(command))
    squeue_result = _run_status_command(command)
    if squeue_result["ok"]:
        for line in str(squeue_result["stdout"]).splitlines():
            fields = line.split("|")
            if len(fields) != 12:
                continue
            job_id, state, reason, elapsed, time_limit, nodes, cpus, tres, qos, account, nice, node_list = fields
            jobs.setdefault(job_id, {}).update(
                {
                    "job_id": job_id,
                    "live_state": state,
                    "reason": reason,
                    "elapsed": elapsed,
                    "time_limit": time_limit,
                    "nodes": nodes,
                    "cpus": cpus,
                    "tres": tres,
                    "qos": qos,
                    "account": account,
                    "nice": nice,
                    "node_list": node_list,
                }
            )
    else:
        status["available"] = False
        status["squeue_error"] = squeue_result["error"]

    command = [
        "sacct",
        "-j",
        ",".join(job_ids),
        "--duplicates",
        "--format=JobID,JobName%80,State,ExitCode,Elapsed,Submit,Start,End",
        "-P",
    ]
    status["commands"].append(" ".join(command))
    sacct_result = _run_status_command(command)
    if sacct_result["ok"]:
        _merge_sacct_rows(jobs, str(sacct_result["stdout"]))
    else:
        status["sacct_error"] = sacct_result["error"]

    live_job_ids = [job_id for job_id in job_ids if "live_state" in jobs.get(job_id, {})]
    for job_id in live_job_ids:
        command = ["scontrol", "show", "job", job_id]
        status["commands"].append(" ".join(command))
        scontrol_result = _run_status_command(command)
        if not scontrol_result["ok"]:
            jobs.setdefault(job_id, {"job_id": job_id})["scontrol_error"] = scontrol_result["error"]
            continue
        fields = _parse_scontrol_fields(str(scontrol_result["stdout"]))
        if fields:
            jobs.setdefault(job_id, {"job_id": job_id}).update(
                {
                    "live_state": fields.get("JobState") or jobs.get(job_id, {}).get("live_state"),
                    "reason": fields.get("Reason") or jobs.get(job_id, {}).get("reason"),
                    "restarts": _parse_int(fields.get("Restarts")),
                    "requeue": _parse_int(fields.get("Requeue")),
                    "priority": _parse_int(fields.get("Priority")),
                    "submit_time": fields.get("SubmitTime"),
                    "eligible_time": fields.get("EligibleTime"),
                    "start_time": fields.get("StartTime"),
                    "end_time": fields.get("EndTime"),
                    "node_list": fields.get("NodeList") or jobs.get(job_id, {}).get("node_list"),
                    "scheduled_node_list": fields.get("SchedNodeList"),
                }
            )
    for job in jobs.values():
        attempts = job.get("attempts")
        if isinstance(attempts, list):
            job["preemptions"] = sum(1 for attempt in attempts if attempt.get("state") == "PREEMPTED")
    status["jobs"] = jobs
    return status


def _collect_status_job_ids(rows: object) -> list[str]:
    job_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("prepare_slurm_job_id", "eval_slurm_job_id", "teardown_slurm_job_id", "slurm_job_id"):
            value = row.get(key)
            if value is None:
                continue
            text = str(value)
            if text:
                job_ids.add(text)
    return sorted(job_ids)


def _run_status_command(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    except FileNotFoundError as exc:
        return {"ok": False, "stdout": "", "stderr": "", "error": str(exc)}
    except subprocess.SubprocessError as exc:
        return {"ok": False, "stdout": "", "stderr": "", "error": str(exc)}
    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.stderr.strip() or f"exit code {result.returncode}",
    }


def _merge_sacct_rows(jobs: dict[str, dict[str, object]], stdout: str) -> None:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if lines and lines[0].startswith("JobID|"):
        lines = lines[1:]
    for line in lines:
        fields = line.split("|")
        if len(fields) != 8:
            continue
        job_id, job_name, state, exit_code, elapsed, submit, start, end = fields
        root_job_id = job_id.split(".", 1)[0]
        if job_id != root_job_id:
            continue
        job = jobs.setdefault(root_job_id, {"job_id": root_job_id})
        attempt = {
            "job_id": job_id,
            "job_name": job_name,
            "state": state,
            "exit_code": exit_code,
            "elapsed": elapsed,
            "submit": submit,
            "start": start,
            "end": end,
        }
        job.setdefault("attempts", []).append(attempt)
        if state not in {"PENDING", "RUNNING"} or "accounting_state" not in job:
            job["accounting_state"] = state
            job["exit_code"] = exit_code
            job["accounting_elapsed"] = elapsed


def _parse_scontrol_fields(stdout: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=(\S+)", stdout))


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _merge_slurm_status(rows: object, slurm_status: dict[str, object]) -> None:
    jobs = slurm_status.get("jobs")
    if not isinstance(jobs, dict):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        for phase in ("prepare", "eval", "teardown"):
            job_id = row.get(f"{phase}_slurm_job_id")
            if job_id is None:
                continue
            job = jobs.get(str(job_id))
            if not isinstance(job, dict):
                continue
            row[f"{phase}_slurm_live_state"] = job.get("live_state") or job.get("accounting_state")
            row[f"{phase}_slurm_reason"] = job.get("reason")
            row[f"{phase}_slurm_restarts"] = job.get("restarts")
            row[f"{phase}_slurm_preemptions"] = job.get("preemptions")
            row[f"{phase}_slurm_elapsed"] = job.get("elapsed") or job.get("accounting_elapsed")


def _load_json_artifact(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Malformed orchestrator status artifact: {path}") from exc


def _require_source(args: argparse.Namespace) -> None:
    if args.plan is None and args.suite is None:
        raise SystemExit("medarc-orchestrate run requires --plan or --suite with at least one --endpoint.")
    if args.plan is not None and args.endpoint:
        raise SystemExit("medarc-orchestrate run --endpoint is only valid with --suite shorthand, not --plan.")
    if args.suite is not None and not args.endpoint:
        raise SystemExit("medarc-orchestrate run --suite requires at least one --endpoint.")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "launch":
        from medarc_verifiers.orchestrate.worker import main as worker_main

        return worker_main(argv[1:])
    if argv and argv[0] == "prepare":
        from medarc_verifiers.orchestrate.lifecycle import build_prepare_parser, run_prepare

        args = build_prepare_parser().parse_args(argv[1:])
        return run_prepare(
            model=args.model,
            result_path=args.result.expanduser().resolve() if args.result is not None else None,
            env_file=args.env_file.expanduser().resolve() if args.env_file is not None else None,
            hf_home=args.hf_home.expanduser().resolve() if args.hf_home is not None else None,
            hub_cache=args.hub_cache.expanduser().resolve() if args.hub_cache is not None else None,
            image=args.image,
            image_dir=args.image_dir.expanduser().resolve() if args.image_dir is not None else None,
            image_output=args.image_output.expanduser().resolve() if args.image_output is not None else None,
            latest_link=bool(args.latest_link),
            prefetch_model_flag=bool(args.prefetch_model) if (args.prefetch_model or args.materialize_image) else None,
            materialize_image_flag=(
                bool(args.materialize_image) if (args.prefetch_model or args.materialize_image) else None
            ),
        )
    if argv and argv[0] == "teardown":
        from medarc_verifiers.orchestrate.lifecycle import build_teardown_parser, run_teardown

        args = build_teardown_parser().parse_args(argv[1:])
        return run_teardown(
            result_path=args.result.expanduser().resolve(),
            model=args.model,
            env_file=args.env_file.expanduser().resolve() if args.env_file is not None else None,
            hub_cache=args.hub_cache.expanduser().resolve() if args.hub_cache is not None else None,
            remove_model_weights=bool(args.remove_model_weights),
            remove_image=args.remove_image.expanduser().resolve() if args.remove_image is not None else None,
            image_root=args.image_root.expanduser().resolve() if args.image_root is not None else None,
            prepare_result=args.prepare_result.expanduser().resolve() if args.prepare_result is not None else None,
        )
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


__all__ = ["build_parser", "main"]


def _cleanup_orphans(*, runtime: str, run_id: str | None) -> list[str]:
    if runtime == "podman":
        return cleanup_podman_orphans(run_id=run_id)
    if runtime == "docker":
        return cleanup_docker_orphans(run_id=run_id)
    raise ValueError(f"cleanup --runtime {runtime!r} is not supported.")
