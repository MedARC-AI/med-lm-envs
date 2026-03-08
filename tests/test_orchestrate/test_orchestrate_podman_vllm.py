from medarc_verifiers.orchestrate.podman_vllm import ORCHESTRATOR_LABEL_KEY, PodmanLogStreamer, PodmanRuntimeAdapter


def test_podman_log_streamer_stop_closes_sink(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self) -> None:
            captured["terminated"] = True

        def wait(self, timeout: float | None = None) -> int:
            captured["wait_timeout"] = timeout
            return 0

    def fake_popen(command, stdout, stderr):
        captured["command"] = command
        captured["sink"] = stdout
        return FakeProcess()

    monkeypatch.setattr("medarc_verifiers.orchestrate.podman_vllm.subprocess.Popen", fake_popen)

    sink_path = tmp_path / "logs.txt"
    streamer = PodmanLogStreamer("container-123", str(sink_path))
    streamer.start()
    sink = captured["sink"]

    assert sink.closed is False

    streamer.stop(timeout=1.0)

    assert sink.closed is True
    assert captured["terminated"] is True
    assert captured["command"] == ["podman", "logs", "-f", "container-123"]


def test_podman_runtime_adapter_labels_managed_containers(monkeypatch) -> None:
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "container-123\n"
        stderr = ""

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return Result()

    monkeypatch.setattr("medarc_verifiers.orchestrate.podman_vllm.subprocess.run", fake_run)
    adapter = PodmanRuntimeAdapter()

    adapter.launch(
        task_id="task-1",
        model_id="Foo/Bar",
        container_args=["--model", "Foo/Bar"],
        image="fake",
        container_port=8000,
        volume_mounts=[],
        gpus_required=1,
        gpu_ids=[0],
        server_port=8100,
        env={},
        labels={"orchestrator.task_id": "task-1"},
        name="podman-task-1",
        ipc_mode="host",
    )

    command = calls[0]
    assert "--label" in command
    assert f"{ORCHESTRATOR_LABEL_KEY}=true" in command
    assert "orchestrator.task_id=task-1" in command
