import pytest

from medarc_verifiers.orchestrate.pyxis_vllm import PyxisLaunchError, PyxisRuntimeAdapter


class FakeStdout:
    def __init__(self, payload: bytes = b"") -> None:
        self._payload = payload
        self._read_done = False

    def read(self) -> bytes:
        if self._read_done:
            return b""
        self._read_done = True
        return self._payload

    def readline(self) -> bytes:
        if self._read_done:
            return b""
        self._read_done = True
        return self._payload


class FakeProcess:
    def __init__(self, *, pid: int = 1234, returncode: int | None = None, output: bytes = b"") -> None:
        self.pid = pid
        self._returncode = returncode
        self.stdout = FakeStdout(output)
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        return 0 if self._returncode is None else self._returncode


def test_pyxis_launch_renders_expected_srun_command(monkeypatch) -> None:
    adapter = PyxisRuntimeAdapter()
    captured: dict[str, object] = {}

    def fake_popen(command, stdout, stderr, env):
        captured["command"] = command
        captured["env"] = env
        return FakeProcess(pid=4321)

    monkeypatch.setattr("medarc_verifiers.orchestrate.pyxis_vllm.subprocess.Popen", fake_popen)
    monkeypatch.setattr("medarc_verifiers.orchestrate.pyxis_vllm.time.sleep", lambda _: None)
    monkeypatch.setenv("OPENAI_API_KEY", "host-key")

    handle = adapter.launch(
        task_id="task-1",
        model_id="Foo/Bar",
        container_args=["--model", "Foo/Bar", "--tensor-parallel-size", "2"],
        image="docker://vllm/vllm-openai:latest",
        container_port=8000,
        volume_mounts=["/cache:/root/.cache/huggingface:ro"],
        gpus_required=2,
        gpu_ids=[],
        server_port=8100,
        env={"OPENAI_API_KEY": "host-key", "SAFETENSORS_FAST_GPU": "1", "UNSET_VAR": "value"},
        labels={"orchestrator.task_id": "task-1"},
        srun_extra_args=["--container-entrypoint"],
    )

    command = captured["command"]
    assert handle.base_url == "http://127.0.0.1:8100/v1"
    assert handle.identifier == "4321"
    assert command[:5] == [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--gpus-per-task=2",
        "--container-image=docker://vllm/vllm-openai:latest",
    ]
    assert "--container-mounts=/cache:/root/.cache/huggingface:ro" in command
    assert "--container-env=OPENAI_API_KEY,SAFETENSORS_FAST_GPU" in command
    assert "--container-entrypoint" in command
    assert "--no-container-entrypoint" not in command
    assert "--no-container-mount-home" in command
    assert command[-8:] == [
        "vllm",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "8100",
        "--model",
        "Foo/Bar",
    ] or command[-10:] == [
        "vllm",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "8100",
        "--model",
        "Foo/Bar",
        "--tensor-parallel-size",
        "2",
    ]


def test_pyxis_launch_classifies_common_cluster_error(monkeypatch) -> None:
    adapter = PyxisRuntimeAdapter()

    def fake_popen(command, stdout, stderr, env):
        return FakeProcess(returncode=1, output=b"srun: error: unknown option --container-image")

    monkeypatch.setattr("medarc_verifiers.orchestrate.pyxis_vllm.subprocess.Popen", fake_popen)
    monkeypatch.setattr("medarc_verifiers.orchestrate.pyxis_vllm.time.sleep", lambda _: None)

    with pytest.raises(PyxisLaunchError, match="container support is unavailable"):
        adapter.launch(
            task_id="task-1",
            model_id="Foo/Bar",
            container_args=["--model", "Foo/Bar"],
            image="docker://image",
            container_port=8000,
            volume_mounts=[],
            gpus_required=1,
            gpu_ids=[],
            server_port=8000,
            env={},
            labels={},
            srun_extra_args=[],
        )


def test_pyxis_teardown_terminates_process(monkeypatch) -> None:
    adapter = PyxisRuntimeAdapter()
    process = FakeProcess(pid=999)

    def fake_popen(command, stdout, stderr, env):
        return process

    monkeypatch.setattr("medarc_verifiers.orchestrate.pyxis_vllm.subprocess.Popen", fake_popen)
    monkeypatch.setattr("medarc_verifiers.orchestrate.pyxis_vllm.time.sleep", lambda _: None)

    handle = adapter.launch(
        task_id="task-1",
        model_id="Foo/Bar",
        container_args=["--model", "Foo/Bar"],
        image="docker://image",
        container_port=8000,
        volume_mounts=[],
        gpus_required=1,
        gpu_ids=[],
        server_port=8000,
        env={},
        labels={},
        srun_extra_args=[],
    )
    adapter.teardown(handle)

    assert process.terminated is True
