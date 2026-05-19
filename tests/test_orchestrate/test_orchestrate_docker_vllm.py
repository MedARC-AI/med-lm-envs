import queue
import sys
import time
import types

import pytest

from medarc_verifiers.orchestrate.docker_vllm import (
    ContainerLogStreamer,
    DockerLaunchError,
    DockerRuntimeAdapter,
    create_and_start_container,
    normalize_volumes,
)


def test_normalize_volumes_parses_mount_strings():
    volumes = normalize_volumes(["/host/cache:/root/.cache/huggingface:ro", "/host/data:/data"])
    assert volumes["/host/cache"]["bind"] == "/root/.cache/huggingface"
    assert volumes["/host/cache"]["mode"] == "ro"
    assert volumes["/host/data"]["bind"] == "/data"
    assert volumes["/host/data"]["mode"] == "rw"


def test_normalize_volumes_rejects_bad_mount_string():
    with pytest.raises(DockerLaunchError):
        normalize_volumes(["/host/only"])


def test_container_log_streamer_stop_does_not_hang(tmp_path):
    class BlockingStream:
        def __init__(self):
            self._queue = queue.Queue()

        def __iter__(self):
            return self

        def __next__(self):
            item = self._queue.get()
            if item is None:
                raise StopIteration
            return item

        def send(self, payload: bytes) -> None:
            self._queue.put(payload)

        def close(self) -> None:
            self._queue.put(None)

    class FakeContainer:
        def __init__(self, stream):
            self._stream = stream

        def logs(self, stream: bool, follow: bool):
            return self._stream

    stream = BlockingStream()
    container = FakeContainer(stream)
    sink_path = tmp_path / "logs.txt"
    streamer = ContainerLogStreamer(container, str(sink_path))
    streamer.start()
    stream.send(b"hello\n")
    time.sleep(0.1)
    streamer.stop(timeout=1.0)

    assert streamer.is_alive() is False
    assert sink_path.read_text(encoding="utf-8") == "hello\n"


def test_docker_runtime_adapter_returns_local_base_url(monkeypatch) -> None:
    class FakeContainer:
        id = "abc123"

    captured: dict[str, object] = {}

    def fake_create_and_start_container(**kwargs):
        captured.update(kwargs)
        return FakeContainer()

    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.docker_vllm.create_and_start_container",
        fake_create_and_start_container,
    )
    adapter = DockerRuntimeAdapter()

    handle = adapter.launch(
        task_id="task-1",
        model_id="Foo/Bar",
        container_args=["--model", "Foo/Bar"],
        image="fake",
        container_port=8000,
        volume_mounts=["/cache:/root/.cache/huggingface:ro"],
        gpus_required=1,
        gpu_ids=[0],
        server_port=8123,
        env={"OPENAI_API_KEY": "x"},
        labels={"orchestrator.task_id": "task-1"},
        name="container-name",
        ipc_mode="host",
    )

    assert handle.base_url == "http://127.0.0.1:8123/v1"
    assert handle.identifier == "abc123"
    assert captured["host_port"] == 8123
    assert captured["volumes"] == ["/cache:/root/.cache/huggingface:ro"]


def test_create_and_start_container_recovers_from_owned_name_conflict(monkeypatch) -> None:
    class FakeContainer:
        def __init__(
            self, *, status: str = "created", labels: dict[str, str] | None = None, container_id: str = "abc123"
        ) -> None:
            self.status = status
            self.labels = labels or {"orchestrator.managed": "true", "orchestrator.task_id": "task-1"}
            self.id = container_id
            self.removed = False
            self.started = False

        def reload(self) -> None:
            return None

        def remove(self, v: bool, force: bool) -> None:
            self.removed = True

        def start(self) -> None:
            self.started = True

    class FakeContainers:
        def __init__(self) -> None:
            self.create_calls = 0
            self.existing = FakeContainer(status="exited")

        def get(self, name: str):
            assert name == "task-1"
            return self.existing

        def create(self, **kwargs):
            self.create_calls += 1
            if self.create_calls == 1:
                raise RuntimeError("Conflict. The container name is already in use.")
            return FakeContainer(labels=kwargs["labels"])

    class FakeClient:
        def __init__(self) -> None:
            self.containers = FakeContainers()
            self.images = types.SimpleNamespace(pull=lambda image: None)

    fake_client = FakeClient()
    fake_docker = types.SimpleNamespace(from_env=lambda timeout=600: fake_client)
    fake_docker_types = types.SimpleNamespace(DeviceRequest=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "docker", fake_docker)
    monkeypatch.setitem(sys.modules, "docker.types", fake_docker_types)

    container = create_and_start_container(
        image="fake",
        name="task-1",
        container_port=8000,
        host_port=8123,
        env={"OPENAI_API_KEY": "x"},
        volumes=["/cache:/root/.cache/huggingface:ro"],
        ipc_mode="host",
        gpu_ids=[0],
        command=["--model", "Foo/Bar"],
        labels={"orchestrator.task_id": "task-1"},
    )

    assert fake_client.containers.create_calls == 2
    assert fake_client.containers.existing.removed is True
    assert container.started is True
