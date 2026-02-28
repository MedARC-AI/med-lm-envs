import queue
import time

import pytest

from medarc_verifiers.orchestrate.docker_vllm import (
    ContainerLogStreamer,
    DockerLaunchError,
    DockerRuntimeAdapter,
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
