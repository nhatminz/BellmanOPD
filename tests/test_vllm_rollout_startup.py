import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

import b200_experiment.vllm_rollout as rollout
from b200_experiment.config import load_config
from test_vllm_rollout import _config


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def prepare_start(monkeypatch, tmp_path, timeout=900):
    clock = Clock()
    config = _config()
    config["rollout"]["vllm"].update(startup_timeout=timeout, startup_log_interval=30)
    engine = rollout.VLLMRolloutEngine(
        config, tmp_path, local_rank=1, world_size=2, port=12345
    )
    process = MagicMock(pid=123456789)
    process.poll.return_value = None
    process.wait.return_value = 0
    monkeypatch.setattr(rollout.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(rollout.time, "sleep", clock.sleep)
    monkeypatch.setattr(engine, "_server_command", lambda: ["fake-vllm", "serve"])
    monkeypatch.setattr(
        engine,
        "_server_environment",
        lambda: {"CUDA_VISIBLE_DEVICES": "1", "VLLM_LOGGING_LEVEL": "INFO"},
    )
    monkeypatch.setattr(rollout.subprocess, "Popen", lambda *a, **kw: process)
    signals = []
    monkeypatch.setattr(
        rollout.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    posts = []
    monkeypatch.setattr(engine, "_post", lambda path, **kwargs: posts.append(path))
    logs = []
    monkeypatch.setattr(rollout.tqdm, "write", logs.append)
    return engine, clock, process, signals, posts, logs


def test_cold_start_beyond_old_600_second_limit_completes_then_initializes_ipc(
    monkeypatch, tmp_path
):
    engine, clock, process, signals, posts, logs = prepare_start(monkeypatch, tmp_path)
    monkeypatch.setattr(
        rollout.requests,
        "get",
        lambda *a, **kw: SimpleNamespace(
            ok=clock.now >= 700, status_code=200 if clock.now >= 700 else 503
        ),
    )
    engine.start()
    report = json.loads(
        (tmp_path / "vllm_rollout_server.rank-00001.startup.json").read_text()
    )
    assert clock.now == 700 and report["status"] == "ready"
    assert report["elapsed_sec"] == 700
    assert report["gpu"] == "1" and report["pid"] == process.pid
    assert posts == ["/init_weight_transfer_engine", "/sleep"] and engine.sleeping
    assert any("still starting" in line for line in logs)
    engine.close()
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    engine.close()  # idempotent, never targets another process group
    assert len(signals) == 2


def test_timeout_reports_http_state_and_log_then_reaps_entire_process_group(
    monkeypatch, tmp_path
):
    engine, clock, process, signals, posts, logs = prepare_start(
        monkeypatch, tmp_path, timeout=2
    )
    monkeypatch.setattr(
        rollout.requests,
        "get",
        lambda *a, **kw: SimpleNamespace(ok=False, status_code=503),
    )
    with pytest.raises(TimeoutError, match="within 2s.*GPU 1.*HTTP 503") as error:
        engine.start()
    assert "ROLLOUT_VLLM_STARTUP_TIMEOUT" in str(error.value)
    assert not posts and engine.closed
    assert engine.log_handle.closed
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    report = json.loads(
        (tmp_path / "vllm_rollout_server.rank-00001.startup.json").read_text()
    )
    assert report["status"] == "failed" and report["last_health"] == "HTTP 503"


def test_http_connection_error_is_visible_in_timeout_report(monkeypatch, tmp_path):
    engine, _, _, _, _, _ = prepare_start(monkeypatch, tmp_path, timeout=2)

    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(rollout.requests, "get", unavailable)
    with pytest.raises(TimeoutError, match="ConnectionError: connection refused"):
        engine.start()


def test_server_crash_fails_immediately_and_cleans_descendants_after_launcher_exit(
    monkeypatch, tmp_path
):
    engine, clock, process, signals, _, _ = prepare_start(monkeypatch, tmp_path)
    process.poll.return_value = 9
    process.returncode = 9
    with pytest.raises(RuntimeError, match="rank 1 exited.*exit=9"):
        engine.start()
    assert clock.now == 0
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    assert engine.log_handle.closed


def test_wait_timeout_escalates_to_sigkill(monkeypatch, tmp_path):
    engine, _, process, signals, _, _ = prepare_start(monkeypatch, tmp_path)
    engine.process = process
    process.wait.side_effect = [subprocess.TimeoutExpired("fake-vllm", 20), 0]
    engine.close()
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


def test_real_local_http_server_health_ipc_init_sleep_and_teardown(
    monkeypatch, tmp_path
):
    script = tmp_path / "server.py"
    script.write_text(
        "import http.server,json,sys\n"
        "class Handler(http.server.BaseHTTPRequestHandler):\n"
        " def log_message(self,*args): pass\n"
        " def do_GET(self):\n"
        "  self.send_response(200);self.end_headers()\n"
        " def do_POST(self):\n"
        "  self.rfile.read(int(self.headers.get('Content-Length',0)))\n"
        "  self.send_response(200);self.end_headers();self.wfile.write(b'{}')\n"
        "http.server.HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()\n"
    )
    config = _config()
    config["rollout"]["vllm"]["startup_timeout"] = 10
    engine = rollout.VLLMRolloutEngine(config, tmp_path)
    monkeypatch.setattr(
        engine,
        "_server_command",
        lambda: [sys.executable, str(script), str(engine.port)],
    )
    try:
        engine.start()
        assert engine.sleeping
        report = json.loads((tmp_path / "vllm_rollout_server.startup.json").read_text())
        assert report["status"] == "ready" and report["last_health"] == "HTTP 200"
    finally:
        engine.close()
    assert engine.process.poll() is not None


def test_real_orphaned_child_is_reaped_even_if_api_launcher_has_exited(tmp_path):
    ready = tmp_path / "child_ready"
    child_code = (
        "import signal,time,pathlib,os;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid()));"
        "time.sleep(20)"
    )
    parent_code = (
        "import subprocess,sys,time,pathlib;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        f"p=pathlib.Path({str(ready)!r});"
        "exec('while not p.exists(): time.sleep(.01)')"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], start_new_session=True
    )
    engine = rollout.VLLMRolloutEngine(_config(), tmp_path)
    engine.process = parent
    try:
        parent.wait(timeout=10)
        child_pid = int(ready.read_text())
        assert Path(f"/proc/{child_pid}/status").is_file()
        engine.close()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child_pid}/status")
            if not status.exists() or "State:\tZ" in status.read_text():
                break
            time.sleep(0.01)
        else:
            pytest.fail("The orphaned engine worker remained alive after close()")
    finally:
        engine.close()
        try:
            os.killpg(parent.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize(
    "key,value",
    [
        ("startup_timeout", 0),
        ("startup_timeout", float("inf")),
        ("startup_log_interval", -1),
    ],
)
def test_invalid_startup_settings_do_not_spawn(monkeypatch, tmp_path, key, value):
    config = _config()
    config["rollout"]["vllm"][key] = value
    engine = rollout.VLLMRolloutEngine(config, tmp_path)
    spawned = MagicMock()
    monkeypatch.setattr(rollout.subprocess, "Popen", spawned)
    with pytest.raises(ValueError, match="finite and positive"):
        engine.start()
    spawned.assert_not_called()
    engine.close()


def test_rollout_environment_filters_only_project_logprob_controls(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("VLLM_LOGPROB_SANITY_FAIL", "true")
    monkeypatch.setenv("VLLM_LOGPROB_SANITY_ENABLED", "false")
    monkeypatch.setenv("VLLM_RETURN_LOG_PROBS", "false")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("ROLLOUT_VLLM_LOGGING_LEVEL", "INFO")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "WARNING")
    engine = rollout.VLLMRolloutEngine(_config(), tmp_path, local_rank=1, world_size=2)
    env = engine._server_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["VLLM_LOGGING_LEVEL"] == "INFO"
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert not any(name.startswith("VLLM_LOGPROB_SANITY_") for name in env)
    assert "VLLM_RETURN_LOG_PROBS" not in env
    assert os.environ["VLLM_LOGPROB_SANITY_FAIL"] == "true"
    engine.close()


def test_server_startup_is_identical_for_opd_ta_cmt_iw(monkeypatch, tmp_path):
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/alfworld_multiturn.yaml"
    )
    monkeypatch.setattr(rollout.shutil, "which", lambda *a: "/venv/bin/vllm")
    commands = []
    for method in ["opd", "ta", "cmt", "iw"]:
        selected = copy.deepcopy(config)
        selected["experiment"]["method"] = method
        engine = rollout.VLLMRolloutEngine(selected, tmp_path, port=12345)
        commands.append(engine._server_command())
        engine.close()
    assert all(command == commands[0] for command in commands)


@pytest.mark.parametrize("timeout", [None, "1800", "2400"])
def test_common_launcher_maps_explicit_timeout_without_overriding_yaml_defaults(
    tmp_path, timeout
):
    root = Path(__file__).resolve().parents[1]
    capture = tmp_path / "args.json"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0,1",
        "PYTHON_BIN": sys.executable,
        "PPO_MINI_BATCH_SIZE": "8",
        "CAPTURE": str(capture),
    }
    env.pop("ROLLOUT_VLLM_STARTUP_TIMEOUT", None)
    if timeout is not None:
        env["ROLLOUT_VLLM_STARTUP_TIMEOUT"] = timeout
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/common_b200.sh\nbuild_training_args /tmp/test-output\n"
            '"$PYTHON_BIN" - "${COMMON_TRAIN_ARGS[@]}" <<\'PY\'\n'
            "import os,sys,json\n"
            "with open(os.environ['CAPTURE'],'w') as f: json.dump(sys.argv[1:],f)\nPY\n",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    values = [arg for arg in args if arg.startswith("rollout.vllm.startup_timeout=")]
    assert values == (
        [] if timeout is None else [f"rollout.vllm.startup_timeout={timeout}"]
    )
