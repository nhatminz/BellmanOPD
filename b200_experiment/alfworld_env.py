"""ALFWorld raw parquet -> real TextWorld PDDL simulator, with bounded RPCs.

No stored response/solution is ever used to produce environment feedback.
This module stays torch-free so environment subprocesses do not import models.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import select
import socket
import subprocess
import sys
import tempfile
from pathlib import Path


SPLITS = ("train", "eval_in_distribution", "eval_out_of_distribution")


class EnvironmentSetupError(RuntimeError):
    pass


def native_site_path():
    """Optional Fast Downward build for this host/Python, outside shared venvs."""
    explicit = os.environ.get("ALFWORLD_NATIVE_SITE")
    if explicit:
        # Do not resolve the last symlink: rebuilds atomically replace it.
        return Path(explicit).expanduser().absolute()
    hostname = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())
    return (
        Path(__file__).resolve().parents[1]
        / ".alfworld-native"
        / hostname
        / sys.implementation.cache_tag
    )


def configure_native_site():
    site = native_site_path()
    if not (site / "fast_downward/libdownward.so").is_file():
        if os.environ.get("ALFWORLD_NATIVE_SITE"):
            raise EnvironmentSetupError(
                f"ALFWORLD_NATIVE_SITE has no fast_downward/libdownward.so: {site}. Run bash scripts/rebuild_alfworld_native.sh on this node."
            )
        return None
    loaded = sys.modules.get("fast_downward")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(
        site.resolve()
    ):
        raise EnvironmentSetupError(
            "Fast Downward is already imported from another location. Restart this Python process to use the host-local ALFWorld native build."
        )
    if str(site) not in sys.path:
        sys.path.insert(0, str(site))
    return site


def validate_settings(config):
    settings = config["alfworld"]
    from .alfworld_protocol import protocol_manifest

    protocol_manifest(config)
    for key in (
        "max_turns",
        "max_turn_tokens",
        "max_episode_tokens",
        "env_workers",
        "env_timeout_sec",
    ):
        if float(settings.get(key, 0)) <= 0:
            raise ValueError(f"alfworld.{key} must be positive")
    if int(config["rollout"]["max_new_tokens"]) <= 0:
        raise ValueError(
            "rollout.max_new_tokens must be a positive episode actor-token cap"
        )


def _mapping(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def normalize_task(row, split, source):
    merged = dict(row)
    for key in ("metadata", "extra_info", "mcp", "game"):
        merged = {**_mapping(row.get(key)), **merged}
    content = merged.get("game_content")
    reference = merged.get("game_file_path", merged.get("game_file"))
    if content is None and reference:
        path = Path(reference).expanduser()
        if path.suffix not in {".tw-pddl", ".ulx", ".z8"}:
            raise EnvironmentSetupError(f"Unsupported ALFWorld game asset: {path}")
        candidates = [path] if path.is_absolute() else [Path(source).parent / path]
        path = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if path is None:
            raise EnvironmentSetupError(
                f"Missing ALFWorld game asset {reference!r}; restore it relative to {Path(source).parent}, or use the raw parquet with embedded game_content."
            )
        if path.suffix != ".tw-pddl":
            return {
                "id": str(merged.get("id") or reference),
                "task_type": str(merged.get("task_type", "unknown")),
                "split": split,
                "game_file_path": str(path.resolve()),
                "game_content": None,
                "game_identity": str(reference),
                "game_fingerprint": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        content = path.read_text(encoding="utf-8")
    game = _mapping(content)
    required = {"pddl_domain", "pddl_problem", "grammar"}
    if not required.issubset(game):
        raise EnvironmentSetupError(
            f"ALFWorld record {merged.get('id')} needs embedded game_content with {sorted(required)}, or a real game_file_path; static prompts/responses are not an environment."
        )
    # The raw export includes walkthroughs. They are reserved for an explicit
    # environment readiness smoke test, never for policy prompts or training.
    serialized = json.dumps(
        {key: game[key] for key in sorted(required)}, ensure_ascii=False, sort_keys=True
    )
    fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
    identity = "/".join(Path(str(reference)).parts[-3:]) if reference else fingerprint
    return {
        "id": str(merged.get("id") or fingerprint),
        "task_type": str(merged.get("task_type", "unknown")),
        "split": split,
        "game_file_path": str(reference or ""),
        "game_content": serialized,
        "game_identity": identity,
        "game_fingerprint": fingerprint,
    }


def load_task_splits(config):
    from .data import read_records

    section = config["alfworld"]
    validate_settings(config)
    paths = {"train": config["data"]["path"], **section["eval_paths"]}
    result, seen_ids, seen_games, seen_content = {}, {}, {}, {}
    for split in SPLITS:
        if split not in paths:
            raise EnvironmentSetupError(f"Missing alfworld split path: {split}")
        rows, files = read_records(paths[split])
        if not rows:
            raise EnvironmentSetupError(f"ALFWorld {split} is empty: {paths[split]}")
        tasks = [normalize_task(row, split, paths[split]) for row in rows]
        for task in tasks:
            for seen, value in (
                (seen_ids, task["id"]),
                (seen_games, task["game_identity"]),
                (seen_content, task["game_fingerprint"]),
            ):
                if value in seen:
                    raise EnvironmentSetupError(
                        f"Duplicate/leaked ALFWorld task {task['id']} between {seen[value]} and {split}"
                    )
                seen[value] = split
        result[split] = tasks
    return result


def materialize_game(task, cache_dir):
    if task["game_content"] is None:
        return Path(task["game_file_path"])
    directory = Path(cache_dir) / task["split"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (task["game_fingerprint"] + ".tw-pddl")
    if not path.exists():
        fd, temporary = tempfile.mkstemp(prefix=".game-", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(task["game_content"])
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return path


class NativeEnvironment:
    """Compatible with ALFWorld's AlfredDemangler + TextWorld PDDL backend."""

    def __init__(self):
        self.env = None
        self.state = None

    def reset(self, game_file, seed):
        self.close()
        self.native_site = configure_native_site()
        try:
            import textworld
        except ImportError as error:
            raise EnvironmentSetupError(
                "Install requirements-alfworld.txt in the active venv: python3 -m pip install -r requirements-alfworld.txt (TextWorld PDDL + Fast Downward are required)."
            ) from error

        class AlfredNames(textworld.core.Wrapper):
            # Same deterministic entity-name numbering as official ALFWorld's
            # AlfredDemangler (shuffle=False); no planner/expert is requested.
            def load(self, *args, **kwargs):
                from collections import Counter

                super().load(*args, **kwargs)
                identifiers = sorted(info.id for info in self._entity_infos.values())
                names = {
                    identifier: identifier.split("_bar_", 1)[0]
                    + (
                        "basin"
                        if "_bar_" in identifier and "basin" in identifier
                        else ""
                    )
                    for identifier in identifiers
                }
                counts = Counter(names[i] for i in identifiers if "_bar_" in i)
                self.aliases = {}
                for identifier in identifiers:
                    name = names[identifier]
                    if "_bar_" in identifier:
                        alias = f"{name} {counts[name]}"
                        counts[name] -= 1
                    else:
                        alias = identifier
                    self.aliases[identifier] = alias
                for info in self._entity_infos.values():
                    info.name = self.aliases[info.id]

        try:
            self.env = textworld.start(
                str(game_file),
                textworld.EnvInfos(won=True, lost=True, admissible_commands=True),
                wrappers=[AlfredNames],
            )
            self.env.seed(int(seed))
            self.state = self.env.reset()
        except Exception as error:
            self.close()
            hint = (
                "Native C++ runtime mismatch (GLIBCXX/CXXABI). Run bash scripts/rebuild_alfworld_native.sh on THIS node, then python3 -m b200_experiment.alfworld_cli smoke. The isolated build does not replace the shared training venv."
                if any(name in str(error) for name in ("GLIBCXX_", "CXXABI_"))
                else "Verify TextWorld PDDL/Fast Downward installation; run the readiness smoke test."
            )
            raise EnvironmentSetupError(
                f"Cannot initialize real ALFWorld game {game_file}: {error}. {hint}"
            ) from error
        return self._observation(0.0, False)

    def _observation(self, reward, done):
        return {
            "observation": str(self.state.feedback),
            "reward": float(reward),
            "terminated": bool(done or self.state.won or self.state.lost),
            "success": bool(self.state.won),
            "admissible_actions": list(self.state.admissible_commands or []),
            "native_site": str(self.native_site) if self.native_site else None,
        }

    def step(self, action):
        if self.env is None:
            raise RuntimeError("Call reset before step")
        self.state, reward, done = self.env.step(action)
        return self._observation(reward, done)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


class EnvironmentClient:
    """A disposable simulator process; timeouts cannot leave a stuck env thread."""

    def __init__(self, task, config):
        self.task, self.config = task, config
        self.process = None

    def _request(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select(
            [self.process.stdout], [], [], float(self.config.get("env_timeout_sec", 60))
        )
        if not ready:
            self.close()
            raise TimeoutError(
                f"ALFWorld simulator timed out on {request['op']} for {self.task['id']}"
            )
        line = self.process.stdout.readline()
        if not line:
            self.close()
            raise EnvironmentSetupError(
                "ALFWorld worker exited; check stderr and install requirements-alfworld.txt"
            )
        response = json.loads(line)
        if "error" in response:
            self.close()
            raise EnvironmentSetupError(response["error"])
        return response

    def reset(self, seed):
        game = materialize_game(self.task, self.config["game_cache_dir"])
        self.process = subprocess.Popen(
            [sys.executable, "-m", "b200_experiment.alfworld_env", "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self._request({"op": "reset", "game": str(game), "seed": int(seed)})

    def step(self, action):
        return self._request({"op": "step", "action": action})

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for handle in (self.process.stdin, self.process.stdout):
                handle.close()
            self.process = None


def worker_main():
    env = NativeEnvironment()
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    result = (
                        env.reset(request["game"], request["seed"])
                        if request["op"] == "reset"
                        else env.step(request["action"])
                    )
                print(json.dumps(result), flush=True)
            except Exception as error:
                print(
                    json.dumps({"error": f"{type(error).__name__}: {error}"}),
                    flush=True,
                )
    finally:
        env.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["--native-site"]:
        print(native_site_path())
    elif sys.argv[1:] == ["--worker"]:
        worker_main()
    else:
        raise SystemExit("Use python3 -m b200_experiment.alfworld_cli ready/smoke")
