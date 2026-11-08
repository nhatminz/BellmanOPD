import sys
from types import SimpleNamespace

import pytest

from b200_experiment import alfworld_env as env


def test_native_runtime_is_scoped_to_host_and_python(monkeypatch):
    monkeypatch.delenv("ALFWORLD_NATIVE_SITE", raising=False)
    monkeypatch.setattr(env.socket, "gethostname", lambda: "node-a")
    first = env.native_site_path()
    monkeypatch.setattr(env.socket, "gethostname", lambda: "node-b")
    second = env.native_site_path()
    assert first != second
    assert first.name == sys.implementation.cache_tag
    assert first.parent.name == "node-a"


def test_override_must_contain_a_real_native_package(monkeypatch, tmp_path):
    monkeypatch.setenv("ALFWORLD_NATIVE_SITE", str(tmp_path))
    with pytest.raises(env.EnvironmentSetupError, match="rebuild_alfworld_native.sh"):
        env.configure_native_site()


def test_native_override_is_only_added_to_simulator_import_path(monkeypatch, tmp_path):
    package = tmp_path / "fast_downward"
    package.mkdir()
    (package / "libdownward.so").touch()
    monkeypatch.setenv("ALFWORLD_NATIVE_SITE", str(tmp_path))
    monkeypatch.delitem(sys.modules, "fast_downward", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert env.configure_native_site() == tmp_path
    assert sys.path[0] == str(tmp_path)


def test_native_path_preserves_atomic_publish_symlink(monkeypatch, tmp_path):
    built = tmp_path / "version-1"
    built.mkdir()
    link = tmp_path / "current"
    link.symlink_to(built, target_is_directory=True)
    monkeypatch.setenv("ALFWORLD_NATIVE_SITE", str(link))
    assert env.native_site_path() == link  # rebuild replaces current, not version-1


def test_glibcxx_error_has_host_local_rebuild_instructions(monkeypatch):
    def broken_start(*args, **kwargs):
        raise OSError("libstdc++.so.6: version `GLIBCXX_3.4.32' not found")

    monkeypatch.setattr(env, "configure_native_site", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "textworld",
        SimpleNamespace(
            core=SimpleNamespace(Wrapper=object),
            EnvInfos=lambda **kwargs: kwargs,
            start=broken_start,
        ),
    )
    native = env.NativeEnvironment()
    with pytest.raises(
        env.EnvironmentSetupError, match="GLIBCXX_3.4.32.*rebuild_alfworld_native.sh"
    ):
        native.reset("real-game.tw-pddl", 42)
    assert native.env is None
