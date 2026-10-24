from types import SimpleNamespace

import pytest

from rsenv.rlvr.bridge import VerifiedBridgeClient
from rsenv.rlvr.contracts import InfrastructureError, TaskSpec


@pytest.mark.parametrize(
    "change",
    [
        {"episode_id": "old"},
        {"world_id": "other"},
        {"request_id": "2"},
        {"version": 2},
        {"status": "infra_error"},
    ],
)
def test_response_correlation_poisoning(change):
    client = VerifiedBridgeClient(SimpleNamespace(world_id="world"), TaskSpec(), "episode")
    response = dict(version=1, episode_id="episode", world_id="world", request_id="1", status="ok")
    client.responses.put({**response, **change})
    with pytest.raises(InfrastructureError):
        client._receive("1", 0.01)
    assert client.poisoned
    with pytest.raises(InfrastructureError, match="unavailable"):
        client.rpc("state", 0.01)


def test_timeout_poisoning_rejects_late_result():
    client = VerifiedBridgeClient(SimpleNamespace(world_id="world"), TaskSpec(), "episode")
    with pytest.raises(InfrastructureError):
        client._receive("1", 0.001)
    client.responses.put(dict(version=1, episode_id="episode", world_id="world", request_id="1", status="ok"))
    with pytest.raises(InfrastructureError, match="unavailable"):
        client.rpc("state", 0.01)


def test_reader_rejects_non_json_and_eof():
    client = VerifiedBridgeClient(SimpleNamespace(world_id="world"), TaskSpec(), "episode")
    client._process = SimpleNamespace(stdout=iter(["not json\n"]))
    client._reader()
    with pytest.raises(InfrastructureError):
        client._receive("1", 0.01)


def test_partial_world_preparation_cleans_up(tmp_path, monkeypatch):
    from rsenv.rlvr import worlds

    lease = worlds.WorldLease(tmp_path)
    closed = []
    monkeypatch.setattr(worlds, "doctor", lambda sdk: {"ready": True})
    monkeypatch.setattr(lease.processes, "close", lambda: closed.append(True))

    def fail(spec):
        raise InfrastructureError("partial_reset")

    monkeypatch.setattr(lease, "_prepare", fail)
    with pytest.raises(InfrastructureError, match="partial_reset"):
        lease.acquire("episode", TaskSpec().reset)
    assert closed == [True] and not lease._acquired


def test_engine_patch_rejects_ambiguous_source(tmp_path):
    from rsenv.rlvr.worlds import replace_once

    source = tmp_path / "engine.ts"
    source.write_text("target target")
    with pytest.raises(InfrastructureError, match="engine_patch_mismatch"):
        replace_once(source, "target", "patched")
    assert source.read_text() == "target target"


def test_missing_browser_fails_before_world_preparation(tmp_path, monkeypatch):
    from rsenv.rlvr import worlds

    lease = worlds.WorldLease(tmp_path)
    monkeypatch.setattr(worlds, "doctor", lambda sdk: {"ready": False, "checks": {"browser_executable": False}})
    with pytest.raises(InfrastructureError, match="browser_executable_missing.*puppeteer browsers install chrome"):
        lease.acquire("episode", TaskSpec().reset)
    assert not lease._acquired
    assert lease.workdir is None
    assert not lease.processes.processes
