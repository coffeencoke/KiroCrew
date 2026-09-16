"""The production host factory builds an EXECUTABLE DriveOperations, proven end to
end on the real W01 seam with no Google account.

This is not an import smoke test. It composes the factory with:

* a REAL on-disk AES-256-GCM ``SecretVault`` holding the binding's credential,
* a REAL on-disk L04 ``BindingStore``,
* a real ``SubjectTenantVerifier``,

and then RUNS a Drive read (``files.list`` — a paged walk) through the factory's
runner, which drives the real ``execute`` / ``PageWalk`` and the real
``build_production_transport`` (custody gate + store + vault + this package's real
``locator`` + per-op ``decode``). Only the socket is scripted, via the documented
``http_send`` seam W01 itself exposes — everything from the gate through the
locator/decode is the true path. The vault guard asserts the token is not stored
in plaintext, so the custody claim is not vacuous.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from kiro_crew.connections.control_plane.binding import binding_secret_ref
from kiro_crew.connections.control_plane.lifecycle import BindingStore
from kiro_crew.connections.control_plane.production import HttpReply, HttpRequest
from kiro_crew.connections.vendors.google_drive.host_factory import (
    build_drive_operations,
    make_drive_operations_factory,
)
from kiro_crew.connections.vendors.google_drive.operations import DriveOperations
from kiro_crew.secrets import SecretVault

_T0 = 1_000_000.0
_TOKEN = "drive-oauth-access-token-value"


def _verifier(*, claimed_subject: str, claimed_tenant: str, service_id: str) -> Dict[str, str]:
    return {
        "subject_ref": f"subject://verified/{claimed_subject}",
        "tenant_ref": f"tenant://verified/{claimed_tenant}",
    }


def _real_vault(tmp_path: Path) -> SecretVault:
    """A REAL AES-256-GCM vault holding the Drive binding's credential."""
    vault = SecretVault(tmp_path / "crewhome")
    # create_binding(slug="google_drive") records this deterministic secret name.
    vault.set_sync(binding_secret_ref("google_drive")["name"], _TOKEN)
    enc = tmp_path / "crewhome" / ".vault" / "secrets.enc"
    # Guard the guard: a stub vault would make the custody claim vacuous.
    assert enc.is_file() and _TOKEN.encode() not in enc.read_bytes()
    return vault


def _binding_store(tmp_path: Path) -> BindingStore:
    return BindingStore(tmp_path / "connections" / "control_plane_bindings.json")


class _ScriptedSender:
    """An HttpSend that records the requests it was asked to send and replies from
    a URL-predicate script — the documented socket seam, so the gate/custody/
    locator all run for real; only the wire is canned. The Authorization header
    W01's transport adds is asserted, proving the vault credential reached the
    request (custody actually resolved), without echoing the token anywhere else.
    """

    def __init__(self) -> None:
        self.requests: List[HttpRequest] = []
        self._routes: List[Any] = []

    def route(self, predicate, reply: HttpReply) -> "_ScriptedSender":
        self._routes.append((predicate, reply))
        return self

    def __call__(self, request: HttpRequest, *, timeout_seconds: float) -> HttpReply:
        self.requests.append(request)
        for predicate, reply in self._routes:
            if predicate(request.url):
                return reply
        raise AssertionError(f"no scripted reply for {request.url}")


def _json_reply(status: int, obj) -> HttpReply:
    import json

    return HttpReply(status=status, headers={}, body=json.dumps(obj).encode())


def test_factory_builds_executable_runner_and_runs_a_read(tmp_path: Path):
    vault = _real_vault(tmp_path)
    store = _binding_store(tmp_path)
    sender = _ScriptedSender()
    # A two-page files.list, so the paged walk actually advances through PageWalk.
    sender.route(
        lambda u: "/files" in u and "pageToken" not in u,
        _json_reply(200, {"files": [{"id": "F1", "name": "a"}], "nextPageToken": "p2"}),
    ).route(
        lambda u: "/files" in u and "pageToken=p2" in u,
        _json_reply(200, {"files": [{"id": "F2", "name": "b"}]}),
    )

    ops = build_drive_operations(
        subject="alice",
        tenant="ws-acme",
        verifier=_verifier,  # type: ignore[arg-type]
        binding_store=store,
        vault=vault,
        deployment_id="deployment://test/google_drive/0",
        kiro_principal="kiro://test/owner",
        now=_T0,
        http_send=sender,
    )
    # It is a real DriveOperations, not a hook or a flag.
    assert isinstance(ops, DriveOperations)

    # RUN a real read through it: execute/PageWalk -> gate -> vault -> locator ->
    # scripted sender -> decode. Two pages of files come back.
    files = ops.list_files()
    assert [f["id"] for f in files] == ["F1", "F2"]

    # The request was really built by the real locator (both shared-drive flags)
    # and really carried the vault credential as a Bearer token (custody resolved).
    assert sender.requests, "the runner never reached the transport"
    first = sender.requests[0]
    assert "supportsAllDrives=true" in first.url
    assert "includeItemsFromAllDrives=true" in first.url
    assert first.headers.get("Authorization") == f"Bearer {_TOKEN}"


def test_operations_factory_reads_subject_tenant_from_source(tmp_path: Path):
    # make_drive_operations_factory returns the source->DriveOperations callable the
    # connector's operations_factory contract expects, and it enters a real read.
    vault = _real_vault(tmp_path)
    store = _binding_store(tmp_path)
    sender = _ScriptedSender().route(
        lambda u: "/files" in u,
        _json_reply(200, {"files": [{"id": "OK", "name": "ok"}]}),
    )
    factory = make_drive_operations_factory(
        verifier=_verifier,  # type: ignore[arg-type]
        binding_store=store,
        vault=vault,
        deployment_id="deployment://test/google_drive/1",
        kiro_principal="kiro://test/owner",
        clock=lambda: _T0,
        http_send=sender,
    )
    ops = factory({"subject": "bob", "tenant": "ws-acme", "account": "my-drive"})
    assert isinstance(ops, DriveOperations)
    assert [f["id"] for f in ops.list_files()] == ["OK"]


def test_factory_requires_subject_and_tenant(tmp_path: Path):
    factory = make_drive_operations_factory(
        verifier=_verifier,  # type: ignore[arg-type]
        binding_store=_binding_store(tmp_path),
        vault=_real_vault(tmp_path),
        deployment_id="deployment://test/google_drive/2",
        kiro_principal="kiro://test/owner",
        clock=lambda: _T0,
    )
    with pytest.raises(ValueError):
        factory({"tenant": "ws-acme"})  # no subject
    with pytest.raises(ValueError):
        factory({"subject": "bob"})  # no tenant


def test_connector_registered_with_factory_validates_and_syncs(tmp_path: Path):
    # The end-to-end contract the landed handler seam relies on: constructing the
    # connector WITH this factory yields a connector whose validate_config passes
    # and whose fetch_rows drives a real read (snapshot) through the factory's
    # W01-backed runner.
    from kiro_crew.knowledge.connectors.google_drive import GoogleDriveConnector

    vault = _real_vault(tmp_path)
    store = _binding_store(tmp_path)
    sender = (
        _ScriptedSender()
        .route(
            lambda u: "/files" in u
            and "/export" not in u
            and "startPageToken" not in u
            and "alt=media" not in u,
            _json_reply(
                200,
                {
                    "files": [
                        {"id": "D1", "name": "doc.txt", "mimeType": "text/plain", "version": "1"}
                    ]
                },
            ),
        )
        .route(
            lambda u: "/files/D1" in u and "alt=media" in u,
            HttpReply(status=200, headers={}, body=b"hello"),
        )
        .route(
            lambda u: "/changes/startPageToken" in u, _json_reply(200, {"startPageToken": "cp-1"})
        )
    )
    factory = make_drive_operations_factory(
        verifier=_verifier,  # type: ignore[arg-type]
        binding_store=store,
        vault=vault,
        deployment_id="deployment://test/google_drive/3",
        kiro_principal="kiro://test/owner",
        clock=lambda: _T0,
        http_send=sender,
    )
    # Positional construction, exactly as _register_optional_connector does:
    # connector_cls(runner_factory).
    connector = GoogleDriveConnector(factory)
    ok, msg = connector.validate_config({"account": "my-drive", "tenant": "ws-acme"})
    assert ok and msg == ""

    import asyncio

    rows, snapshot, checkpoint = asyncio.run(
        connector.fetch_rows({"account": "my-drive", "tenant": "ws-acme", "subject": "carol"})
    )
    assert snapshot is True
    assert checkpoint == "cp-1"
    assert [r.key for r in rows] == ["D1"]
    assert rows[0].subjects == ()  # no permissions field -> empty deny-all
