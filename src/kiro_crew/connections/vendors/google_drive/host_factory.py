"""Production factory: compose a :class:`DriveOperations` from W01's public seam.

This is the piece the host installs into
``app["knowledge_connector_runners"]["google_drive"]`` so the shared knowledge
handler's landed runner-injection seam can construct
``GoogleDriveConnector(operations_factory)`` with a REAL, executable runner
(rather than a no-arg dead registration, which its own ``validate_config``
refuses).

It composes ONLY W01's public control-plane entry points --
:func:`~kiro_crew.connections.control_plane.binding.create_binding`,
:func:`~kiro_crew.connections.control_plane.handle.derive_handle` /
:func:`~kiro_crew.connections.control_plane.handle.ensure_usable`,
:class:`~kiro_crew.connections.control_plane.production.BindingCustodyGate`,
:class:`~kiro_crew.connections.control_plane.lifecycle.BindingStore`,
:func:`~kiro_crew.connections.control_plane.production.build_production_transport`,
and :func:`~kiro_crew.connections.control_plane.executor.execute` /
:class:`~kiro_crew.connections.control_plane.executor.PageWalk` -- plus this
package's own request assembly (:mod:`.locator`, :mod:`.decode`,
:mod:`.descriptors`). There is NO second auth, NO self-built sender, NO HTTP, NO
vault and NO token here: credential custody is single-sourced in W01. The Drive
side only says WHICH operation with WHICH args and reads the neutral outcome.

The host supplies the pieces that are its own, not the vendor's: the
:class:`~kiro_crew.connections.control_plane.binding.SubjectTenantVerifier` (real
identity verification against Google is a host/directory concern), the live
:class:`~kiro_crew.connections.control_plane.lifecycle.BindingStore`, and the
:class:`~kiro_crew.connections.control_plane.production.SecretStore` (the existing
``SecretVault``). The per-source subject / tenant / Drive account / granted scopes
come from the source row. Given those, this module builds the binding, derives a
narrowed expiring handle, fences it with a custody gate, composes the real
transport, and returns a :class:`DriveOperations` whose every call runs through
``execute`` / ``PageWalk``.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Tuple

from kiro_crew.connections.control_plane.auth_modes import declare_permitted_modes
from kiro_crew.connections.control_plane.binding import (
    SubjectTenantVerifier,
    create_binding,
)
from kiro_crew.connections.control_plane.executor import (
    ExecutionOutcome,
    PageWalk,
    execute,
)
from kiro_crew.connections.control_plane.handle import (
    DerivedHandle,
    derive_handle,
    ensure_usable,
)
from kiro_crew.connections.control_plane.lifecycle import BindingStore
from kiro_crew.connections.control_plane.operation import (
    CredentialMode,
    OperationDescriptor,
)
from kiro_crew.connections.control_plane.policy import LayerCeilings
from kiro_crew.connections.control_plane.production import (
    BindingCustodyGate,
    HttpSend,
    SecretStore,
    Transport,
    build_production_transport,
    urllib_http_send,
)

from . import decode as drive_decode
from . import locator as drive_locator
from .operations import DriveOperations, OperationRunner

#: Drive read operations authenticate as the querying/owning user's OAuth grant.
_CREDENTIAL_MODE: CredentialMode = "oauth_user"

#: The governed catalog scope a Drive operation is authorized under (verified live
#: SCOPE_CATALOG member; "knowledge" is NOT one and is deny-by-default), with the
#: operation id as the governed item.
_GOVERNANCE_SCOPE = "tools"

#: Default OAuth scopes a Drive read grant is expected to carry. The handle is
#: NARROWED to the read scope it actually requests; a grant that does not carry it
#: fails handle derivation (never silently widened).
_DEFAULT_GRANTED_SCOPES: Tuple[str, ...] = ("https://www.googleapis.com/auth/drive.readonly",)
_DEFAULT_REQUESTED_SCOPES: Tuple[str, ...] = ("https://www.googleapis.com/auth/drive.readonly",)

#: The provider slug the binding's secret family belongs to (matches the
#: ``google_drive`` provider slug used across the stack).
_SLUG = "google_drive"

#: Default handle lifetime. A per-source sync round is well within this; the
#: transport re-resolves the secret every call, so the TTL bounds the handle, not
#: the credential freshness.
_DEFAULT_TTL_SECONDS = 300.0


class _W01OperationRunner(OperationRunner):
    """An :class:`OperationRunner` that runs each Drive op through W01's executor.

    Holds the derived handle and a per-descriptor transport composer. It NEVER
    touches a socket, a token, or the vault directly: ``execute`` / ``PageWalk``
    drive the composed production transport, which is where W01 owns custody. The
    transport is composed PER descriptor because a decode is bound to one
    operation (W01 calls ``decode`` with the reply alone), so ``decode.for_operation``
    selects the right shape for the op being run.
    """

    def __init__(
        self,
        *,
        handle: DerivedHandle,
        transport_for: Callable[[OperationDescriptor], Transport],
        clock: Callable[[], float],
    ) -> None:
        self._handle = handle
        self._transport_for = transport_for
        self._clock = clock

    def _kwargs(self, descriptor: OperationDescriptor) -> dict:
        return dict(
            now=self._clock(),
            offered_mode=_CREDENTIAL_MODE,
            permitted=declare_permitted_modes((_CREDENTIAL_MODE,)),
            layers=LayerCeilings(),
            governance_scope=_GOVERNANCE_SCOPE,
            governance_item=descriptor["operation_id"],
        )

    def run(
        self, descriptor: OperationDescriptor, request_args: Mapping[str, object]
    ) -> ExecutionOutcome:
        return execute(
            descriptor,
            self._handle,
            self._transport_for(descriptor),
            request_args=dict(request_args),
            **self._kwargs(descriptor),
        )

    def walk(self, descriptor: OperationDescriptor, base_args: Mapping[str, object]):
        walk = PageWalk(
            descriptor=descriptor,
            handle=self._handle,
            transport=self._transport_for(descriptor),
            base_args=dict(base_args),
            clock=self._clock,
            **{k: v for k, v in self._kwargs(descriptor).items() if k != "now"},
        )
        while not walk.done:
            yield walk.next()


def build_drive_operations(
    *,
    subject: str,
    tenant: str,
    verifier: SubjectTenantVerifier,
    binding_store: BindingStore,
    vault: SecretStore,
    deployment_id: str,
    kiro_principal: str,
    now: float,
    granted_scopes: Tuple[str, ...] = _DEFAULT_GRANTED_SCOPES,
    requested_scopes: Tuple[str, ...] = _DEFAULT_REQUESTED_SCOPES,
    ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    page_size: int = 100,
    clock: Optional[Callable[[], float]] = None,
    http_send: HttpSend = urllib_http_send,
) -> DriveOperations:
    """Compose one executable :class:`DriveOperations` for a single source identity.

    Steps, all on W01's public seam:

    1. ``create_binding`` verifies the claimed subject/tenant through the
       host-supplied ``verifier`` (which does the real Google-side identity check)
       and mints a binding recording only the VERIFIED identity + a secret
       REFERENCE (never a value).
    2. The binding is inserted into the live ``binding_store`` (L04) so the
       transport resolves its credential from the store's own record, and
       ``derive_handle`` mints a narrowed, expiring handle for the read scope.
    3. ``BindingCustodyGate`` fences that handle's identity; a call routed for a
       different binding is refused before any secret is read.
    4. ``build_production_transport`` composes the real transport (gate + store +
       vault + this package's ``locator`` + per-op ``decode``). The Drive
       ``OperationRunner`` drives ``execute`` / ``PageWalk`` over it.

    Raises whatever W01 raises on a bad identity/scope (``BindingVerificationError``,
    ``HandleScopeError``, ...) rather than swallowing it -- a source that cannot be
    bound must fail loudly, not produce a dead runner.
    """
    tick = clock if clock is not None else (lambda: now)

    binding = create_binding(
        service_id="google_drive",
        claimed_subject=subject,
        claimed_tenant=tenant,
        credential_mode=_CREDENTIAL_MODE,
        verifier=verifier,
        slug=_SLUG,
    )
    binding_store.insert(
        binding,
        deployment_id=deployment_id,
        kiro_principal=kiro_principal,
    )
    handle = derive_handle(
        binding,
        granted_scopes=granted_scopes,
        requested_scopes=requested_scopes,
        now=now,
        ttl_seconds=ttl_seconds,
    )
    view = ensure_usable(handle, now=now)
    gate = BindingCustodyGate(binding=binding, binding_fingerprint=view.binding_fingerprint)

    def _transport_for(descriptor: OperationDescriptor) -> Transport:
        return build_production_transport(
            gate=gate,
            store=binding_store,
            vault=vault,
            locator=drive_locator.locate,
            decode=drive_decode.for_operation(descriptor),
            http_send=http_send,
        )

    runner = _W01OperationRunner(handle=handle, transport_for=_transport_for, clock=tick)
    return DriveOperations(runner, page_size=page_size)


def make_drive_operations_factory(
    *,
    verifier: SubjectTenantVerifier,
    binding_store: BindingStore,
    vault: SecretStore,
    deployment_id: str,
    kiro_principal: str,
    clock: Callable[[], float],
    granted_scopes: Tuple[str, ...] = _DEFAULT_GRANTED_SCOPES,
    requested_scopes: Tuple[str, ...] = _DEFAULT_REQUESTED_SCOPES,
    ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    page_size: int = 100,
    http_send: HttpSend = urllib_http_send,
) -> Callable[[Mapping[str, object]], DriveOperations]:
    """Return the ``operations_factory`` the connector expects: ``source -> DriveOperations``.

    This is what the host installs at
    ``app["knowledge_connector_runners"]["google_drive"]``. It captures the
    host-owned dependencies (verifier, live binding store, vault, principal
    identifiers, clock) ONCE and reads the per-source subject / tenant from each
    ``source`` row when called. The connector calls this per sync;
    :func:`build_drive_operations` does the W01 composition each time so the handle
    is freshly derived (bounded by its TTL) for that round.

    The subject/tenant are taken from the source row: ``subject`` (the querying/
    owning identity) and ``tenant`` (the KiroCrew workspace/org the grant belongs
    to). Both are required -- the connector's ``validate_config`` already refuses a
    source without a tenant, and a binding cannot be minted without a subject.
    """

    def _factory(source: Mapping[str, object]) -> DriveOperations:
        subject = str(source.get("subject") or "").strip()
        tenant = str(source.get("tenant") or "").strip()
        if not subject:
            raise ValueError(
                "Google Drive source requires a 'subject' (the verified identity "
                "the credential grant belongs to) to bind a runner"
            )
        if not tenant:
            raise ValueError("Google Drive source requires a 'tenant' to bind a runner")
        return build_drive_operations(
            subject=subject,
            tenant=tenant,
            verifier=verifier,
            binding_store=binding_store,
            vault=vault,
            deployment_id=deployment_id,
            kiro_principal=kiro_principal,
            now=clock(),
            granted_scopes=granted_scopes,
            requested_scopes=requested_scopes,
            ttl_seconds=ttl_seconds,
            page_size=page_size,
            clock=clock,
            http_send=http_send,
        )

    return _factory


__all__ = ["build_drive_operations", "make_drive_operations_factory"]
