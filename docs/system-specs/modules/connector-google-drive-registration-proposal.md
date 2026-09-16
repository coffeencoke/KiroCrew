# Registration — status + the one remaining host-side hook

**Update (base advanced):** the shared knowledge handler now carries a landed
**runner-injection registration seam** (commit `40caf8031`, on this PR's base
`feat/connector-control-plane-executor`), authored by the ACL/handler owner. It
already registers `google_drive` by module + class, so there is **no handler edit
to propose** and **no from-scratch hunk**. This doc records what the seam does,
confirms my connector satisfies its contract, and states the ONE host-side hook
that remains (which needs W01's executor and is the host's, not mine).

## What the landed seam already does (not mine to change)

`src/kiro_crew/dashboard/handlers/knowledge.py`:

- `_register_optional_connector(connectors, module_path, class_name, runner_factory)`
  imports the vendor module (guarded) and constructs the connector **with** the
  injected runner: `connector = connector_cls(runner_factory)` (positional).
- `setup_knowledge_routes` reads `app["knowledge_connector_runners"]`
  (a `{source_type: runner_factory}` map the host installs once W01's executor —
  PR #11286 — is available) and registers each vendor **only when** its module
  imports **and** its runner factory is installed. Absent module or absent runner
  ⇒ not registered (fail-closed, never public).
- The loop already lists my connector:
  `("google_drive", "kiro_crew.knowledge.connectors.google_drive", "GoogleDriveConnector")`.

## My connector satisfies that contract (verified)

- `GoogleDriveConnector.__init__(self, operations_factory=None)` — a positional
  `connector_cls(runner_factory)` binds `operations_factory = runner_factory`.
- `fetch_rows` calls `self._operations_factory(source) -> DriveOperations`, so the
  installed factory must be `Callable[[source_dict], DriveOperations]`.
- `validate_config` returns False when no factory is wired, so a no-arg
  registration is impossible-by-construction — matching the seam's fail-closed
  intent (an empty registration is not `code_complete`, and the seam enforces it).
- Verified live: `test/test_knowledge_query_acl_wiring.py` +
  `test/test_knowledge_rows_ingest.py` (the seam's own tests, on this base) pass
  with my connector in the tree.

## The ONE remaining host-side hook (host/ACL owner, needs W01 PR #11286)

Install a `google_drive` runner factory into `app["knowledge_connector_runners"]`.
The factory itself is now a **shipped product**, not a proposal:
`kiro_crew.connections.vendors.google_drive.host_factory.make_drive_operations_factory`
composes the whole W01 path (binding → handle → custody gate → live binding store
→ vault → production transport → `execute`/`PageWalk`) and returns the
`source -> DriveOperations` callable the connector expects. It is proven
executable end to end in `test/test_google_drive_host_factory.py` (a real
`files.list` walk runs through it over a real vault + store + scripted socket).

So the host hook is a single call, supplying the host-owned dependencies
(the verifier, the live `BindingStore`, the `SecretVault`, the deployment/principal
ids, the clock — none of which the vendor package holds):

```python
# host setup, once the W01 executor is available:
from kiro_crew.connections.vendors.google_drive.host_factory import (
    make_drive_operations_factory,
)

app.setdefault("knowledge_connector_runners", {})["google_drive"] = (
    make_drive_operations_factory(
        verifier=<host SubjectTenantVerifier>,       # real Google-side identity check
        binding_store=<the live BindingStore>,       # L04 store
        vault=<the SecretVault>,                      # existing custody
        deployment_id=<this deployment's id>,
        kiro_principal=<the authorized Kiro principal>,
        clock=time.time,
    )
)
```

The landed `_register_optional_connector` loop then constructs
`GoogleDriveConnector(<that factory>)` and registers it (fail-closed until the
factory is installed). No handler edit; no second host assembly point; the factory
holds no token/session/HTTP and copies no W01 code — custody stays single-sourced
in W01.

The host-owned pieces named honestly (still the host's, not vendor work): the
`SubjectTenantVerifier` (real identity verification against Google) and the live
`BindingStore` + `SecretVault` wiring. `host_factory` composes them; it does not
invent them.

## The query-time ACL revalidator (segment 4) — same pattern, host install

`app["knowledge_revalidator"]` (read by the retriever) should be set to
`kiro_crew.connections.vendors.google_drive.acl_probe.GoogleDriveRevalidationHook(
store=<store>, runner=<subject-scoped SubjectOperationRunner>)`. The runner runs
`files.get` AS the querying subject (no impersonation; a subject with no binding
→ UNVERIFIABLE; revocation effective next query). The hook already resolves the
item's `ProviderResourceRef` by `item_id` via `store.get_item_grants` and persists
via `store.revoke_item_acl` / `store.mark_item_acl_revalidated` (all verified to
exist).

## Not proposed / not touched

No change to W01 `control_plane/**`/`exports`, container anchors, `setup.cfg`,
workflows/permissions/hooks, S2 validator, fast-gate, or the main manifest. No
new governance scope. No live-secret read, no real Google request, no business
write.
