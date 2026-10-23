# KMesh registered-state controls

Status: proposed. These controls do not enable engine recovery or storage I/O.

Expose runtime discovery, registration, worker binding, heartbeat, state and
revocation through the existing typed request-client contract. Both ZeroMQ and
gRPC must preserve nested semantic/domain/registration identities; protobuf
messages must represent those fields directly. Do not accept opaque digest
echoes as registration proof or add client-facing physical-completion reports.

Enable the control module explicitly on a blend server. Populate its capability
inventory from the actual assembled handlers. Keep ordinary servers unchanged
and refuse invalid configuration before constructing storage resources. Require
the actual receiving client identity for every control call; derive worker
ownership from transport context, never a payload session string.

Retire expired/revoked registration holds through server liveness maintenance.
Before clearing storage, disarm the logical generation and refuse physical clear
while accepted work remains. Module shutdown must revoke registrations and must
not discard accepted operations. A zero pending-transaction count does not prove
exporter teardown; producer buffer release still requires its separate positive
drain and unregistration acknowledgement.

Qualification must use actual CPU tensor registrations and both transports,
including a separate server process and replacement clients. Verify matching
wire codecs, missing identity refusal, rank readiness, expiry, store reset and
teardown. Device fingerprint completion, reserved storage namespaces, native
span execution and engine metadata consumption remain closed until qualified.
