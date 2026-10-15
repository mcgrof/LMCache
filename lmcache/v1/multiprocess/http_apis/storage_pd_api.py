# SPDX-License-Identifier: Apache-2.0
"""Opt-in single-writer shared-NVMe request controls for MP serving."""

# Standard
from concurrent.futures import CancelledError
from dataclasses import asdict, dataclass
from http import HTTPStatus

# Third Party
from fastapi import APIRouter, HTTPException, Request

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.raw_block_handoff import RawBlockMPHandoff
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import RawBlockL2Adapter
from lmcache.v1.multiprocess.cache_control.key_resolver import resolve_object_keys
from lmcache.v1.multiprocess.http_apis.dependencies import get_context
from lmcache.v1.storage_backend.raw_block import (
    RawBlockPublicationReceipt,
    ReadAckIdentity,
)

router = APIRouter()


@dataclass(frozen=True)
class ManifestRequest:
    """Complete full-attention prefix, one tensor/pipeline rank, group zero.

    The server resolves tokens with its configured hasher, and verifies that
    the explicit raw-block derivation matches that hasher. Other groups and
    sliding/recurrent layouts are refused by this initial API.
    """

    request_id: str
    model_name: str
    token_ids: list[int]
    cache_salt: str = ""


@dataclass(frozen=True)
class PublishRequest:
    """Previously prepared request and bounded checkpoint wait in seconds."""

    request_id: str
    timeout_s: float = 30.0


@dataclass(frozen=True)
class AdoptRequest:
    """Writer receipt and exact token manifest to adopt before model dispatch."""

    manifest: ManifestRequest
    receipt: RawBlockPublicationReceipt
    timeout_ms: int = 1000


@dataclass(frozen=True)
class LeaseRequest:
    """Exact receipt/consumer identity and group incarnation for claim or ACK."""

    identity: ReadAckIdentity
    session_id: str


@dataclass(frozen=True)
class CancelRequest:
    """Prepared identifier to retire before successful publication."""

    request_id: str


@dataclass(frozen=True)
class UnreadRequest:
    """Exact unclaimed receipt and dispatch group incarnation to release."""

    request_id: str
    receipt: RawBlockPublicationReceipt
    session_id: str


def get_handoff(request: Request) -> RawBlockMPHandoff:
    """Resolve exactly one opted-in raw adapter; refuse ambiguous configurations.

    Args:
        request: Initialized MP server request.

    Returns:
        Borrowed adapter controller.

    Raises:
        HTTPException: 503 for absent/ambiguous raw adapters or disabled P/D.
    """
    adapters = [
        a
        for _, a in get_context(request).engine.storage_manager.l2_adapters()
        if isinstance(a, RawBlockL2Adapter)
    ]
    if len(adapters) != 1:
        raise HTTPException(
            HTTPStatus.SERVICE_UNAVAILABLE, "one raw_block adapter required"
        )
    try:
        handoff = adapters[0].handoff
        expected = get_context(
            request
        ).engine.context.token_hasher.raw_block_derivation()
        if handoff.derivation != expected:
            raise RuntimeError("raw_block pd_derivation differs from MP token hasher")
        return handoff
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE, str(exc)) from exc


def resolve_manifest(
    request: Request, manifest: ManifestRequest
) -> tuple[list[ObjectKey], MemoryLayoutDesc]:
    """Resolve the registered single-group model's complete prefix chunks.

    Args:
        request: Initialized MP server request.
        manifest: Model, complete token prefix and tenant salt.

    Returns:
        Ordered object keys and the registered canonical chunk layout.

    Raises:
        HTTPException: 400 for invalid content; 409 for unsupported layout;
            503 before model registration.
    """
    ctx = get_context(request).engine.context
    layouts = ctx.layout_desc_registry.find_group_layout_descs(manifest.model_name, 1)
    if not layouts:
        raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE, "model not registered")
    try:
        attention = ctx.layout_desc_registry.find_attn_desc(manifest.model_name, 1)
    except ValueError as exc:
        raise HTTPException(
            HTTPStatus.SERVICE_UNAVAILABLE, "model unregistered"
        ) from exc
    if (
        set(layouts) != {0}
        or attention.num_chunks_in_sw != [-1]
        or attention.group_kinds not in ((), ("attention",))
    ):
        raise HTTPException(
            HTTPStatus.CONFLICT, "P/D requires one full-attention group"
        )
    try:
        keys, _ = resolve_object_keys(
            ctx.token_hasher,
            manifest.model_name,
            1,
            manifest.token_ids,
            manifest.cache_salt,
        )
    except ValueError as exc:
        raise HTTPException(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    if not keys:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "manifest has no complete chunks")
    return keys, layouts[0]


@router.post("/storage-pd/prepare")
def prepare_publication(body: ManifestRequest, request: Request) -> dict[str, object]:
    """Protect resolved extents before prefill; exact duplicate admission is safe.

    Args:
        body: Request identity and token-addressed manifest.
        request: Initialized MP server request.

    Returns:
        Prepared identifier and resolved keys; no completed-write claim.

    Raises:
        HTTPException: Invalid manifest, unsupported backend or refused admission.
    """
    try:
        keys, _ = resolve_manifest(request, body)
        get_handoff(request).prepare(body.request_id, keys)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(HTTPStatus.CONFLICT, str(exc)) from exc
    return {
        "request_id": body.request_id,
        "status": "prepared",
        "keys": [asdict(key.to_encoded_object_key()) for key in keys],
    }


@router.post("/storage-pd/publish")
def publish_publication(body: PublishRequest, request: Request) -> dict[str, object]:
    """Return READY only after all payloads and the forced checkpoint complete.

    Args:
        body: Prepared identifier and finite positive deadline in seconds.
        request: Initialized MP writer request.

    Returns:
        Scalar receipt, without local payload-provenance records.

    Raises:
        HTTPException: 409 for unknown/failed request, 504 for incomplete writes.
    """
    try:
        receipt = get_handoff(request).publish(body.request_id, body.timeout_s)
    except TimeoutError as exc:
        raise HTTPException(HTTPStatus.GATEWAY_TIMEOUT, str(exc)) from exc
    except (CancelledError, KeyError, ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(HTTPStatus.CONFLICT, str(exc)) from exc
    result = asdict(receipt)
    result.pop("payload_provenance")
    return {"request_id": body.request_id, "status": "ready", "receipt": result}


@router.post("/storage-pd/adopt")
def adopt_publication(body: AdoptRequest, request: Request) -> dict[str, str]:
    """Adopt an exact namespace/epoch/manifest before allowing decode lookup.

    A caller must claim at the writer first and must keep its lease until GPU
    restore/decode has completed. This endpoint never releases writer extents.

    Args:
        body: Receipt, exact manifest and checkpoint adoption deadline.
        request: Initialized MP reader request.

    Returns:
        Adopted identifier after generation and tensor-layout checks pass.

    Raises:
        HTTPException: 409 for a mismatched or unavailable publication.
    """
    try:
        keys, layout = resolve_manifest(request, body.manifest)
        adopted = get_handoff(request).adopt(
            body.receipt, keys, body.timeout_ms, layout
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(HTTPStatus.CONFLICT, str(exc)) from exc
    if not adopted:
        raise HTTPException(HTTPStatus.CONFLICT, "publication does not match storage")
    return {"request_id": body.manifest.request_id, "status": "adopted"}


@router.post("/storage-pd/claim")
def claim_publication(body: LeaseRequest, request: Request) -> dict[str, object]:
    """Bind the writer lease to a reader before it accesses storage.

    Args:
        body: Complete receipt/consumer identity and group incarnation.
        request: Initialized MP writer request.

    Returns:
        Claimed status; no extent is released.

    Raises:
        HTTPException: 409 for refused claim, 503 for unavailable backend.
    """
    result = get_handoff(request).claim(body.identity, body.session_id)
    if not result.granted:
        raise HTTPException(HTTPStatus.CONFLICT, result.reason)
    return {"status": "claimed"}


@router.post("/storage-pd/ack")
def acknowledge_publication(body: LeaseRequest, request: Request) -> dict[str, str]:
    """Release an exact claimed lease, keeping uncertainty explicit.

    Args:
        body: Exact successful claim identity after the reader has finished.
        request: Initialized MP writer request.

    Returns:
        Applied, duplicate, rejected or unresolved outcome from the writer.

    Raises:
        HTTPException: 503 when the configured backend is unavailable.
    """
    result = get_handoff(request).acknowledge(body.identity, body.session_id)
    return {"outcome": result.value}


@router.post("/storage-pd/cancel")
def cancel_prepared(body: CancelRequest, request: Request) -> dict[str, bool]:
    """Cancel preparation without reclaiming a published or claimed request.

    Args:
        body: Prepared identifier whose model work was abandoned.
        request: Initialized MP writer request.

    Returns:
        Whether the request was retired without issuing READY. False means
        it is unknown or published; it does not authorize slot reclamation.

    Raises:
        HTTPException: 503 for an unavailable backend.
    """
    return {"cancelled": get_handoff(request).cancel_prepared(body.request_id)}


@router.post("/storage-pd/unread")
def release_unread(body: UnreadRequest, request: Request) -> dict[str, object]:
    """Release an exact published receipt only if nobody claimed its read.

    Args:
        body: Published request, exact receipt and dispatch group incarnation.
        request: Initialized MP writer request.

    Returns:
        Released/refused status, reason and whether the answer is final.
        Exact retries are safe; a claimed read must complete through ACK.

    Raises:
        HTTPException: 503 for an unavailable backend.
    """
    result = get_handoff(request).release_unread(
        body.request_id, body.receipt, body.session_id
    )
    return {"released": result.released, "reason": result.reason, "final": result.final}
