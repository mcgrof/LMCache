# SPDX-License-Identifier: Apache-2.0
"""Refuse unsupported KMesh profiles before MP adapters acquire resources."""

KMESH_MP_HANDOFF_UNSUPPORTED = (
    "The multiprocess LMCache connector does not provide a qualified KMesh "
    "handoff: server-bound runtime evidence, explicit-span execution and "
    "completion/exporter lifetime are not integrated. Use the in-process "
    "LMCacheConnectorV1 for the supported KMesh position-recovery profiles."
)


def reject_kmesh_mp_target_profile(profile: object) -> None:
    """Allow ordinary MP use and refuse every explicitly requested KMesh profile.

    Args:
        profile: Value of the KMesh target-profile key. None means absent;
            false or empty values are explicit invalid profiles, not absence.

    Raises:
        NotImplementedError: A target profile is present without a qualified MP
            metadata/span consumer. No adapter or server must be created first.
    """
    if profile is not None:
        raise NotImplementedError(
            f"KMesh target profile {profile!r}: {KMESH_MP_HANDOFF_UNSUPPORTED}"
        )
