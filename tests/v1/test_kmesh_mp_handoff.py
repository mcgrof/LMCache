# SPDX-License-Identifier: Apache-2.0
"""Keep ordinary MP use distinct from unsupported KMesh target profiles."""

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.kmesh_mp_handoff import reject_kmesh_mp_target_profile


def test_absent_profile_preserves_ordinary_mp_use() -> None:
    reject_kmesh_mp_target_profile(None)


@pytest.mark.parametrize(
    "profile", ["serial_pic_v1", "collective_lockstep_v1", "", False]
)
def test_explicit_profile_refuses_without_claiming_handoff(profile: object) -> None:
    with pytest.raises(NotImplementedError, match="server-bound runtime evidence"):
        reject_kmesh_mp_target_profile(profile)
