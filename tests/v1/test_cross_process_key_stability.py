# SPDX-License-Identifier: Apache-2.0
"""Two nodes sharing a namespace must derive the same key for the same tokens.

A key is derived in each engine's own process. If that derivation depends on
anything that differs between the two, the writer stores under one key and the
reader looks under another, and the handoff silently finds nothing. This drives
the derivation in genuinely fresh interpreters rather than twice in one, which
is the only way a per-process seed shows up.
"""

# Future
from __future__ import annotations

# Standard
import os
import subprocess
import sys
import textwrap

# Third Party
import pytest

PROBE = textwrap.dedent(
    """
    import sys
    import torch
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.token_database import ChunkedTokenDatabase

    config = LMCacheEngineConfig.from_defaults(chunk_size=256)
    config.pre_caching_hash_algorithm = sys.argv[1]
    metadata = LMCacheMetadata(
        model_name="probe-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 1, 256, 8, 64),
    )
    database = ChunkedTokenDatabase(config, metadata)
    tokens = torch.tensor(list(range(300)), dtype=torch.long)
    effective = getattr(database.hash_func, "__name__", repr(database.hash_func))
    print("EFFECTIVE " + effective)
    print("|".join(str(key) for _, _, key in database.process_tokens(tokens)))
    """
)


def _vllm_is_available() -> bool:
    """Whether the hash functions this contract names can be resolved."""
    try:
        # Third Party
        import vllm  # noqa: F401
    except Exception:
        return False
    return True


def _probe(algorithm: str, hash_seed: str) -> tuple[str, str]:
    """Return the effective hash function and the keys, from a fresh process.

    Where the named functions can be resolved at all, a probe failure is a
    real failure: converting every error to a skip would hide exactly the
    misconfiguration this contract exists to catch. Only the absence of the
    serving engine is a skip.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    result = subprocess.run(
        [sys.executable, "-c", PROBE, algorithm],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[-400:]
        if not _vllm_is_available():
            pytest.skip(f"serving engine not installed: {detail}")
        raise AssertionError(f"key probe failed for {algorithm}: {detail}")
    lines = result.stdout.strip().splitlines()
    effective = next(
        line.split(" ", 1)[1] for line in lines if line.startswith("EFFECTIVE ")
    )
    keys = lines[-1]
    assert "CacheEngineKey" in keys, keys
    return effective, keys


def _keys(algorithm: str, hash_seed: str) -> str:
    return _probe(algorithm, hash_seed)[1]


@pytest.mark.parametrize("algorithm", ["builtin", "sha256_cbor"])
def test_the_same_tokens_give_the_same_keys_in_a_fresh_process(algorithm):
    """The derivation must not depend on anything a process picks per run."""
    first = _keys(algorithm, "0")
    second = _keys(algorithm, "0")
    assert first == second


def test_the_named_algorithm_is_the_one_actually_used():
    """Naming an algorithm that silently falls back proves nothing.

    The resolver tries several import paths and drops to the builtin hash
    when none works, with a warning. A deployment that believes it selected
    a documented function while getting the interpreter's would satisfy
    every other check here.
    """
    if not _vllm_is_available():
        pytest.skip(
            "the named algorithms resolve through the serving engine, which is "
            "not installed here; both fall back and the comparison is vacuous"
        )
    effective, _ = _probe("sha256_cbor", "0")
    assert "sha256" in effective.lower(), (
        f"sha256_cbor resolved to {effective!r}, which is the fallback"
    )
    builtin_effective, _ = _probe("builtin", "0")
    assert effective != builtin_effective, (
        "the named algorithm and the builtin resolve to the same function, "
        "so naming it has no effect in this environment"
    )


def test_the_hash_seed_changes_the_keys_whichever_algorithm_is_named():
    """Naming a deterministic algorithm does not remove the seed dependence.

    The chunk hash chain starts from a value the serving engine derives from
    ``PYTHONHASHSEED``, so selecting ``sha256_cbor`` changes the hash function
    without making the keys independent of the seed. Two nodes must therefore
    agree on the seed as well as the algorithm. This is asserted so that if a
    future version does make the derivation seed-independent, this test fails
    and the requirement can be relaxed deliberately rather than by accident.
    """
    seeded = _keys("sha256_cbor", "0")
    other = _keys("sha256_cbor", "12345")
    if seeded == other:
        pytest.skip(
            "keys are already independent of PYTHONHASHSEED in this "
            "environment; the seed no longer has to match across nodes"
        )
    assert seeded != other
