# SPDX-License-Identifier: Apache-2.0
"""
Factory for creating SerdeProcessor instances from SerdeConfig.

Each serde type registers itself here so it can be referenced by name
in L2 adapter configs:

    {
      "type": "fs",
      "base_path": "/cache",
      "serde": {"type": "fp8", "fp8_dtype": "float8_e4m3fn"}
    }
"""

# Standard
from typing import TYPE_CHECKING, Callable

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.serde.base import SerdeSizeContract

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.serde.base import SerdeConfig, SerdeProcessor

logger = init_logger(__name__)

# name -> factory(kwargs) -> SerdeProcessor.
# Factories receive the type-specific kwargs (everything except "type").
_SERDE_FACTORY_REGISTRY: dict[str, Callable[[dict[str, object]], "SerdeProcessor"]] = {}
_SERDE_SIZE_CONTRACT_REGISTRY: dict[str, SerdeSizeContract] = {}


def register_serde_factory(
    name: str,
    factory: Callable[[dict[str, object]], "SerdeProcessor"],
    *,
    size_contract: SerdeSizeContract = SerdeSizeContract.UPPER_BOUND,
) -> None:
    """Register a serde factory under a type name.

    Args:
        name: Serde type name (used in the JSON config ``"type"`` field).
        factory: Callable that takes the type-specific kwargs dict and
            returns a SerdeProcessor instance.
        size_contract: Whether the serializer's estimate is exact or only an
            upper bound. Custom serdes default to ``UPPER_BOUND`` so backends
            without an actual-used-length load contract fail closed.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    if name in _SERDE_FACTORY_REGISTRY:
        raise ValueError(f"Serde type already registered: {name!r}")
    _SERDE_FACTORY_REGISTRY[name] = factory
    _SERDE_SIZE_CONTRACT_REGISTRY[name] = size_contract


def get_registered_serde_types() -> list[str]:
    """Return the list of registered serde type names."""
    return list(_SERDE_FACTORY_REGISTRY)


def get_serde_size_contract(name: str) -> SerdeSizeContract:
    """Return the registered serialized-size contract for ``name``.

    Args:
        name: Registered serde type name.

    Returns:
        The serde's exact-size or upper-bound contract.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    contract = _SERDE_SIZE_CONTRACT_REGISTRY.get(name)
    if contract is None:
        known = ", ".join(sorted(_SERDE_FACTORY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown serde type {name!r}. Registered: {known}")
    return contract


def create_serde_processor(config: "SerdeConfig") -> "SerdeProcessor":
    """Build a SerdeProcessor from a SerdeConfig.

    Args:
        config: Serde configuration. ``config.type`` must name a registered
            serde; ``config.kwargs`` is forwarded to the factory.

    Returns:
        A SerdeProcessor instance ready to be passed to a controller.

    Raises:
        ValueError: If ``config.type`` names an unregistered serde.
    """
    factory = _SERDE_FACTORY_REGISTRY.get(config.type)
    if factory is None:
        known = ", ".join(sorted(_SERDE_FACTORY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown serde type {config.type!r}. Registered: {known}")
    return factory(config.kwargs)
