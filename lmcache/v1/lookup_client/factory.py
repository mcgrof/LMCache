# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union

# First Party
from lmcache.integration.base_service_factory import ServiceResultPublisher
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.construction import (
    CloseMethodCleanupOwner,
    drain_lookup_construction_cleanup,
    get_lookup_construction_cleanup_receipt,
    retain_lookup_construction_cleanup,
)
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.chunk_statistics_lookup_client import (
    ChunkStatisticsLookupClient,
)
from lmcache.v1.lookup_client.hit_limit_lookup_client import HitLimitLookupClient
from lmcache.v1.lookup_client.lmcache_lookup_client_bypass import (
    LMCacheBypassLookupClient,
)
from lmcache.v1.lookup_client.mooncake_lookup_client import MooncakeLookupClient
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc.zmq_transport import (
    SocketParams,
    ZmqReqRepClientTransport,
    ZmqRouterServerTransport,
)
from lmcache.v1.rpc_utils import get_zmq_rpc_path_lmcache

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer

logger = init_logger(__name__)


def _component_close_retry_safe(error: BaseException, component: object) -> bool:
    """Accept a failed close retry only from an audited component contract."""
    del error
    return getattr(component, "construction_cleanup_close_retry_safe", False) is True


def _close_transport_after_factory_failure(
    transport: Union[ZmqReqRepClientTransport, ZmqRouterServerTransport],
    construction_error: BaseException,
) -> None:
    """Retain a transport until containing-service rollback completes."""
    cleanup_owner: CloseMethodCleanupOwner | None = None

    def attach_and_drain() -> None:
        nonlocal cleanup_owner
        if cleanup_owner is None:
            cleanup_owner = CloseMethodCleanupOwner(
                transport,
                retry_safe_failure=lambda error: _component_close_retry_safe(
                    error,
                    transport,
                ),
            )
        retain_lookup_construction_cleanup(
            construction_error,
            cleanup_owner,
        )
        drain_lookup_construction_cleanup(construction_error)

    try:
        attach_and_drain()
    except BaseException as cleanup_error:
        if cleanup_error is construction_error:
            raise
        attach_and_drain()


def _close_client_after_wrapper_failure(
    client: LookupClientInterface,
    construction_error: BaseException,
) -> None:
    """Retain the exact current client when its next wrapper fails."""
    cleanup_owner: CloseMethodCleanupOwner | None = None

    def attach_and_drain() -> None:
        nonlocal cleanup_owner
        if cleanup_owner is None:
            # Arbitrary external clients remain sticky after a real close
            # error. Internal clients may opt in only by publishing the
            # audited marker.
            cleanup_owner = CloseMethodCleanupOwner(
                client,
                retry_safe_failure=lambda error: _component_close_retry_safe(
                    error,
                    client,
                ),
            )
        retain_lookup_construction_cleanup(
            construction_error,
            cleanup_owner,
        )
        drain_lookup_construction_cleanup(construction_error)

    try:
        attach_and_drain()
    except BaseException as cleanup_error:
        if cleanup_error is construction_error:
            raise
        attach_and_drain()


class LookupClientFactory:
    """Factory for creating lookup clients and servers based on configuration."""

    @staticmethod
    def create_lookup_client(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        lmcache_engine: Optional[LMCacheEngine] = None,
        *,
        result_publisher: Optional[ServiceResultPublisher] = None,
    ) -> LookupClientInterface:
        """
        Create a lookup client based on the configuration.

        Args:
            config: The LMCache engine configuration
            metadata: The LMCache engine metadata (includes engine_id,
                world_size, kv_connector_extra_config)
            lmcache_engine: Optional LMCacheEngine instance for
                bypass lookup client

        Returns:
            A lookup client instance
        """

        client: LookupClientInterface
        # Check if external_lookup_client is configured
        if config.external_lookup_client is not None:
            if config.enable_async_loading:
                raise ValueError(
                    "Asynchronous loading is not supported for external lookup clients."
                )
            client = LookupClientFactory._create_external_lookup_client(
                config.external_lookup_client, config, metadata
            )
        else:
            # First Party
            from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
                LMCacheAsyncLookupClient,
            )
            from lmcache.v1.lookup_client.lmcache_lookup_client import (
                LMCacheLookupClient,
            )

            # Check if bypass lookup is enabled and lmcache_engine is provided
            if config.enable_scheduler_bypass_lookup and lmcache_engine is not None:
                client = LMCacheBypassLookupClient(config, metadata, lmcache_engine)
            elif config.enable_async_loading:
                client = LMCacheAsyncLookupClient(config, metadata)
            else:
                transport = LookupClientFactory._create_zmq_client_transport(
                    config,
                    metadata,
                    result_publisher=result_publisher,
                )
                try:
                    client = LMCacheLookupClient(
                        config,
                        metadata,
                        transport,
                        result_publisher=result_publisher,
                    )
                except BaseException as construction_error:
                    # The client constructor may have moved exact ownership
                    # from ``transport`` into a published client before an
                    # interrupt reached this caller's STORE_FAST. Never close
                    # beneath that committed successor.
                    if result_publisher is None or result_publisher.owns_intermediate(
                        transport
                    ):
                        _close_transport_after_factory_failure(
                            transport,
                            construction_error,
                        )
                    raise

        try:
            if config.hit_miss_ratio is not None and 0 <= config.hit_miss_ratio <= 1:
                client = HitLimitLookupClient(client, config)

            # ``client`` is a moving ownership token. If this second wrapper
            # fails, close the successfully constructed first wrapper rather
            # than reaching through it and closing the base client directly.
            if config.enable_chunk_statistics:
                client = ChunkStatisticsLookupClient(
                    client,
                    config,
                    metadata,
                )
        except BaseException as construction_error:
            _close_client_after_wrapper_failure(client, construction_error)
            raise
        if result_publisher is not None:
            result_publisher.publish_final(client)
        return client

    @staticmethod
    def create_lookup_server(
        lmcache_engine: LMCacheEngine,
        metadata: LMCacheMetadata,
        *,
        result_publisher: Optional[ServiceResultPublisher] = None,
    ) -> Optional[Union["LMCacheLookupServer", "LMCacheAsyncLookupServer"]]:
        """
        Create a lookup server based on the configuration.

        Args:
            lmcache_engine: The LMCache engine instance
            metadata: The LMCache engine metadata (includes engine_id,
                world_size, kv_connector_extra_config, worker_id)

        Returns:
            A lookup server instance, or None if no server should be created
        """
        config = lmcache_engine.config
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 config is expected for lookup server and client"
        )

        lookup_server_worker_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )

        if config.external_lookup_client is None and (
            len(lookup_server_worker_ids) == 0
            or metadata.worker_id in lookup_server_worker_ids
        ):
            # First Party
            from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
                LMCacheAsyncLookupServer,
            )
            from lmcache.v1.lookup_client.lmcache_lookup_client import (
                LMCacheLookupServer,
            )

            if config.enable_async_loading:
                async_server = LMCacheAsyncLookupServer(lmcache_engine, metadata)
                if result_publisher is not None:
                    result_publisher.publish_final(async_server)
                return async_server
            else:
                transport = LookupClientFactory._create_zmq_server_transport(
                    metadata,
                    result_publisher=result_publisher,
                )
                try:
                    sync_server = LMCacheLookupServer(
                        lmcache_engine,
                        metadata,
                        transport,
                        result_publisher=result_publisher,
                    )
                except BaseException as construction_error:
                    # The real synchronous server transfers transport ownership
                    # at constructor entry and attaches its partial server when
                    # rollback is incomplete. Never close beneath a retained,
                    # nonquiesced request thread. A constructor double or an
                    # earlier transfer failure has no receipt, so the factory
                    # remains the exact transport owner in that case.
                    if get_lookup_construction_cleanup_receipt(
                        construction_error
                    ) is None and (
                        result_publisher is None
                        or result_publisher.owns_intermediate(transport)
                    ):
                        _close_transport_after_factory_failure(
                            transport,
                            construction_error,
                        )
                    raise
                if result_publisher is not None:
                    result_publisher.publish_final(sync_server)
                return sync_server

        if result_publisher is not None:
            result_publisher.publish_final(None)
        return None

    @staticmethod
    def _create_external_lookup_client(
        external_lookup_uri: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> LookupClientInterface:
        """
        Create an external lookup client based on the URI format.

        Args:
            external_lookup_uri: URI in format <scheme>://<address>
            config: The LMCache engine configuration
            metadata: The LMCache engine metadata

        Returns:
            A lookup client instance

        Raises:
            ValueError: If the URI format is unsupported
        """
        # Parse URI scheme and address
        if "://" not in external_lookup_uri:
            raise ValueError(
                f"Invalid external lookup client URI format: {external_lookup_uri}. "
                "Expected format: <scheme>://<address>"
            )

        scheme, address = external_lookup_uri.split("://", 1)

        # Route to appropriate client based on scheme
        if scheme == "mooncakestore":
            return LookupClientFactory._create_mooncake_lookup_client(
                address, config, metadata
            )
        else:
            raise ValueError(
                f"Unsupported external lookup client scheme: {scheme}. "
                "Supported schemes: mooncakestore"
            )

    @staticmethod
    def _create_mooncake_lookup_client(
        master_address: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> "MooncakeLookupClient":
        """Create a MooncakeLookupClient instance."""
        return MooncakeLookupClient(config, metadata, master_address)

    @staticmethod
    def _create_zmq_client_transport(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        *,
        result_publisher: Optional[ServiceResultPublisher] = None,
    ) -> ZmqReqRepClientTransport:
        """Create a ZMQ REQ-REP client transport."""
        kv_extra = metadata.kv_connector_extra_config or {}
        rpc_port = kv_extra.get("lmcache_rpc_port", 0)
        assert metadata.engine_id is not None, (
            "engine_id is required for RPC communication"
        )

        lookup_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        ranks = lookup_ids if len(lookup_ids) > 0 else list(range(metadata.world_size))

        socket_params = [
            SocketParams(
                socket_path=get_zmq_rpc_path_lmcache(
                    metadata.engine_id,
                    "lookup",
                    rpc_port,
                    rank,
                ),
                rank=rank,
            )
            for rank in ranks
        ]
        return ZmqReqRepClientTransport(
            socket_params=socket_params,
            timeout_ms=config.lookup_timeout_ms,
            result_publisher=result_publisher,
        )

    @staticmethod
    def _create_zmq_server_transport(
        metadata: LMCacheMetadata,
        *,
        result_publisher: Optional[ServiceResultPublisher] = None,
    ) -> ZmqRouterServerTransport:
        """Create a ZMQ ROUTER server transport."""
        kv_extra = metadata.kv_connector_extra_config or {}
        rpc_port = kv_extra.get("lmcache_rpc_port", 0)
        assert metadata.engine_id is not None, (
            "engine_id is required for RPC communication"
        )
        socket_path = get_zmq_rpc_path_lmcache(
            metadata.engine_id,
            "lookup",
            rpc_port,
            metadata.worker_id,
        )
        return ZmqRouterServerTransport(
            socket_path=socket_path,
            result_publisher=result_publisher,
        )
