# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import abc
import threading

# Third Party
import torch

# First Party
from lmcache.v1.token_span import ExplicitTokenSpans


class LookupWrapperLifecycleBarrier:
    """Allow concurrent lookups while fencing wrapper teardown.

    An inner client's close barrier cannot see work a wrapper performs after
    the inner lookup returns. Wrappers use this separate admission counter so
    close/force-close waits for the complete outer operation without routing
    serving concurrency through an unrelated statistics/configuration lock.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active_lookups = 0
        self._closing = False
        self._closed = False
        self._terminal = False

    def enter_lookup(self) -> None:
        with self._condition:
            if self._closing or self._closed or self._terminal:
                raise RuntimeError("lookup wrapper is closing or closed")
            self._active_lookups += 1

    def exit_lookup(self) -> None:
        with self._condition:
            self._active_lookups -= 1
            if self._active_lookups == 0:
                self._condition.notify_all()

    def begin_close(self, *, terminal: bool) -> bool:
        with self._condition:
            while self._closing:
                self._condition.wait()
            if self._closed:
                return False
            if self._terminal and not terminal:
                raise RuntimeError("lookup wrapper terminal close is pending")
            if terminal:
                # A failed terminal close remains admission-fenced but can be
                # retried by a later terminal manager shutdown attempt.
                self._terminal = True
            self._closing = True
            while self._active_lookups:
                self._condition.wait()
            return True

    def end_close(self, *, success: bool, terminal_failure: bool = False) -> None:
        with self._condition:
            if success:
                self._closed = True
            elif terminal_failure:
                self._terminal = True
            self._closing = False
            self._condition.notify_all()


class LookupClientInterface(metaclass=abc.ABCMeta):
    """Abstract interface for lookup clients."""

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        """
        Lookup the cache for the given lookup ID.

        Args:
            lookup_id: The lookup ID to lookup

        Returns:
            -1 means not found;
            None means ongoing; (this semantic is not supported in sync lookup clients)
            int >= 0 means number of hit tokens
        """
        return None

    def validate_lookup_request(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
    ) -> None:
        """Validate current content against any cached request binding.

        The default is intentionally a no-op for clients whose cache/status
        lifecycle is managed by an external or asynchronous protocol.
        """
        return

    @abc.abstractmethod
    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
    ) -> Optional[int]:
        """
        Perform lookup for the given token IDs.
        Should be called for first lookup and pinning. Subsequent lookups for the same
        request should call lookup_cache instead.

        Caller should handle overlaps between tokens that exist in LMCache
        and tokens that are already computed by the caller.

        Args:
            token_ids: The token IDs to lookup

            lookup_id: The lookup ID to associate with the lookup

            request_configs: The configs of the request,
            includes tags and the other configs

            explicit_spans: Optional exact token-span declaration. Callers
            must check :meth:`supports_explicit_spans` before supplying it.

        Returns:
            The number of tokens that exist inside LMCache.
            None indicates the lookup/prefetch is in progress.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Close the lookup client and clean up resources."""
        raise NotImplementedError

    def force_close(self) -> None:
        """Quiesce the client for terminal engine shutdown.

        Stateful clients whose normal ``close`` can fail without losing
        cleanup ownership override this hook. Other implementations have no
        separate ownership lifecycle and can use their ordinary close path.
        This hook is for terminal manager shutdown only; dynamic lookup-client
        replacement must call :meth:`close`.
        """
        self.close()

    def supports_producer_reuse(self) -> bool:
        """
        Return whether this lookup client supports producer KV cache reuse.

        Returns:
            True if producer reuse is supported, False otherwise
        """
        return False

    def supports_explicit_spans(self) -> bool:
        """Return whether lookup accepts an explicit token-span declaration."""
        return False

    def clear_lookup_status(self, lookup_id: str) -> None:
        """
        Clear temporary lookup status for a given lookup ID.

        Args:
            lookup_id: The lookup ID whose status needs to be cleared.
        """
        return

    def finish_lookup(self, lookup_id: str) -> None:
        """Retire all local state after the request lifecycle finishes.

        ``clear_lookup_status`` may intentionally preserve cleanup identity
        across scheduling and allocation. Clients that retain such identity
        override this lifecycle hook.
        """
        self.clear_lookup_status(lookup_id)

    def transfer_lookup(self, lookup_id: str) -> None:
        """Transfer cleanup ownership to an allocated worker request.

        The scheduler must retire its local status and correlation state
        without cancelling worker resources. The worker receives its separate
        per-attempt lookup ID through connector metadata.
        """
        self.transfer_lookups([lookup_id])

    def transfer_lookups(self, lookup_ids: list[str]) -> None:
        """Atomically transfer a batch when the client retains ownership.

        The default client has no correlation state beyond status, and these
        local dictionary-style clears are non-failing. Stateful clients
        override this method to preflight the full batch before mutation.
        """
        for lookup_id in lookup_ids:
            self.clear_lookup_status(lookup_id)

    def cancel_lookup(self, lookup_id: str) -> None:
        """Cancel a lookup and release any request-owned worker resources.

        Clients without worker-side resources fall back to clearing only their
        local status. RPC and bypass clients override this method to unpin the
        engine that executed the lookup.
        """
        self.clear_lookup_status(lookup_id)
