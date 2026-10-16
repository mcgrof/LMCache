# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lookup_client.abstract_client import (
    LookupClientInterface,
    LookupWrapperLifecycleBarrier,
)
from lmcache.v1.token_span import ExplicitTokenSpans

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


"""
HitLimitLookupClient now is used for test, when lookup is called, cal the cache hit,
- if the cache hit <= (1 - hit_miss_ratio), direct return the result
- if the cache hit > (1 - hit_miss_ratio), re-compute the result by hit_miss_ratio
"""


class HitLimitLookupClient(LookupClientInterface):
    def __init__(
        self, actual_lookup_client: LookupClientInterface, config: "LMCacheEngineConfig"
    ):
        assert config.hit_miss_ratio is not None and 0 <= config.hit_miss_ratio <= 1
        self.actual_lookup_client = actual_lookup_client
        self.config = config
        self._outer_lifecycle = LookupWrapperLifecycleBarrier()
        logger.info(
            "create HitLimitLookupClient succeed, "
            "the hit ratio upper is %s, chunk size is %s",
            1 - config.hit_miss_ratio,
            config.chunk_size,
        )

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        return self.actual_lookup_client.lookup_cache(lookup_id)

    def validate_lookup_request(
        self,
        token_ids: Union["torch.Tensor", list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
    ) -> None:
        self.actual_lookup_client.validate_lookup_request(
            token_ids,
            lookup_id,
            request_configs,
            explicit_spans,
        )

    def lookup(
        self,
        token_ids: Union["torch.Tensor", list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
        explicit_spans: Optional[ExplicitTokenSpans] = None,
    ) -> Optional[int]:
        if explicit_spans is not None:
            raise NotImplementedError(
                "hit-limit lookup cannot preserve explicit span results"
            )
        self._outer_lifecycle.enter_lookup()
        try:
            # get real hit tokens
            result = self.actual_lookup_client.lookup(
                token_ids,
                lookup_id,
                request_configs,
            )
            if result is not None:
                total_tokens_length = len(token_ids)
                assert result <= total_tokens_length
                current_hit_ratio = 0.0
                if total_tokens_length > 0:
                    current_hit_ratio = result / total_tokens_length
                # limit the hit tokens
                hit_ratio_upper = 1 - self.config.hit_miss_ratio
                if current_hit_ratio > hit_ratio_upper:
                    origin_result = result
                    # align to chunk size
                    chunk_size = self.config.chunk_size
                    new_result = (
                        int(total_tokens_length * hit_ratio_upper)
                        // chunk_size
                        * chunk_size
                    )
                    # check again
                    result = min(result, new_result)
                    logger.debug(
                        f"hit ratio upper: {hit_ratio_upper} is smaller than "
                        f"the real hit ratio {current_hit_ratio}, "
                        f"the origin result is {origin_result}, "
                        f"the new result is {new_result}, the final result is {result}"
                    )
            return result
        finally:
            self._outer_lifecycle.exit_lookup()

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.actual_lookup_client.clear_lookup_status(lookup_id)

    def finish_lookup(self, lookup_id: str) -> None:
        self.actual_lookup_client.finish_lookup(lookup_id)

    def transfer_lookup(self, lookup_id: str) -> None:
        self.actual_lookup_client.transfer_lookup(lookup_id)

    def transfer_lookups(self, lookup_ids: list[str]) -> None:
        self.actual_lookup_client.transfer_lookups(lookup_ids)

    def cancel_lookup(self, lookup_id: str) -> None:
        self.actual_lookup_client.cancel_lookup(lookup_id)

    def supports_producer_reuse(self) -> bool:
        return self.actual_lookup_client.supports_producer_reuse()

    def supports_explicit_spans(self) -> bool:
        return False

    def close(self) -> None:
        if not self._outer_lifecycle.begin_close(terminal=False):
            return
        try:
            self.actual_lookup_client.close()
        except Exception:
            self._outer_lifecycle.end_close(success=False)
            raise
        self._outer_lifecycle.end_close(success=True)

    def force_close(self) -> None:
        if not self._outer_lifecycle.begin_close(terminal=True):
            return
        try:
            self.actual_lookup_client.force_close()
        except Exception:
            self._outer_lifecycle.end_close(
                success=False,
                terminal_failure=True,
            )
            raise
        self._outer_lifecycle.end_close(success=True)
