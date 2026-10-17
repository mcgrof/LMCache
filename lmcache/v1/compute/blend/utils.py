# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Dict

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.models.utils import VLLMModelTracker

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.gpu_connector import GPUConnectorInterface

logger = init_logger(__name__)


class LMCBlenderBuilder:
    _blenders: Dict[str, LMCBlender] = {}

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        cache_engine: "LMCacheEngine",
        gpu_connector: "GPUConnectorInterface",
        config: "LMCacheEngineConfig",
    ) -> LMCBlender:
        """
        Get or create a blender for the given instance_id.
        """

        if instance_id not in cls._blenders:
            logger.info("Creating blender for %s", instance_id)
            vllm_model = VLLMModelTracker.get_model(instance_id)
            blender = LMCBlender(
                cache_engine=cache_engine,
                gpu_connector=gpu_connector,
                vllm_model=vllm_model,
                config=config,
            )
            cls._blenders[instance_id] = blender
        else:
            logger.info(
                "Blender for %s already exists, returning the original one.",
                instance_id,
            )
        return cls._blenders[instance_id]

    @classmethod
    def get(
        cls,
        instance_id: str,
    ) -> LMCBlender:
        """
        Get the blender by instance_id.
        """
        if instance_id not in cls._blenders:
            raise ValueError(f"Blender for {instance_id} not found.")
        return cls._blenders[instance_id]

    @classmethod
    def drop(cls, instance_id: str) -> bool:
        """Remove the blender retained for an engine instance.

        Args:
            instance_id: Engine instance whose blender should be forgotten.

        Returns:
            ``True`` when an entry existed, otherwise ``False``.
        """
        blender = cls._blenders.pop(instance_id, None)
        if blender is None:
            return False
        logger.info("Dropped blender for %s", instance_id)
        return True
