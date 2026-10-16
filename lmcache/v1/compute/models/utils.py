# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict

# Third Party
from torch import nn

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def infer_model_from_vllm(vllm_model, blender, enable_sparse: bool = False):
    model_name = type(vllm_model).__name__
    if model_name == "LlamaForCausalLM":
        # First Party
        from lmcache.v1.compute.models.llama import LMCLlamaModel

        return LMCLlamaModel(vllm_model, blender, enable_sparse)
    elif model_name == "Qwen2ForCausalLM":
        # First Party
        from lmcache.v1.compute.models.llama import LMCLlamaModel

        return LMCLlamaModel(vllm_model, blender, enable_sparse)
    elif model_name == "Qwen3ForCausalLM":
        # First Party
        from lmcache.v1.compute.models.qwen3 import LMCQwen3Model

        return LMCQwen3Model(vllm_model, blender, enable_sparse)
    else:
        # TODO(Jiayi): Add support for more models
        raise NotImplementedError(
            f"Model type {model_name} is not supported in LMCache."
        )


class VLLMModelTracker:
    _vllm_models: Dict[str, nn.Module] = {}

    @classmethod
    def register_model(
        cls,
        instance_id: str,
        vllm_model: nn.Module,
    ) -> None:
        """
        Register a vllm model by instance_id.

        Args:
            instance_id: Engine instance that owns ``vllm_model``.
            vllm_model: The loaded vLLM model used by CacheBlend.

        Raises:
            ValueError: If another model is already registered for the same
                live engine instance.
        """
        logger.info(f"Registering vllm model for {instance_id}")
        if instance_id not in cls._vllm_models:
            cls._vllm_models[instance_id] = vllm_model
        elif cls._vllm_models[instance_id] is vllm_model:
            logger.info("vllm model for %s is already registered", instance_id)
        else:
            raise ValueError(
                f"A different vllm model is already registered for {instance_id}. "
                "Destroy the old LMCache engine before registering a replacement."
            )

    @classmethod
    def get_model(
        cls,
        instance_id: str,
    ) -> nn.Module:
        """
        Get the vllm model by instance_id.
        """
        if instance_id not in cls._vllm_models:
            raise ValueError(f"vllm model for {instance_id} not found.")
        return cls._vllm_models[instance_id]

    @classmethod
    def unregister_model(cls, instance_id: str) -> bool:
        """Remove the model retained for an engine instance.

        Args:
            instance_id: Engine instance whose model should be forgotten.

        Returns:
            ``True`` when an entry existed, otherwise ``False``.
        """
        model = cls._vllm_models.pop(instance_id, None)
        if model is None:
            return False
        logger.info("Unregistered vllm model for %s", instance_id)
        return True
