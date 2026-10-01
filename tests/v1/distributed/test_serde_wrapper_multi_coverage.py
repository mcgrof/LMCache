# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for complete multi-output serde coverage."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.serde_wrapper import SerdeL2AdapterWrapper
from lmcache.v1.distributed.serde import SerdeConfig, create_serde_processor
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)


def _make_grouped_object(group_count: int) -> TensorMemoryObj:
    """Create a tiny grouped BF16 object with distinct group contents."""
    shapes = [torch.Size([4])] * group_count
    dtypes = [torch.bfloat16] * group_count
    size = sum(
        shape.numel() * dtype.itemsize
        for shape, dtype in zip(shapes, dtypes, strict=True)
    )
    obj = TensorMemoryObj(
        raw_data=torch.zeros(size, dtype=torch.uint8),
        metadata=MemoryObjMetadata(
            shape=shapes[0],
            dtype=dtypes[0],
            address=0,
            phy_size=size,
            ref_count=1,
            pin_count=0,
            fmt=MemoryFormat.KV_2LTD,
            shapes=shapes,
            dtypes=dtypes,
        ),
        parent_allocator=None,
    )
    for index in range(group_count):
        tensor = obj.get_tensor(index)
        assert tensor is not None
        tensor.fill_(index + 1)
    return obj


@pytest.mark.parametrize(
    ("serde_type", "group_count"),
    [
        pytest.param("asym_k16_v8_v_only", 2, id="standalone-v-only"),
        pytest.param("asym_k16_v8", 4, id="second-kv-pair"),
    ],
)
def test_incomplete_multi_output_mapping_fails_store(
    tmp_path, serde_type: str, group_count: int
) -> None:
    """A wrapper must not report success when a serde omits parent groups."""
    l1 = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1 << 20,
                use_lazy=False,
                init_size_in_bytes=1 << 20,
            )
        )
    )
    wrapper = SerdeL2AdapterWrapper(
        FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path))),
        create_serde_processor(SerdeConfig(type=serde_type)),
        l1,
    )
    obj = _make_grouped_object(group_count)
    try:
        key = ObjectKey(
            chunk_hash=b"\x91" * 32,
            model_name="coverage-test",
            kv_rank=0,
        )
        task_id = wrapper.submit_store_task([key], [obj])

        result = wrapper.pop_completed_store_tasks()[task_id]
        assert not result.is_successful()
        assert list(tmp_path.glob("*.data")) == []
    finally:
        obj.ref_count_down()
        wrapper.close()
        l1.close()
