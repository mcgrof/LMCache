#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
role=${1:?"usage: launch_vllm.sh <writer|reader> [model]"}
model=${2:-meta-llama/Llama-3.1-8B-Instruct}
device=${LMCACHE_RAW_DEVICE:?"set LMCACHE_RAW_DEVICE to a dedicated by-id path"}
gpu_buffer_bytes=${LMCACHE_GPU_BUFFER_BYTES:-4294967296}
# One slot must hold a full KV chunk plus its header; size it for the model.
slot_bytes=${LMCACHE_RAW_SLOT_BYTES:-9437184}

"$script_dir/verify_raw_device.sh" "$device"

case "$role" in
    writer)
        service_port=${PREFILLER_PORT:-7100}
        visible_device=${PREFILLER_DEVICE_ID:-0}
        kv_role=kv_producer
        raw_role=writer
        pd_role=sender
        connector_extra='{"discard_partial_chunks":false}'
        ;;
    reader)
        service_port=${DECODER_PORT:-7200}
        visible_device=${DECODER_DEVICE_ID:-1}
        kv_role=kv_consumer
        raw_role=reader
        pd_role=receiver
        connector_extra='{"discard_partial_chunks":false,"skip_last_n_tokens":1}'
        ;;
    *)
        echo "role must be writer or reader" >&2
        exit 1
        ;;
esac

printf -v extra_config \
    '{"storage_plugin.raw_block.module_path":"lmcache.v1.storage_backend.plugins.rust_raw_block_backend","storage_plugin.raw_block.class_name":"RustRawBlockBackend","storage_plugin.raw_block.required":true,"rust_raw_block.device_path":"%s","rust_raw_block.role":"%s","rust_raw_block.slot_bytes":%s,"rust_raw_block.storage_pd_mode":true,"rust_raw_block.io_engine":"io_uring","rust_raw_block.use_odirect":true,"rust_raw_block.use_uring_cmd":false,"rust_raw_block.gpu_buffer_bytes":%s,"rust_raw_block.require_dmabuf_registration":true,"rust_raw_block.publish_after_put":false,"rust_raw_block.publish_min_interval_ms":0,"rust_raw_block.meta_enable_periodic":false,"rust_raw_block.index_refresh_min_ms":1,"rust_raw_block.publication_adopt_timeout_ms":30000,"rust_raw_block.status_send_timeout_s":5}' \
    "$device" "$raw_role" "$slot_bytes" "$gpu_buffer_bytes"

export PYTHONHASHSEED=0
export LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE:-256}
export LMCACHE_LOCAL_CPU=false
export LMCACHE_MAX_LOCAL_CPU_SIZE=0
# Two spellings reach the same route. LMCACHE_PD_DATA_PATH=raw_block says it
# through LMCache's own prefill/decode switch, which then derives the storage
# role from LMCACHE_PD_ROLE and requires the plugin. Leaving it unset keeps
# the older spelling, where the plugin settings carry everything and the
# prefill/decode switch stays off.
export LMCACHE_PD_DATA_PATH=${LMCACHE_PD_DATA_PATH:-raw_block}
if [ "$LMCACHE_PD_DATA_PATH" = "raw_block" ]; then
    export LMCACHE_ENABLE_PD=true
else
    export LMCACHE_ENABLE_PD=false
fi
export LMCACHE_PD_ROLE=$pd_role
export LMCACHE_PD_PROXY_HOST=${LMCACHE_PD_PROXY_HOST:-localhost}
export LMCACHE_PD_PROXY_PORT=${LMCACHE_PD_PROXY_PORT:-7500}
export LMCACHE_SAVE_UNFULL_CHUNK=true
export LMCACHE_ENABLE_ASYNC_LOADING=false
export LMCACHE_STORAGE_PLUGINS=raw_block
export LMCACHE_EXTRA_CONFIG=$extra_config

exec env CUDA_VISIBLE_DEVICES="$visible_device" \
    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    vllm serve "$model" \
    --port "$service_port" \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-transfer-config \
    "{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"$kv_role\",\"kv_connector_extra_config\":$connector_extra}"
