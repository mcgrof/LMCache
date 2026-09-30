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
# Where the writer answers the acknowledgements that release its extents,
# and the address it tells the reader to use. A wildcard is a bind, not an
# address, so the two are separate; loopback is enough for one host.
ack_port=${LMCACHE_RAW_ACK_PORT:-7600}
ack_bind_host=${LMCACHE_RAW_ACK_BIND_HOST:-127.0.0.1}
ack_advertise_host=${LMCACHE_RAW_ACK_ADVERTISE_HOST:-127.0.0.1}

# Names this producer/consumer pair's run, and both nodes carry it. The
# writer binds the first consumer incarnation that acknowledges under a
# session and refuses any other one for it, so a consumer restarted on its
# own cannot release holds for reads it never made. Restarting the whole
# group is a new session and rebinds, which is why this defaults to a value
# that changes per launch and has to be set explicitly -- to the same value
# on both nodes -- for a pair started separately.
if [ -z "${LMCACHE_STORAGE_PD_SESSION:-}" ]; then
    echo "LMCACHE_STORAGE_PD_SESSION is unset." >&2
    echo "Set it to the same value on the writer and the reader; changing" >&2
    echo "it is how a whole-group restart is distinguished from a reader" >&2
    echo "that came back alone." >&2
    exit 1
fi
export LMCACHE_STORAGE_PD_SESSION

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

# Two spellings reach the same route. LMCACHE_PD_DATA_PATH=raw_block says it
# through LMCache's own prefill/decode switch, which derives the storage role
# from LMCACHE_PD_ROLE and requires the plugin, so neither is repeated here.
# Any other value keeps the older spelling, where the plugin settings carry
# everything and the prefill/decode switch stays off.
pd_data_path=${LMCACHE_PD_DATA_PATH:-raw_block}
if [ "$pd_data_path" = "raw_block" ]; then
    role_settings=""
else
    printf -v role_settings \
        '"rust_raw_block.role":"%s","rust_raw_block.storage_pd_mode":true,' \
        "$raw_role"
fi

printf -v extra_config \
    '{"storage_plugin.raw_block.module_path":"lmcache.v1.storage_backend.plugins.rust_raw_block_backend","storage_plugin.raw_block.class_name":"RustRawBlockBackend","storage_plugin.raw_block.required":true,%s"rust_raw_block.device_path":"%s","rust_raw_block.slot_bytes":%s,"rust_raw_block.io_engine":"io_uring","rust_raw_block.use_odirect":true,"rust_raw_block.use_uring_cmd":false,"rust_raw_block.gpu_buffer_bytes":%s,"rust_raw_block.require_dmabuf_registration":true,"rust_raw_block.publish_after_put":false,"rust_raw_block.publish_min_interval_ms":0,"rust_raw_block.meta_enable_periodic":false,"rust_raw_block.index_refresh_min_ms":1,"rust_raw_block.publication_adopt_timeout_ms":30000,"rust_raw_block.status_send_timeout_s":5,"rust_raw_block.ack_listen_port":%s,"rust_raw_block.ack_listen_host":"%s","rust_raw_block.ack_advertise_host":"%s","rust_raw_block.require_extent_reuse":true}' \
    "$role_settings" "$device" "$slot_bytes" "$gpu_buffer_bytes" \
    "$ack_port" "$ack_bind_host" "$ack_advertise_host"

# Both nodes must derive the same key for the same tokens, in separate
# processes. Two things decide that, and both have to match.
#
# The algorithm: name one, rather than leaving it at the builtin fallback,
# so the derivation is a documented function instead of whatever the
# interpreter provides.
#
# The seed: measured, the keys still follow PYTHONHASHSEED under either
# algorithm, because the chunk hash chain starts from a value the serving
# engine derives from it. So the seed is part of the contract, not a
# leftover for other hashing. A deterministic function may legitimately
# take a configured seed; what matters is that both nodes agree on it and
# that a mismatch is refused rather than read as a miss.
export LMCACHE_PRE_CACHING_HASH_ALGORITHM=${LMCACHE_PRE_CACHING_HASH_ALGORITHM:-sha256_cbor}
export PYTHONHASHSEED=0
export LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE:-256}
export LMCACHE_LOCAL_CPU=false
export LMCACHE_MAX_LOCAL_CPU_SIZE=0
export LMCACHE_PD_DATA_PATH=$pd_data_path
if [ "$pd_data_path" = "raw_block" ]; then
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
